"""MaxEnt species modeling pipeline extracted from maxent_model.ipynb."""

from __future__ import annotations

import glob
import json
import os
import re
import warnings
from contextlib import contextmanager

import elapid
import geopandas as gpd
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import rasterio
import seaborn as sns
from elapid import GeographicKFold
from joblib import Parallel, delayed
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from scipy import stats
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    auc,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

# Worker-process cache for batch runs (populated via init_worker_cache).
_BG_CACHE: dict[str, pl.DataFrame] = {}
_AOI_CACHE: gpd.GeoDataFrame | None = None

LABEL_MAP = {
    "0": "water",
    "1": "trees",
    "2": "grass",
    "3": "flooded_vegetation",
    "4": "crops",
    "5": "shrub_and_scrub",
    "6": "built",
    "7": "bare",
    "8": "snow_and_ice",
}

_RENAME_PATTERN = re.compile(r"^dw_class_(\d)_pct_(\d+)m$")

METADATA_DROP_COLS = [
    "basisofrecord",
    "coordinateuncertaintyinmeters",
    "species",
    "rand",
    "date",
    "day",
    "obs_date",
    "observation_date",
    "protocol_name",
    "scientific_name",
    "year",
    "label",
    "srad_mean_mean",
    "swe_mean_mean",
    "vp_mean_mean",
]

STATIC_PREDICTOR_DROPS = [
    "dayl_10000m",
    "dw_class_8_pct_10000m",
    "dw_class_8_pct_500m",
]

DEFAULT_REG_VALUES = [0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0]
SPARSE_REG_VALUES = [1.5, 2.0, 3.0, 4.0, 5.0, 6.0]

SEASON_LABEL_TO_CODE = {"winter": 0, "spring": 1, "summer": 2, "fall": 3}
REPRESENTATIVE_MONTH_FOR_SEASON = {"winter": 1, "spring": 4, "summer": 7, "fall": 10}
INJECTABLE_INFERENCE_FEATURES = frozenset({"month", "season"})


def month_to_season(month: int) -> int:
    """Map calendar month to GEE season code (0=winter … 3=fall)."""
    if month in (12, 1, 2):
        return 0
    if month <= 5:
        return 1
    if month <= 8:
        return 2
    return 3


def ensure_bird_temporal_pl(df: pl.DataFrame) -> pl.DataFrame:
    """Derive month from obs_date when needed and sync season from month."""
    if "month" not in df.columns:
        if "obs_date" not in df.columns:
            raise ValueError("birds require month or obs_date")
        df = df.with_columns(
            pl.col("obs_date").str.to_datetime(strict=False).dt.month().alias("month")
        )
    return df.with_columns(
        pl.when(pl.col("month").is_in([12, 1, 2]))
        .then(0)
        .when(pl.col("month") <= 5)
        .then(1)
        .when(pl.col("month") <= 8)
        .then(2)
        .otherwise(3)
        .cast(pl.Int64)
        .alias("season")
    )


def rename_col(column_name: str) -> str:
    match = _RENAME_PATTERN.match(column_name)
    if not match:
        return column_name
    class_id, scale = match.group(1), match.group(2)
    label = LABEL_MAP.get(class_id)
    if label is None:
        return column_name
    return f"{label}_{scale}m"


def prepare_predictors(
    data: pd.DataFrame, excel_group: str
) -> tuple[pd.DataFrame, list[str]]:
    """Build model matrix with group-specific temporal covariates."""
    df = data.copy()
    group = excel_group.strip().lower()

    if group == "birds":
        if "month" not in df.columns:
            raise ValueError("month column required for birds")
        if "season" not in df.columns:
            raise ValueError("season column required for birds")
        df["month"] = df["month"].astype(int)
        df["season"] = df["season"].astype(int)
        categorical_features = ["month", "season"]
    else:
        df = df.drop(columns=["month", "season"], errors="ignore")
        categorical_features = []

    X = df.drop(columns=METADATA_DROP_COLS, errors="ignore").copy()
    X = X.fillna(0)
    X.columns = [col.replace("_mean_mean", "_10000m") for col in X.columns]
    X = X.drop(columns=STATIC_PREDICTOR_DROPS, errors="ignore")
    # elapid 1.0.x only treats pandas category columns as categorical for DataFrames.
    for col in categorical_features:
        if col in X.columns:
            X[col] = X[col].astype(int).astype("category")
    return X, categorical_features


def _categorical_indices(
    X: pd.DataFrame, categorical_features: list[str] | None
) -> list[int] | None:
    """Column indices for elapid fit(categorical=...) — elapid 1.0.x API."""
    if not categorical_features:
        return None
    indices = [X.columns.get_loc(name) for name in categorical_features if name in X.columns]
    return indices or None


def load_model_config(outputdir: str, sp: str) -> dict:
    """Load saved model config, falling back to predictors list for legacy models."""
    sp = sp.lower().replace(" ", "_")
    config_path = os.path.join(outputdir, f"model_config_{sp}.json")
    if os.path.isfile(config_path):
        with open(config_path, encoding="utf-8") as handle:
            return json.load(handle)

    predictors_path = os.path.join(outputdir, f"predictors_{sp}.txt")
    with open(predictors_path, encoding="utf-8") as handle:
        predictor_columns = [line.strip() for line in handle if line.strip()]
    return {
        "scientific_name": sp,
        "excel_group": "birds",
        "spgroup": "avian",
        "categorical_features": [],
        "predictor_columns": predictor_columns,
    }


def deploy_model_artifact_paths(models_dir: str, sp: str) -> dict[str, str]:
    """Paths to model artifacts in flat BioAPI deploy models/ directory."""
    sp = sp.lower().replace(" ", "_")
    return {
        "model_path": os.path.join(models_dir, f"elapid_maxent_model_tuned_{sp}.pkl"),
        "metrics_path": os.path.join(models_dir, f"accuracy_tuned_{sp}.csv"),
        "rangefile_path": os.path.join(models_dir, f"convex_hull_{sp}.json"),
        "predictors_path": os.path.join(models_dir, f"predictors_{sp}.txt"),
        "config_path": os.path.join(models_dir, f"model_config_{sp}.json"),
    }


def load_model_config_from_deploy(models_dir: str, sp: str) -> dict:
    """Load model config from flat BioAPI deploy models/ directory."""
    return load_model_config(models_dir, sp)


def normalize_band_name(name: str) -> str:
    return (name or "").strip().lower().replace("_", "").replace(" ", "")


def match_band_index(feature: str, raster_band_names: list[str]) -> int | None:
    """Return 0-based band index for a training feature, or None if not in raster."""
    target = normalize_band_name(feature)
    for i, nm in enumerate(raster_band_names):
        if normalize_band_name(nm) == target:
            return i
    return None


def inference_injected_values(season_label: str, features: list[str]) -> dict[str, float]:
    """Constant temporal covariate values for seasonal inference tiles."""
    label = season_label.strip().lower()
    if label not in SEASON_LABEL_TO_CODE:
        raise ValueError(
            f"Unknown season label {season_label!r}; expected one of {sorted(SEASON_LABEL_TO_CODE)}"
        )
    values: dict[str, float] = {}
    for feat in features:
        if feat == "month":
            values[feat] = float(REPRESENTATIVE_MONTH_FOR_SEASON[label])
        elif feat == "season":
            values[feat] = float(SEASON_LABEL_TO_CODE[label])
        else:
            raise ValueError(
                f"Cannot inject feature {feat!r} at inference; only {sorted(INJECTABLE_INFERENCE_FEATURES)} "
                "are supported as injected constants."
            )
    return values


def resolve_inference_band_mapping(
    training_columns: list[str],
    raster_band_names: list[str],
    season_label: str,
) -> tuple[dict[str, int], dict[str, float], list[str], list[str]]:
    """Map raster bands and injected constants for inference.

    Returns:
        band_mapping: feature -> 1-based rasterio band index
        injected_values: feature -> constant value for non-raster predictors
        raster_features: training columns read from raster bands (in read order)
        injected_features: training columns filled with constants
    """
    band_mapping: dict[str, int] = {}
    to_inject: list[str] = []

    for feature in training_columns:
        band_idx = match_band_index(feature, raster_band_names)
        if band_idx is not None:
            band_mapping[feature] = band_idx + 1
        elif feature in INJECTABLE_INFERENCE_FEATURES:
            to_inject.append(feature)
        else:
            raise ValueError(
                f"Feature '{feature}' not found among raster band descriptions "
                f"{raster_band_names!r}. Set band descriptions to match training column names, "
                "or ensure month/season are listed in model_config categorical_features."
            )

    injected_values = inference_injected_values(season_label, to_inject)
    raster_features = [f for f in training_columns if f in band_mapping]
    injected_features = [f for f in training_columns if f in injected_values]
    return band_mapping, injected_values, raster_features, injected_features


def assemble_inference_matrix(
    X_raster: np.ndarray,
    training_columns: list[str],
    raster_features: list[str],
    injected_values: dict[str, float],
) -> np.ndarray:
    """Combine raster bands and injected constants in training column order."""
    n_pixels = X_raster.shape[0]
    X = np.full((n_pixels, len(training_columns)), np.nan, dtype=np.float32)
    col_lookup = {name: idx for idx, name in enumerate(training_columns)}
    for j, feat in enumerate(raster_features):
        X[:, col_lookup[feat]] = X_raster[:, j]
    for feat, value in injected_values.items():
        X[:, col_lookup[feat]] = value
    return X


def prepare_inference_batch(
    X: np.ndarray,
    training_columns: list[str],
    categorical_features: list[str] | None,
) -> pd.DataFrame:
    """Build a prediction batch with the same dtypes used during training."""
    df = pd.DataFrame(X, columns=training_columns)
    for col in categorical_features or []:
        if col in df.columns:
            df[col] = df[col].astype(int).astype("category")
    return df


FULL_RASTER_BYTE_THRESHOLD = 500 * 1024 * 1024
INFERENCE_STACK_NODATA = -9999.0

_gdal_configured = False


def _ensure_gdal_quiet() -> None:
    """Silence GDAL FutureWarning and configure legacy exception handling once."""
    global _gdal_configured
    if _gdal_configured:
        return
    from osgeo import gdal

    gdal.DontUseExceptions()
    warnings.filterwarnings(
        "ignore",
        message="Neither gdal.UseExceptions.*",
        category=FutureWarning,
        module=r"osgeo\.gdal",
    )
    _gdal_configured = True


@contextmanager
def _gdal_quiet():
    """Suppress GDAL stderr noise during warp/VRT operations."""
    _ensure_gdal_quiet()
    from osgeo import gdal

    gdal.PushErrorHandler("CPLQuietErrorHandler")
    old_cpl_log = os.environ.get("CPL_LOG")
    os.environ["CPL_LOG"] = "OFF"
    try:
        yield
    finally:
        gdal.PopErrorHandler()
        if old_cpl_log is None:
            os.environ.pop("CPL_LOG", None)
        else:
            os.environ["CPL_LOG"] = old_cpl_log


def inference_stack_cache_path(spdir: str, season_label: str) -> str:
    """Path for cached warped covariate stack used at inference."""
    season = season_label.strip().lower()
    return os.path.join(spdir, f"inference_stack_{season}.tif")


def _inference_stack_source_paths(
    raster_dir: str,
    season_label: str,
    pattern: str = "all_months*.tif",
) -> list[str]:
    season = season_label.strip().lower()
    paths = sorted(glob.glob(os.path.join(raster_dir, pattern)))
    season_tif = os.path.join(raster_dir, f"{season}.tif")
    if not os.path.isfile(season_tif):
        raise FileNotFoundError(f"Could not find {season_tif}")
    paths.append(season_tif)
    return paths


def _stack_cache_is_fresh(
    cache_path: str,
    source_paths: list[str],
    rangefile_path: str,
) -> bool:
    if not os.path.isfile(cache_path):
        return False
    cache_mtime = os.path.getmtime(cache_path)
    for path in [*source_paths, rangefile_path]:
        if not os.path.isfile(path) or os.path.getmtime(path) > cache_mtime:
            return False
    return True


def build_inference_covariate_stack(
    raster_dir: str,
    rangefile_path: str,
    season_label: str,
    cache_path: str,
    *,
    pattern: str = "all_months*.tif",
    force: bool = False,
    verbose: bool = False,
) -> str:
    """Build or reuse a flat GeoTIFF covariate stack cropped to the species hull."""
    from osgeo import gdal

    _ensure_gdal_quiet()
    source_paths = _inference_stack_source_paths(raster_dir, season_label, pattern)
    if not force and _stack_cache_is_fresh(cache_path, source_paths, rangefile_path):
        if verbose:
            print(f"Using cached inference stack: {cache_path}")
        return cache_path

    season = season_label.strip().lower()
    spectral_tifs = source_paths[:-1]
    season_tif = source_paths[-1]
    if verbose:
        print(f"Building inference stack from {len(spectral_tifs)} spectral + 1 seasonal raster(s)")

    mosaic_vrt = f"/vsimem/inference_{season}_mosaic.vrt"
    stacked_vrt = f"/vsimem/inference_{season}_stacked.vrt"
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    warp_opts = gdal.WarpOptions(
        format="GTiff",
        cutlineDSName=rangefile_path,
        cropToCutline=True,
        dstNodata=INFERENCE_STACK_NODATA,
        outputType=gdal.GDT_Float32,
        creationOptions=["COMPRESS=LZW", "TILED=YES", "BIGTIFF=IF_SAFER"],
    )

    with _gdal_quiet():
        mosaic = gdal.BuildVRT(mosaic_vrt, spectral_tifs, separate=False)
        if mosaic is None:
            raise RuntimeError("BuildVRT failed for spectral mosaic.")
        mosaic.FlushCache()

        stacked = gdal.BuildVRT(stacked_vrt, [mosaic_vrt, season_tif], separate=True)
        if stacked is None:
            raise RuntimeError("BuildVRT failed for stacked covariates.")
        stacked.FlushCache()

        result = gdal.Warp(cache_path, stacked_vrt, options=warp_opts)
        if result is None:
            raise RuntimeError(f"gdal.Warp failed writing {cache_path}")
        result.FlushCache()

        band_names = read_inference_band_names(raster_dir, season_label)
        for band_idx, name in enumerate(band_names, start=1):
            band = result.GetRasterBand(band_idx)
            band.SetDescription(name)
            band.SetNoDataValue(INFERENCE_STACK_NODATA)
        result.FlushCache()
        result = None

    if verbose:
        print(f"Wrote inference stack: {cache_path}")
    return cache_path


def read_inference_band_names(band_source_dir: str, season_label: str) -> list[str]:
    """Read GDAL band descriptions from spectral + seasonal env inference rasters."""
    from osgeo import gdal

    _ensure_gdal_quiet()
    spectral_tifs = sorted(glob.glob(os.path.join(band_source_dir, "*all_months*.tif")))
    if not spectral_tifs:
        raise FileNotFoundError(
            f"No *all_months*.tif files found in {band_source_dir} to read band descriptions."
        )
    ds = gdal.Open(spectral_tifs[0])
    band_names = [ds.GetRasterBand(i).GetDescription() for i in range(1, ds.RasterCount + 1)]

    env_tifs = sorted(glob.glob(os.path.join(band_source_dir, f"*{season_label}.tif")))
    if not env_tifs:
        raise FileNotFoundError(
            f"No *{season_label}.tif files found in {band_source_dir} to read band descriptions."
        )
    ds = gdal.Open(env_tifs[0])
    band_names.extend(ds.GetRasterBand(i).GetDescription() for i in range(1, ds.RasterCount + 1))
    return band_names


def _predict_model_batch(
    model,
    X: np.ndarray,
    training_columns: list[str],
    categorical_features: list[str] | None,
) -> np.ndarray:
    """Run model.predict on a batch, skipping DataFrame when no categoricals."""
    if categorical_features:
        batch = prepare_inference_batch(X, training_columns, categorical_features)
        return model.predict(batch).astype(np.float32)
    return model.predict(X).astype(np.float32)


def _spatial_valid_mask(
    data: np.ndarray,
    *,
    skip_geometry_mask: bool,
    hull_geom,
    window_transform,
    height: int,
    width: int,
    stack_nodata: float = INFERENCE_STACK_NODATA,
) -> np.ndarray:
    """Return flat boolean mask of pixels to predict."""
    if skip_geometry_mask:
        return (~np.all(np.isclose(data, stack_nodata), axis=0)).flatten()
    from rasterio import features

    mask = features.geometry_mask(
        hull_geom,
        out_shape=(height, width),
        transform=window_transform,
        invert=True,
    )
    return mask.flatten()


def _load_inference_threshold(metrics_path: str) -> float:
    metrics = pd.read_csv(metrics_path, header=None, index_col=0)
    if "BestThreshold" in metrics.index:
        return float(metrics.loc["BestThreshold"][1])
    if "Threshold" in metrics.index:
        return float(metrics.loc["Threshold"][1])
    raise ValueError("BestThreshold not found in metrics CSV.")


def _write_prediction_window(
    dst_prob,
    dst_bin,
    preds_prob: np.ndarray,
    preds_bin: np.ndarray,
    window,
    height: int,
    width: int,
    nodata_value: float,
) -> None:
    prob_block = np.where(
        np.isnan(preds_prob.reshape((height, width))),
        nodata_value,
        preds_prob.reshape((height, width)),
    )
    bin_block = np.where(
        np.isnan(preds_bin.reshape((height, width))),
        nodata_value,
        preds_bin.reshape((height, width)),
    )
    dst_prob.write(prob_block.astype(np.float32), indexes=1, window=window)
    dst_bin.write(bin_block.astype(np.float32), indexes=1, window=window)


def _predict_raster_full(
    src,
    model,
    read_indices: list[int],
    training_columns: list[str],
    raster_features: list[str],
    injected_values: dict[str, float],
    categorical_features: list[str] | None,
    best_threshold: float,
    dst_prob,
    dst_bin,
    *,
    hull_geom,
    skip_geometry_mask: bool,
    batch_size: int,
    nodata_value: float,
    sample_per_tile: int,
) -> tuple[int, int, int, list[float]]:
    data = src.read(indexes=read_indices, out_dtype="float32")
    height, width = data.shape[1], data.shape[2]
    spatial_valid = _spatial_valid_mask(
        data,
        skip_geometry_mask=skip_geometry_mask,
        hull_geom=hull_geom,
        window_transform=src.transform,
        height=height,
        width=width,
    )

    X_raster = np.moveaxis(data, 0, -1).reshape(-1, len(raster_features))
    X = assemble_inference_matrix(
        X_raster, training_columns, raster_features, injected_values
    )

    preds_prob = np.full(X.shape[0], np.nan, dtype=np.float32)
    preds_bin = np.full(X.shape[0], np.nan, dtype=np.float32)
    valid = (~np.isnan(X).any(axis=1)) & spatial_valid
    valid_idx = np.where(valid)[0]
    total_valid = int(valid.sum())
    total_nan = int((~valid).sum())
    total_above_thr = 0
    sampled_probs: list[float] = []

    if valid_idx.size > 0:
        for start in range(0, valid_idx.size, batch_size):
            batch_sel = valid_idx[start : start + batch_size]
            prob = _predict_model_batch(
                model, X[batch_sel, :], training_columns, categorical_features
            )
            binary = (prob >= best_threshold).astype(np.float32)
            preds_prob[batch_sel] = prob
            preds_bin[batch_sel] = binary
        total_above_thr = int(np.nansum(preds_bin == 1))
        k = min(sample_per_tile, preds_prob[valid].size)
        if k > 0:
            sel = np.random.choice(preds_prob[valid].size, size=k, replace=False)
            sampled_probs.extend(preds_prob[valid][sel].tolist())

    prob_block = np.where(
        np.isnan(preds_prob.reshape((height, width))),
        nodata_value,
        preds_prob.reshape((height, width)),
    )
    bin_block = np.where(
        np.isnan(preds_bin.reshape((height, width))),
        nodata_value,
        preds_bin.reshape((height, width)),
    )
    dst_prob.write(prob_block.astype(np.float32), indexes=1)
    dst_bin.write(bin_block.astype(np.float32), indexes=1)
    return total_valid, total_nan, total_above_thr, sampled_probs


def _predict_raster_tiled(
    src,
    model,
    read_indices: list[int],
    training_columns: list[str],
    raster_features: list[str],
    injected_values: dict[str, float],
    categorical_features: list[str] | None,
    best_threshold: float,
    dst_prob,
    dst_bin,
    *,
    hull_geom,
    skip_geometry_mask: bool,
    batch_size: int,
    nodata_value: float,
    log_every: int,
    sample_per_tile: int,
    verbose: bool = False,
) -> tuple[int, int, int, list[float]]:
    tile_idx = 0
    total_valid = 0
    total_nan = 0
    total_above_thr = 0
    sampled_probs: list[float] = []

    for _, window in src.block_windows(1):
        tile_idx += 1
        data = src.read(indexes=read_indices, window=window, out_dtype="float32")
        height, width = data.shape[1], data.shape[2]
        window_transform = src.window_transform(window)
        spatial_valid = _spatial_valid_mask(
            data,
            skip_geometry_mask=skip_geometry_mask,
            hull_geom=hull_geom,
            window_transform=window_transform,
            height=height,
            width=width,
        )

        X_raster = np.moveaxis(data, 0, -1).reshape(-1, len(raster_features))
        X = assemble_inference_matrix(
            X_raster, training_columns, raster_features, injected_values
        )

        preds_prob = np.full(X.shape[0], np.nan, dtype=np.float32)
        preds_bin = np.full(X.shape[0], np.nan, dtype=np.float32)
        valid = (~np.isnan(X).any(axis=1)) & spatial_valid
        valid_idx = np.where(valid)[0]
        total_nan += (~valid).sum()
        total_valid += valid.sum()

        if valid_idx.size > 0:
            for start in range(0, valid_idx.size, batch_size):
                batch_sel = valid_idx[start : start + batch_size]
                prob = _predict_model_batch(
                    model, X[batch_sel, :], training_columns, categorical_features
                )
                binary = (prob >= best_threshold).astype(np.float32)
                preds_prob[batch_sel] = prob
                preds_bin[batch_sel] = binary

            if tile_idx % log_every == 0 and verbose:
                q = np.percentile(preds_prob[valid], [0, 25, 50, 75, 95, 99])
                print(f"Tile {tile_idx}: prob pct [min,25,50,75,95,99] = {q}")

            total_above_thr += np.nansum(preds_bin == 1)

            k = min(sample_per_tile, preds_prob[valid].size)
            if k > 0:
                sel = np.random.choice(preds_prob[valid].size, size=k, replace=False)
                sampled_probs.extend(preds_prob[valid][sel].tolist())

        _write_prediction_window(
            dst_prob,
            dst_bin,
            preds_prob,
            preds_bin,
            window,
            height,
            width,
            nodata_value,
        )

    return total_valid, total_nan, total_above_thr, sampled_probs


def predict_raster_with_elapid(
    model_path: str,
    metrics_path: str,
    raster_path: str,
    rangefile_path: str,
    output_prob_path: str,
    output_bin_path: str,
    training_columns: list[str],
    band_source_dir: str,
    *,
    season_label: str,
    categorical_features: list[str] | None = None,
    batch_size: int = 1000,
    nodata_value: float = -9999.0,
    log_every: int = 50,
    sample_per_tile: int = 1000,
    skip_geometry_mask: bool = False,
    full_raster_threshold_bytes: int = FULL_RASTER_BYTE_THRESHOLD,
    verbose: bool = False,
) -> None:
    """Apply a tuned elapid MaxEnt model over a raster stack."""
    from shapely.geometry import mapping

    model = joblib.load(model_path)
    best_threshold = _load_inference_threshold(metrics_path)

    raster_band_names = read_inference_band_names(band_source_dir, season_label)

    band_mapping, injected_values, raster_features, injected_features = resolve_inference_band_mapping(
        training_columns, raster_band_names, season_label
    )
    read_indices = [band_mapping[f] for f in raster_features]

    with rasterio.open(raster_path) as src:
        gdf_hull = gpd.read_file(rangefile_path).to_crs(src.crs)
        hull_geom = [mapping(gdf_hull.geometry.union_all())]

        profile = src.profile.copy()
        profile.update(
            driver="GTiff",
            dtype="float32",
            count=1,
            compress="lzw",
            tiled=True,
            blockxsize=128,
            blockysize=128,
            nodata=nodata_value,
        )

        raster_bytes = src.width * src.height * len(read_indices) * np.dtype(np.float32).itemsize
        use_full_raster = raster_bytes <= full_raster_threshold_bytes
        if verbose:
            print(f"Loaded tuned elapid model and metrics: threshold={best_threshold:.4f}")
            print("Raster band names:", raster_band_names)
            print("Training columns:", training_columns)
            print("Band mapping:", band_mapping)
            if injected_features:
                print(f"Injected constants for {season_label}:", injected_values)
            print(f"Raster bands={src.count}; reading {len(read_indices)} mapped bands for model features")
            print(
                f"Predict path: {'full-raster' if use_full_raster else 'tiled'} "
                f"({raster_bytes / 1e6:.1f} MB read budget)"
            )

        predict_kwargs = dict(
            read_indices=read_indices,
            training_columns=training_columns,
            raster_features=raster_features,
            injected_values=injected_values,
            categorical_features=categorical_features,
            best_threshold=best_threshold,
            hull_geom=hull_geom,
            skip_geometry_mask=skip_geometry_mask,
            batch_size=batch_size,
            nodata_value=nodata_value,
            sample_per_tile=sample_per_tile,
        )

        with rasterio.open(output_prob_path, "w", **profile) as dst_prob, rasterio.open(
            output_bin_path, "w", **profile
        ) as dst_bin:
            if use_full_raster:
                total_valid, total_nan, total_above_thr, sampled_probs = _predict_raster_full(
                    src, model, dst_prob=dst_prob, dst_bin=dst_bin, **predict_kwargs
                )
            else:
                total_valid, total_nan, total_above_thr, sampled_probs = _predict_raster_tiled(
                    src,
                    model,
                    dst_prob=dst_prob,
                    dst_bin=dst_bin,
                    log_every=log_every,
                    verbose=verbose,
                    **predict_kwargs,
                )

    if verbose:
        if sampled_probs:
            gq = np.percentile(np.array(sampled_probs, dtype=np.float32), [0, 25, 50, 75, 95, 99])
            print(f"Global prob pct [min,25,50,75,95,99] = {gq}")
        print(f"Total valid pixels: {total_valid}; nodata/masked: {total_nan}; above threshold: {total_above_thr}")
        print(f"Probability raster saved to {output_prob_path}")
        print(f"Binary raster saved to {output_bin_path}")


INFERENCE_SEASONS = ["spring", "summer", "fall", "winter"]


def _maxent_failed_lookup() -> dict[str, str]:
    from gap_species import MAXENT_FAILED_SPECIES

    return MAXENT_FAILED_SPECIES


def is_maxent_excluded(sp: str) -> bool:
    key = sp.lower().replace(" ", "_")
    return key in _maxent_failed_lookup()


def species_output_dir(baseoutputdir: str, sp: str) -> str:
    if "HUGE" in baseoutputdir:
        return baseoutputdir
    return os.path.join(baseoutputdir, "ppp_paramsoutput", sp.lower().replace(" ", "_"))


def species_has_trained_model(spdir: str, sp: str) -> bool:
    return os.path.isfile(os.path.join(spdir, f"elapid_maxent_model_tuned_{sp}.pkl"))


def inference_output_paths(spdir: str, sp: str, season: str) -> tuple[str, str]:
    prob = os.path.join(spdir, f"predictions_prob_{sp}_{season}.tif")
    binary = os.path.join(spdir, f"predictions_binary_{sp}_{season}.tif")
    return prob, binary


def inference_season_complete(spdir: str, sp: str, season: str) -> bool:
    prob, binary = inference_output_paths(spdir, sp, season)
    return os.path.isfile(prob) and os.path.isfile(binary)


def audit_inference_outputs(
    species: list[str],
    baseoutputdir: str,
    seasons: list[str] | None = None,
) -> dict:
    """Summarize inference raster completeness for manifest species."""
    seasons = seasons or INFERENCE_SEASONS
    failed_lookup = _maxent_failed_lookup()
    complete: list[str] = []
    partial: list[dict] = []
    need_training: list[str] = []
    maxent_excluded: list[dict] = []

    for sp in species:
        sp = sp.lower().replace(" ", "_")
        if sp in failed_lookup:
            maxent_excluded.append({"species": sp, "reason": failed_lookup[sp]})
            continue

        spdir = species_output_dir(baseoutputdir, sp)
        if not os.path.isdir(spdir):
            need_training.append(sp)
            continue
        if not species_has_trained_model(spdir, sp):
            need_training.append(sp)
            continue

        missing_seasons = [
            season
            for season in seasons
            if not inference_season_complete(spdir, sp, season)
        ]
        if not missing_seasons:
            complete.append(sp)
        else:
            partial.append({"species": sp, "missing_seasons": missing_seasons})

    modelable = [s.lower().replace(" ", "_") for s in species if s.lower().replace(" ", "_") not in failed_lookup]
    expected_tifs = len(modelable) * len(seasons) * 2
    present_tifs = 0
    ppp = os.path.join(baseoutputdir, "ppp_paramsoutput")
    if os.path.isdir(ppp):
        present_tifs = sum(1 for _ in glob.glob(os.path.join(ppp, "**", "predictions_*.tif"), recursive=True))

    return {
        "complete": complete,
        "partial": partial,
        "need_training": sorted(set(need_training)),
        "maxent_excluded": maxent_excluded,
        "modelable_species": len(modelable),
        "expected_tifs": expected_tifs,
        "present_tifs": present_tifs,
    }


def inference_jobs(
    species: list[str],
    baseoutputdir: str,
    *,
    missing_only: bool = True,
    seasons: list[str] | None = None,
) -> list[tuple[str, str]]:
    """Return (species, season) jobs ready to run (trained model required)."""
    seasons = seasons or INFERENCE_SEASONS
    jobs: list[tuple[str, str]] = []
    for sp in species:
        sp = sp.lower().replace(" ", "_")
        if is_maxent_excluded(sp):
            continue
        spdir = species_output_dir(baseoutputdir, sp)
        if not species_has_trained_model(spdir, sp):
            continue
        for season in seasons:
            if missing_only and inference_season_complete(spdir, sp, season):
                continue
            jobs.append((sp, season))
    return jobs


def run_inference_species_season(
    sp: str,
    season: str,
    data_root: str,
    *,
    force_stack: bool = False,
    verbose: bool = False,
) -> dict:
    """Build covariate stack and run MaxEnt inference for one species and season."""
    import time

    sp = sp.lower().replace(" ", "_")
    season = season.strip().lower()
    spdir = species_output_dir(data_root, sp)
    raster_path = os.path.join(data_root, "rasters", "inference")
    rangefile = os.path.join(spdir, f"convex_hull_{sp}.json")
    model_path = os.path.join(spdir, f"elapid_maxent_model_tuned_{sp}.pkl")
    metrics_path = os.path.join(spdir, f"accuracy_tuned_{sp}.csv")
    output_prob_path, output_bin_path = inference_output_paths(spdir, sp, season)

    cfg = load_model_config(spdir, sp)
    training_columns = cfg["predictor_columns"]
    categorical_features = cfg.get("categorical_features", [])

    t0 = time.perf_counter()
    stack_path = build_inference_covariate_stack(
        raster_path,
        rangefile,
        season,
        inference_stack_cache_path(spdir, season),
        force=force_stack,
        verbose=verbose,
    )
    stack_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    predict_raster_with_elapid(
        model_path=model_path,
        metrics_path=metrics_path,
        raster_path=stack_path,
        rangefile_path=rangefile,
        output_prob_path=output_prob_path,
        output_bin_path=output_bin_path,
        training_columns=training_columns,
        band_source_dir=raster_path,
        season_label=season,
        categorical_features=categorical_features,
        batch_size=2000,
        skip_geometry_mask=True,
        verbose=verbose,
    )
    predict_seconds = time.perf_counter() - t0

    if not verbose:
        print(
            f"  {sp} {season}: stack {stack_seconds:.1f}s + predict {predict_seconds:.1f}s",
            flush=True,
        )

    return {
        "species": sp,
        "season": season,
        "status": "ok",
        "stack_path": stack_path,
        "output_prob_path": output_prob_path,
        "output_bin_path": output_bin_path,
        "stack_seconds": stack_seconds,
        "predict_seconds": predict_seconds,
    }


def run_inference_species(
    sp: str,
    data_root: str,
    seasons: list[str] | None = None,
    *,
    force_stack: bool = False,
    verbose: bool = False,
) -> list[dict]:
    """Run inference for one species across multiple seasons (serial within species)."""
    sp = sp.lower().replace(" ", "_")
    season_list = seasons or INFERENCE_SEASONS
    return [
        run_inference_species_season(sp, season, data_root, force_stack=force_stack, verbose=verbose)
        for season in season_list
    ]


def init_worker_cache(parambasedir: str, aoi_filename: str) -> None:
    """Load shared AOI/background data once per worker process."""
    global _AOI_CACHE
    _AOI_CACHE = gpd.read_parquet(os.path.join(parambasedir, aoi_filename))
    _BG_CACHE.clear()


def _get_aoi(parambasedir: str, aoi_filename: str) -> gpd.GeoDataFrame:
    if _AOI_CACHE is not None:
        return _AOI_CACHE
    return gpd.read_parquet(os.path.join(parambasedir, aoi_filename))


def _load_background(parambasedir: str, spgroup: str) -> pl.DataFrame:
    if spgroup in _BG_CACHE:
        return _BG_CACHE[spgroup].clone()

    expected = os.path.join(parambasedir, f"background_{spgroup}.csv")
    csv_files = glob.glob(os.path.join(parambasedir, f"background*{spgroup}*.csv"))
    if not csv_files and os.path.isfile(expected):
        csv_files = [expected]

    if not csv_files:
        pts = os.path.join(parambasedir, "background_pts.csv")
        hint = (
            f"No background CSV found for spgroup={spgroup!r} under {parambasedir}. "
            f"Expected e.g. {expected}. "
        )
        if os.path.isfile(pts):
            hint += (
                f"Found {pts} but not group files — run dataprep/groupBGpoints.ipynb "
                "to create background_avian.csv, background_herp.csv, background_mammal.csv."
            )
        else:
            hint += (
                "Run geeBackgroundToCSV.ipynb first (background_pts.csv), "
                "then groupBGpoints.ipynb."
            )
        raise FileNotFoundError(hint)

    bgs = []
    for file_path in csv_files:
        print(f"Reading: {file_path}")
        bgs.append(pl.read_csv(file_path))

    combined_bg = pl.concat(bgs, how="diagonal")
    combined_bg = combined_bg.with_columns(pl.lit(0).alias("label"))
    _BG_CACHE[spgroup] = combined_bg
    return combined_bg.clone()


def _extract_lon_lat(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns([
        pl.col(".geo")
        .str.json_path_match("$['coordinates'][0]")
        .cast(pl.Float64)
        .alias("longitude"),
        pl.col(".geo")
        .str.json_path_match("$['coordinates'][1]")
        .cast(pl.Float64)
        .alias("latitude"),
    ]).drop(["system:index", ".geo"])


def _load_presence_points(parambasedir: str, sp: str) -> pl.DataFrame:
    csv_files = glob.glob(os.path.join(parambasedir, f"*{sp}*.csv"))
    dfs = []
    for file_path in csv_files:
        print(f"Reading: {file_path}")
        df = pl.read_csv(file_path)
        df = _extract_lon_lat(df)
        dfs.append(df)

    combined_df = pl.concat(dfs, how="diagonal")
    return combined_df.with_columns(pl.lit(1).alias("label"))


@contextmanager
def _batch_plot_context(batch_mode: bool):
    if batch_mode:
        plt.switch_backend("Agg")
    try:
        yield
    finally:
        if batch_mode:
            plt.close("all")


def _show_or_close(batch_mode: bool) -> None:
    if batch_mode:
        plt.close()
    else:
        plt.show()


def plot_categorical_responses(
    model,
    training_data: pd.DataFrame,
    target_var: str,
    title: str = "Response Curve",
    *,
    outputdir: str | None = None,
    sp: str | None = None,
    batch_mode: bool = False,
) -> None:
    levels = sorted(training_data[target_var].dropna().unique())
    template = training_data.iloc[[0]].copy()

    preds = []
    for level in levels:
        eval_df = template.copy()
        if pd.api.types.is_categorical_dtype(training_data[target_var]):
            eval_df[target_var] = pd.Categorical(
                [level], categories=training_data[target_var].cat.categories
            )
        else:
            eval_df[target_var] = level
        preds.append(float(model.predict(eval_df)[0]))

    display_name = rename_col(target_var)
    plt.figure(figsize=(6, 4))
    plt.bar([str(level) for level in levels], preds, color="steelblue")
    plt.xlabel(f"{display_name} level")
    plt.ylabel("Suitability")
    plt.title(f"{title}: {display_name}")
    plt.grid(True, axis="y", alpha=0.3)

    if outputdir and sp:
        safe_name = target_var.replace("/", "_")
        plt.savefig(os.path.join(outputdir, f"Response_{sp}_{safe_name}.png"))
    _show_or_close(batch_mode)


def plot_elapid_responses(
    model,
    training_data: pd.DataFrame,
    target_var: str,
    title: str = "Response Curve",
    *,
    outputdir: str | None = None,
    sp: str | None = None,
    batch_mode: bool = False,
) -> None:
    v_min, v_max = training_data[target_var].min(), training_data[target_var].max()
    v_range = np.linspace(v_min, v_max, 100)

    template = training_data.iloc[[0]].copy()
    eval_df = pd.concat([template] * 100, ignore_index=True)
    eval_df[target_var] = v_range

    preds = model.predict(eval_df)
    display_name = rename_col(target_var)

    plt.figure(figsize=(6, 4))
    plt.plot(v_range, preds)
    plt.xlabel(f"{display_name} value")
    plt.ylabel("Suitability")
    plt.title(f"{title}: {display_name}")
    plt.grid(True, alpha=0.3)

    if outputdir and sp:
        safe_name = target_var.replace("/", "_")
        plt.savefig(os.path.join(outputdir, f"Response_{sp}_{safe_name}.png"))
    _show_or_close(batch_mode)


def _build_maxent_model(feature_types, base_tau: float, reg: float, n_cpus: int):
    return elapid.MaxentModel(
        feature_types=feature_types,
        tau=base_tau,
        clamp=True,
        scorer="roc_auc",
        beta_multiplier=reg,
        beta_lqp=1.0,
        beta_hinge=1.0,
        beta_threshold=1.0,
        beta_categorical=1.0,
        n_hinge_features=10,
        n_threshold_features=10,
        convergence_tolerance=1e-4,
        use_lambdas="best",
        n_cpus=n_cpus,
    )


def _eval_reg(
    reg: float,
    X: pd.DataFrame,
    y: pd.Series,
    folds,
    feature_types,
    base_tau: float,
    n_cpus: int,
    categorical_features: list[str] | None = None,
) -> dict:
    model_alpha = _build_maxent_model(feature_types, base_tau, reg, n_cpus)
    cat_idx = _categorical_indices(X, categorical_features)

    all_y_true_alpha, all_y_pred_prob_alpha = [], []
    aucs_alpha = []

    for train_idx, test_idx in folds:
        X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
        y_train, y_test = y.iloc[train_idx].copy(), y.iloc[test_idx].copy()

        model_alpha.fit(X_train, y_train, categorical=cat_idx)
        y_pred_prob_alpha = model_alpha.predict(X_test)

        all_y_true_alpha.extend(y_test)
        all_y_pred_prob_alpha.extend(y_pred_prob_alpha)

        if len(np.unique(y_test)) > 1:
            fpr, tpr, _ = roc_curve(y_test, y_pred_prob_alpha)
            aucs_alpha.append(auc(fpr, tpr))

    all_y_true_alpha = np.array(all_y_true_alpha)
    all_y_pred_prob_alpha = np.array(all_y_pred_prob_alpha)

    precisions_alpha, recalls_alpha, _ = precision_recall_curve(
        all_y_true_alpha, all_y_pred_prob_alpha
    )
    f1_scores_alpha = (
        2 * (precisions_alpha * recalls_alpha) / (precisions_alpha + recalls_alpha + 1e-9)
    )
    best_f1_alpha = float(np.max(f1_scores_alpha))
    mean_auc_alpha = float(np.mean(aucs_alpha)) if len(aucs_alpha) else float("nan")

    return {
        "beta_multiplier": reg,
        "best_f1": best_f1_alpha,
        "mean_auc": mean_auc_alpha,
    }


def fit_maxent_with_tuning(
    X: pd.DataFrame,
    y: pd.Series,
    coords: np.ndarray,
    outputdir: str,
    sp: str,
    selection_metric: str = "best_f1",
    target_vars=None,
    base_tau: float = 0.5,
    n_splits: int = 5,
    random_state: int = 42,
    reg_values: list[float] | None = None,
    feature_types: list[str] | None = None,
    categorical_features: list[str] | None = None,
    excel_group: str = "birds",
    spgroup: str = "avian",
    n_cpus: int = 8,
    inner_parallel: bool = False,
    batch_mode: bool = False,
) -> dict:
    os.makedirs(outputdir, exist_ok=True)

    X = X.copy()
    y = y.astype(int).copy()
    n_pres = int(y.sum())

    if feature_types is None:
        feature_types = (
            ["linear", "quadratic"] if n_pres < 10 else ["linear", "quadratic", "hinge"]
        )
    feature_types = ["linear", "quadratic", "product"]

    if reg_values is None:
        reg_values = SPARSE_REG_VALUES if n_pres < 10 else DEFAULT_REG_VALUES

    try:
        gkf = GeographicKFold(n_splits=n_splits)
        geometry = gpd.points_from_xy(X["longitude"], X["latitude"])
        gdf_coords = gpd.GeoDataFrame(
            X[["longitude", "latitude"]], geometry=geometry, crs="EPSG:4326"
        )
        folds = list(gkf.split(gdf_coords))
        cv_name = "GeographicKFold"
    except Exception as exc:
        print("Not using GeographicKFold:", exc)
        folds = list(
            StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state).split(
                X, y
            )
        )
        cv_name = "StratifiedKFold"
    print("Fold type:", cv_name)

    X = X.drop(columns=["longitude", "latitude"])

    if inner_parallel and len(reg_values) > 1:
        reg_jobs = min(len(reg_values), n_cpus)
        reg_results = Parallel(n_jobs=reg_jobs)(
            delayed(_eval_reg)(
                reg, X, y, folds, feature_types, base_tau, max(1, n_cpus // reg_jobs), categorical_features
            )
            for reg in reg_values
        )
    else:
        reg_results = [
            _eval_reg(reg, X, y, folds, feature_types, base_tau, n_cpus, categorical_features)
            for reg in reg_values
        ]

    reg_df = pd.DataFrame(reg_results)
    reg_df.to_csv(os.path.join(outputdir, f"Regularization_{sp}.csv"), index=False)

    plt.figure(figsize=(10, 5))
    plt.plot(reg_df["beta_multiplier"], reg_df["best_f1"], marker="o", label="Best F1")
    plt.xscale("log")
    plt.xlabel("beta_multiplier (log scale)")
    plt.ylabel("Best F1")
    plt.title(f"Effect of Regularization on F1 ({cv_name})")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(outputdir, f"Regularization_F1_{sp}.png"))
    _show_or_close(batch_mode)

    plt.figure(figsize=(10, 5))
    plt.plot(reg_df["beta_multiplier"], reg_df["mean_auc"], marker="o", color="orange", label="Mean AUC")
    plt.xscale("log")
    plt.xlabel("beta_multiplier (log scale)")
    plt.ylabel("Mean AUC")
    plt.title(f"Effect of Regularization on AUC ({cv_name})")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(outputdir, f"Regularization_AUC_{sp}.png"))
    _show_or_close(batch_mode)

    sel = selection_metric if selection_metric in {"best_f1", "mean_auc"} else "best_f1"
    if sel == "mean_auc" and np.all(np.isnan(reg_df["mean_auc"].values)):
        sel = "best_f1"

    best_entry = max(reg_results, key=lambda r: r[sel] if not np.isnan(r[sel]) else -np.inf)
    best_beta = best_entry["beta_multiplier"]
    best_score = best_entry[sel]
    print(f"Selected beta_multiplier={best_beta} (CV {sel}={best_score:.4f})")

    with open(os.path.join(outputdir, f"best_beta_{sp}.txt"), "w", encoding="utf-8") as handle:
        handle.write(str(best_beta))

    final_model = _build_maxent_model(feature_types, base_tau, best_beta, n_cpus)
    cat_idx = _categorical_indices(X, categorical_features)
    final_model.fit(X, y, categorical=cat_idx)
    y_pred_prob_full = final_model.predict(X)

    precisions_f, recalls_f, thresholds_f = precision_recall_curve(y, y_pred_prob_full)
    f1_scores_f = 2 * (precisions_f * recalls_f) / (precisions_f + recalls_f + 1e-9)
    idx_f = int(np.argmax(f1_scores_f))
    final_threshold = thresholds_f[idx_f]
    final_f1 = float(f1_scores_f[idx_f])

    if len(np.unique(y)) > 1:
        fpr_f, tpr_f, _ = roc_curve(y, y_pred_prob_full)
        final_auc = float(auc(fpr_f, tpr_f))
    else:
        final_auc = float("nan")

    final_precision = precision_score(y, (y_pred_prob_full >= final_threshold).astype(int))
    final_recall = recall_score(y, (y_pred_prob_full >= final_threshold).astype(int))
    final_logloss = log_loss(y, y_pred_prob_full, labels=[0, 1])
    prevalence = y.mean()

    print(f"Final tuned model (beta_multiplier={best_beta})")
    print(
        f"AUC={final_auc:.4f}  Precision={final_precision:.4f}  Recall={final_recall:.4f}  "
        f"F1={final_f1:.4f}"
    )
    print(
        f"Best threshold={final_threshold:.4f}  Log-loss={final_logloss:.4f}  "
        f"Prevalence={prevalence:.4f}"
    )

    pd.Series(X.columns).to_csv(
        os.path.join(outputdir, f"predictors_{sp}.txt"), index=False, header=False
    )

    config = {
        "scientific_name": sp,
        "excel_group": excel_group,
        "spgroup": spgroup,
        "categorical_features": categorical_features or [],
        "predictor_columns": list(X.columns),
    }
    with open(os.path.join(outputdir, f"model_config_{sp}.json"), "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    joblib.dump(final_model, os.path.join(outputdir, f"elapid_maxent_model_tuned_{sp}.pkl"))
    np.savetxt(
        os.path.join(outputdir, f"accuracy_tuned_{sp}.csv"),
        [
            ["beta_multiplier", best_beta],
            ["SelectionMetric", sel],
            ["AUC", final_auc],
            ["Precision", final_precision],
            ["Recall", final_recall],
            ["F1", final_f1],
            ["BestThreshold", final_threshold],
            ["LogLoss", final_logloss],
            ["Prevalence", prevalence],
        ],
        delimiter=",",
        fmt="%s",
    )

    if target_vars is not None and len(target_vars) > 0:
        for var in target_vars:
            plot_elapid_responses(
                final_model,
                X,
                var,
                title=f"Beta={best_beta}",
                outputdir=outputdir,
                sp=sp,
                batch_mode=batch_mode,
            )

    for var in categorical_features or []:
        if var in X.columns:
            plot_categorical_responses(
                final_model,
                X,
                var,
                title=f"Beta={best_beta}",
                outputdir=outputdir,
                sp=sp,
                batch_mode=batch_mode,
            )

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(recalls_f, precisions_f, label="PR (final tuned)", color="navy")
    plt.scatter(recalls_f[idx_f], precisions_f[idx_f], color="red", s=80, label=f"Best F1={final_f1:.2f}")
    plt.title(f"PR Curve (beta={best_beta})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()

    plt.subplot(1, 2, 2)
    if not np.isnan(final_auc):
        plt.plot(*roc_curve(y, y_pred_prob_full)[:2], label=f"ROC (AUC={final_auc:.2f})", color="darkorange")
    else:
        plt.text(0.5, 0.5, "AUC undefined (single class in y)", ha="center", va="center")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.title(f"ROC Curve (beta={best_beta})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outputdir, f"Curves_tuned_{sp}.png"))
    _show_or_close(batch_mode)

    lambdas = getattr(final_model, "lambdas_", None)
    names = getattr(final_model, "feature_names_", None)

    if lambdas is not None:
        lambdas = np.asarray(lambdas).ravel()
        if names is None or len(names) != len(lambdas):
            names = [f"feature_{i}" for i in range(len(lambdas))]
        df_imp = pd.DataFrame(
            {
                "feature": names,
                "lambda_weight": lambdas,
                "abs_importance": np.abs(lambdas),
            }
        ).sort_values("abs_importance", ascending=True)
        df_imp.to_csv(os.path.join(outputdir, f"FeatureImportance_Lambdas_{sp}.csv"), index=False)
        display_labels = df_imp["feature"].map(rename_col)
        plt.figure(figsize=(10, max(6, 0.3 * len(names))))
        plt.barh(display_labels, df_imp["lambda_weight"], color="skyblue")
        plt.axvline(0, color="gray", linewidth=1)
        plt.title("MaxEnt Feature Importance (lambda weights)")
        plt.xlabel("Lambda weight (positive ↑ suitability)")
        plt.tight_layout()
        plt.savefig(os.path.join(outputdir, f"FeatureImportance_Lambdas_{sp}.png"))
        _show_or_close(batch_mode)
    else:
        model_pi = _build_maxent_model(feature_types, base_tau, best_beta, n_cpus)
        try:
            X_train_pi, X_holdout_pi, y_train_pi, y_holdout_pi = train_test_split(
                X, y, test_size=0.2, random_state=random_state, stratify=y
            )
        except ValueError:
            split = int(0.8 * len(X))
            X_train_pi, X_holdout_pi = X.iloc[:split].copy(), X.iloc[split:].copy()
            y_train_pi, y_holdout_pi = y.iloc[:split].copy(), y.iloc[split:].copy()

        model_pi.fit(X_train_pi, y_train_pi, categorical=cat_idx)

        def baseline_score(y_true, y_prob):
            if len(np.unique(y_true)) > 1:
                return roc_auc_score(y_true, y_prob)
            return -log_loss(y_true, y_prob, labels=[0, 1])

        base_score = baseline_score(y_holdout_pi, model_pi.predict(X_holdout_pi))

        def robust_prob_scorer(est, X_eval, y_eval):
            y_prob = est.predict(X_eval)
            if len(np.unique(y_eval)) > 1:
                return roc_auc_score(y_eval, y_prob)
            return -log_loss(y_eval, y_prob, labels=[0, 1])

        perm = permutation_importance(
            model_pi,
            X_holdout_pi,
            y_holdout_pi,
            scoring=robust_prob_scorer,
            n_repeats=10,
        )

        imp = pd.DataFrame(
            {
                "feature": X.columns,
                "importance_mean": perm.importances_mean,
                "importance_std": perm.importances_std,
            }
        ).sort_values("importance_mean", ascending=True)

        imp.to_csv(os.path.join(outputdir, f"PermutationImportance_{sp}.csv"), index=False)
        display_labels = imp["feature"].map(rename_col)
        plt.figure(figsize=(10, max(6, 0.3 * len(imp))))
        plt.barh(display_labels, imp["importance_mean"], xerr=imp["importance_std"], color="steelblue")
        plt.axvline(0, color="gray", linewidth=1)
        metric_label = "AUC" if len(np.unique(y_holdout_pi)) > 1 else "−LogLoss"
        plt.title(f"Permutation Importance ({metric_label} change). Baseline={base_score:.3f}")
        plt.xlabel(f"Mean importance (Δ{metric_label})")
        plt.tight_layout()
        plt.savefig(os.path.join(outputdir, f"PermutationImportance_{sp}.png"))
        _show_or_close(batch_mode)

    print("Tuned final MaxEnt (PPP-equivalent) model saved, with diagnostics & importance.")
    return {
        "best_beta": best_beta,
        "selection_metric": sel,
        "final_auc": final_auc,
        "final_f1": final_f1,
        "final_precision": final_precision,
        "final_recall": final_recall,
        "final_threshold": final_threshold,
        "final_logloss": final_logloss,
        "prevalence": prevalence,
        "cv_name": cv_name,
        "reg_results": reg_results,
    }


def run_maxent_species(
    sp: str,
    spgroup: str,
    parambasedir: str,
    baseoutputdir: str,
    aoi_filename: str,
    *,
    n_cpus: int = 4,
    batch_mode: bool = True,
    inner_parallel: bool = False,
    selection_metric: str = "mean_auc",
    excel_group: str = "birds",
) -> dict:
    """Run the full MaxEnt modeling pipeline for one species."""
    sp = sp.lower().replace(" ", "_")

    if "HUGE" in baseoutputdir:
        outputdir = baseoutputdir
    else:
        outputdir = os.path.join(baseoutputdir, "ppp_paramsoutput")
        os.makedirs(outputdir, exist_ok=True)
        outputdir = os.path.join(outputdir, sp)
        os.makedirs(outputdir, exist_ok=True)

    print(outputdir)
    aoi = _get_aoi(parambasedir, aoi_filename)

    with _batch_plot_context(batch_mode):
        combined_df = _load_presence_points(parambasedir, sp)

        gdf = gpd.GeoDataFrame(
            combined_df.to_pandas(),
            geometry=gpd.points_from_xy(combined_df["longitude"], combined_df["latitude"]),
            crs="EPSG:4326",
        )
        convex_hull_geom = gdf.geometry.union_all().convex_hull
        hull_gdf = gpd.GeoDataFrame({"id": [1], "geometry": [convex_hull_geom]}, crs="EPSG:4326")
        hull_gdf.to_file(os.path.join(outputdir, f"convex_hull_{sp}.json"), driver="GeoJSON")

        if excel_group.strip().lower() == "birds":
            combined_df = ensure_bird_temporal_pl(combined_df)

        month_counts = combined_df.group_by("month").len().sort("month")
        month_counts_pd = month_counts.to_pandas()
        plt.figure(figsize=(8, 5))
        plt.bar(month_counts_pd["month"], month_counts_pd["len"], color="skyblue")
        plt.title(f"Row Count by Month for {sp}")
        plt.xlabel("Month")
        plt.ylabel("Row Count")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(outputdir, f"CountByMonth_{sp}.png"))
        _show_or_close(batch_mode)

        combined_bg = _load_background(parambasedir, spgroup)

        x = combined_df["longitude"]
        y = combined_df["latitude"]
        xy = np.vstack([x, y])
        kde = stats.gaussian_kde(xy)
        xmin, ymin = x.min(), y.min()
        xmax, ymax = x.max(), y.max()
        X_grid, Y_grid = np.meshgrid(
            np.linspace(xmin, xmax, 100), np.linspace(ymin, ymax, 100)
        )
        Z = kde(np.vstack([X_grid.ravel(), Y_grid.ravel()])).reshape(X_grid.shape)

        plt.figure(figsize=(8, 6))
        plt.imshow(Z, extent=[xmin, xmax, ymin, ymax], origin="lower", cmap="viridis")
        plt.scatter(x, y, s=5, c="red")
        plt.title(f"Presence KDE for {sp}")
        plt.tight_layout()
        plt.savefig(os.path.join(outputdir, f"KDensity_{sp}.png"))
        _show_or_close(batch_mode)

        xres = (xmax - xmin) / Z.shape[1]
        yres = (ymax - ymin) / Z.shape[0]
        transform = from_origin(xmin, ymax, xres, yres)
        Z_clean = np.nan_to_num(Z, nan=0.0)
        Z_clean[Z_clean < 0] = 0.0
        Z_raster = np.flipud(Z_clean)

        with rasterio.Env():
            with MemoryFile() as memfile:
                with memfile.open(
                    driver="GTiff",
                    height=Z_raster.shape[0],
                    width=Z_raster.shape[1],
                    count=1,
                    dtype=Z_raster.dtype,
                    crs="EPSG:4326",
                    transform=transform,
                ) as dataset:
                    dataset.write(Z_raster, 1)
                vsi_path = memfile.name
                pseudoabsence_bias = elapid.sample_bias_file(vsi_path, 10000)

        fig, ax = plt.subplots(figsize=(8, 6))
        pseudoabsence_bias.plot(ax=ax, markersize=0.5)
        plt.title(f"Bias surface for {sp}")
        plt.tight_layout()
        plt.savefig(os.path.join(outputdir, f"BiasSurface_{sp}.png"))
        _show_or_close(batch_mode)

        geometry = gpd.points_from_xy(combined_bg["longitude"], combined_bg["latitude"])
        bgdf = gpd.GeoDataFrame(combined_bg.to_pandas().copy(), geometry=geometry, crs="EPSG:4326")
        pseudo = gpd.GeoDataFrame(geometry=pseudoabsence_bias, crs="EPSG:4326")

        A_with_nearest_B = gpd.sjoin_nearest(
            pseudo,
            bgdf,
            how="left",
            distance_col="dist_m",
        )
        unique_b_idxs = A_with_nearest_B["index_right"].dropna().astype(int).unique()
        B_for_each_A = bgdf.loc[unique_b_idxs].reset_index(drop=True)

        fig, ax = plt.subplots(figsize=(8, 6))
        aoi.plot(ax=ax, color="red")
        B_for_each_A.plot(ax=ax, markersize=0.5)
        plt.title(f"Background points for {sp}")
        plt.tight_layout()
        plt.savefig(os.path.join(outputdir, f"BackgroundPoints_{sp}.png"))
        _show_or_close(batch_mode)

        print("Total presence points:", len(combined_df))
        print("Total background points:", len(B_for_each_A))
        pd.Series(
            [
                f"Total presence points: {len(combined_df)}",
                f"Total background points: {len(B_for_each_A)}",
            ]
        ).to_csv(os.path.join(outputdir, f"PointNums_{sp}.txt"), index=False, header=False)

        combined_bg = pl.from_pandas(B_for_each_A.drop(columns="geometry"))
        if excel_group.strip().lower() == "birds":
            combined_bg = ensure_bird_temporal_pl(combined_bg)
        month_counts = combined_bg.group_by("month").len().sort("month")
        month_counts_pd = month_counts.to_pandas()
        plt.figure(figsize=(8, 5))
        plt.bar(month_counts_pd["month"], month_counts_pd["len"], color="skyblue")
        plt.title("Background count by month")
        plt.xlabel("Month")
        plt.ylabel("Row Count")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(outputdir, f"CountByMonth_bg_{sp}.png"))
        _show_or_close(batch_mode)

        data = pd.concat([combined_df.to_pandas(), combined_bg.to_pandas()], ignore_index=True)
        coords = data[["longitude", "latitude"]].to_numpy()
        X, categorical_features = prepare_predictors(data, excel_group)
        y = data["label"].astype(int)

        target_vars = [
            col
            for col in X.drop(columns=["latitude", "longitude"]).columns
            if col not in categorical_features
        ]
        if len(combined_df) <= len(target_vars):
            raise ValueError("Not enough presence records")

        results = fit_maxent_with_tuning(
            X=X,
            y=y,
            coords=coords,
            outputdir=outputdir,
            sp=sp,
            selection_metric=selection_metric,
            target_vars=target_vars,
            categorical_features=categorical_features,
            excel_group=excel_group,
            spgroup=spgroup,
            n_cpus=n_cpus,
            inner_parallel=inner_parallel,
            batch_mode=batch_mode,
        )

        dfX = X.drop(columns=["longitude", "latitude"]).rename(columns=rename_col)
        num_df = dfX.select_dtypes(include="number")
        corr = num_df.corr(method="spearman")
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, cmap="coolwarm", center=0, vmin=-1, vmax=1, annot=False, square=True)
        plt.title("Variable Correlations (Spearman)")
        plt.tight_layout()
        plt.savefig(os.path.join(outputdir, f"Correlation_{sp}.png"))
        _show_or_close(batch_mode)

    return results
