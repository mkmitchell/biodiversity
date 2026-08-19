"""Precompute 1 km habitat-conversion biodiversity-index uplift (GeoParquet/Parquet).

Matches BioAPI compare rules: presence is any pixel > 0; biodiversity index is
100 * present_count / catalog_total; Dynamic World swaps move 100% FROM→TO on
both 500 m and 10 km percent bands.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Cap GDAL's per-process cache so several predict workers fit in RAM.
os.environ.setdefault("GDAL_CACHEMAX", "512")

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely
from joblib import Parallel, delayed
from pyproj import CRS
from rasterio.enums import Resampling
from rasterio.errors import WindowError
from rasterio.features import geometry_mask, rasterize
from rasterio.merge import merge
from rasterio.warp import reproject
from rasterio.windows import Window, from_bounds, intersection
from shapely.geometry import box, mapping

DATAPREP = Path(__file__).resolve().parent
REPO_ROOT = DATAPREP.parent
if str(DATAPREP) not in sys.path:
    sys.path.insert(0, str(DATAPREP))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gap_species import MAXENT_FAILED_SPECIES  # noqa: E402
from paths import DATA_ROOT, DEPLOY_API_ROOT, INFERENCE_RASTERS  # noqa: E402
from species_manifest import load_species_manifest  # noqa: E402

HABITAT_TO_CLASS = {
    "water": 0,
    "trees": 1,
    "grass": 2,
    "flooded_vegetation": 3,
    "crops": 4,
    "shrub_and_scrub": 5,
    "built": 6,
    "bare": 7,
    "snow_and_ice": 8,
}
CLASS_TO_HABITAT = {v: k for k, v in HABITAT_TO_CLASS.items()}

DEFAULT_CONVERSIONS: list[tuple[str, str]] = [
    ("crops", "trees"),
    ("crops", "flooded_vegetation"),
    ("trees", "crops"),
    ("flooded_vegetation", "crops"),
    ("grass", "flooded_vegetation"),
]

FROM_COVER_FLOOR = 5.0
CELL_SIZE_M = 1000
GROUP_ORDER = ("amphibians", "birds", "mammals", "reptiles")
INFERENCE_SEASONS = ["spring", "summer", "fall", "winter"]
PILOT_BBOX_Wsen = (-91.52, 34.15, -91.22, 34.45)
DW_SCALES = ("500m", "10000m")

ALBERS_5070_PROJ4 = (
    "+proj=aea +lat_0=23 +lon_0=-96 +lat_1=29.5 +lat_2=45.5 +x_0=0 +y_0=0 "
    "+ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs +type=crs"
)


def albers_5070_crs() -> CRS:
    """CONUS Albers equivalent to EPSG:5070 without NADCON grid files."""
    return CRS.from_proj4(ALBERS_5070_PROJ4)


def biodiversity_index(present_count: int, total_species: int) -> float:
    if total_species <= 0:
        return 0.0
    return round(100.0 * present_count / total_species, 1)


def conversion_id(from_class: str, to_class: str) -> str:
    return f"{from_class}__{to_class}"


def species_output_dir(data_root: Path | str, sp: str) -> Path:
    return Path(data_root) / "ppp_paramsoutput" / sp.lower().replace(" ", "_")


def species_has_trained_model(spdir: Path | str, sp: str) -> bool:
    return (Path(spdir) / f"elapid_maxent_model_tuned_{sp}.pkl").is_file()


def is_maxent_excluded(sp: str) -> bool:
    return sp.lower().replace(" ", "_") in MAXENT_FAILED_SPECIES


def dw_band_description(class_id: int, scale: str) -> str:
    return f"dw_class_{class_id}_pct_{scale}"


def swap_class_pair(
    from_arr: np.ndarray,
    to_arr: np.ndarray,
    pct: float = 100.0,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Move (pct)% of FROM cover into TO; conserves mass when pct is 100."""
    from_arr = from_arr.astype(np.float32, copy=False)
    to_arr = to_arr.astype(np.float32, copy=False)
    orig = from_arr if mask is None else from_arr * mask.astype(np.float32)
    moved = orig * (float(pct) / 100.0)
    new_from = np.clip(from_arr - moved, 0.0, 100.0)
    new_to = np.clip(to_arr + moved, 0.0, 100.0)
    return new_from.astype(np.float32), new_to.astype(np.float32)


def apply_dw_swap_to_arrays(
    bands: dict[str, np.ndarray],
    from_class: str,
    to_class: str,
    pct: float = 100.0,
) -> dict[str, np.ndarray]:
    """Swap habitat arrays keyed ``{class}_{scale}`` (500m and 10000m)."""
    out = {key: np.array(value, copy=True) for key, value in bands.items()}
    for scale in DW_SCALES:
        from_key = f"{from_class}_{scale}"
        to_key = f"{to_class}_{scale}"
        if from_key not in out or to_key not in out:
            continue
        out[from_key], out[to_key] = swap_class_pair(out[from_key], out[to_key], pct=pct)
    return out


def _to_albers(gdf: gpd.GeoDataFrame, dest: CRS | None = None) -> gpd.GeoDataFrame:
    dest = dest or albers_5070_crs()
    out = gdf.copy()
    out["geometry"] = out.geometry.make_valid()
    if out.crs is None:
        out = out.set_crs(4326)
    if out.crs == dest:
        return out
    return out.to_crs(dest)


def build_fishnet(
    aoi: gpd.GeoDataFrame,
    cell_size_m: float = CELL_SIZE_M,
    crs: CRS | None = None,
) -> gpd.GeoDataFrame:
    """1 km squares in CONUS Albers whose centroids fall inside the AOI."""
    dest = crs or albers_5070_crs()
    aoi_albers = _to_albers(aoi, dest)
    union = aoi_albers.geometry.union_all()
    if union.is_empty:
        raise ValueError("AOI is empty after projection")
    union = shapely.make_valid(union)
    if cell_size_m >= 100:
        union = union.simplify(min(50.0, cell_size_m / 20.0), preserve_topology=True)
    shapely.prepare(union)
    minx, miny, maxx, maxy = union.bounds
    origin_x = math.floor(minx / cell_size_m) * cell_size_m
    origin_y = math.floor(miny / cell_size_m) * cell_size_m
    ncols = max(1, int(math.ceil((maxx - origin_x) / cell_size_m - 1e-9)))
    nrows = max(1, int(math.ceil((maxy - origin_y) / cell_size_m - 1e-9)))
    cols = np.arange(ncols, dtype=np.int32)
    rows = np.arange(nrows, dtype=np.int32)
    col_grid, row_grid = np.meshgrid(cols, rows, indexing="xy")
    centroids_x = origin_x + (col_grid.astype(np.float64) + 0.5) * cell_size_m
    centroids_y = origin_y + (row_grid.astype(np.float64) + 0.5) * cell_size_m
    inside = shapely.contains_xy(union, centroids_x, centroids_y)
    sel_col = col_grid[inside].astype(np.int32)
    sel_row = row_grid[inside].astype(np.int32)
    if sel_col.size == 0:
        raise ValueError("Fishnet produced no cells")
    minx_cell = origin_x + sel_col.astype(np.float64) * cell_size_m
    miny_cell = origin_y + sel_row.astype(np.float64) * cell_size_m
    geoms = shapely.box(minx_cell, miny_cell, minx_cell + cell_size_m, miny_cell + cell_size_m)
    return gpd.GeoDataFrame(
        {
            "cell_id": np.arange(sel_col.size, dtype=np.int32),
            "col": sel_col,
            "row": sel_row,
            "geometry": geoms,
        },
        crs=dest,
    )


def rasterize_cell_ids(
    cells: gpd.GeoDataFrame,
    transform,
    out_shape: tuple[int, int],
) -> np.ndarray:
    """Raster of cell_id; background is -1."""
    shapes = (
        (geom, int(cid) + 1)
        for geom, cid in zip(cells.geometry, cells["cell_id"], strict=True)
    )
    burned = rasterize(
        shapes,
        out_shape=out_shape,
        transform=transform,
        fill=0,
        dtype=np.int32,
        all_touched=False,
    )
    return burned.astype(np.int32) - 1


def zonal_any_positive(
    raster_path: Path | str,
    cell_index: np.ndarray,
    n_cells: int,
    threshold: float = 0.0,
    window: Window | None = None,
) -> np.ndarray:
    """True where any pixel in the cell is > threshold (BioAPI current scan)."""
    with rasterio.open(raster_path) as ds:
        arr = ds.read(1, window=window)
        nodata = ds.nodata
    valid = cell_index >= 0
    pos = arr > threshold
    if nodata is not None:
        pos = pos & (arr != nodata)
    ids = cell_index[pos & valid]
    out = np.zeros(n_cells, dtype=bool)
    if ids.size:
        out[np.unique(ids)] = True
    return out


def zonal_mean(
    raster_path: Path | str,
    cell_index: np.ndarray,
    n_cells: int,
    window: Window | None = None,
    band: int = 1,
) -> np.ndarray:
    with rasterio.open(raster_path) as ds:
        arr = ds.read(band, window=window).astype(np.float64)
        nodata = ds.nodata
    valid = cell_index >= 0
    if nodata is not None:
        valid = valid & (arr != nodata)
    valid = valid & np.isfinite(arr)
    ids = cell_index[valid]
    vals = arr[valid]
    sums = np.bincount(ids, weights=vals, minlength=n_cells)
    counts = np.bincount(ids, minlength=n_cells)
    mean = np.full(n_cells, np.nan, dtype=np.float64)
    nz = counts > 0
    mean[nz] = sums[nz] / counts[nz]
    return mean.astype(np.float32)


def build_blocks_table(
    cell_ids: np.ndarray,
    from_class: str,
    to_class: str,
    from_cover_pct: np.ndarray,
    present_before: np.ndarray,
    present_after: np.ndarray,
    catalog_total: int,
    cover_floor: float = FROM_COVER_FLOOR,
    groups: list[str] | None = None,
) -> pd.DataFrame:
    n_before = present_before.sum(axis=1).astype(int)
    n_after = present_after.sum(axis=1).astype(int)
    n_gained = ((~present_before) & present_after).sum(axis=1).astype(int)
    n_lost = (present_before & (~present_after)).sum(axis=1).astype(int)
    eligible = from_cover_pct >= cover_floor
    index_before = np.array(
        [biodiversity_index(int(n), catalog_total) for n in n_before], dtype=np.float32
    )
    index_after = np.array(
        [biodiversity_index(int(n), catalog_total) for n in n_after], dtype=np.float32
    )
    delta = np.where(eligible, index_after - index_before, np.nan)
    data: dict[str, object] = {
        "cell_id": cell_ids,
        "from_class": from_class,
        "to_class": to_class,
        "conversion": conversion_id(from_class, to_class),
        "from_cover_pct": from_cover_pct,
        "n_present_before": n_before,
        "n_present_after": np.where(eligible, n_after, np.nan),
        "n_gained": np.where(eligible, n_gained, np.nan),
        "n_lost": np.where(eligible, n_lost, np.nan),
        "index_before": index_before,
        "index_after": np.where(eligible, index_after, np.nan),
        "delta_index": delta,
    }
    if groups is not None:
        group_arr = np.asarray(groups)
        for group in GROUP_ORDER:
            mask = group_arr == group
            data[f"n_{group}_before"] = present_before[:, mask].sum(axis=1).astype(int)
            after_g = present_after[:, mask].sum(axis=1).astype(float)
            data[f"n_{group}_after"] = np.where(eligible, after_g, np.nan)
    return pd.DataFrame(data)


def build_species_change_table(
    cell_ids: np.ndarray,
    from_class: str,
    to_class: str,
    present_before: np.ndarray,
    present_after: np.ndarray,
    species: list[str],
    from_cover_pct: np.ndarray,
    cover_floor: float = FROM_COVER_FLOOR,
    catalog_total: int | None = None,
) -> pd.DataFrame:
    eligible = from_cover_pct >= cover_floor
    n_species = present_before.shape[1]
    total = int(catalog_total) if catalog_total is not None else n_species
    rows: list[dict] = []
    conv = conversion_id(from_class, to_class)
    n_cells = present_before.shape[0]
    empty_cols = [
        "cell_id",
        "from_class",
        "to_class",
        "conversion",
        "species",
        "before",
        "after",
        "change",
        "index_before",
        "index_after",
        "delta_index",
    ]
    for i in range(n_cells):
        if not eligible[i]:
            continue
        for j, sp in enumerate(species):
            before = bool(present_before[i, j])
            after = bool(present_after[i, j])
            if before and after:
                change = "unchanged"
            elif (not before) and after:
                change = "gained"
            elif before and (not after):
                change = "lost"
            else:
                change = "unchanged"
            index_before = biodiversity_index(int(before), total)
            index_after = biodiversity_index(int(after), total)
            rows.append(
                {
                    "cell_id": int(cell_ids[i]),
                    "from_class": from_class,
                    "to_class": to_class,
                    "conversion": conv,
                    "species": sp,
                    "before": before,
                    "after": after,
                    "change": change,
                    "index_before": index_before,
                    "index_after": index_after,
                    "delta_index": round(index_after - index_before, 1),
                }
            )
    if not rows:
        return pd.DataFrame(columns=empty_cols)
    return pd.DataFrame(rows)


def _species_index_layer(cells: gpd.GeoDataFrame, blocks: pd.DataFrame) -> gpd.GeoDataFrame:
    """One polygon per cell × conversion with the API total biodiversity index."""
    table = blocks.copy()
    if "conversion" not in table.columns:
        table["conversion"] = [
            conversion_id(frm, to) for frm, to in zip(table["from_class"], table["to_class"])
        ]
    table = table.merge(cells[["cell_id", "geometry"]], on="cell_id", how="left")
    return gpd.GeoDataFrame(
        {
            "before": pd.to_numeric(table["n_present_before"], errors="coerce").astype("Int64"),
            "after": pd.to_numeric(table["n_present_after"], errors="coerce").astype("Int64"),
            "conversion": table["conversion"],
            "species_index_before": pd.to_numeric(table["index_before"], errors="coerce").astype("float64").round(1),
            "species_index_after": pd.to_numeric(table["index_after"], errors="coerce").astype("float64").round(1),
            "species_index_delta": pd.to_numeric(table["delta_index"], errors="coerce").astype("float64").round(1),
            "geometry": table["geometry"],
        },
        geometry="geometry",
        crs=cells.crs,
    )


def layer_name_suffix(cell_size_m: float) -> str:
    metres = int(round(cell_size_m))
    if metres == CELL_SIZE_M:
        return ""
    return f"_{metres}m"


def conversion_change_path(out_dir: Path, conv: str, cell_size_m: float) -> Path:
    suffix = layer_name_suffix(cell_size_m)
    if suffix:
        return out_dir / f"species_change{suffix}_{conv}.parquet"
    return out_dir / f"species_change_{conv}.parquet"


def conversion_blocks_path(out_dir: Path, conv: str, cell_size_m: float) -> Path:
    suffix = layer_name_suffix(cell_size_m)
    return out_dir / f"blocks{suffix}_{conv}.parquet"


def write_conversion_checkpoint(
    out_dir: Path,
    cells: gpd.GeoDataFrame,
    blocks: pd.DataFrame,
    conv: str,
    cell_size_m: float = CELL_SIZE_M,
) -> dict[str, Path]:
    """Write one conversion so an 8-hour session can stop without losing the map."""
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = layer_name_suffix(cell_size_m)
    cells_path = out_dir / f"cells{suffix}.parquet"
    blocks_path = conversion_blocks_path(out_dir, conv, cell_size_m)
    change_path = conversion_change_path(out_dir, conv, cell_size_m)
    cells.to_parquet(cells_path, index=False)
    blocks.to_parquet(blocks_path, index=False)
    _species_index_layer(cells, blocks).to_parquet(change_path, index=False)
    return {"cells": cells_path, "blocks": blocks_path, "species_change": change_path}


def write_uplift_parquets(
    out_dir: Path,
    cells: gpd.GeoDataFrame,
    blocks: pd.DataFrame,
    cell_size_m: float = CELL_SIZE_M,
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = layer_name_suffix(cell_size_m)
    cells_path = out_dir / f"cells{suffix}.parquet"
    blocks_path = out_dir / f"blocks{suffix}.parquet"
    cells.to_parquet(cells_path, index=False)
    blocks.to_parquet(blocks_path, index=False)
    layer = _species_index_layer(cells, blocks)
    written: dict[str, Path] = {"cells": cells_path, "blocks": blocks_path}
    conversions = list(dict.fromkeys(layer["conversion"].tolist()))
    for conv in conversions:
        part = layer.loc[layer["conversion"] == conv]
        path = conversion_change_path(out_dir, conv, cell_size_m)
        part.to_parquet(path, index=False)
        written[f"species_change_{conv}"] = path
    if not suffix:
        change_path = out_dir / "species_change.parquet"
        layer.to_parquet(change_path, index=False)
        written["species_change"] = change_path
    return written


def load_catalog_species(catalog_path: Path) -> list[str]:
    data = json.loads(catalog_path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "species" in data:
        return [str(s).lower().replace(" ", "_") for s in data["species"]]
    if isinstance(data, list):
        return [str(s).lower().replace(" ", "_") for s in data]
    raise ValueError(f"Unrecognized catalog JSON: {catalog_path}")


def load_aoi(aoi_path: Path, bbox_wsen: tuple[float, float, float, float] | None = None) -> gpd.GeoDataFrame:
    if aoi_path.suffix.lower() == ".parquet":
        gdf = gpd.read_parquet(aoi_path)
    else:
        gdf = gpd.read_file(aoi_path)
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    else:
        gdf = gdf.to_crs(4326)
    gdf["geometry"] = gdf.geometry.make_valid()
    if bbox_wsen is not None:
        west, south, east, north = bbox_wsen
        gdf = gdf.clip(box(west, south, east, north))
        if gdf.empty:
            raise ValueError(f"AOI clip to bbox {bbox_wsen} is empty")
    return gdf


def _intersect_bounds(
    cell_bounds: tuple[float, float, float, float],
    raster_bounds,
) -> tuple[float, float, float, float] | None:
    minx = max(cell_bounds[0], raster_bounds.left)
    miny = max(cell_bounds[1], raster_bounds.bottom)
    maxx = min(cell_bounds[2], raster_bounds.right)
    maxy = min(cell_bounds[3], raster_bounds.top)
    if minx >= maxx or miny >= maxy:
        return None
    return (minx, miny, maxx, maxy)


def _window_for_bounds(
    ds: rasterio.io.DatasetReader,
    bounds_wsen: tuple[float, float, float, float],
    pad: int = 2,
) -> Window | None:
    try:
        window = from_bounds(*bounds_wsen, transform=ds.transform)
    except Exception:
        return None
    window = window.round_offsets().round_lengths()
    padded = Window(
        col_off=window.col_off - pad,
        row_off=window.row_off - pad,
        width=window.width + 2 * pad,
        height=window.height + 2 * pad,
    )
    full = Window(0, 0, ds.width, ds.height)
    try:
        clipped = intersection(padded, full)
    except WindowError:
        return None
    if clipped.width <= 0 or clipped.height <= 0:
        return None
    return clipped.round_offsets().round_lengths()


def zonal_any_positive_cells(
    raster_path: Path | str,
    cells: gpd.GeoDataFrame,
    threshold: float = 0.0,
) -> np.ndarray:
    """Any-pixel>0 presence per cell on that raster's own grid."""
    n_cells = len(cells)
    with rasterio.open(raster_path) as ds:
        cells_r = cells.to_crs(ds.crs)
        overlap = _intersect_bounds(tuple(cells_r.total_bounds), ds.bounds)
        if overlap is None:
            return np.zeros(n_cells, dtype=bool)
        window = _window_for_bounds(ds, overlap)
        if window is None:
            return np.zeros(n_cells, dtype=bool)
        transform = ds.window_transform(window)
        shape = (int(window.height), int(window.width))
        cell_index = rasterize_cell_ids(cells_r, transform=transform, out_shape=shape)
        return zonal_any_positive(
            raster_path, cell_index, n_cells, threshold=threshold, window=window
        )


def zonal_mean_cells(
    raster_path: Path | str,
    cells: gpd.GeoDataFrame,
    band: int = 1,
) -> np.ndarray:
    n_cells = len(cells)
    with rasterio.open(raster_path) as ds:
        cells_r = cells.to_crs(ds.crs)
        overlap = _intersect_bounds(tuple(cells_r.total_bounds), ds.bounds)
        if overlap is None:
            return np.full(n_cells, np.nan, dtype=np.float32)
        window = _window_for_bounds(ds, overlap)
        if window is None:
            return np.full(n_cells, np.nan, dtype=np.float32)
        transform = ds.window_transform(window)
        shape = (int(window.height), int(window.width))
        cell_index = rasterize_cell_ids(cells_r, transform=transform, out_shape=shape)
        return zonal_mean(raster_path, cell_index, n_cells, window=window, band=band)


def _copy_band_descriptions(src_path: Path | str, dst_path: Path | str) -> None:
    with rasterio.open(src_path) as src, rasterio.open(dst_path, "r+") as dst:
        for idx, desc in enumerate(src.descriptions, start=1):
            if desc:
                dst.set_band_description(idx, desc)


def _default_dw_descriptions(count: int) -> list[str]:
    names: list[str] = []
    for scale in DW_SCALES:
        for class_id in range(9):
            names.append(dw_band_description(class_id, scale))
    if count <= len(names):
        return names[:count]
    names.extend(f"band_{i}" for i in range(len(names) + 1, count + 1))
    return names


def _ensure_dw_descriptions(path: Path) -> list[str]:
    with rasterio.open(path, "r+") as ds:
        names = list(ds.descriptions)
        if not any(names):
            names = _default_dw_descriptions(ds.count)
            for idx, desc in enumerate(names, start=1):
                ds.set_band_description(idx, desc)
        return [n or "" for n in names]


def _write_multiband(
    path: Path,
    data: np.ndarray,
    transform,
    crs,
    descriptions: list[str],
    nodata: float | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "height": data.shape[1],
        "width": data.shape[2],
        "count": data.shape[0],
        "dtype": "float32",
        "crs": crs,
        "transform": transform,
        "compress": "LZW",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "BIGTIFF": "IF_SAFER",
    }
    if nodata is not None:
        profile["nodata"] = nodata
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data.astype(np.float32))
        for idx, desc in enumerate(descriptions, start=1):
            if desc:
                dst.set_band_description(idx, desc)
    return path


def mosaic_clip_geotiff(
    src_paths: list[Path],
    dst_path: Path,
    bounds_wsen: tuple[float, float, float, float],
) -> Path:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    datasets = [rasterio.open(path) for path in src_paths]
    try:
        mosaic, transform = merge(
            datasets,
            bounds=bounds_wsen,
            nodata=datasets[0].nodata,
            dtype="float32",
        )
        crs = datasets[0].crs
        descriptions = list(datasets[0].descriptions)
    finally:
        for ds in datasets:
            ds.close()
    if not any(descriptions):
        descriptions = _default_dw_descriptions(mosaic.shape[0])
    return _write_multiband(dst_path, mosaic, transform, crs, descriptions)


def reproject_to_template(
    src_path: Path,
    template_path: Path,
    dst_path: Path,
    resampling: Resampling = Resampling.bilinear,
) -> Path:
    with rasterio.open(template_path) as tmpl, rasterio.open(src_path) as src:
        dest = np.full((src.count, tmpl.height, tmpl.width), np.nan, dtype=np.float32)
        reproject(
            source=src.read(),
            destination=dest,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=tmpl.transform,
            dst_crs=tmpl.crs,
            resampling=resampling,
            src_nodata=src.nodata,
            dst_nodata=np.nan,
        )
        descriptions = list(src.descriptions)
        crs = tmpl.crs
        transform = tmpl.transform
    return _write_multiband(dst_path, dest, transform, crs, descriptions)


def clip_geotiff(
    src_path: Path,
    dst_path: Path,
    bounds_wsen: tuple[float, float, float, float],
) -> Path:
    return mosaic_clip_geotiff([src_path], dst_path, bounds_wsen)


def apply_dw_swap_to_geotiff(
    src_path: Path,
    dst_path: Path,
    from_class: str,
    to_class: str,
    pct: float = 100.0,
) -> Path:
    if from_class not in HABITAT_TO_CLASS or to_class not in HABITAT_TO_CLASS:
        raise ValueError(f"Unknown habitat class in {from_class}->{to_class}")
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(src_path) as src:
        profile = src.profile.copy()
        profile.update(dtype="float32", compress="LZW", tiled=True, BIGTIFF="IF_SAFER")
        data = src.read().astype(np.float32)
        descriptions = list(src.descriptions)
    if not any(descriptions):
        descriptions = _default_dw_descriptions(data.shape[0])

    for scale in DW_SCALES:
        from_desc = dw_band_description(HABITAT_TO_CLASS[from_class], scale)
        to_desc = dw_band_description(HABITAT_TO_CLASS[to_class], scale)
        try:
            from_idx = next(i for i, d in enumerate(descriptions) if d == from_desc)
            to_idx = next(i for i, d in enumerate(descriptions) if d == to_desc)
        except StopIteration as exc:
            raise ValueError(f"Missing {scale} band for {from_class}->{to_class}") from exc
        data[from_idx], data[to_idx] = swap_class_pair(data[from_idx], data[to_idx], pct=pct)

    with rasterio.open(dst_path, "w", **profile) as dst:
        dst.write(data)
        for idx, desc in enumerate(descriptions, start=1):
            if desc:
                dst.set_band_description(idx, desc)
    return dst_path


def presence_matrix_for_rasters(
    species: list[str],
    raster_for_species,
    cells: gpd.GeoDataFrame,
) -> np.ndarray:
    present = np.zeros((len(cells), len(species)), dtype=bool)
    for j, sp in enumerate(species):
        path = raster_for_species(sp)
        if path is None or not Path(path).is_file():
            continue
        present[:, j] = zonal_any_positive_cells(path, cells)
    return present


def presence_matrix_for_season_binaries(
    species: list[str],
    pred_dir: Path,
    seasons: list[str],
    cells: gpd.GeoDataFrame,
) -> np.ndarray:
    present = np.zeros((len(cells), len(species)), dtype=bool)
    for j, sp in enumerate(species):
        for season in seasons:
            path = pred_dir / sp / f"predictions_binary_{sp}_{season}.tif"
            if path.is_file():
                present[:, j] |= zonal_any_positive_cells(path, cells)
    return present


def _excel_groups_for_species(species: list[str]) -> list[str]:
    try:
        manifest = load_species_manifest()
        lookup = dict(zip(manifest["scientific_name"], manifest["excel_group"]))
    except Exception:
        lookup = {}
    return [str(lookup.get(sp, "unknown")) for sp in species]


def prepare_inference_rasters(
    inference_dir: Path,
    work_dir: Path,
    bounds_wsen: tuple[float, float, float, float],
    force: bool = False,
) -> Path:
    """Clip DW mosaic + seasonal env rasters to AOI bounds (shared across species)."""
    work_dir.mkdir(parents=True, exist_ok=True)
    needed = ["all_months.tif", *[f"{season}.tif" for season in INFERENCE_SEASONS]]
    if not force and all((work_dir / name).is_file() for name in needed):
        return work_dir
    tiles = sorted(inference_dir.glob("all_months*.tif"))
    if not tiles:
        raise FileNotFoundError(f"No all_months*.tif in {inference_dir}")
    mosaic_clip_geotiff(tiles, work_dir / "all_months.tif", bounds_wsen)
    _ensure_dw_descriptions(work_dir / "all_months.tif")
    for season in INFERENCE_SEASONS:
        src = inference_dir / f"{season}.tif"
        if not src.is_file():
            raise FileNotFoundError(src)
        reproject_to_template(src, work_dir / "all_months.tif", work_dir / f"{season}.tif")
    return work_dir


def build_bbox_covariate_stack(
    raster_dir: Path,
    season: str,
    cache_path: Path,
    rangefile_path: Path,
    *,
    nodata: float = -9999.0,
) -> Path:
    """Stack clipped all_months + season (same grid) and mask outside the species hull."""
    mosaic = raster_dir / "all_months.tif"
    env = raster_dir / f"{season}.tif"
    if not mosaic.is_file() or not env.is_file():
        raise FileNotFoundError(f"Need {mosaic.name} and {env.name} in {raster_dir}")
    with rasterio.open(mosaic) as src_m, rasterio.open(env) as src_e:
        if src_m.shape != src_e.shape or src_m.transform != src_e.transform:
            raise ValueError(f"{env} is not on the same grid as {mosaic}")
        stacked = np.concatenate(
            [src_m.read().astype(np.float32), src_e.read().astype(np.float32)], axis=0
        )
        descriptions = [d or "" for d in src_m.descriptions] + [d or "" for d in src_e.descriptions]
        transform = src_m.transform
        crs = src_m.crs
        height, width = src_m.height, src_m.width
    hull = gpd.read_file(rangefile_path).to_crs(crs)
    geom = hull.geometry.union_all()
    if geom.is_empty:
        stacked[:] = nodata
    else:
        outside = geometry_mask(
            [mapping(geom)],
            out_shape=(height, width),
            transform=transform,
            invert=False,
        )
        stacked[:, outside] = nodata
    return _write_multiband(cache_path, stacked, transform, crs, descriptions, nodata=nodata)


def or_season_binaries(binary_paths: list[Path], out_tif: Path) -> None:
    """Pixel-wise OR of seasonal binaries → uint8 0/1 (small grids, fully flushed)."""
    if not binary_paths:
        raise ValueError("binary_paths is empty")
    with rasterio.open(binary_paths[0]) as ref:
        out = np.zeros((ref.height, ref.width), dtype=np.uint8)
        profile = {
            "driver": "GTiff",
            "height": ref.height,
            "width": ref.width,
            "count": 1,
            "dtype": "uint8",
            "crs": ref.crs,
            "transform": ref.transform,
            "compress": "DEFLATE",
            "tiled": False,
        }
        arr = ref.read(1)
        nodata = ref.nodata
    if nodata is not None:
        arr = np.where(arr == nodata, 0, arr)
    out = np.maximum(out, (arr > 0).astype(np.uint8))
    for path in binary_paths[1:]:
        with rasterio.open(path) as ds:
            arr = ds.read(1)
            if ds.nodata is not None:
                arr = np.where(arr == ds.nodata, 0, arr)
            out = np.maximum(out, (arr > 0).astype(np.uint8))
    out_tif.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_tif, "w", **profile) as dst:
        dst.write(out, 1)


def _predict_one_species(
    sp: str,
    data_root: Path,
    raster_dir: Path,
    work_spdir: Path,
    seasons: list[str],
    force: bool,
) -> Path:
    """Write seasonal binaries for one species. Loads MaxEnt only in this process."""
    from maxent_model import load_model_config, predict_raster_with_elapid

    work_spdir.mkdir(parents=True, exist_ok=True)
    spdir = species_output_dir(data_root, sp)
    rangefile = spdir / f"convex_hull_{sp}.json"
    model_path = spdir / f"elapid_maxent_model_tuned_{sp}.pkl"
    metrics_path = spdir / f"accuracy_tuned_{sp}.csv"
    if not rangefile.is_file() or not model_path.is_file():
        raise FileNotFoundError(f"{sp}: missing model or hull")
    print(f"  {sp}", flush=True)
    cfg = load_model_config(str(spdir), sp)
    training_columns = cfg["predictor_columns"]
    categorical_features = cfg.get("categorical_features", [])
    last = work_spdir / f"predictions_binary_{sp}_{seasons[-1]}.tif"
    for season in seasons:
        bin_path = work_spdir / f"predictions_binary_{sp}_{season}.tif"
        if season_binary_ready(bin_path) and not force:
            last = bin_path
            continue
        stack_path = work_spdir / f"inference_stack_{season}.tif"
        build_bbox_covariate_stack(raster_dir, season, stack_path, rangefile)
        predict_raster_with_elapid(
            model_path=str(model_path),
            metrics_path=str(metrics_path),
            raster_path=str(stack_path),
            rangefile_path=str(rangefile),
            output_prob_path=str(work_spdir / f"predictions_prob_{sp}_{season}.tif"),
            output_bin_path=str(bin_path),
            training_columns=training_columns,
            band_source_dir=str(raster_dir),
            season_label=season,
            categorical_features=categorical_features,
            batch_size=250_000,
            skip_geometry_mask=True,
            verbose=False,
        )
        last = bin_path
    return last


def season_binary_ready(path: Path) -> bool:
    """True when a seasonal binary exists and is non-empty (0-byte files after a kill do not count)."""
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _run_predict_subprocess(
    sp: str,
    data_root: Path,
    raster_dir: Path,
    work_spdir: Path,
    seasons: list[str],
    force: bool,
) -> Path:
    """Ensure seasonal MaxEnt binaries exist; returns dummy path for the first season."""
    work_spdir.mkdir(parents=True, exist_ok=True)
    binaries = [
        work_spdir / f"predictions_binary_{sp}_{season}.tif" for season in seasons
    ]
    need_predict = force or any(not season_binary_ready(p) for p in binaries)
    if need_predict:
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--predict-one",
            "--predict-sp",
            sp,
            "--data-root",
            str(data_root),
            "--predict-raster-dir",
            str(raster_dir),
            "--predict-work-dir",
            str(work_spdir),
            "--seasons",
            ",".join(seasons),
        ]
        if force:
            cmd.append("--force")
        print(f"  predicting {sp}", flush=True)
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if proc.stdout:
            print(proc.stdout.rstrip(), flush=True)
        if proc.returncode != 0:
            err = (proc.stderr or "").strip() or "(no stderr)"
            print(
                f"  predict worker {sp} exited {proc.returncode}: {err[-2000:]}",
                flush=True,
            )
    missing = [str(p) for p in binaries if not season_binary_ready(p)]
    if missing:
        raise RuntimeError(f"{sp}: predict worker missing {missing}")
    return binaries[0]


def run_conversion_predictions(
    species: list[str],
    data_root: Path,
    raster_dir: Path,
    work_dir: Path,
    seasons: list[str],
    jobs: int,
    force: bool,
) -> dict[str, Path]:
    work_dir.mkdir(parents=True, exist_ok=True)

    def _one(sp: str) -> tuple[str, Path | None, str | None]:
        if is_maxent_excluded(sp):
            return sp, None, "excluded"
        spdir = Path(species_output_dir(str(data_root), sp))
        if not species_has_trained_model(str(spdir), sp):
            return sp, None, "no_model"
        try:
            out = _run_predict_subprocess(
                sp, data_root, raster_dir, work_dir / sp, seasons, force
            )
            return sp, out, None
        except Exception as exc:
            print(f"FAILED {sp}: {exc}", flush=True)
            return sp, None, str(exc)

    if jobs <= 1:
        completed = [_one(sp) for sp in species]
    else:
        completed = Parallel(n_jobs=jobs, verbose=1)(delayed(_one)(sp) for sp in species)

    paths: dict[str, Path] = {}
    for sp, path, err in completed:
        if path is not None:
            paths[sp] = path
        elif err:
            print(f"skip {sp}: {err}", flush=True)
    return paths


def _parse_conversions(raw: str | None) -> list[tuple[str, str]]:
    if not raw:
        return list(DEFAULT_CONVERSIONS)
    pairs: list[tuple[str, str]] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Conversion must be from:to, got {item!r}")
        frm, to = (p.strip() for p in item.split(":", 1))
        if frm not in HABITAT_TO_CLASS or to not in HABITAT_TO_CLASS:
            raise ValueError(f"Unknown class in {frm}:{to}")
        pairs.append((frm, to))
    return pairs


def _parse_bbox(raw: str | None) -> tuple[float, float, float, float] | None:
    if not raw:
        return None
    parts = [float(x) for x in raw.replace(",", " ").split()]
    if len(parts) != 4:
        raise ValueError("--bbox needs west south east north")
    return parts[0], parts[1], parts[2], parts[3]


def run_uplift(
    *,
    data_root: Path,
    aoi_path: Path,
    current_dir: Path,
    inference_dir: Path,
    catalog_path: Path,
    out_dir: Path,
    conversions: list[tuple[str, str]],
    bbox_wsen: tuple[float, float, float, float] | None,
    species_filter: list[str] | None,
    cover_floor: float,
    seasons: list[str],
    jobs: int,
    skip_predict: bool,
    force: bool,
    cell_size_m: float = CELL_SIZE_M,
    work_dir: Path | None = None,
) -> dict[str, Path]:
    catalog = load_catalog_species(catalog_path)
    species = species_filter or catalog
    species = [s.lower().replace(" ", "_") for s in species]
    catalog_total = len(catalog) if not species_filter else len(species)
    groups = _excel_groups_for_species(species)

    aoi = load_aoi(aoi_path, bbox_wsen)
    print(f"Building {cell_size_m:g} m fishnet ({aoi.crs})...", flush=True)
    cells = build_fishnet(aoi, cell_size_m=cell_size_m)
    print(f"  {len(cells)} cells", flush=True)

    current_files = [p for p in (current_dir / f"{sp}.tif" for sp in species) if p.is_file()]
    if not current_files:
        raise FileNotFoundError(f"No current rasters in {current_dir}")

    print("Baseline zonal presence...", flush=True)
    present_before = presence_matrix_for_rasters(
        species,
        lambda sp: current_dir / f"{sp}.tif",
        cells,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    work_base = Path(work_dir) if work_dir is not None else out_dir / "work"
    cells_4326 = cells.to_crs(4326)
    bounds = tuple(float(x) for x in cells_4326.total_bounds)
    print("Clipping inference covariates to AOI...", flush=True)
    baseline_rasters = prepare_inference_rasters(
        inference_dir, work_base / "_baseline_covars", bounds, force=force
    )

    descriptions = _ensure_dw_descriptions(baseline_rasters / "all_months.tif")
    cover_lookup = {desc: i + 1 for i, desc in enumerate(descriptions)}

    from_cover_by_class: dict[str, np.ndarray] = {}
    unique_from = sorted({frm for frm, _ in conversions})
    for frm in unique_from:
        desc = dw_band_description(HABITAT_TO_CLASS[frm], "500m")
        if desc not in cover_lookup:
            raise ValueError(f"Missing Dynamic World band {desc}")
        from_cover_by_class[frm] = zonal_mean_cells(
            baseline_rasters / "all_months.tif",
            cells,
            band=cover_lookup[desc],
        )

    block_frames: list[pd.DataFrame] = []
    for frm, to in conversions:
        label = conversion_id(frm, to)
        print(f"Conversion {label}...", flush=True)
        t0 = time.perf_counter()
        change_ckpt = conversion_change_path(out_dir, label, cell_size_m)
        blocks_ckpt = conversion_blocks_path(out_dir, label, cell_size_m)
        if not force and change_ckpt.is_file() and blocks_ckpt.is_file():
            print(f"  reuse checkpoint {change_ckpt}", flush=True)
            block_frames.append(pd.read_parquet(blocks_ckpt))
            continue
        swapped_dir = work_base / label / "covars"
        swapped_dir.mkdir(parents=True, exist_ok=True)
        swapped_dw = swapped_dir / "all_months.tif"
        if force or not swapped_dw.is_file():
            apply_dw_swap_to_geotiff(
                baseline_rasters / "all_months.tif",
                swapped_dw,
                frm,
                to,
            )
            print(f"  swapped DW bands -> {swapped_dw}", flush=True)
        else:
            print(f"  reuse swapped DW -> {swapped_dw}", flush=True)
        for season in seasons:
            src = baseline_rasters / f"{season}.tif"
            dst = swapped_dir / f"{season}.tif"
            if not dst.exists() or force:
                dst.write_bytes(src.read_bytes())
                _copy_band_descriptions(src, dst)

        if skip_predict:
            present_after = present_before.copy()
        else:
            pred_dir = work_base / label / "pred"
            run_conversion_predictions(
                species, data_root, swapped_dir, pred_dir, seasons, jobs, force
            )
            present_after = presence_matrix_for_season_binaries(
                species, pred_dir, seasons, cells
            )

        blocks = build_blocks_table(
            cell_ids=cells["cell_id"].to_numpy(),
            from_class=frm,
            to_class=to,
            from_cover_pct=from_cover_by_class[frm],
            present_before=present_before,
            present_after=present_after,
            catalog_total=catalog_total,
            cover_floor=cover_floor,
            groups=groups,
        )
        block_frames.append(blocks)
        ckpt = write_conversion_checkpoint(out_dir, cells, blocks, label, cell_size_m)
        print(f"  wrote {ckpt['species_change']}", flush=True)
        n_scored = int((from_cover_by_class[frm] >= cover_floor).sum())
        if n_scored:
            scored = blocks.loc[blocks["delta_index"].notna(), "delta_index"]
            print(
                f"  scored {n_scored} cells  median_delta={float(scored.median()):.2f}  "
                f"max={float(scored.max()):.1f}  ({time.perf_counter() - t0:.0f}s)",
                flush=True,
            )
        else:
            print(f"  no cells above cover floor {cover_floor}", flush=True)

    blocks_all = pd.concat(block_frames, ignore_index=True)
    written = write_uplift_parquets(out_dir, cells, blocks_all, cell_size_m=cell_size_m)
    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "catalog_total": catalog_total,
        "n_cells": int(len(cells)),
        "cell_size_m": cell_size_m,
        "cover_floor": cover_floor,
        "conversions": [list(p) for p in conversions],
        "bbox_wsen": list(bbox_wsen) if bbox_wsen else None,
        "skip_predict": skip_predict,
        "species": species,
        "work_dir": str(work_base),
    }
    suffix = layer_name_suffix(cell_size_m)
    meta_name = "meta.json" if not suffix else f"meta{suffix}.json"
    (out_dir / meta_name).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    for path in written.values():
        print(f"Wrote {path}", flush=True)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="1 km habitat-conversion biodiversity uplift")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--aoi", type=Path, default=DATAPREP / "mav_counties_4326.parquet")
    parser.add_argument("--current-dir", type=Path, default=DEPLOY_API_ROOT / "rasters" / "current")
    parser.add_argument("--inference-dir", type=Path, default=INFERENCE_RASTERS)
    parser.add_argument(
        "--catalog",
        type=Path,
        default=DEPLOY_API_ROOT / "species_catalog.json",
    )
    parser.add_argument("--out-dir", type=Path, default=DATA_ROOT / "uplift")
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Reuse MaxEnt work trees (default: <out-dir>/work)",
    )
    parser.add_argument(
        "--cell-size",
        type=float,
        default=CELL_SIZE_M,
        help="Fishnet cell size in metres (default 1000; 500 writes *_500m.parquet)",
    )
    parser.add_argument(
        "--conversions",
        type=str,
        default="",
        help="from:to,from:to (default: all five). Re-run the same command to resume; finished conversions are skipped.",
    )
    parser.add_argument("--bbox", type=str, default="", help="west south east north")
    parser.add_argument("--pilot", action="store_true", help="Clip to a small central-MAV bbox")
    parser.add_argument("--species", type=str, default="", help="Comma-separated species override")
    parser.add_argument("--cover-floor", type=float, default=FROM_COVER_FLOOR)
    parser.add_argument("--seasons", type=str, default="")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--skip-predict", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--predict-one", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--predict-sp", type=str, default="", help=argparse.SUPPRESS)
    parser.add_argument("--predict-raster-dir", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--predict-work-dir", type=Path, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    seasons = (
        [s.strip().lower() for s in args.seasons.split(",") if s.strip()]
        if args.seasons.strip()
        else list(INFERENCE_SEASONS)
    )
    if args.predict_one:
        if not args.predict_sp or args.predict_raster_dir is None or args.predict_work_dir is None:
            raise SystemExit("--predict-one requires --predict-sp, --predict-raster-dir, --predict-work-dir")
        _predict_one_species(
            args.predict_sp,
            args.data_root,
            args.predict_raster_dir,
            args.predict_work_dir,
            seasons,
            args.force,
        )
        os._exit(0)

    bbox = _parse_bbox(args.bbox) if args.bbox else None
    if args.pilot and bbox is None:
        bbox = PILOT_BBOX_Wsen
    species_filter = (
        [s.strip().lower().replace(" ", "_") for s in args.species.split(",") if s.strip()]
        if args.species.strip()
        else None
    )
    run_uplift(
        data_root=args.data_root,
        aoi_path=args.aoi,
        current_dir=args.current_dir,
        inference_dir=args.inference_dir,
        catalog_path=args.catalog,
        out_dir=args.out_dir,
        conversions=_parse_conversions(args.conversions or None),
        bbox_wsen=bbox,
        species_filter=species_filter,
        cover_floor=args.cover_floor,
        seasons=seasons,
        jobs=args.jobs,
        skip_predict=args.skip_predict,
        force=args.force,
        cell_size_m=args.cell_size,
        work_dir=args.work_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
