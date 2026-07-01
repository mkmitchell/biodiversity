"""Build group-specific background CSVs (avian / herp / mammal) for MaxEnt."""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import elapid
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import rasterio
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from scipy import stats

from ebird_polars_io import log
from gap_species import has_param_csvs
from paths import PARAM_CSV_DIR
from species_manifest import DEFAULT_EXCEL, load_species_manifest

SPGROUPS = ("avian", "herp", "mammal")
# NAD83 / Conus Albers — used elsewhere in MAV pipeline (meters for nearest join)
PROJECTED_CRS = "EPSG:5070"


def extract_lon_lat(df: pl.DataFrame) -> pl.DataFrame:
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


def load_presence_by_spgroup(
    manifest: pd.DataFrame,
    spgroup: str,
    param_dir: Path,
) -> pl.DataFrame | None:
    """Concatenate GEE param CSV presence points for all manifest species in spgroup."""
    species = manifest.loc[manifest["spgroup"] == spgroup, "scientific_name"].tolist()

    dfs: list[pl.DataFrame] = []
    for sp in species:
        if not has_param_csvs(param_dir, sp):
            log(f"  skip {sp}: missing param CSVs")
            continue
        for file_path in sorted(param_dir.glob(f"{sp}_subset*.csv")):
            log(f"  Reading presence: {file_path.name}")
            dfs.append(extract_lon_lat(pl.read_csv(file_path)))

    if not dfs:
        log(f"  no presence param CSVs for spgroup={spgroup}")
        return None
    return pl.concat(dfs, how="diagonal").with_columns(pl.lit(1).alias("label"))


def load_background_pool(param_dir: Path) -> pl.DataFrame:
    path = param_dir / "background_pts.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path} — run geeBackgroundToCSV.ipynb first"
        )
    log(f"Reading: {path}")
    return extract_lon_lat(pl.read_csv(path)).with_columns(pl.lit(0).alias("label"))


def sample_background_for_group(
    presence: pl.DataFrame,
    background_pool: pl.DataFrame,
    *,
    n_bias_samples: int = 25000,
    show_plot: bool = False,
    aoi: gpd.GeoDataFrame | None = None,
    title: str = "",
) -> gpd.GeoDataFrame:
    """KDE bias surface from group presences; nearest background points from pool."""
    x = presence["longitude"]
    y = presence["latitude"]
    xy = np.vstack([x, y])
    kde = stats.gaussian_kde(xy)

    xmin, ymin = float(x.min()), float(y.min())
    xmax, ymax = float(x.max()), float(y.max())
    grid_x, grid_y = np.meshgrid(
        np.linspace(xmin, xmax, 100), np.linspace(ymin, ymax, 100)
    )
    z = kde(np.vstack([grid_x.ravel(), grid_y.ravel()])).reshape(grid_x.shape)

    fig, ax = (plt.subplots() if show_plot else (None, None))
    if show_plot:
        ax.imshow(z, extent=[xmin, xmax, ymin, ymax], origin="lower", cmap="viridis")
        ax.scatter(x, y, s=8, c="red")

    xres = (xmax - xmin) / z.shape[1]
    yres = (ymax - ymin) / z.shape[0]
    transform = from_origin(xmin, ymax, xres, yres)
    z_clean = np.nan_to_num(z, nan=0.0)
    z_clean[z_clean < 0] = 0.0
    z_raster = np.flipud(z_clean)

    with rasterio.Env():
        with MemoryFile() as memfile:
            with memfile.open(
                driver="GTiff",
                height=z_raster.shape[0],
                width=z_raster.shape[1],
                count=1,
                dtype=z_raster.dtype,
                crs="EPSG:4326",
                transform=transform,
            ) as dataset:
                dataset.write(z_raster, 1)
            pseudoabsence_bias = elapid.sample_bias_file(memfile.name, n_bias_samples)

    geometry = gpd.points_from_xy(background_pool["longitude"], background_pool["latitude"])
    bgdf = gpd.GeoDataFrame(background_pool.to_pandas().copy(), geometry=geometry, crs="EPSG:4326")
    pseudo = gpd.GeoDataFrame(geometry=pseudoabsence_bias, crs="EPSG:4326")

    # Nearest-neighbor in geographic CRS uses degrees; project to meters first.
    bgdf_p = bgdf.to_crs(PROJECTED_CRS)
    pseudo_p = pseudo.to_crs(PROJECTED_CRS)
    joined = gpd.sjoin_nearest(pseudo_p, bgdf_p, how="left", distance_col="dist_m")
    unique_idx = joined["index_right"].dropna().astype(int).unique()
    selected = bgdf.loc[unique_idx].reset_index(drop=True)

    if show_plot and ax is not None:
        if aoi is not None:
            aoi.plot(ax=ax, color="black", alpha=0.3)
        selected.plot(ax=ax, markersize=0.5)
        ax.set_title(title)
        ax.legend(["Presence KDE", "Presence", "Background"])
        plt.show()
        plt.close()

    return selected


def build_group_backgrounds(
    param_dir: Path | str = PARAM_CSV_DIR,
    *,
    excel_path: Path | str = DEFAULT_EXCEL,
    output_dir: Path | str | None = None,
    aoi_path: Path | str | None = None,
    show_plot: bool = False,
    n_bias_samples: int = 25000,
) -> dict[str, Path]:
    """
    Write background_avian.csv, background_herp.csv, background_mammal.csv.

    Returns mapping spgroup -> output path for groups that were built.
    """
    param_dir = Path(param_dir)
    output_dir = Path(output_dir) if output_dir is not None else param_dir
    manifest = load_species_manifest(excel_path)
    background_pool = load_background_pool(param_dir)

    aoi = None
    if aoi_path is not None:
        aoi = gpd.read_parquet(aoi_path)
    elif (param_dir / "mav_counties_4326.parquet").is_file():
        aoi = gpd.read_parquet(param_dir / "mav_counties_4326.parquet")

    written: dict[str, Path] = {}
    for spgroup in SPGROUPS:
        log(f"\n=== {spgroup} ===")
        presence = load_presence_by_spgroup(manifest, spgroup, param_dir)
        if presence is None:
            continue

        selected = sample_background_for_group(
            presence,
            background_pool,
            n_bias_samples=n_bias_samples,
            show_plot=show_plot,
            aoi=aoi,
            title=spgroup,
        )

        out_path = output_dir / f"background_{spgroup}.csv"
        out_df = selected.drop(columns="geometry", errors="ignore")
        out_df.to_csv(out_path, index=False)
        log(f"Wrote {out_path} ({len(out_df)} background points from {len(presence)} presence)")
        written[spgroup] = out_path

    if written:
        log("\nDone. Group background files:")
        for spgroup, path in written.items():
            log(f"  {spgroup}: {path}")
    else:
        log("\nNo group background files written (check param CSVs and background_pts.csv).")

    return written


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build group background CSVs for MaxEnt")
    parser.add_argument("--param-dir", type=Path, default=PARAM_CSV_DIR)
    parser.add_argument("--excel", type=Path, default=DEFAULT_EXCEL)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--aoi", type=Path, default=None)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    build_group_backgrounds(
        args.param_dir,
        excel_path=args.excel,
        output_dir=args.output_dir,
        aoi_path=args.aoi,
        show_plot=args.plot,
    )
