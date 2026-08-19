"""Start GEE Drive exports for one or more species (Daymet/DW/GMTED → paramcsv_daymet).

Uses the same covariate path as geeDataFromPoints.ipynb / export_ursus_gee.py.
Daymet bands are renamed to ``{b}_mean`` before reduceNeighborhood so final CSV
columns are ``{b}_mean_mean`` (→ ``{b}_10000m`` in MaxEnt).

After COMPLETED, download all ``{species}_subset*.csv`` from Drive folder
``paramcsv_daymet`` into /mnt/f/biodiversity/param_csvs/ (large species may
have more than two subsets).
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import duckdb
import ee
import geopandas as gpd
import pandas as pd

DATAPREP = Path(__file__).resolve().parent
sys.path.insert(0, str(DATAPREP))

from gee_covariates import compute_gmted_std_500m  # noqa: E402
from gee_species_io import excel_group_for_species, load_species_data  # noqa: E402
from paths import EBIRD_PARQUET, GBIF_ROOT, PARAM_CSV_DIR  # noqa: E402

N_SUBSETS = 2
DRIVE_FOLDER = "paramcsv_daymet"
EE_PROJECT = "biodiversity-478015"
LOG_FILE = DATAPREP / "process_log.txt"

WATERFOWL_12 = [
    "aix_sponsa",
    "anas_acuta",
    "anas_crecca",
    "anas_platyrhynchos",
    "anser_albifrons",
    "aythya_collaris",
    "branta_canadensis",
    "lophodytes_cucullatus",
    "mareca_americana",
    "mareca_strepera",
    "spatula_clypeata",
    "spatula_discors",
]


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", encoding="utf-8") as handle:
        handle.write(f"[{timestamp}] {message}\n")
    print(message, flush=True)


def df_to_ee_fc(df, datefield, lon_col="longitude", lat_col="latitude", properties=None):
    df = df.copy()
    df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")
    df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
    df = df.dropna(subset=[lon_col, lat_col])
    df = df[df[lon_col].between(-180, 180) & df[lat_col].between(-90, 90)]
    log(f"converting to fc ({len(df)} points)")
    if properties is None:
        properties = [c for c in df.columns if c not in [lon_col, lat_col, datefield]]
    features = []
    for _, row in df.iterrows():
        lon = float(row[lon_col])
        lat = float(row[lat_col])
        geom = ee.Geometry.Point([lon, lat])
        props = {}
        for key in properties:
            val = row[key]
            if hasattr(val, "item"):
                val = val.item()
            if isinstance(val, pd.Timestamp):
                val = val.strftime("%Y-%m-%d")
            props[key] = val
        props["obs_date"] = pd.Timestamp(row[datefield]).strftime("%Y-%m-%d")
        features.append(ee.Feature(geom, props))
    if not features:
        raise ValueError("No valid point geometries after coordinate cleanup")
    return ee.FeatureCollection(features)


def split_fc(fc, n_subsets=N_SUBSETS):
    log("splitting fc")
    n_points = fc.size().getInfo()
    points_list = fc.toList(n_points)
    subsets = []
    step = max(n_points // n_subsets + 1, 1)
    for i in range(0, n_points, step):
        subsets.append(ee.FeatureCollection(points_list.slice(i, i + step)))
    log(f"split into {len(subsets)} subsets (n={n_points})")
    return subsets


def get_dw_mode_image(obs_date):
    obs_date = ee.Date(obs_date)
    start_date = obs_date.advance(-1, "month")
    end_date = obs_date.advance(1, "month")
    dw_collection = ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1").select("label")
    return dw_collection.filterDate(start_date, end_date).reduce(ee.Reducer.mode())


def compute_dw_percent_cover(dw_img, radius_m):
    kernel = ee.Kernel.circle(radius=radius_m, units="meters", normalize=True)
    cover_images = []
    for class_id in range(9):
        mask = dw_img.eq(class_id)
        pct = (
            mask.reduceNeighborhood(ee.Reducer.mean(), kernel)
            .multiply(100)
            .rename(f"dw_class_{class_id}_pct_{radius_m}m")
        )
        cover_images.append(pct)
    return ee.Image.cat(cover_images)


def build_daymet_composite(obs_date, bands=("dayl", "prcp", "tmax", "tmin")):
    """Match geeDataFromPoints: ±1 month mean, then neighborhood mean → ``*_mean_mean``."""
    obs_date = ee.Date(obs_date)
    start_date = obs_date.advance(-1, "month")
    end_date = obs_date.advance(1, "month")
    daymet = (
        ee.ImageCollection("NASA/ORNL/DAYMET_V4")
        .filterDate(start_date, end_date)
        .select(list(bands))
        .mean()
    )
    return daymet.rename([f"{b}_mean" for b in bands])


def reduce_all_bands_with_radius(img, radius_m, reducer=None, units="meters", skip_masked=True):
    reducer = reducer or ee.Reducer.mean()
    kernel = ee.Kernel.circle(radius=radius_m, units=units, normalize=True)
    return img.reduceNeighborhood(reducer=reducer, kernel=kernel, skipMasked=skip_masked)


def export_subset(sub_fc, species, subset_index):
    log(f"Setting up export {species}_subset{subset_index}")
    bands = ("dayl", "prcp", "tmax", "tmin")

    def process_feature(f):
        obs_date = f.get("obs_date")
        dw_img = get_dw_mode_image(obs_date)
        env_img = build_daymet_composite(obs_date, bands)
        is_number = ee.Algorithms.IsEqual(obs_date, ee.Number(obs_date))
        date = ee.Date(ee.Algorithms.If(is_number, ee.Number(obs_date), ee.String(obs_date)))
        month = ee.Number(date.get("month"))
        season = ee.Number(
            ee.Algorithms.If(
                month.lte(2).Or(month.eq(12)),
                0,
                ee.Algorithms.If(month.lte(5), 1, ee.Algorithms.If(month.lte(8), 2, 3)),
            )
        ).toInt()
        f_with_props = f.set({"month": month, "season": season})
        cover_100m = compute_dw_percent_cover(dw_img, radius_m=500)
        cover_10km = compute_dw_percent_cover(dw_img, radius_m=10000)
        env_metrics_10km = reduce_all_bands_with_radius(env_img, 10000)
        terrain_500m = compute_gmted_std_500m(500)
        full_img = ee.Image.cat([cover_100m, cover_10km, env_metrics_10km, terrain_500m])
        return full_img.sampleRegions(
            collection=ee.FeatureCollection([f_with_props]),
            scale=100,
            geometries=True,
            tileScale=4,
        )

    sampled_fc = sub_fc.map(process_feature).flatten()
    export_desc = f"{species}_subset{subset_index}"
    task = ee.batch.Export.table.toDrive(
        collection=sampled_fc,
        description=export_desc,
        fileNamePrefix=export_desc,
        folder=DRIVE_FOLDER,
    )
    task.start()
    log(f"Export started for {species} subset {subset_index} id={task.id}")
    return task


def wait_for_tasks(tasks: list, poll_seconds: int = 60) -> int:
    pending = {t.id: t for t in tasks}
    while pending:
        done_ids = []
        for task_id, task in pending.items():
            status = task.status()
            state = status.get("state")
            desc = status.get("description")
            log(f"  {desc}: {state}")
            if state in {"COMPLETED", "FAILED", "CANCELLED"}:
                done_ids.append(task_id)
                if state != "COMPLETED":
                    log(f"  ERROR detail: {status.get('error_message')}")
        for task_id in done_ids:
            pending.pop(task_id, None)
        if pending:
            time.sleep(poll_seconds)
    failed = sum(1 for t in tasks if t.status().get("state") != "COMPLETED")
    return failed


def export_one_species(
    species: str,
    con,
    *,
    dry_run: bool,
    n_subsets: int,
    max_points_per_subset: int = 2500,
) -> list:
    species = species.lower().replace(" ", "_")

    df = load_species_data(
        species,
        excel_group_for_species(species),
        gbif_root=GBIF_ROOT,
        parquet_folder=EBIRD_PARQUET,
        con=con,
        log=log,
    )
    if df is None or df.empty:
        raise SystemExit(f"No occurrence rows loaded for {species}")

    # Scale subsets for large eBird species so Drive tasks stay tractable.
    needed = max(n_subsets, (len(df) + max_points_per_subset - 1) // max_points_per_subset)
    for i in range(needed):
        path = PARAM_CSV_DIR / f"{species}_subset{i}.csv"
        if path.exists():
            log(f"Skip {species}: {path.name} already exists")
            return []

    training_fc = df_to_ee_fc(df, datefield="date")
    subsets = split_fc(training_fc, n_subsets=needed)

    if dry_run:
        log(f"Dry-run: would export {len(subsets)} subsets for {species}")
        return []

    with ThreadPoolExecutor(max_workers=min(4, len(subsets))) as executor:
        tasks = list(
            executor.map(
                lambda pair: export_subset(pair[1], species, pair[0]),
                enumerate(subsets),
            )
        )
    return tasks


def parse_species_arg(raw: str) -> list[str]:
    raw = raw.strip()
    if raw.lower() == "waterfowl12":
        return list(WATERFOWL_12)
    return [s.strip().lower().replace(" ", "_") for s in raw.split(",") if s.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--species",
        default="waterfowl12",
        help="Comma-separated keys, or 'waterfowl12' for the 12 planned ducks/geese",
    )
    parser.add_argument("--poll-seconds", type=int, default=90)
    parser.add_argument("--n-subsets", type=int, default=N_SUBSETS)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--start-only",
        action="store_true",
        help="Start tasks then exit without polling",
    )
    args = parser.parse_args(argv)

    ee.Initialize(project=EE_PROJECT)
    species_list = parse_species_arg(args.species)
    log(f"Exporting {len(species_list)} species → Drive/{DRIVE_FOLDER}")

    aoi_path = DATAPREP / "mav_counties_4326.parquet"
    _ = gpd.read_parquet(aoi_path)

    con = duckdb.connect()
    con.execute("INSTALL spatial; LOAD spatial;")
    con.execute(
        f"""
        CREATE OR REPLACE TABLE aoi AS
        SELECT * FROM read_parquet('{aoi_path}');
        """
    )

    all_tasks: list = []
    for species in species_list:
        log(f"\n--- {species} ---")
        try:
            tasks = export_one_species(
                species, con, dry_run=args.dry_run, n_subsets=args.n_subsets
            )
            all_tasks.extend(tasks)
        except Exception as exc:
            log(f"FAILED {species}: {exc}")
            return 1

    if args.dry_run or args.start_only or not all_tasks:
        log(f"Started {len(all_tasks)} task(s); not waiting")
        return 0

    log(f"Waiting for {len(all_tasks)} GEE Drive tasks...")
    failed = wait_for_tasks(all_tasks, poll_seconds=args.poll_seconds)
    if failed:
        log(f"{failed} task(s) failed")
        return 1

    log(
        "All GEE tasks COMPLETED. Download from Drive folder "
        f"'{DRIVE_FOLDER}' into {PARAM_CSV_DIR} then retrain MaxEnt."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
