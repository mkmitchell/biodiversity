"""Build DU + Capture presence for ursus_americanus as GBIF-shaped yearly CSVs.

Default: only replaces /mnt/f/biodiversity/gbif/ursus_americanus_{year}.csv so
geeDataFromPoints.ipynb / export_ursus_gee.py can export real covariates to Drive.

Does NOT write param CSVs unless --write-local-params is passed (discouraged;
prefer waiting for GEE Drive exports). Does NOT touch background_mammal.csv.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

DATAPREP = Path(__file__).resolve().parent
sys.path.insert(0, str(DATAPREP))

from paths import DATA_ROOT, GBIF_ROOT, INFERENCE_RASTERS, PARAM_CSV_DIR  # noqa: E402

DEFAULT_DL = Path("/mnt/c/Users/mmitchell/Downloads")
DU_XLSX = "DU LA Bear Data 1-1-2017to8-5-2026.xlsx"
CAPTURE_XLSX = "Capture_Mortality Georeferenced Final Corrected (1).xlsx"
SPECIES = "ursus_americanus"
SPECIES_LABEL = "Ursus americanus"
YEAR_START = 2017
YEAR_END = 2024
UNCERTAINTY_M = 50.0

SEASON_BY_MONTH = {
    12: 0,
    1: 0,
    2: 0,
    3: 1,
    4: 1,
    5: 1,
    6: 2,
    7: 2,
    8: 2,
    9: 3,
    10: 3,
    11: 3,
}
SEASON_NAME = {0: "winter", 1: "spring", 2: "summer", 3: "fall"}


def _aoi_union(aoi_path: Path):
    aoi = gpd.read_parquet(aoi_path).set_crs("EPSG:4326", allow_override=True)
    return aoi.union_all()


def _in_aoi(lats, lons, aoi_geom) -> np.ndarray:
    pts = gpd.GeoSeries(gpd.points_from_xy(lons, lats), crs="EPSG:4326")
    return pts.within(aoi_geom).to_numpy()


def load_du_capture(downloads: Path, aoi_geom) -> pd.DataFrame:
    """Return presence rows: lat, lon, date, year, month, day, source."""
    rows: list[pd.DataFrame] = []

    du = pd.read_excel(downloads / DU_XLSX)
    du = du.rename(columns={"Latitude": "latitude", "Longitude": "longitude"})
    du["date"] = pd.to_datetime(du["GenericDate"], errors="coerce")
    du["source"] = "du_la"
    du["basisofrecord"] = "HUMAN_OBSERVATION"
    rows.append(du[["latitude", "longitude", "date", "source", "basisofrecord"]])

    cap = pd.read_excel(downloads / CAPTURE_XLSX, sheet_name="Capture ")
    cap = cap.rename(columns={"Y": "latitude", "X": "longitude"})
    cap["date"] = pd.to_datetime(cap["Capture Date"], errors="coerce")
    cap["source"] = "capture"
    cap["basisofrecord"] = "HUMAN_OBSERVATION"
    rows.append(cap[["latitude", "longitude", "date", "source", "basisofrecord"]])

    df = pd.concat(rows, ignore_index=True)
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude", "date"])
    df["year"] = df["date"].dt.year.astype(int)
    df["month"] = df["date"].dt.month.astype(int)
    df["day"] = df["date"].dt.day.astype(int)
    df = df[(df["year"] >= YEAR_START) & (df["year"] <= YEAR_END)]
    df = df[_in_aoi(df["latitude"].to_numpy(), df["longitude"].to_numpy(), aoi_geom)]

    df["lat_r"] = df["latitude"].round(4)
    df["lon_r"] = df["longitude"].round(4)
    df["day_key"] = df["date"].dt.floor("D")
    before = len(df)
    df = df.drop_duplicates(subset=["lat_r", "lon_r", "day_key"], keep="first")
    print(
        f"Presence after MAV/{YEAR_START}-{YEAR_END} filter: "
        f"{before} → {len(df)} unique day-locations"
    )
    print(df["source"].value_counts().to_string())
    return df.drop(columns=["lat_r", "lon_r", "day_key"]).reset_index(drop=True)


def write_gbif_yearly(df: pd.DataFrame, gbif_root: Path, *, backup: bool) -> None:
    gbif_root.mkdir(parents=True, exist_ok=True)
    backup_dir = gbif_root / f"_backup_ursus_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
    existing = sorted(gbif_root.glob(f"{SPECIES}_*.csv"))
    if backup and existing:
        backup_dir.mkdir(parents=True, exist_ok=True)
        for path in existing:
            shutil.move(str(path), str(backup_dir / path.name))
        print(f"Backed up {len(existing)} old GBIF files → {backup_dir}")
    else:
        for path in existing:
            path.unlink()

    for year, group in df.groupby("year"):
        out = gbif_root / f"{SPECIES}_{year}.csv"
        payload = pd.DataFrame(
            {
                "basisofrecord": group["basisofrecord"].values,
                "species": SPECIES_LABEL,
                "latitude": group["latitude"].values,
                "longitude": group["longitude"].values,
                "coordinateuncertaintyinmeters": UNCERTAINTY_M,
                "year": group["year"].values,
                "month": group["month"].values,
                "day": group["day"].values,
                "source": group["source"].values,
            }
        )
        payload.to_csv(out, index=False)
        print(f"Wrote {out.name}: {len(payload)} rows")


def _sample_point(ds, lon: float, lat: float) -> np.ndarray | None:
    try:
        vals = np.array(list(ds.sample([(lon, lat)]))[0], dtype=np.float64)
    except Exception:
        return None
    if ds.nodata is not None:
        vals = np.where(vals == ds.nodata, np.nan, vals)
    if np.isnan(vals).all():
        return None
    return vals


def build_param_csvs(
    df: pd.DataFrame,
    raster_dir: Path,
    param_dir: Path,
    *,
    n_subsets: int = 2,
) -> None:
    """Optional local raster sampling — prefer GEE Drive exports instead."""
    dw_paths = sorted(raster_dir.glob("all_months*.tif"))
    if not dw_paths:
        raise FileNotFoundError(f"No all_months*.tif in {raster_dir}")

    dw_datasets = [rasterio.open(p) for p in dw_paths]
    with rasterio.open(dw_paths[0]) as ref:
        dw_names = [ref.descriptions[i] or f"band_{i+1}" for i in range(ref.count)]
    print(f"DW tiles={len(dw_datasets)} bands={dw_names}")

    season_datasets: dict[str, rasterio.DatasetReader] = {}
    season_names: dict[str, list[str]] = {}
    for _code, name in SEASON_NAME.items():
        path = raster_dir / f"{name}.tif"
        ds = rasterio.open(path)
        season_datasets[name] = ds
        season_names[name] = [ds.descriptions[i] or f"band_{i+1}" for i in range(ds.count)]

    rename_env = {
        "dayl_10000m": "dayl_mean_mean",
        "prcp_10000m": "prcp_mean_mean",
        "tmax_10000m": "tmax_mean_mean",
        "tmin_10000m": "tmin_mean_mean",
        "gmted_std_500m": "gmted_std_500m",
    }

    records: list[dict] = []
    skipped = 0
    try:
        for i, (_, row) in enumerate(df.iterrows()):
            if i % 100 == 0:
                print(f"  sampling {i}/{len(df)}...", flush=True)
            month = int(row["month"])
            season_code = SEASON_BY_MONTH[month]
            season_name = SEASON_NAME[season_code]
            lon, lat = float(row["longitude"]), float(row["latitude"])

            dw_vals = None
            for ds in dw_datasets:
                left, bottom, right, top = ds.bounds
                if left <= lon <= right and bottom <= lat <= top:
                    dw_vals = _sample_point(ds, lon, lat)
                    if dw_vals is not None:
                        break
            if dw_vals is None:
                skipped += 1
                continue

            s_vals = _sample_point(season_datasets[season_name], lon, lat)
            if s_vals is None:
                skipped += 1
                continue

            rec = {
                "system:index": str(i),
                "basisofrecord": row["basisofrecord"],
                "coordinateuncertaintyinmeters": UNCERTAINTY_M,
                "species": SPECIES_LABEL,
                "month": month,
                "season": season_code,
                "obs_date": row["date"].isoformat(),
                ".geo": json.dumps(
                    {"geodesic": False, "type": "Point", "coordinates": [lon, lat]}
                ),
            }
            for name, val in zip(dw_names, dw_vals):
                if name:
                    rec[name] = float(val) if not np.isnan(val) else 0.0
            for name, val in zip(season_names[season_name], s_vals):
                if not name or name == "season":
                    continue
                key = rename_env.get(name, name)
                rec[key] = float(val) if not np.isnan(val) else 0.0
            records.append(rec)
    finally:
        for ds in dw_datasets:
            ds.close()
        for ds in season_datasets.values():
            ds.close()

    print(f"Sampled {len(records)} points; skipped {skipped} (outside raster / nodata)")
    if len(records) < 20:
        raise RuntimeError(f"Too few sampled points ({len(records)}) for MaxEnt")

    out_df = pd.DataFrame.from_records(records)
    preferred = [
        "system:index",
        "basisofrecord",
        "coordinateuncertaintyinmeters",
        "dayl_mean_mean",
        "dw_class_0_pct_10000m",
        "dw_class_0_pct_500m",
        "dw_class_1_pct_10000m",
        "dw_class_1_pct_500m",
        "dw_class_2_pct_10000m",
        "dw_class_2_pct_500m",
        "dw_class_3_pct_10000m",
        "dw_class_3_pct_500m",
        "dw_class_4_pct_10000m",
        "dw_class_4_pct_500m",
        "dw_class_5_pct_10000m",
        "dw_class_5_pct_500m",
        "dw_class_6_pct_10000m",
        "dw_class_6_pct_500m",
        "dw_class_7_pct_10000m",
        "dw_class_7_pct_500m",
        "dw_class_8_pct_10000m",
        "dw_class_8_pct_500m",
        "gmted_std_500m",
        "month",
        "obs_date",
        "prcp_mean_mean",
        "season",
        "species",
        "tmax_mean_mean",
        "tmin_mean_mean",
        ".geo",
    ]
    cols = [c for c in preferred if c in out_df.columns] + [
        c for c in out_df.columns if c not in preferred
    ]
    out_df = out_df[cols]

    for old in param_dir.glob(f"{SPECIES}_subset*.csv"):
        old.unlink()
        print(f"Removed {old.name}")

    out_df = out_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    n = len(out_df)
    edges = [int(round(i * n / n_subsets)) for i in range(n_subsets + 1)]
    for i in range(n_subsets):
        part = out_df.iloc[edges[i] : edges[i + 1]]
        path = param_dir / f"{SPECIES}_subset{i}.csv"
        part.to_csv(path, index=False)
        print(f"Wrote {path.name}: {len(part)} rows")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--downloads", type=Path, default=DEFAULT_DL)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--aoi", type=Path, default=DATAPREP / "mav_counties_4326.parquet")
    parser.add_argument("--skip-gbif", action="store_true")
    parser.add_argument(
        "--write-local-params",
        action="store_true",
        help="(Discouraged) sample local inference rasters instead of waiting for GEE",
    )
    parser.add_argument("--no-backup", action="store_true")
    args = parser.parse_args(argv)

    aoi_geom = _aoi_union(args.aoi)
    presence = load_du_capture(args.downloads, aoi_geom)

    gbif_root = args.data_root / "gbif" if args.data_root != DATA_ROOT else GBIF_ROOT
    param_dir = args.data_root / "param_csvs" if args.data_root != DATA_ROOT else PARAM_CSV_DIR
    raster_dir = (
        args.data_root / "rasters" / "inference"
        if args.data_root != DATA_ROOT
        else INFERENCE_RASTERS
    )

    if not args.skip_gbif:
        write_gbif_yearly(presence, gbif_root, backup=not args.no_backup)

    if args.write_local_params:
        build_param_csvs(presence, raster_dir, param_dir)
        note = "Param CSVs sampled from local inference rasters (not GEE Drive export)."
    else:
        note = (
            "GBIF presence only. Run export_ursus_gee.py or geeDataFromPoints.ipynb; "
            "download ursus_americanus_subset{0,1}.csv from Drive folder paramcsv_daymet "
            f"into {param_dir}, then retrain MaxEnt."
        )
        print(note)

    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "species": SPECIES,
        "sources": ["du_la", "capture"],
        "excluded": ["collar", "mortality_sheet"],
        "year_range": [YEAR_START, YEAR_END],
        "n_presence": int(len(presence)),
        "note": note,
    }
    meta_path = param_dir / f"{SPECIES}_du_capture_prep.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
