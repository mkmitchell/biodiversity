"""Pull GBIF occurrence CSVs for MAV species (extracted from pullgbif.ipynb)."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import Point

from ebird_polars_io import log
from paths import GBIF_ROOT
from species_manifest import DEFAULT_EXCEL, normalize_scientific_name

DATAPREP = Path(__file__).resolve().parent
DEFAULT_AOI = DATAPREP / "mav_counties_4326.parquet"
DEFAULT_UNCERTAINTY_LIMIT = 100


def scientific_name_lookup(excel_path: Path = DEFAULT_EXCEL) -> dict[str, str]:
    """Map normalized species keys to scientific names for GBIF API queries."""
    df = pd.read_excel(excel_path)
    lookup: dict[str, str] = {}
    for name in df["Scientific name"].astype(str):
        cleaned = name.strip()
        if not cleaned:
            continue
        lookup[normalize_scientific_name(cleaned)] = cleaned
    return lookup


def load_aoi(aoi_path: Path = DEFAULT_AOI) -> gpd.GeoDataFrame:
    return gpd.read_parquet(aoi_path).set_crs("EPSG:4326", allow_override=True)


def query_species(
    scientific_name: str,
    year: int,
    aoi: gpd.GeoDataFrame,
    *,
    max_records: int = 50_000,
) -> pd.DataFrame:
    base_url = "https://api.gbif.org/v1/occurrence/search"
    limit = 300
    offset = 0
    all_records: list[pd.DataFrame] = []

    bounds = aoi.total_bounds
    bbox = {
        "decimalLatitude": f"{bounds[1]},{bounds[3]}",
        "decimalLongitude": f"{bounds[0]},{bounds[2]}",
    }

    while True:
        params = {
            "scientificName": scientific_name,
            "hasCoordinate": "true",
            "limit": limit,
            "offset": offset,
            "year": year,
            **bbox,
        }
        response = requests.get(base_url, params=params, timeout=120)
        response.raise_for_status()
        data = response.json()

        records = data.get("results", [])
        if not records:
            break

        all_records.append(pd.json_normalize(records))
        offset += limit
        if offset >= data.get("count", 0) or offset >= max_records:
            break

    if not all_records:
        return pd.DataFrame()
    return pd.concat(all_records, ignore_index=True)


def pull_gbif_year(
    scientific_name: str,
    year: int,
    output_dir: Path,
    aoi: gpd.GeoDataFrame,
    *,
    dry_run: bool = False,
) -> Path | None:
    """Query GBIF, clip to AOI, and write {species_key}_{year}.csv."""
    safe_name = normalize_scientific_name(scientific_name)
    output_path = output_dir / f"{safe_name}_{year}.csv"
    if output_path.is_file():
        log(f"GBIF skip (exists): {output_path.name}")
        return output_path

    if dry_run:
        log(f"GBIF dry-run: would pull {scientific_name} {year} → {output_path}")
        return None

    df = query_species(scientific_name, year, aoi)
    if df.empty:
        log(f"GBIF no records: {scientific_name} {year}")
        return None

    df = df[df["decimalLatitude"].notna() & df["decimalLongitude"].notna()]
    if df.empty:
        log(f"GBIF no coordinates: {scientific_name} {year}")
        return None

    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["decimalLongitude"], df["decimalLatitude"]),
        crs="EPSG:4326",
    )
    gdf = gdf.rename(
        columns={"decimalLatitude": "Latitude", "decimalLongitude": "Longitude"}
    )
    clipped = gpd.overlay(gdf, aoi, how="intersection")
    if clipped.empty:
        log(f"GBIF no AOI records: {scientific_name} {year}")
        return None

    clipped = clipped.rename(columns=lambda column: column.lower().strip().replace(" ", "_"))
    if "species" in clipped.columns:
        clipped["species"] = clipped["species"].str.lower().str.strip().str.replace(" ", "_")

    output_dir.mkdir(parents=True, exist_ok=True)
    clipped.drop(columns="geometry").to_csv(output_path, index=False)
    log(f"GBIF saved {len(clipped)} records: {output_path}")
    return output_path


def pull_gbif_for_species(
    species_key: str,
    years: list[int],
    *,
    name_lookup: dict[str, str],
    output_dir: Path = GBIF_ROOT,
    aoi: gpd.GeoDataFrame | None = None,
    dry_run: bool = False,
) -> int:
    scientific_name = name_lookup.get(species_key)
    if not scientific_name:
        log(f"GBIF ERROR: no scientific name in Excel for key {species_key!r}")
        return 1

    aoi_gdf = None
    if not dry_run:
        aoi_gdf = aoi if aoi is not None else load_aoi()
    errors = 0
    for year in years:
        try:
            pull_gbif_year(
                scientific_name,
                year,
                output_dir,
                aoi_gdf,
                dry_run=dry_run,
            )
        except requests.RequestException as exc:
            log(f"GBIF ERROR {scientific_name} {year}: {exc}")
            errors += 1
    return 1 if errors else 0
