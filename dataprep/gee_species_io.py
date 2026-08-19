"""Load occurrence data for geeDataFromPoints (GBIF CSV vs eBird parquet)."""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Callable

import pandas as pd

from check_occurrences import gbif_stems_for_lookup, occurrence_source_for_group


def excel_group_for_species(species_key: str, manifest=None) -> str:
    """Return manifest Excel group (birds/amphibians/reptiles/mammals) for a species key."""
    if manifest is None:
        from species_manifest import load_species_manifest

        manifest = load_species_manifest()
    key = species_key.strip().lower().replace(" ", "_")
    rows = manifest[manifest["scientific_name"] == key]
    if rows.empty:
        return "birds"
    return str(rows.iloc[0]["excel_group"])


def gbif_csv_paths(gbif_root: Path | str, species_key: str) -> list[str]:
    """Return GBIF CSV paths for a manifest species (includes legacy filename stems)."""
    root = Path(gbif_root)
    paths: list[str] = []
    seen: set[str] = set()
    for stem in gbif_stems_for_lookup(species_key):
        for path in sorted(root.glob(f"{stem}_*.csv")):
            if path.is_file() and path.name not in seen:
                seen.add(path.name)
                paths.append(str(path))
    return paths


def load_species_data(
    species: str,
    excel_group: str,
    *,
    gbif_root: Path | str,
    parquet_folder: Path | str,
    con,
    eb_start_year: int = 2017,
    eb_end_year: int = 2024,
    log: Callable[[str], None] | None = None,
) -> pd.DataFrame | None:
    """Load MAV-filtered occurrence rows for GEE export."""
    log = log or print
    species = species.strip().lower().replace(" ", "_")
    group = excel_group.strip().lower()
    source = occurrence_source_for_group(group)

    if source.value == "gbif":
        return _load_gbif_csv(species, gbif_root=gbif_root, log=log)
    return _load_ebird_parquet(
        species,
        parquet_folder=parquet_folder,
        con=con,
        eb_start_year=eb_start_year,
        eb_end_year=eb_end_year,
        log=log,
    )


def _load_gbif_csv(
    species: str,
    *,
    gbif_root: Path | str,
    log: Callable[[str], None],
) -> pd.DataFrame | None:
    csv_files = gbif_csv_paths(gbif_root, species)
    if not csv_files:
        stems = gbif_stems_for_lookup(species)
        log(
            f"❌ No GBIF CSV for {species} (tried stems: {', '.join(stems)}) — "
            f"run: python -u check_occurrences.py --acquire"
        )
        return None

    log(f"Reading {len(csv_files)} GBIF CSV(s) for {species}")
    try:
        df_list = [pd.read_csv(f) for f in csv_files]
        df = pd.concat(df_list, ignore_index=True)
        df["date"] = pd.to_datetime(df[["year", "month", "day"]])
        df = df[
            [
                "basisofrecord",
                "species",
                "latitude",
                "longitude",
                "coordinateuncertaintyinmeters",
                "date",
            ]
        ]
        df = df.dropna(how="any")
        df = df[df["coordinateuncertaintyinmeters"] <= 100]
        if df.empty:
            log(f"❌ No GBIF records passed filters for {species}")
            return None
        log(f"✔ Loaded GBIF CSV for {species} ({len(df)} rows)")
        return df
    except Exception as exc:
        log(f"⚠️ Failed to read GBIF CSV for {species}: {exc}")
        return None


def _load_ebird_parquet(
    species: str,
    *,
    parquet_folder: Path | str,
    con,
    eb_start_year: int,
    eb_end_year: int,
    log: Callable[[str], None],
) -> pd.DataFrame | None:
    log("Reading parquet")
    query = f"""
            SELECT
                lower(eb.scientific_name) AS scientific_name,
                eb.observation_date,
                eb.protocol_name,
                eb.longitude,
                eb.latitude
            FROM read_parquet('{parquet_folder}/scientific_name={species}/*.parquet', hive_partitioning = true) AS eb
            JOIN aoi
                ON ST_Intersects(
                    ST_Point(CAST(eb.longitude AS DOUBLE), CAST(eb.latitude AS DOUBLE)),
                    aoi.geometry
                )
            WHERE lower(eb.scientific_name) = '{species}'
              AND year(CAST(eb.observation_date AS DATE)) BETWEEN {eb_start_year} AND {eb_end_year}
              AND (
                    CAST(eb.effort_distance_km AS DOUBLE) < 0.5
                    OR eb.protocol_name = 'Stationary'
              );
    """
    try:
        df = con.execute(query).fetchdf()
        df["date"] = pd.to_datetime(df["observation_date"])
        df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
        df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month
        df["day"] = df["date"].dt.day
        if df.empty:
            log(f"❌ No matching eBird records in MAV AOI for {species}")
            return None
        log(f"✔ Loaded Parquet data for {species} ({len(df)} rows)")
        return df
    except Exception as exc:
        err = str(exc)
        if "No files found" in err and f"scientific_name={species}" in err:
            log(
                f"❌ No eBird parquet for {species} — request EBD Custom Download, "
                f"convert to {parquet_folder}, then re-run. "
                f"Check gaps: python download_ebird_api_mav.py --mode gaps"
            )
        else:
            log(f"⚠️ DuckDB query failed for {species}: {exc}")
        return None
