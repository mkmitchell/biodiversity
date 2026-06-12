"""Pipeline gap detection and test helpers for MAV species."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import polars as pl

from ebird_polars_io import COMPLETE_MARKER, PARTITION_PREFIX, log

DATAPREP = Path(__file__).resolve().parent
DEFAULT_EXCEL = DATAPREP / "mavBiodiversityToolSpeciesList.xlsx"
DEFAULT_EBD_ROOT = Path("/mnt/f/ebird")
DEFAULT_EBIRD_PARQUET = Path("/mnt/c/ebirdpolars")
DEFAULT_PARAM_CSV = Path("/mnt/f/readyparams/param_csvs")

# Species that failed MaxEnt due to missing param CSVs (or missing upstream data).
# Use for end-to-end pipeline testing: EBD → convert → geeDataFromPoints → MaxEnt.
GAP_TEST_SPECIES: tuple[str, ...] = (
    "agelaius_phoeniceus",  # no EBD folder → no ebirdpolars partition
    "egretta_caerulea",  # no EBD folder → no ebirdpolars partition
    "anthus_spragueii",  # parquet exists; geeDataFromPoints not run
    "coturnicops_noveboracensis",
    "geothlypis_trichas",
    "melospiza_georgiana",
    "melospiza_melodia",
)


class PipelineStage(str, Enum):
    MISSING_EBD = "need_ebd_download"
    NEED_CONVERT = "need_convert"
    NEED_GEE = "need_gee_export"
    READY_MAXENT = "ready_for_maxent"
    GBIF_ONLY = "gbif_only"
    NOT_IN_TAXONOMY = "not_in_ebird_taxonomy"


@dataclass(frozen=True)
class SpeciesPipelineStatus:
    species_key: str
    stage: PipelineStage
    ebird_code: str | None = None
    ebd_folders: tuple[str, ...] = ()
    note: str = ""


def has_parquet_partition(output_dir: Path, species_key: str) -> bool:
    part = output_dir / f"{PARTITION_PREFIX}{species_key}"
    return part.is_dir() and any(part.glob("*.parquet"))


def has_param_csvs(param_dir: Path, species_key: str, n_subsets: int = 2) -> bool:
    return all(
        (param_dir / f"{species_key}_subset{i}.csv").is_file() for i in range(n_subsets)
    )


def ebd_folders_for_code(ebd_root: Path, species_code: str) -> list[str]:
    code = species_code.strip().lower()
    if not code or not ebd_root.is_dir():
        return []
    return sorted(
        d.name
        for d in ebd_root.iterdir()
        if d.is_dir() and d.name.startswith("ebd_") and code in d.name.lower()
    )


def seed_test_parquet(
    target_species: str,
    template_species: str,
    output_dir: Path = DEFAULT_EBIRD_PARQUET,
    *,
    n_rows: int = 500,
    dry_run: bool = False,
) -> Path:
    """Copy a small parquet sample for pipeline testing without a real EBD download."""
    target = target_species.strip().lower().replace(" ", "_")
    template = template_species.strip().lower().replace(" ", "_")
    src_dir = output_dir / f"{PARTITION_PREFIX}{template}"
    dst_dir = output_dir / f"{PARTITION_PREFIX}{target}"

    if not src_dir.is_dir():
        raise FileNotFoundError(f"Template partition not found: {src_dir}")

    src_files = sorted(src_dir.glob("*.parquet"))
    if not src_files:
        raise FileNotFoundError(f"No parquet files in template partition: {src_dir}")

    log(f"Seeding test parquet: {target} ← {template} ({n_rows} rows from {src_files[0].name})")
    if dry_run:
        return dst_dir / "0.parquet"

    sample = pl.read_parquet(src_files[0], n_rows=n_rows).with_columns(
        pl.lit(target).alias("scientific_name"),
        pl.lit(target.replace("_", " ").title()).alias("common_name"),
    )

    dst_dir.mkdir(parents=True, exist_ok=True)
    out_path = dst_dir / "0.parquet"
    sample.write_parquet(out_path)
    (dst_dir / COMPLETE_MARKER).touch()
    log(f"Wrote {out_path} ({len(sample)} rows) — TEST DATA ONLY, replace with real EBD when available")
    return out_path


def classify_species(
    species_key: str,
    *,
    resolved: dict[str, dict] | None,
    ebd_root: Path,
    ebird_parquet: Path,
    param_dir: Path,
    n_subsets: int = 2,
) -> SpeciesPipelineStatus:
    key = species_key.strip().lower().replace(" ", "_")

    if resolved is not None and key not in resolved:
        if has_param_csvs(param_dir, key, n_subsets):
            return SpeciesPipelineStatus(key, PipelineStage.GBIF_ONLY, note="GBIF/herp CSV present")
        return SpeciesPipelineStatus(key, PipelineStage.NOT_IN_TAXONOMY)

    code = None
    folders: tuple[str, ...] = ()
    if resolved and key in resolved:
        code = str(resolved[key].get("speciesCode") or "")
        folders = tuple(ebd_folders_for_code(ebd_root, code))

    has_pq = has_parquet_partition(ebird_parquet, key)
    has_csv = has_param_csvs(param_dir, key, n_subsets)

    if has_csv:
        return SpeciesPipelineStatus(key, PipelineStage.READY_MAXENT, ebird_code=code, ebd_folders=folders)
    if has_pq:
        return SpeciesPipelineStatus(
            key,
            PipelineStage.NEED_GEE,
            ebird_code=code,
            ebd_folders=folders,
            note="Run geeDataFromPoints for this species",
        )
    if folders:
        return SpeciesPipelineStatus(
            key,
            PipelineStage.NEED_CONVERT,
            ebird_code=code,
            ebd_folders=folders,
            note=f"Run: python -u convert_ebird_downloads.py --only {folders[0]}",
        )
    return SpeciesPipelineStatus(
        key,
        PipelineStage.MISSING_EBD,
        ebird_code=code,
        note="Request EBD Custom Download or seed test parquet",
    )


def report_pipeline_gaps(
    species_keys: list[str],
    *,
    resolved: dict[str, dict] | None = None,
    ebd_root: Path = DEFAULT_EBD_ROOT,
    ebird_parquet: Path = DEFAULT_EBIRD_PARQUET,
    param_dir: Path = DEFAULT_PARAM_CSV,
) -> list[SpeciesPipelineStatus]:
    statuses = [
        classify_species(
            key,
            resolved=resolved,
            ebd_root=ebd_root,
            ebird_parquet=ebird_parquet,
            param_dir=param_dir,
        )
        for key in species_keys
    ]

    by_stage: dict[PipelineStage, list[SpeciesPipelineStatus]] = {}
    for status in statuses:
        by_stage.setdefault(status.stage, []).append(status)

    log(f"Pipeline status for {len(species_keys)} species")
    for stage in PipelineStage:
        items = by_stage.get(stage, [])
        if not items:
            continue
        log(f"  {stage.value}: {len(items)}")
        for item in items:
            extra = f" ({item.ebird_code})" if item.ebird_code else ""
            log(f"    {item.species_key}{extra}")
            if item.note:
                log(f"      → {item.note}")
            if item.ebd_folders:
                log(f"      ebd: {', '.join(item.ebd_folders)}")

    return statuses


def gap_test_species_for_gee(
    *,
    ebird_parquet: Path = DEFAULT_EBIRD_PARQUET,
    param_dir: Path = DEFAULT_PARAM_CSV,
) -> list[str]:
    """Gap test species that already have parquet and still need geeDataFromPoints."""
    return [
        key
        for key in GAP_TEST_SPECIES
        if has_parquet_partition(ebird_parquet, key)
        and not has_param_csvs(param_dir, key)
    ]


def gap_test_species_need_ebd(
    *,
    ebird_parquet: Path = DEFAULT_EBIRD_PARQUET,
) -> list[str]:
    """Gap test species with no parquet partition yet."""
    return [
        key
        for key in GAP_TEST_SPECIES
        if not has_parquet_partition(ebird_parquet, key)
    ]
