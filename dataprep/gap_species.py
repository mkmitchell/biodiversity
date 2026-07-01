"""Pipeline gap detection and GEE export queue for MAV species."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

from ebird_polars_io import has_production_parquet_partition, log
from paths import DEFAULT_EBD_ROOT, DEFAULT_EBIRD_PARQUET, DEFAULT_PARAM_CSV, GBIF_ROOT, LOGS_DIR

if TYPE_CHECKING:
    import pandas as pd

DATAPREP = Path(__file__).resolve().parent
DEFAULT_EXCEL = DATAPREP / "mavBiodiversityToolSpeciesList.xlsx"

# Species used in end-to-end pipeline tests (EBD → convert → geeDataFromPoints → MaxEnt).
GAP_TEST_SPECIES: tuple[str, ...] = (
    "agelaius_phoeniceus",
    "egretta_caerulea",
    "anthus_spragueii",
    "coturnicops_noveboracensis",
    "geothlypis_trichas",
    "melospiza_georgiana",
    "melospiza_melodia",
)


# Species that failed MaxEnt training — excluded from inference batch expectations.
# See runBatch_maxent.ipynb "Failed species" for diagnostics.
MAXENT_FAILED_SPECIES: dict[str, str] = {
    "dryophytes_squirella": "cannot concat empty list",
    "anthus_spragueii": "Not enough presence records",
    "centronyx_henslowii": "Not enough presence records",
    "coturnicops_noveboracensis": "singular KDE covariance (too few presence points)",
    "ursus_americanus": "Not enough presence records",
    "nerodia_cyclopion": "Not enough presence records",
    "regina_grahamii": "Not enough presence records",
}


def is_maxent_failed(species_key: str) -> bool:
    key = species_key.strip().lower().replace(" ", "_")
    return key in MAXENT_FAILED_SPECIES


def maxent_failure_reason(species_key: str) -> str | None:
    key = species_key.strip().lower().replace(" ", "_")
    return MAXENT_FAILED_SPECIES.get(key)


class PipelineStage(str, Enum):
    MISSING_EBD = "need_ebd_download"
    NEED_GBIF = "need_gbif_pull"
    NEED_CONVERT = "need_convert"
    NEED_GEE = "need_gee_export"
    READY_MAXENT = "ready_for_maxent"
    GBIF_ONLY = "gbif_only"
    NOT_IN_TAXONOMY = "not_in_ebird_taxonomy"


@dataclass(frozen=True)
class SpeciesPipelineStatus:
    species_key: str
    stage: PipelineStage
    excel_group: str = ""
    ebird_code: str | None = None
    ebd_folders: tuple[str, ...] = ()
    note: str = ""


def has_parquet_partition(output_dir: Path, species_key: str) -> bool:
    return has_production_parquet_partition(output_dir, species_key)


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


def has_gbif_occurrence_data(
    species_key: str,
    gbif_root: Path,
    *,
    gbif_index: dict[tuple[str, int], Path] | None = None,
) -> bool:
    """True when at least one GBIF year CSV exists for this species (incl. aliases)."""
    from check_occurrences import build_gbif_csv_index, gbif_stems_for_lookup

    index = gbif_index if gbif_index is not None else build_gbif_csv_index(gbif_root)
    for stem in gbif_stems_for_lookup(species_key):
        if any(species_part == stem for species_part, _year in index):
            return True
    return False


def _classify_bird(
    key: str,
    *,
    resolved: dict[str, dict] | None,
    ebd_root: Path,
    ebird_parquet: Path,
    param_dir: Path,
    excel_group: str,
    n_subsets: int,
) -> SpeciesPipelineStatus:
    if resolved is not None and key not in resolved:
        if has_param_csvs(param_dir, key, n_subsets):
            return SpeciesPipelineStatus(
                key,
                PipelineStage.READY_MAXENT,
                excel_group=excel_group,
                note="param CSVs present (not in eBird taxonomy cache)",
            )
        return SpeciesPipelineStatus(
            key,
            PipelineStage.NOT_IN_TAXONOMY,
            excel_group=excel_group,
            note="not found in eBird taxonomy — check scientific name in Excel",
        )

    code = None
    folders: tuple[str, ...] = ()
    if resolved and key in resolved:
        code = str(resolved[key].get("speciesCode") or "")
        folders = tuple(ebd_folders_for_code(ebd_root, code))

    has_pq = has_parquet_partition(ebird_parquet, key)
    has_csv = has_param_csvs(param_dir, key, n_subsets)

    if has_csv:
        return SpeciesPipelineStatus(
            key,
            PipelineStage.READY_MAXENT,
            excel_group=excel_group,
            ebird_code=code,
            ebd_folders=folders,
        )
    if has_pq:
        return SpeciesPipelineStatus(
            key,
            PipelineStage.NEED_GEE,
            excel_group=excel_group,
            ebird_code=code,
            ebd_folders=folders,
            note="Run geeDataFromPoints for this species",
        )
    if folders:
        return SpeciesPipelineStatus(
            key,
            PipelineStage.NEED_CONVERT,
            excel_group=excel_group,
            ebird_code=code,
            ebd_folders=folders,
            note=f"Run: python -u convert_ebird_downloads.py --only {folders[0]}",
        )
    return SpeciesPipelineStatus(
        key,
        PipelineStage.MISSING_EBD,
        excel_group=excel_group,
        ebird_code=code,
        note="Request EBD Custom Download",
    )


def _classify_gbif_group(
    key: str,
    *,
    gbif_root: Path,
    param_dir: Path,
    excel_group: str,
    gbif_index: dict[tuple[str, int], Path] | None,
    n_subsets: int,
) -> SpeciesPipelineStatus:
    if has_param_csvs(param_dir, key, n_subsets):
        return SpeciesPipelineStatus(
            key,
            PipelineStage.READY_MAXENT,
            excel_group=excel_group,
        )
    if has_gbif_occurrence_data(key, gbif_root, gbif_index=gbif_index):
        return SpeciesPipelineStatus(
            key,
            PipelineStage.NEED_GEE,
            excel_group=excel_group,
            note="Run geeDataFromPoints for this species",
        )
    return SpeciesPipelineStatus(
        key,
        PipelineStage.NEED_GBIF,
        excel_group=excel_group,
        note="Run: python -u check_occurrences.py --acquire",
    )


def classify_species(
    species_key: str,
    *,
    excel_group: str = "birds",
    resolved: dict[str, dict] | None,
    ebd_root: Path,
    ebird_parquet: Path,
    param_dir: Path,
    gbif_root: Path = GBIF_ROOT,
    gbif_index: dict[tuple[str, int], Path] | None = None,
    n_subsets: int = 2,
) -> SpeciesPipelineStatus:
    key = species_key.strip().lower().replace(" ", "_")
    group = excel_group.strip().lower()

    from check_occurrences import GBIF_GROUPS, occurrence_source_for_group

    source = occurrence_source_for_group(group)
    if source.value == "gbif":
        return _classify_gbif_group(
            key,
            gbif_root=gbif_root,
            param_dir=param_dir,
            excel_group=group,
            gbif_index=gbif_index,
            n_subsets=n_subsets,
        )
    return _classify_bird(
        key,
        resolved=resolved,
        ebd_root=ebd_root,
        ebird_parquet=ebird_parquet,
        param_dir=param_dir,
        excel_group=group,
        n_subsets=n_subsets,
    )


def classify_manifest(
    manifest: pd.DataFrame,
    *,
    resolved: dict[str, dict] | None = None,
    ebd_root: Path = DEFAULT_EBD_ROOT,
    ebird_parquet: Path = DEFAULT_EBIRD_PARQUET,
    param_dir: Path = DEFAULT_PARAM_CSV,
    gbif_root: Path = GBIF_ROOT,
    n_subsets: int = 2,
) -> list[SpeciesPipelineStatus]:
    from check_occurrences import build_gbif_csv_index

    gbif_index = build_gbif_csv_index(gbif_root)
    return [
        classify_species(
            row.scientific_name,
            excel_group=row.excel_group,
            resolved=resolved,
            ebd_root=ebd_root,
            ebird_parquet=ebird_parquet,
            param_dir=param_dir,
            gbif_root=gbif_root,
            gbif_index=gbif_index,
            n_subsets=n_subsets,
        )
        for row in manifest.itertuples(index=False)
    ]


def species_for_gee_export(
    manifest: pd.DataFrame,
    *,
    gbif_root: Path = GBIF_ROOT,
    ebird_parquet: Path = DEFAULT_EBIRD_PARQUET,
    param_dir: Path = DEFAULT_PARAM_CSV,
    ebd_root: Path = DEFAULT_EBD_ROOT,
    resolved: dict[str, dict] | None = None,
    n_subsets: int = 2,
) -> list[str]:
    """Manifest species with loadable occurrence data and missing param CSVs."""
    statuses = classify_manifest(
        manifest,
        resolved=resolved,
        ebd_root=ebd_root,
        ebird_parquet=ebird_parquet,
        param_dir=param_dir,
        gbif_root=gbif_root,
        n_subsets=n_subsets,
    )
    return sorted(
        status.species_key
        for status in statuses
        if status.stage is PipelineStage.NEED_GEE
    )


def report_pipeline_gaps(
    species_keys: list[str],
    *,
    excel_group: str = "birds",
    resolved: dict[str, dict] | None = None,
    ebd_root: Path = DEFAULT_EBD_ROOT,
    ebird_parquet: Path = DEFAULT_EBIRD_PARQUET,
    param_dir: Path = DEFAULT_PARAM_CSV,
    gbif_root: Path = GBIF_ROOT,
) -> list[SpeciesPipelineStatus]:
    statuses = [
        classify_species(
            key,
            excel_group=excel_group,
            resolved=resolved,
            ebd_root=ebd_root,
            ebird_parquet=ebird_parquet,
            param_dir=param_dir,
            gbif_root=gbif_root,
        )
        for key in species_keys
    ]
    return _log_pipeline_statuses(statuses, label=f"{len(species_keys)} species")


def report_manifest_pipeline(
    manifest: pd.DataFrame,
    *,
    resolved: dict[str, dict] | None = None,
    ebd_root: Path = DEFAULT_EBD_ROOT,
    ebird_parquet: Path = DEFAULT_EBIRD_PARQUET,
    param_dir: Path = DEFAULT_PARAM_CSV,
    gbif_root: Path = GBIF_ROOT,
) -> list[SpeciesPipelineStatus]:
    statuses = classify_manifest(
        manifest,
        resolved=resolved,
        ebd_root=ebd_root,
        ebird_parquet=ebird_parquet,
        param_dir=param_dir,
        gbif_root=gbif_root,
    )
    return _log_pipeline_statuses(statuses, label=f"{len(manifest)} manifest species")


def _log_pipeline_statuses(
    statuses: list[SpeciesPipelineStatus],
    *,
    label: str,
) -> list[SpeciesPipelineStatus]:
    by_stage: dict[PipelineStage, list[SpeciesPipelineStatus]] = {}
    for status in statuses:
        by_stage.setdefault(status.stage, []).append(status)

    log(f"Pipeline status for {label}")
    for stage in PipelineStage:
        items = by_stage.get(stage, [])
        if not items:
            continue
        log(f"  {stage.value}: {len(items)}")
        for item in items:
            extra = f" ({item.ebird_code})" if item.ebird_code else ""
            group = f" [{item.excel_group}]" if item.excel_group else ""
            log(f"    {item.species_key}{extra}{group}")
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


def _load_ebird_taxonomy_for_cli(excel_path: Path) -> dict[str, dict] | None:
    import os

    from check_occurrences import _load_ebird_taxonomy

    bird_count = 0
    try:
        from species_manifest import load_species_manifest

        manifest = load_species_manifest(excel_path)
        bird_count = int((manifest["excel_group"] == "birds").sum())
    except Exception:
        pass
    if not bird_count:
        return None
    verify_ssl = os.environ.get("EBIRD_INSECURE_SSL", "1").lower() not in ("1", "true", "yes")
    return _load_ebird_taxonomy(excel_path, DEFAULT_EBIRD_PARQUET, verify_ssl=verify_ssl)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="MAV pipeline stage report and GEE export queue",
    )
    parser.add_argument("--excel", type=Path, default=DEFAULT_EXCEL)
    parser.add_argument("--gbif-root", type=Path, default=GBIF_ROOT)
    parser.add_argument("--ebd-root", type=Path, default=DEFAULT_EBD_ROOT)
    parser.add_argument("--ebird-parquet", type=Path, default=DEFAULT_EBIRD_PARQUET)
    parser.add_argument("--param-dir", type=Path, default=DEFAULT_PARAM_CSV)
    parser.add_argument(
        "--gee-queue",
        action="store_true",
        help="Print species keys ready for geeDataFromPoints (occurrence OK, param CSVs missing)",
    )
    parser.add_argument(
        "--pipeline-report",
        action="store_true",
        help="Print full pipeline stage breakdown for all manifest species",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="With --gee-queue, write species list JSON to this path",
    )
    args = parser.parse_args(argv)

    if not args.excel.is_file():
        log(f"ERROR: Excel not found: {args.excel}")
        return 1

    if not args.gee_queue and not args.pipeline_report:
        parser.error("Specify --gee-queue and/or --pipeline-report")

    from species_manifest import load_species_manifest

    manifest = load_species_manifest(args.excel)
    resolved = _load_ebird_taxonomy_for_cli(args.excel)

    if args.pipeline_report:
        report_manifest_pipeline(
            manifest,
            resolved=resolved,
            ebd_root=args.ebd_root,
            ebird_parquet=args.ebird_parquet,
            param_dir=args.param_dir,
            gbif_root=args.gbif_root,
        )

    if args.gee_queue:
        queue = species_for_gee_export(
            manifest,
            gbif_root=args.gbif_root,
            ebird_parquet=args.ebird_parquet,
            param_dir=args.param_dir,
            ebd_root=args.ebd_root,
            resolved=resolved,
        )
        log("")
        log(f"GEE export queue: {len(queue)} species")
        for key in queue:
            log(f"  {key}")

        if args.json_out:
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            payload = {"species": queue, "count": len(queue)}
            args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            log(f"Wrote {args.json_out}")
        elif not args.pipeline_report:
            default_json = LOGS_DIR / "gee_queue.json"
            default_json.parent.mkdir(parents=True, exist_ok=True)
            default_json.write_text(
                json.dumps({"species": queue, "count": len(queue)}, indent=2),
                encoding="utf-8",
            )
            log(f"Wrote {default_json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
