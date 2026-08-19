"""Check GBIF and eBird occurrence data against the Excel species manifest."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from ebird_polars_io import log
from gap_species import ebd_folders_for_code, has_parquet_partition
from paths import DEFAULT_EBD_ROOT, DEFAULT_EBIRD_PARQUET, GBIF_ROOT, NATIONAL_EBD_TSV
from species_manifest import DEFAULT_EXCEL, load_species_manifest

DATAPREP = Path(__file__).resolve().parent
CONVERT_SCRIPT = DATAPREP / "convert_ebird_downloads.py"

GBIF_GROUPS = frozenset({"amphibians", "reptiles", "mammals"})
EBIRD_GROUPS = frozenset({"birds"})
DEFAULT_GBIF_YEAR_MIN = 2017
DEFAULT_GBIF_YEAR_MAX = 2024
DEFAULT_EBIRD_YEAR_MIN = 1990
DEFAULT_EBIRD_YEAR_MAX = 2025
TAXONOMY_CACHE_NAME = "_ebird_taxonomy_species.json"

# Manifest uses current taxonomy; older GBIF pulls used prior genus/epithet spellings.
GBIF_FILENAME_ALIASES: dict[str, tuple[str, ...]] = {
    "dryophytes_avivoca": ("hyla_avivoca",),
    "dryophytes_chrysoscelis": ("hyla_chrysoscelis",),
    "dryophytes_cinereus": ("hyla_cinerea",),
    "dryophytes_squirella": ("hyla_squirella",),
}


class OccurrenceSource(str, Enum):
    GBIF = "gbif"
    EBIRD = "ebird"


class OccurrenceStatus(str, Enum):
    OK = "ok"
    MISSING = "missing"
    PARTIAL = "partial"
    NEED_CONVERT = "need_convert"
    NOT_IN_TAXONOMY = "not_in_taxonomy"


@dataclass(frozen=True)
class SpeciesOccurrenceStatus:
    scientific_name: str
    excel_group: str
    source: OccurrenceSource
    status: OccurrenceStatus
    detail: str = ""
    ebird_code: str | None = None
    ebd_folders: tuple[str, ...] = ()
    gbif_years_found: tuple[int, ...] = ()
    gbif_years_missing: tuple[int, ...] = ()


def occurrence_source_for_group(excel_group: str) -> OccurrenceSource:
    group = excel_group.strip().lower()
    if group in GBIF_GROUPS:
        return OccurrenceSource.GBIF
    if group in EBIRD_GROUPS:
        return OccurrenceSource.EBIRD
    raise ValueError(
        f"Unknown Excel group {excel_group!r}; expected one of "
        f"{sorted(GBIF_GROUPS | EBIRD_GROUPS)}"
    )


def expected_gbif_years(year_min: int, year_max: int) -> list[int]:
    if year_max < year_min:
        raise ValueError(f"year_max ({year_max}) must be >= year_min ({year_min})")
    return list(range(year_min, year_max + 1))


def normalize_gbif_species_key(species_key: str) -> str:
    return species_key.strip().lower().replace(" ", "_")


def gbif_stems_for_lookup(species_key: str) -> tuple[str, ...]:
    key = normalize_gbif_species_key(species_key)
    return (key, *GBIF_FILENAME_ALIASES.get(key, ()))


def build_gbif_csv_index(gbif_root: Path) -> dict[tuple[str, int], Path]:
    """Map (normalized species stem, year) -> CSV path anywhere under gbif_root."""
    index: dict[tuple[str, int], Path] = {}
    if not gbif_root.is_dir():
        return index
    for path in gbif_root.rglob("*.csv"):
        stem = path.stem.lower()
        if "_" not in stem:
            continue
        species_part, year_str = stem.rsplit("_", 1)
        if not year_str.isdigit() or len(year_str) != 4:
            continue
        key = (species_part, int(year_str))
        existing = index.get(key)
        if existing is None or len(path.parts) < len(existing.parts):
            index[key] = path
    return index


def locate_gbif_year_csv(
    species_key: str,
    year: int,
    *,
    gbif_index: dict[tuple[str, int], Path],
) -> Path | None:
    for stem in gbif_stems_for_lookup(species_key):
        path = gbif_index.get((stem, year))
        if path is not None:
            return path
    return None


def log_gbif_root_summary(gbif_root: Path) -> None:
    if not gbif_root.is_dir():
        log(f"WARNING: GBIF root does not exist: {gbif_root}")
        log(
            "  Mount the data drive or set BIODIVERSITY_DATA_ROOT / --gbif-root "
            "(default expects /mnt/f/biodiversity/gbif)"
        )
        return
    all_csvs = list(gbif_root.rglob("*.csv"))
    top_level = list(gbif_root.glob("*.csv"))
    log(f"GBIF CSV files: {len(all_csvs)} total ({len(top_level)} in top-level folder)")
    if all_csvs and not top_level:
        subdirs = sorted(
            {p.parent.relative_to(gbif_root) for p in all_csvs if p.parent != gbif_root}
        )
        preview = ", ".join(str(d) for d in subdirs[:3])
        log(f"  CSVs are only in subfolders ({preview}); checker searches recursively")


def check_gbif_species(
    species_key: str,
    excel_group: str,
    gbif_root: Path,
    *,
    year_min: int = DEFAULT_GBIF_YEAR_MIN,
    year_max: int = DEFAULT_GBIF_YEAR_MAX,
    gbif_index: dict[tuple[str, int], Path] | None = None,
) -> SpeciesOccurrenceStatus:
    key = normalize_gbif_species_key(species_key)
    index = gbif_index if gbif_index is not None else build_gbif_csv_index(gbif_root)
    found: list[int] = []
    missing: list[int] = []
    for year in expected_gbif_years(year_min, year_max):
        if locate_gbif_year_csv(key, year, gbif_index=index) is not None:
            found.append(year)
        else:
            missing.append(year)

    if not found:
        status = OccurrenceStatus.MISSING
        detail = f"no GBIF CSV files under {gbif_root} — run check_occurrences.py --acquire"
    elif missing:
        status = OccurrenceStatus.PARTIAL
        detail = f"missing years {missing} — run check_occurrences.py --acquire"
    else:
        status = OccurrenceStatus.OK
        detail = f"{len(found)} year files ({year_min}-{year_max})"

    return SpeciesOccurrenceStatus(
        scientific_name=key,
        excel_group=excel_group,
        source=OccurrenceSource.GBIF,
        status=status,
        detail=detail,
        gbif_years_found=tuple(found),
        gbif_years_missing=tuple(missing),
    )


def check_ebird_species(
    species_key: str,
    excel_group: str,
    *,
    resolved: dict[str, dict],
    ebd_root: Path,
    ebird_parquet: Path,
) -> SpeciesOccurrenceStatus:
    key = species_key.strip().lower().replace(" ", "_")
    record = resolved.get(key)
    if record is None:
        return SpeciesOccurrenceStatus(
            scientific_name=key,
            excel_group=excel_group,
            source=OccurrenceSource.EBIRD,
            status=OccurrenceStatus.NOT_IN_TAXONOMY,
            detail="not found in eBird taxonomy — check scientific name in Excel",
        )

    code = str(record.get("speciesCode") or "")
    folders = tuple(ebd_folders_for_code(ebd_root, code))
    has_pq = has_parquet_partition(ebird_parquet, key)

    if has_pq:
        status = OccurrenceStatus.OK
        detail = "parquet partition present"
        if folders:
            detail += f" (ebd: {folders[0]})"
    elif folders:
        status = OccurrenceStatus.NEED_CONVERT
        detail = f"EBD on disk — run: python -u convert_ebird_downloads.py --only {folders[0]}"
    else:
        status = OccurrenceStatus.MISSING
        detail = (
            f"need EBD Custom Download for code {code} — "
            "https://support.ebird.org/en/support/solutions/articles/48000838205-download-ebird-data"
        )

    return SpeciesOccurrenceStatus(
        scientific_name=key,
        excel_group=excel_group,
        source=OccurrenceSource.EBIRD,
        status=status,
        detail=detail,
        ebird_code=code or None,
        ebd_folders=folders,
    )


def check_manifest_occurrences(
    manifest,
    *,
    gbif_root: Path = GBIF_ROOT,
    ebd_root: Path = DEFAULT_EBD_ROOT,
    ebird_parquet: Path = DEFAULT_EBIRD_PARQUET,
    resolved: dict[str, dict] | None,
    year_min: int = DEFAULT_GBIF_YEAR_MIN,
    year_max: int = DEFAULT_GBIF_YEAR_MAX,
) -> list[SpeciesOccurrenceStatus]:
    statuses: list[SpeciesOccurrenceStatus] = []
    gbif_index = build_gbif_csv_index(gbif_root)
    for row in manifest.itertuples(index=False):
        source = occurrence_source_for_group(row.excel_group)
        if source is OccurrenceSource.GBIF:
            statuses.append(
                check_gbif_species(
                    row.scientific_name,
                    row.excel_group,
                    gbif_root,
                    year_min=year_min,
                    year_max=year_max,
                    gbif_index=gbif_index,
                )
            )
        else:
            if resolved is None:
                raise ValueError("eBird taxonomy required to check bird species")
            statuses.append(
                check_ebird_species(
                    row.scientific_name,
                    row.excel_group,
                    resolved=resolved,
                    ebd_root=ebd_root,
                    ebird_parquet=ebird_parquet,
                )
            )
    return statuses


def _load_ebird_taxonomy(
    excel_path: Path,
    ebird_parquet: Path,
    *,
    verify_ssl: bool,
) -> dict[str, dict]:
    from download_ebird_api_mav import (
        load_taxonomy_cache,
        read_species_from_excel,
        resolve_species_codes,
    )

    api_key = os.environ.get("EBIRD_API_KEY", "").strip()
    tax_cache = ebird_parquet / TAXONOMY_CACHE_NAME
    if not tax_cache.is_file() and not api_key:
        raise RuntimeError(
            "Set EBIRD_API_KEY or run download_ebird_api_mav.py --mode manifest once "
            f"to cache taxonomy at {tax_cache}"
        )

    scientific, common = read_species_from_excel(excel_path)
    if tax_cache.is_file() and not api_key:
        log(f"Using cached taxonomy: {tax_cache}")
        with tax_cache.open(encoding="utf-8") as handle:
            taxa = json.load(handle)
    else:
        taxa = load_taxonomy_cache(ebird_parquet, api_key, verify_ssl=verify_ssl)
    return resolve_species_codes(scientific, common, taxa)


def report_occurrence_statuses(statuses: list[SpeciesOccurrenceStatus]) -> int:
    by_source: dict[OccurrenceSource, list[SpeciesOccurrenceStatus]] = {}
    for status in statuses:
        by_source.setdefault(status.source, []).append(status)

    problems = 0
    for source in (OccurrenceSource.GBIF, OccurrenceSource.EBIRD):
        items = by_source.get(source, [])
        if not items:
            continue
        log("")
        log(f"=== {source.value.upper()} ({len(items)} species) ===")
        for status in sorted(items, key=lambda s: (s.status.value, s.scientific_name)):
            prefix = status.status.value.upper()
            extra = f" [{status.ebird_code}]" if status.ebird_code else ""
            log(f"  {prefix:14} {status.scientific_name}{extra}  ({status.excel_group})")
            log(f"                 {status.detail}")
            if status.status is not OccurrenceStatus.OK:
                problems += 1

    ok = len(statuses) - problems
    log("")
    log(f"Summary: {len(statuses)} species — {ok} ok, {problems} need action")
    return 1 if problems else 0


def _years_to_pull(status: SpeciesOccurrenceStatus, year_min: int, year_max: int) -> list[int]:
    if status.status is OccurrenceStatus.PARTIAL:
        return list(status.gbif_years_missing)
    return expected_gbif_years(year_min, year_max)


def acquire_occurrences(
    statuses: list[SpeciesOccurrenceStatus],
    *,
    excel_path: Path,
    gbif_root: Path,
    ebd_root: Path,
    ebird_parquet: Path,
    resolved: dict[str, dict] | None,
    year_min: int,
    year_max: int,
    dry_run: bool = False,
) -> int:
    """Pull GBIF CSVs and convert on-disk eBird TSVs based on check results."""
    from pull_gbif import pull_gbif_for_species, scientific_name_lookup

    exit_code = 0
    name_lookup = scientific_name_lookup(excel_path)

    gbif_targets = [
        s
        for s in statuses
        if s.source is OccurrenceSource.GBIF
        and s.status in (OccurrenceStatus.MISSING, OccurrenceStatus.PARTIAL)
    ]
    if gbif_targets:
        log("")
        log(f"GBIF acquire: {len(gbif_targets)} species")
        for status in gbif_targets:
            years = _years_to_pull(status, year_min, year_max)
            log(f"  {status.scientific_name}: years {years}")
            code = pull_gbif_for_species(
                status.scientific_name,
                years,
                name_lookup=name_lookup,
                output_dir=gbif_root,
                dry_run=dry_run,
            )
            exit_code = max(exit_code, code)

    convert_folders: list[str] = []
    for status in statuses:
        if status.source is not OccurrenceSource.EBIRD:
            continue
        if status.status is OccurrenceStatus.NEED_CONVERT:
            convert_folders.extend(status.ebd_folders)

    if convert_folders:
        log("")
        log(f"eBird convert: {len(convert_folders)} EBD folder(s)")
        for folder in sorted(set(convert_folders)):
            if dry_run:
                log(f"  dry-run: python -u convert_ebird_downloads.py --only {folder}")
                continue
            env = os.environ.copy()
            env["EBIRD_INPUT_ROOT"] = str(ebd_root)
            env["EBIRD_OUTPUT"] = str(ebird_parquet)
            cmd = [
                sys.executable,
                "-u",
                str(CONVERT_SCRIPT),
                "--only",
                folder,
            ]
            log(f"  running: {' '.join(cmd)}")
            result = subprocess.run(cmd, cwd=DATAPREP, env=env)
            exit_code = max(exit_code, result.returncode)

    missing_birds = [
        s
        for s in statuses
        if s.source is OccurrenceSource.EBIRD and s.status is OccurrenceStatus.MISSING
    ]
    if missing_birds:
        from extract_national_ebird import MAV_STATE_CODES, extract_manifest_birds

        if NATIONAL_EBD_TSV.is_file():
            log("")
            log(
                f"eBird national extract: {len(missing_birds)} species "
                f"(scans entire {NATIONAL_EBD_TSV} — expect hours for ~440 GB)"
            )
            keys = [s.scientific_name for s in missing_birds]
            code = extract_manifest_birds(
                NATIONAL_EBD_TSV,
                ebird_parquet,
                keys,
                state_codes=set(MAV_STATE_CODES),
                skip_complete=True,
                dry_run=dry_run,
            )
            exit_code = max(exit_code, code)
        else:
            from download_ebird_api_mav import write_manifest

            log("")
            log(f"eBird manual download needed: {len(missing_birds)} species")
            log(f"  (no national EBD at {NATIONAL_EBD_TSV})")
            if resolved:
                missing_resolved = {
                    s.scientific_name: resolved[s.scientific_name]
                    for s in missing_birds
                    if s.scientific_name in resolved
                }
                if missing_resolved and not dry_run:
                    manifest_path = write_manifest(
                        ebird_parquet,
                        missing_resolved,
                        start_year=DEFAULT_EBIRD_YEAR_MIN,
                        end_year=DEFAULT_EBIRD_YEAR_MAX,
                    )
                    log(f"  wrote download manifest: {manifest_path}")
            for status in missing_birds:
                log(f"  {status.scientific_name} [{status.ebird_code}]")
                log(
                    "    → EBD Custom Download: "
                    "https://support.ebird.org/en/support/solutions/articles/48000838205-download-ebird-data"
                )
                log(f"    → save under {ebd_root}/ebd_<pkg>/ebd_<pkg>.txt")
                log("    → then: python -u check_occurrences.py --acquire")
                log(
                    f"    → or set EBIRD_NATIONAL_TSV and re-run --acquire "
                    f"to extract from national file"
                )

    if not gbif_targets and not convert_folders and not missing_birds:
        log("")
        log("Acquire: nothing to do (all occurrence files present or only manual eBird steps remain)")

    return exit_code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check GBIF and eBird occurrence files against the Excel species manifest",
    )
    parser.add_argument("--excel", type=Path, default=DEFAULT_EXCEL)
    parser.add_argument("--gbif-root", type=Path, default=GBIF_ROOT)
    parser.add_argument("--ebd-root", type=Path, default=DEFAULT_EBD_ROOT)
    parser.add_argument("--ebird-parquet", type=Path, default=DEFAULT_EBIRD_PARQUET)
    parser.add_argument("--year-min", type=int, default=DEFAULT_GBIF_YEAR_MIN)
    parser.add_argument("--year-max", type=int, default=DEFAULT_GBIF_YEAR_MAX)
    parser.add_argument(
        "--acquire",
        action="store_true",
        help="After checking, pull GBIF gaps and convert on-disk eBird TSVs",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="With --acquire, print actions without downloading or converting",
    )
    parser.add_argument(
        "--insecure",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("EBIRD_INSECURE_SSL", "1").lower() in ("1", "true", "yes"),
    )
    args = parser.parse_args(argv)

    if not args.excel.is_file():
        log(f"ERROR: Excel not found: {args.excel}")
        return 1

    manifest = load_species_manifest(args.excel)
    log(f"Loaded {len(manifest)} species from {args.excel.name}")
    log(f"  GBIF groups: {sorted(GBIF_GROUPS)}")
    log(f"  eBird groups: {sorted(EBIRD_GROUPS)}")
    log(f"GBIF root: {args.gbif_root}")
    log_gbif_root_summary(args.gbif_root)
    log(f"EBD root: {args.ebd_root}")
    log(f"Parquet: {args.ebird_parquet}")

    bird_count = int((manifest["excel_group"] == "birds").sum())
    resolved: dict[str, dict] | None = None
    if bird_count:
        try:
            resolved = _load_ebird_taxonomy(
                args.excel,
                args.ebird_parquet,
                verify_ssl=not args.insecure,
            )
            log(f"Resolved {len(resolved)} eBird taxonomy entries")
        except RuntimeError as exc:
            log(f"ERROR: {exc}")
            return 1

    statuses = check_manifest_occurrences(
        manifest,
        gbif_root=args.gbif_root,
        ebd_root=args.ebd_root,
        ebird_parquet=args.ebird_parquet,
        resolved=resolved,
        year_min=args.year_min,
        year_max=args.year_max,
    )

    if args.acquire:
        log("")
        log("=== ACQUIRE ===")
        acquire_code = acquire_occurrences(
            statuses,
            excel_path=args.excel,
            gbif_root=args.gbif_root,
            ebd_root=args.ebd_root,
            ebird_parquet=args.ebird_parquet,
            resolved=resolved,
            year_min=args.year_min,
            year_max=args.year_max,
            dry_run=args.dry_run,
        )
        if acquire_code != 0:
            return acquire_code
        log("")
        log("=== RE-CHECK ===")
        statuses = check_manifest_occurrences(
            manifest,
            gbif_root=args.gbif_root,
            ebd_root=args.ebd_root,
            ebird_parquet=args.ebird_parquet,
            resolved=resolved,
            year_min=args.year_min,
            year_max=args.year_max,
        )

    return report_occurrence_statuses(statuses)


if __name__ == "__main__":
    sys.exit(main())
