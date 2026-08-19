"""
Extract MAV manifest bird species from the national US EBD TSV into hive parquet.

Unlike partition_ebird.py (all ~10k species), this scans the national file once but
only writes parquet for birds listed in mavBiodiversityToolSpeciesList.xlsx — typically
a few dozen species. Rows are also clipped to MAV states (AR/LA/MS) by default.

Still requires a full linear read of the national TSV (~440 GB) unless resuming from a
checkpoint. Checkpoints are written every 20 chunks and on Ctrl+C to
``ebirdpolars/_national_ebd_extract_checkpoint.json``.

Usage:
  conda activate biodiversity
  python -u extract_national_ebird.py --dry-run
  python -u extract_national_ebird.py --missing-only
  python -u extract_national_ebird.py --missing-only --resume   # default
  python -u extract_national_ebird.py --fresh                  # ignore checkpoint
  EBIRD_NATIONAL_TSV=/path/to/ebd_US_relSep-2025.txt python -u extract_national_ebird.py

Then:
  python -u check_occurrences.py
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("POLARS_MAX_THREADS", "12")

from ebird_polars_io import (
    clear_extract_checkpoint,
    extract_ebird_tsv_chunked,
    has_production_parquet_partition,
    load_extract_checkpoint,
    log,
    mark_species_complete,
    partition_dir,
    source_marker_path,
)
from paths import DEFAULT_EBIRD_PARQUET, NATIONAL_EBD_TSV
from species_manifest import DEFAULT_EXCEL, load_species_manifest

MAV_STATE_CODES = frozenset({"US-AR", "US-LA", "US-MS"})
SOURCE_STEM = "national_ebd_manifest"


def manifest_bird_species(excel_path: Path) -> list[str]:
    manifest = load_species_manifest(excel_path)
    birds = manifest.loc[manifest["excel_group"] == "birds", "scientific_name"]
    return sorted(birds.astype(str).tolist())


def source_is_complete(output_dir: Path) -> bool:
    return source_marker_path(output_dir, SOURCE_STEM).is_file()


def mark_source_complete(output_dir: Path) -> None:
    source_marker_path(output_dir, SOURCE_STEM).touch()


def extract_manifest_birds(
    input_path: Path,
    output_dir: Path,
    species_keys: list[str],
    *,
    state_codes: set[str] | None = None,
    skip_complete: bool = True,
    chunk_rows: int = 250_000,
    dry_run: bool = False,
    resume: bool = True,
    fresh: bool = False,
) -> int:
    if not input_path.is_file():
        log(f"ERROR: national EBD not found: {input_path}")
        return 1
    if not species_keys:
        log("No bird species to extract")
        return 0

    skip: set[str] = set()
    if skip_complete and not fresh:
        skip = {
            key
            for key in species_keys
            if has_production_parquet_partition(output_dir, key)
        }
        if skip:
            log(f"Skipping {len(skip)} species with existing parquet")

    targets = [key for key in species_keys if key not in skip]
    if not targets:
        log("All target species already have parquet partitions")
        return 0

    log(f"National EBD: {input_path}")
    log(f"Output:       {output_dir}")
    log(f"Extract:      {len(targets)} bird species")
    if dry_run:
        for key in targets:
            log(f"  would extract: {key}")
        return 0

    if fresh:
        clear_extract_checkpoint(output_dir)
        for key in targets:
            part = partition_dir(output_dir, key)
            if part.is_dir():
                shutil.rmtree(part)
                log(f"Removed partial partition: {key}")

    _rows, affected = extract_ebird_tsv_chunked(
        input_path,
        output_dir,
        target_species=set(targets),
        state_codes=state_codes,
        skip_species=skip,
        chunk_rows=chunk_rows,
        checkpoint_path=output_dir,
        resume=resume and not fresh,
    )
    marked = mark_species_complete(output_dir, affected)
    mark_source_complete(output_dir)
    log(f"Marked {marked} partition(s) complete")
    missing = set(targets) - affected
    if missing:
        log(f"WARNING: no rows matched for {len(missing)} species: {sorted(missing)}")
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Extract MAV manifest birds from national US EBD TSV"
    )
    parser.add_argument("--excel", type=Path, default=DEFAULT_EXCEL)
    parser.add_argument("--input", type=Path, default=NATIONAL_EBD_TSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_EBIRD_PARQUET)
    parser.add_argument(
        "--missing-only",
        action="store_true",
        help="Only species without production parquet yet (default)",
    )
    parser.add_argument(
        "--all-birds",
        action="store_true",
        help="Extract all manifest birds (re-read national file; may append parquet)",
    )
    parser.add_argument(
        "--species",
        action="append",
        default=None,
        help="Scientific name key(s) to extract (repeatable); default: all manifest birds",
    )
    parser.add_argument(
        "--states",
        action="append",
        default=None,
        help="STATE CODE filter (repeatable); default: US-AR US-LA US-MS",
    )
    parser.add_argument(
        "--no-state-filter",
        action="store_true",
        help="Keep all US rows for target species (larger parquet)",
    )
    parser.add_argument("--chunk-rows", type=int, default=250_000)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume from checkpoint if present (default: true)",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Delete checkpoint and scan from the start of the TSV",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-extract even when _source_national_ebd_manifest.complete exists",
    )
    args = parser.parse_args(argv)

    if not args.excel.is_file():
        log(f"ERROR: Excel not found: {args.excel}")
        return 1

    checkpoint = None if args.fresh else load_extract_checkpoint(args.output)
    if checkpoint and args.resume:
        species_keys = list(checkpoint.target_species)
        log(
            f"Found checkpoint at chunk {checkpoint.chunk_num} "
            f"(byte {checkpoint.byte_offset:,}) — resuming those species"
        )
    elif args.species:
        species_keys = [s.strip().lower().replace(" ", "_") for s in args.species]
    else:
        species_keys = manifest_bird_species(args.excel)

    if not checkpoint and not args.fresh and (args.missing_only or not args.all_birds):
        species_keys = [
            key
            for key in species_keys
            if not has_production_parquet_partition(args.output, key)
        ]

    if not species_keys:
        if args.fresh and args.species:
            log("ERROR: --fresh found no species to extract (check --species names)")
            return 1
        log("No bird species to extract")
        return 0

    if (
        not args.force
        and not args.dry_run
        and source_is_complete(args.output)
        and not species_keys
    ):
        log("National manifest extract already complete — nothing missing")
        return 0

    states = None if args.no_state_filter else set(args.states or MAV_STATE_CODES)

    return extract_manifest_birds(
        args.input,
        args.output,
        species_keys,
        state_codes=states,
        skip_complete=not args.all_birds and not args.force and not checkpoint and not args.fresh,
        chunk_rows=args.chunk_rows,
        dry_run=args.dry_run,
        resume=args.resume,
        fresh=args.fresh,
    )


if __name__ == "__main__":
    sys.exit(main())
