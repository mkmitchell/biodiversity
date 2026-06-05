"""
Convert per-download eBird TSV (and optional root Parquet) files to hive Parquet.

Reads observation files under EBIRD_INPUT_ROOT (default /mnt/f/ebird):
  ebd_<pkg>/ebd_<pkg>.txt

Writes hive layout for geeDataFromPoints:
  EBIRD_OUTPUT/scientific_name=<species>/*.parquet

Requires conda env rapids-25.10 (or any env with Polars installed).

Usage:
  conda activate rapids-25.10
  python -u convert_ebird_downloads.py
  python -u convert_ebird_downloads.py --dry-run
  python -u convert_ebird_downloads.py --only ebd_US_comyel_relApr-2026
  python -u convert_ebird_downloads.py --include-parquets
  EBIRD_INPUT_ROOT=/mnt/f/ebird EBIRD_OUTPUT=/mnt/c/ebirdpolars python -u convert_ebird_downloads.py

Do NOT run the full job inside a Jupyter cell; use a WSL terminal.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("POLARS_MAX_THREADS", "12")
os.environ.setdefault("POLARS_MAX_OPEN_PARTITIONS", "32")

import polars as pl

from ebird_polars_io import (
    PARTITION_COL,
    apply_completed_anti_join,
    convert_ebird_tsv_chunked,
    discover_main_tsvs,
    discover_root_parquets,
    list_completed_species,
    log,
    mark_partitions_complete,
    mark_species_complete,
    prepare_run,
    scan_ebird_parquet,
    sink_hive_partitions,
    source_marker_path,
    sync_manifest,
)

INPUT_ROOT = Path("/mnt/f/ebird")
OUTPUT_DIR = Path("/mnt/c/ebirdpolars")

CLEAN_INCOMPLETE = True


def source_stem(path: Path) -> str:
    if path.suffix == ".txt":
        return path.parent.name
    return path.stem


def source_is_complete(output_dir: Path, stem: str) -> bool:
    return source_marker_path(output_dir, stem).is_file()


def mark_source_complete(output_dir: Path, stem: str) -> None:
    source_marker_path(output_dir, stem).touch()


def convert_tsv(input_path: Path, output_dir: Path, *, chunk_rows: int) -> int:
    """Large EBD TSVs are converted in chunks to avoid OOM (Linux 'Killed')."""
    _rows, affected = convert_ebird_tsv_chunked(
        input_path, output_dir, chunk_rows=chunk_rows
    )
    return mark_species_complete(output_dir, affected)


def convert_parquet(input_path: Path, output_dir: Path) -> int:
    lf = apply_completed_anti_join(scan_ebird_parquet(input_path), output_dir)
    if PARTITION_COL not in lf.collect_schema().names():
        raise ValueError(
            f"{input_path} has no '{PARTITION_COL}' column after normalization"
        )
    sink_hive_partitions(lf, output_dir)
    return mark_partitions_complete(output_dir)


def discover_work(
    root: Path,
    *,
    only: str | None,
    include_parquets: bool,
) -> list[tuple[Path, str]]:
    """Return (path, kind) where kind is 'tsv' or 'parquet'."""
    work: list[tuple[Path, str]] = []

    for tsv in discover_main_tsvs(root):
        stem = source_stem(tsv)
        if only is not None and stem != only:
            continue
        work.append((tsv, "tsv"))

    if include_parquets:
        for pq in discover_root_parquets(root):
            stem = source_stem(pq)
            if only is not None and stem != only:
                continue
            work.append((pq, "parquet"))

    return work


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert per-download eBird files to hive Parquet"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List inputs that would be processed and exit",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Process only this download stem (folder/parquet basename)",
    )
    parser.add_argument(
        "--include-parquets",
        action="store_true",
        help="Also ingest top-level ebd_*.parquet files under the input root",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocess even if _source_<stem>.complete exists",
    )
    parser.add_argument(
        "--chunk-rows",
        type=int,
        default=250_000,
        help="Pandas read_csv chunksize for TSV conversion (lower if still OOM)",
    )
    args = parser.parse_args(argv)

    root = Path(os.environ.get("EBIRD_INPUT_ROOT", INPUT_ROOT))
    output_dir = Path(os.environ.get("EBIRD_OUTPUT", OUTPUT_DIR))

    if not root.is_dir():
        log(f"ERROR: input root not found: {root}")
        return 1

    work = discover_work(
        root, only=args.only, include_parquets=args.include_parquets
    )
    if not work:
        log(f"No matching eBird downloads under {root}")
        return 1

    log(f"Input root: {root}")
    log(f"Output:     {output_dir}")
    log(f"Found {len(work)} file(s) to consider")

    if args.dry_run:
        for path, kind in work:
            stem = source_stem(path)
            skip = (
                not args.force
                and output_dir.is_dir()
                and source_is_complete(output_dir, stem)
            )
            status = "skip (complete)" if skip else "process"
            log(f"  [{kind}] {path.name} -> {stem} ({status})")
        return 0

    prepare_run(output_dir, clean_incomplete=CLEAN_INCOMPLETE)

    processed = 0
    skipped = 0

    for i, (path, kind) in enumerate(work, start=1):
        stem = source_stem(path)
        if (
            not args.force
            and source_is_complete(output_dir, stem)
        ):
            log(f"[{i}/{len(work)}] skip {stem} (_source marker present)")
            skipped += 1
            continue

        log(f"[{i}/{len(work)}] {kind}: {path}")
        try:
            if kind == "tsv":
                marked = convert_tsv(path, output_dir, chunk_rows=args.chunk_rows)
            else:
                marked = convert_parquet(path, output_dir)
        except Exception as exc:
            log(f"ERROR processing {path}: {exc}")
            completed = list_completed_species(output_dir)
            sync_manifest(output_dir, completed)
            raise

        mark_source_complete(output_dir, stem)
        completed = list_completed_species(output_dir)
        sync_manifest(output_dir, completed)
        processed += 1
        log(
            f"  done {stem}: marked {marked} partition(s); "
            f"{len(completed)} species complete overall"
        )

    log(f"Finished: processed={processed}, skipped={skipped}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
