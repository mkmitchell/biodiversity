"""
Partition a large eBird EBD TSV into per-species Parquet (hive layout).

Processes the file in prefix buckets (a-z, 0-9, etc.) to cap concurrent
partition writers and avoid Jupyter/OOM kills on ~10k species at once.

Resumable via .complete markers under scientific_name=<key>/ and
_completed_keys.parquet manifest.

Usage:
  python -u partition_ebird.py
  python -u partition_ebird.py --bucket s          # single bucket (smoke test)
  EBIRD_INPUT=/path/to.tsv EBIRD_OUTPUT=/path/out python -u partition_ebird.py

Do NOT run the full job inside a Jupyter cell; use a WSL terminal or:
  !python -u partition_ebird.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Tune Polars before import
os.environ.setdefault("POLARS_MAX_THREADS", "12")
os.environ.setdefault("POLARS_MAX_OPEN_PARTITIONS", "32")

import polars as pl

from ebird_polars_io import (
    COMPLETE_MARKER,
    PARTITION_COL,
    PARTITION_PREFIX,
    apply_completed_anti_join,
    build_schema,
    list_completed_species,
    log,
    manifest_path,
    prepare_run,
    scan_ebird_tsv_polars,
    sink_hive_partitions,
    sync_manifest,
)

# --- paths (override with EBIRD_INPUT / EBIRD_OUTPUT) ---
INPUT_PATH = Path("/mnt/e/backupfrompc/ebd_US_relSep-2025.txt")
OUTPUT_DIR = Path("/mnt/f/ebirdpolars")

BUCKET_COMPLETE_PREFIX = "_bucket_"

CLEAN_INCOMPLETE = True

# Single-char prefix buckets + catch-all for empty/other first chars
PREFIX_BUCKETS: tuple[str, ...] = tuple(
    "abcdefghijklmnopqrstuvwxyz0123456789_"
) + ("other",)


def bucket_filter(bucket: str) -> pl.Expr:
    if bucket == "other":
        first = (
            pl.col(PARTITION_COL)
            .str.strip_chars()
            .str.to_lowercase()
            .str.slice(0, 1)
        )
        return ~first.is_in(list(PREFIX_BUCKETS[:-1]))
    return (
        pl.col(PARTITION_COL)
        .str.strip_chars()
        .str.to_lowercase()
        .str.slice(0, 1)
        == bucket
    )


def bucket_marker_path(output_dir: Path, bucket: str) -> Path:
    return output_dir / f"{BUCKET_COMPLETE_PREFIX}{bucket}.complete"


def bucket_is_complete(output_dir: Path, bucket: str) -> bool:
    return bucket_marker_path(output_dir, bucket).is_file()


def mark_bucket_complete(output_dir: Path, bucket: str) -> None:
    bucket_marker_path(output_dir, bucket).touch()


def mark_partitions_complete(output_dir: Path, bucket: str | None = None) -> int:
    """Touch .complete on every partition dir that has Parquet data."""
    marked = 0
    if not output_dir.is_dir():
        return marked
    for child in output_dir.iterdir():
        if not child.is_dir() or not child.name.startswith(PARTITION_PREFIX):
            continue
        species = child.name[len(PARTITION_PREFIX) :]
        if bucket is not None and bucket != "other":
            if not species.startswith(bucket):
                continue
        elif bucket == "other":
            first = species[:1] if species else ""
            if first in PREFIX_BUCKETS[:-1]:
                continue
        if (child / COMPLETE_MARKER).is_file():
            continue
        if not any(child.glob("*.parquet")):
            continue
        (child / COMPLETE_MARKER).touch()
        marked += 1
    return marked


def build_lazy_frame(
    input_path: Path,
    schema: dict[str, pl.DataType],
    output_dir: Path,
    bucket: str | None = None,
) -> pl.LazyFrame:
    lf = apply_completed_anti_join(
        scan_ebird_tsv_polars(input_path, schema), output_dir
    )
    if bucket is not None:
        lf = lf.filter(bucket_filter(bucket))
    return lf


def run_bucket(
    input_path: Path,
    output_dir: Path,
    schema: dict[str, pl.DataType],
    bucket: str,
    index: int,
    total: int,
) -> int:
    log(f"Bucket {index}/{total}: '{bucket}' — scanning & sinking ...")
    lf = build_lazy_frame(input_path, schema, output_dir, bucket=bucket)
    sink_hive_partitions(lf, output_dir)
    marked = mark_partitions_complete(output_dir, bucket=bucket)
    mark_bucket_complete(output_dir, bucket)
    log(f"Bucket '{bucket}' done — marked {marked} partition(s) complete")
    return marked


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Partition eBird TSV to Parquet")
    parser.add_argument(
        "--bucket",
        type=str,
        default=None,
        help="Process only this prefix bucket (e.g. 's' or 'other') for smoke tests",
    )
    args = parser.parse_args(argv)

    input_path = Path(os.environ.get("EBIRD_INPUT", INPUT_PATH))
    output_dir = Path(os.environ.get("EBIRD_OUTPUT", OUTPUT_DIR))

    if not input_path.is_file():
        log(f"ERROR: input file not found: {input_path}")
        return 1

    if args.bucket is not None and args.bucket not in PREFIX_BUCKETS:
        log(f"ERROR: --bucket must be one of {PREFIX_BUCKETS}")
        return 1

    log(f"Input:  {input_path}")
    log(f"Output: {output_dir}")

    prepare_run(output_dir, clean_incomplete=CLEAN_INCOMPLETE)
    schema = build_schema(input_path)

    buckets: tuple[str, ...]
    if args.bucket is not None:
        buckets = (args.bucket,)
    else:
        buckets = PREFIX_BUCKETS

    total = len(buckets)
    any_work = False

    for i, bucket in enumerate(buckets, start=1):
        completed = list_completed_species(output_dir)
        sync_manifest(output_dir, completed)

        if bucket_is_complete(output_dir, bucket):
            log(f"Bucket {i}/{total}: '{bucket}' — skip (bucket marker present)")
            continue

        any_work = True
        try:
            run_bucket(input_path, output_dir, schema, bucket, i, total)
        except Exception as exc:
            log(f"ERROR in bucket '{bucket}': {exc}")
            completed = list_completed_species(output_dir)
            sync_manifest(output_dir, completed)
            raise

        completed = list_completed_species(output_dir)
        sync_manifest(output_dir, completed)
        log(f"Progress: {len(completed)} species complete overall")

    if not any_work and args.bucket is None:
        log("All buckets appear complete — nothing to do.")
    elif not any_work and args.bucket is not None:
        log(f"Bucket '{args.bucket}' — nothing to do (already complete).")

    log("Finished.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
