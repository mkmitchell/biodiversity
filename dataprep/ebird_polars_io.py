"""Shared Polars helpers for eBird TSV/Parquet → hive partitions."""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

PARTITION_COL = "scientific_name"
PARTITION_PREFIX = f"{PARTITION_COL}="
COMPLETE_MARKER = ".complete"
MANIFEST_NAME = "_completed_keys.parquet"
SOURCE_COMPLETE_PREFIX = "_source_"

# Metadata sidecar files inside eBird download folders (not observation TSVs).
METADATA_TXT_NAMES: frozenset[str] = frozenset(
    {
        "BCRCodes.txt",
        "BirdLifeKBACodes.txt",
        "IBACodes.txt",
        "Protocols.txt",
        "USFWSCodes.txt",
        "recommended_citation.txt",
        "terms_of_use.txt",
    }
)


def log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{ts}] {msg}", flush=True)


def partition_dir(output_dir: Path, species_key: str) -> Path:
    return output_dir / f"{PARTITION_PREFIX}{species_key}"


def manifest_path(output_dir: Path) -> Path:
    return output_dir / MANIFEST_NAME


def source_marker_path(output_dir: Path, stem: str) -> Path:
    return output_dir / f"{SOURCE_COMPLETE_PREFIX}{stem}.complete"


def list_completed_species(output_dir: Path) -> set[str]:
    """Species keys that finished in a prior run (.complete marker)."""
    completed: set[str] = set()
    if not output_dir.is_dir():
        return completed
    for child in output_dir.iterdir():
        if not child.is_dir() or not child.name.startswith(PARTITION_PREFIX):
            continue
        if (child / COMPLETE_MARKER).is_file():
            completed.add(child.name[len(PARTITION_PREFIX) :])
    return completed


def sync_manifest(output_dir: Path, completed: set[str]) -> None:
    """Persist completed keys for anti-join (scales better than is_in(list))."""
    if not completed:
        mp = manifest_path(output_dir)
        if mp.is_file():
            mp.unlink()
        return
    pl.DataFrame({PARTITION_COL: sorted(completed)}).write_parquet(
        manifest_path(output_dir)
    )


def load_completed_lazy(output_dir: Path) -> pl.LazyFrame | None:
    mp = manifest_path(output_dir)
    if mp.is_file():
        return pl.scan_parquet(mp)
    return None


def promote_valid_partitions(output_dir: Path, completed: set[str]) -> int:
    """
    Parquet written but .complete missing (e.g. kernel died after sink).
    If files read cleanly, treat as done so we do not delete and rewrite.
    """
    promoted = 0
    if not output_dir.is_dir():
        return promoted
    for child in output_dir.iterdir():
        if not child.is_dir() or not child.name.startswith(PARTITION_PREFIX):
            continue
        species = child.name[len(PARTITION_PREFIX) :]
        if species in completed or (child / COMPLETE_MARKER).is_file():
            continue
        parquets = list(child.glob("*.parquet"))
        if not parquets:
            continue
        try:
            pl.scan_parquet(parquets).collect_schema()
        except Exception:
            continue
        (child / COMPLETE_MARKER).touch()
        completed.add(species)
        promoted += 1
    return promoted


def clean_incomplete_partitions(output_dir: Path, completed: set[str]) -> list[str]:
    """Drop crashed partial dirs so those species are rewritten."""
    removed: list[str] = []
    if not output_dir.is_dir():
        return removed
    for child in output_dir.iterdir():
        if not child.is_dir() or not child.name.startswith(PARTITION_PREFIX):
            continue
        species = child.name[len(PARTITION_PREFIX) :]
        if species in completed or (child / COMPLETE_MARKER).is_file():
            continue
        shutil.rmtree(child)
        removed.append(species)
    return removed


def mark_partitions_complete(
    output_dir: Path,
    *,
    species_prefix: str | None = None,
) -> int:
    """Touch .complete on every partition dir that has Parquet data."""
    marked = 0
    if not output_dir.is_dir():
        return marked
    for child in output_dir.iterdir():
        if not child.is_dir() or not child.name.startswith(PARTITION_PREFIX):
            continue
        species = child.name[len(PARTITION_PREFIX) :]
        if species_prefix is not None and not species.startswith(species_prefix):
            continue
        if (child / COMPLETE_MARKER).is_file():
            continue
        if not any(child.glob("*.parquet")):
            continue
        (child / COMPLETE_MARKER).touch()
        marked += 1
    return marked


def prepare_run(output_dir: Path, *, clean_incomplete: bool = True) -> set[str]:
    output_dir.mkdir(parents=True, exist_ok=True)

    completed = list_completed_species(output_dir)
    promoted = promote_valid_partitions(output_dir, completed)
    if promoted:
        log(f"Promoted {promoted} partition(s) with valid Parquet (no .complete yet)")

    if clean_incomplete:
        removed = clean_incomplete_partitions(output_dir, completed)
        if removed:
            log(f"Removed {len(removed)} incomplete partition dir(s)")

    sync_manifest(output_dir, completed)
    log(f"Already complete: {len(completed)} species")
    return completed


def build_schema(input_path: Path) -> dict[str, pl.DataType]:
    header_lf = pl.scan_csv(input_path, separator="\t", n_rows=0)
    return {name: pl.Utf8 for name in header_lf.collect_schema().names()}


def normalize_columns(lf: pl.LazyFrame) -> pl.LazyFrame:
    return lf.rename(lambda c: c.lower().strip().replace(" ", "_")).with_columns(
        pl.col(PARTITION_COL)
        .str.strip_chars()
        .str.to_lowercase()
        .str.replace_all(" ", "_")
    )


def scan_ebird_tsv_polars(
    input_path: Path, schema: dict[str, pl.DataType]
) -> pl.LazyFrame:
    """Streaming Polars CSV — fine for clean/national files; can fail on messy comments."""
    lf = pl.scan_csv(
        input_path,
        separator="\t",
        quote_char='"',
        has_header=True,
        schema_overrides=schema,
        infer_schema_length=0,
        truncate_ragged_lines=True,
        ignore_errors=True,
        low_memory=True,
        rechunk=False,
    )
    return normalize_columns(lf)


def scan_ebird_tsv_pyarrow(input_path: Path) -> pl.LazyFrame:
    """
    Read per-download eBird TSVs (messy quotes / embedded newlines).

    Tries PyArrow first (fast when the file is clean). Falls back to pandas
    ``on_bad_lines='warn'`` when PyArrow hits ragged rows (e.g. 53 vs 28 cols).
    """
    import csv

    import pandas as pd
    import pyarrow.csv as pacsv

    parse_opts = pacsv.ParseOptions(
        delimiter="\t",
        quote_char='"',
        double_quote=True,
        newlines_in_values=True,
    )
    try:
        table = pacsv.read_csv(
            str(input_path),
            read_options=pacsv.ReadOptions(encoding="utf8"),
            parse_options=parse_opts,
        )
        return normalize_columns(pl.from_arrow(table).lazy())
    except Exception as exc:
        log(
            f"PyArrow CSV failed for {input_path.name} ({exc!s:.120}); "
            "falling back to pandas ..."
        )
        df = pd.read_csv(
            input_path,
            sep="\t",
            dtype=str,
            low_memory=False,
            quoting=csv.QUOTE_MINIMAL,
            on_bad_lines="warn",
        )
        log(f"pandas read {input_path.name}: {len(df):,} rows x {len(df.columns)} cols")
        return normalize_columns(pl.from_pandas(df).lazy())


def scan_ebird_tsv(input_path: Path, schema: dict[str, pl.DataType]) -> pl.LazyFrame:
    """Default TSV reader for per-download conversion (PyArrow)."""
    _ = schema  # PyArrow infers from header; kept for API compatibility
    return scan_ebird_tsv_pyarrow(input_path)


def scan_ebird_parquet(input_path: Path) -> pl.LazyFrame:
    return normalize_columns(pl.scan_parquet(input_path))


def apply_completed_anti_join(
    lf: pl.LazyFrame, output_dir: Path
) -> pl.LazyFrame:
    completed_lf = load_completed_lazy(output_dir)
    if completed_lf is not None:
        lf = lf.join(completed_lf, on=PARTITION_COL, how="anti")
    return lf


def sink_hive_partitions(lf: pl.LazyFrame, output_dir: Path) -> None:
    lf.sink_parquet(
        pl.PartitionByKey(str(output_dir), by=[PARTITION_COL]),
        mkdir=True,
        compression="snappy",
    )


def mark_species_complete(output_dir: Path, species_keys: set[str]) -> int:
    """Touch .complete only for the given species keys."""
    marked = 0
    for key in species_keys:
        if not key:
            continue
        part_dir = output_dir / f"{PARTITION_PREFIX}{key}"
        if not part_dir.is_dir() or not any(part_dir.glob("*.parquet")):
            continue
        (part_dir / COMPLETE_MARKER).touch()
        marked += 1
    return marked


def convert_ebird_tsv_chunked(
    input_path: Path,
    output_dir: Path,
    *,
    chunk_rows: int = 250_000,
) -> tuple[int, set[str]]:
    """
    Stream a large eBird TSV into hive parquet without loading the full file.

    Uses pandas ``read_csv(chunksize=...)`` then appends per species partition.
    Avoids OOM kills on multi-million-row custom downloads.
    """
    import csv

    import pandas as pd

    log(f"Chunked read {input_path.name} (chunksize={chunk_rows:,} rows)")
    output_dir.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    affected: set[str] = set()
    chunk_num = 0

    reader = pd.read_csv(
        input_path,
        sep="\t",
        dtype=str,
        chunksize=chunk_rows,
        quoting=csv.QUOTE_MINIMAL,
        on_bad_lines="warn",
        low_memory=True,
    )
    for chunk in reader:
        chunk_num += 1
        chunk.columns = [str(c).lower().strip().replace(" ", "_") for c in chunk.columns]
        if PARTITION_COL not in chunk.columns:
            raise ValueError(f"{input_path} has no '{PARTITION_COL}' column")
        chunk[PARTITION_COL] = (
            chunk[PARTITION_COL].str.strip().str.lower().str.replace(" ", "_")
        )
        chunk = chunk[chunk[PARTITION_COL].notna() & (chunk[PARTITION_COL] != "")]
        if chunk.empty:
            continue

        total_rows += len(chunk)
        pl_chunk = pl.from_pandas(chunk)

        for key in pl_chunk[PARTITION_COL].unique().to_list():
            if not key:
                continue
            sub = pl_chunk.filter(pl.col(PARTITION_COL) == key)
            part_dir = output_dir / f"{PARTITION_PREFIX}{key}"
            part_dir.mkdir(parents=True, exist_ok=True)
            target = part_dir / "0.parquet"
            if target.is_file():
                sub = pl.concat(
                    [pl.read_parquet(target), sub], how="diagonal_relaxed"
                )
            sub.write_parquet(target, compression="snappy")
            affected.add(key)

        if chunk_num % 10 == 0:
            log(f"  ... chunk {chunk_num}: {total_rows:,} rows, {len(affected)} species")

    log(
        f"  wrote {total_rows:,} rows from {input_path.name} "
        f"into {len(affected)} partition(s)"
    )
    return total_rows, affected


def discover_main_tsvs(root: Path) -> list[Path]:
    """Observation TSVs: ebd_<pkg>/ebd_<pkg>.txt matching folder basename."""
    files: list[Path] = []
    if not root.is_dir():
        return files
    for folder in sorted(root.glob("ebd_*")):
        if not folder.is_dir():
            continue
        candidate = folder / f"{folder.name}.txt"
        if candidate.is_file() and candidate.name not in METADATA_TXT_NAMES:
            files.append(candidate)
    return files


def discover_root_parquets(root: Path) -> list[Path]:
    return sorted(p for p in root.glob("ebd_*.parquet") if p.is_file())
