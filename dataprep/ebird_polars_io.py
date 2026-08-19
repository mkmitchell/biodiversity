"""Shared Polars helpers for eBird TSV/Parquet → hive partitions."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

PARTITION_COL = "scientific_name"
PARTITION_PREFIX = f"{PARTITION_COL}="
COMPLETE_MARKER = ".complete"
TEST_ONLY_MARKER = ".test_only"
MANIFEST_NAME = "_completed_keys.parquet"
SOURCE_COMPLETE_PREFIX = "_source_"
NATIONAL_EXTRACT_CHECKPOINT_NAME = "_national_ebd_extract_checkpoint.json"
EXTRACT_CHECKPOINT_INTERVAL = 20


@dataclass(frozen=True)
class ExtractCheckpoint:
    byte_offset: int
    chunk_num: int
    matched_rows: int
    affected: tuple[str, ...]
    target_species: tuple[str, ...]
    state_codes: tuple[str, ...] | None
    chunk_rows: int
    input_path: str
    input_size: int

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, payload: dict) -> ExtractCheckpoint:
        return cls(
            byte_offset=int(payload["byte_offset"]),
            chunk_num=int(payload["chunk_num"]),
            matched_rows=int(payload["matched_rows"]),
            affected=tuple(payload.get("affected") or ()),
            target_species=tuple(payload["target_species"]),
            state_codes=(
                tuple(payload["state_codes"])
                if payload.get("state_codes") is not None
                else None
            ),
            chunk_rows=int(payload["chunk_rows"]),
            input_path=str(payload["input_path"]),
            input_size=int(payload["input_size"]),
        )


def extract_checkpoint_path(output_dir: Path) -> Path:
    return output_dir / NATIONAL_EXTRACT_CHECKPOINT_NAME


def load_extract_checkpoint(output_dir: Path) -> ExtractCheckpoint | None:
    path = extract_checkpoint_path(output_dir)
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as handle:
        return ExtractCheckpoint.from_json(json.load(handle))


def save_extract_checkpoint(output_dir: Path, checkpoint: ExtractCheckpoint) -> None:
    path = extract_checkpoint_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(checkpoint.to_json(), handle, indent=2)
    log(
        f"Checkpoint chunk {checkpoint.chunk_num} "
        f"(byte {checkpoint.byte_offset:,}, {checkpoint.matched_rows:,} matched rows)"
    )


def clear_extract_checkpoint(output_dir: Path) -> None:
    path = extract_checkpoint_path(output_dir)
    if path.is_file():
        path.unlink()


def extract_checkpoint_matches(
    checkpoint: ExtractCheckpoint,
    input_path: Path,
    *,
    target_species: set[str] | None,
    state_codes: set[str] | None,
    chunk_rows: int,
) -> bool:
    if checkpoint.input_path != str(input_path.resolve()):
        return False
    if checkpoint.input_size != input_path.stat().st_size:
        return False
    if checkpoint.chunk_rows != chunk_rows:
        return False
    targets = None if target_species is None else sorted(target_species)
    if list(checkpoint.target_species) != targets:
        return False
    states = None if state_codes is None else sorted(state_codes)
    cp_states = None if checkpoint.state_codes is None else sorted(checkpoint.state_codes)
    return states == cp_states


def is_test_parquet_partition(part_dir: Path) -> bool:
    return (part_dir / TEST_ONLY_MARKER).is_file()


def has_production_parquet_partition(output_dir: Path, species_key: str) -> bool:
    """True when a species partition has real parquet (not synthetic test data)."""
    part = output_dir / f"{PARTITION_PREFIX}{species_key.strip().lower().replace(' ', '_')}"
    if not part.is_dir() or is_test_parquet_partition(part):
        return False
    return any(part.glob("*.parquet"))


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
    """Stream a large eBird TSV into hive parquet without loading the full file."""
    return extract_ebird_tsv_chunked(
        input_path,
        output_dir,
        target_species=None,
        state_codes=None,
        skip_species=None,
        chunk_rows=chunk_rows,
    )


def extract_ebird_tsv_chunked(
    input_path: Path,
    output_dir: Path,
    *,
    target_species: set[str] | None,
    state_codes: set[str] | None = None,
    skip_species: set[str] | None = None,
    chunk_rows: int = 250_000,
    checkpoint_path: Path | None = None,
    resume: bool = False,
) -> tuple[int, set[str]]:
    """
    Stream an eBird TSV into hive parquet, optionally filtering rows.

    When ``target_species`` is set, only those normalized scientific_name keys are
    kept (still requires a full linear scan of the input file unless ``resume`` seeks
    to a saved byte offset). ``state_codes`` filters on the STATE CODE column.
    """
    import csv

    import pandas as pd

    targets = None if target_species is None else {s.strip().lower().replace(" ", "_") for s in target_species}
    skip = skip_species or set()
    states = None if state_codes is None else {s.strip().upper() for s in state_codes}

    log(f"Chunked read {input_path.name} (chunksize={chunk_rows:,} rows)")
    if targets is not None:
        log(f"  species filter: {len(targets)} target(s)")
    if states is not None:
        log(f"  state filter: {', '.join(sorted(states))}")

    output_dir.mkdir(parents=True, exist_ok=True)

    matched_rows = 0
    affected: set[str] = set()
    chunk_num = 0
    start_offset = 0

    checkpoint: ExtractCheckpoint | None = None
    if checkpoint_path is not None and resume:
        checkpoint = load_extract_checkpoint(output_dir)
        if checkpoint and extract_checkpoint_matches(
            checkpoint,
            input_path,
            target_species=targets,
            state_codes=states,
            chunk_rows=chunk_rows,
        ):
            start_offset = checkpoint.byte_offset
            chunk_num = checkpoint.chunk_num
            matched_rows = checkpoint.matched_rows
            affected = set(checkpoint.affected)
            log(
                f"Resuming at chunk {chunk_num} "
                f"(byte {start_offset:,}, {matched_rows:,} matched rows so far)"
            )
        elif checkpoint:
            log("WARNING: checkpoint does not match this run — starting from beginning")
            clear_extract_checkpoint(output_dir)
            checkpoint = None

    def _write_checkpoint(byte_offset: int) -> None:
        if checkpoint_path is None:
            return
        save_extract_checkpoint(
            output_dir,
            ExtractCheckpoint(
                byte_offset=byte_offset,
                chunk_num=chunk_num,
                matched_rows=matched_rows,
                affected=tuple(sorted(affected)),
                target_species=tuple(sorted(targets or ())),
                state_codes=tuple(sorted(states)) if states is not None else None,
                chunk_rows=chunk_rows,
                input_path=str(input_path.resolve()),
                input_size=input_path.stat().st_size,
            ),
        )

    byte_after_chunk = start_offset

    try:
        with input_path.open("rb") as handle:
            if start_offset:
                handle.seek(start_offset)

            reader = pd.read_csv(
                handle,
                sep="\t",
                dtype=str,
                chunksize=chunk_rows,
                quoting=csv.QUOTE_MINIMAL,
                on_bad_lines="warn",
                low_memory=True,
            )
            for chunk in reader:
                chunk_num += 1
                chunk.columns = [
                    str(c).lower().strip().replace(" ", "_") for c in chunk.columns
                ]
                if PARTITION_COL not in chunk.columns:
                    raise ValueError(f"{input_path} has no '{PARTITION_COL}' column")
                chunk[PARTITION_COL] = (
                    chunk[PARTITION_COL].str.strip().str.lower().str.replace(" ", "_")
                )
                chunk = chunk[chunk[PARTITION_COL].notna() & (chunk[PARTITION_COL] != "")]
                if chunk.empty:
                    byte_after_chunk = handle.tell()
                    if chunk_num % EXTRACT_CHECKPOINT_INTERVAL == 0:
                        _write_checkpoint(byte_after_chunk)
                    continue

                if states is not None and "state_code" in chunk.columns:
                    chunk = chunk[chunk["state_code"].str.strip().str.upper().isin(states)]
                if chunk.empty:
                    byte_after_chunk = handle.tell()
                    if chunk_num % EXTRACT_CHECKPOINT_INTERVAL == 0:
                        _write_checkpoint(byte_after_chunk)
                    continue

                if targets is not None:
                    chunk = chunk[chunk[PARTITION_COL].isin(targets)]
                if chunk.empty:
                    byte_after_chunk = handle.tell()
                    if chunk_num % EXTRACT_CHECKPOINT_INTERVAL == 0:
                        _write_checkpoint(byte_after_chunk)
                    continue

                if skip:
                    chunk = chunk[~chunk[PARTITION_COL].isin(skip)]
                if chunk.empty:
                    byte_after_chunk = handle.tell()
                    if chunk_num % EXTRACT_CHECKPOINT_INTERVAL == 0:
                        _write_checkpoint(byte_after_chunk)
                    continue

                matched_rows += len(chunk)
                pl_chunk = pl.from_pandas(chunk)

                for key in pl_chunk[PARTITION_COL].unique().to_list():
                    if not key or key in skip:
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

                if chunk_num % EXTRACT_CHECKPOINT_INTERVAL == 0:
                    byte_after_chunk = handle.tell()
                    _write_checkpoint(byte_after_chunk)
                    log(
                        f"  ... chunk {chunk_num}: {matched_rows:,} matched rows, "
                        f"{len(affected)} species written"
                    )
                else:
                    byte_after_chunk = handle.tell()

    except KeyboardInterrupt:
        if checkpoint_path is not None and chunk_num:
            _write_checkpoint(byte_after_chunk)
        log("Interrupted — checkpoint saved; re-run with --resume to continue")
        raise

    if checkpoint_path is not None:
        clear_extract_checkpoint(output_dir)

    log(
        f"  wrote {matched_rows:,} rows from {input_path.name} "
        f"into {len(affected)} partition(s)"
    )
    return matched_rows, affected


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
