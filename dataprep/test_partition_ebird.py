"""Quick tests for partition_ebird (no 440GB file required)."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import polars as pl

DATAPREP = Path(__file__).resolve().parent
SCRIPT = DATAPREP / "partition_ebird.py"


def test_bucket_filter_and_manifest() -> None:
    from partition_ebird import (
        PARTITION_COL,
        bucket_is_complete,
        build_lazy_frame,
        build_schema,
        list_completed_species,
        manifest_path,
        mark_bucket_complete,
        mark_partitions_complete,
        prepare_run,
        sync_manifest,
    )

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        tsv = tmp_path / "sample.tsv"
        out = tmp_path / "out"
        tsv.write_text(
            "Scientific Name\tCommon Name\tCount\n"
            "Setophaga petechia\tYellow Warbler\t1\n"
            "Spiza americana\tDickcissel\t2\n"
            "Setophaga coronata\tYellow-rumped Warbler\t3\n",
            encoding="utf-8",
        )

        schema = build_schema(tsv)
        assert "Scientific Name" in schema or len(schema) >= 2

        os.environ["EBIRD_INPUT"] = str(tsv)
        os.environ["EBIRD_OUTPUT"] = str(out)

        prepare_run(out)
        lf = build_lazy_frame(tsv, schema, out, bucket="s")
        lf.sink_parquet(
            pl.PartitionByKey(str(out), by=[PARTITION_COL]),
            mkdir=True,
            compression="snappy",
        )
        mark_partitions_complete(out, bucket="s")
        mark_bucket_complete(out, "s")

        completed = list_completed_species(out)
        assert "setophaga_petechia" in completed or any(
            "setophaga" in s for s in completed
        ), completed

        sync_manifest(out, completed)
        assert manifest_path(out).is_file()

        # Anti-join skips completed on re-run setup
        prepare_run(out)
        assert bucket_is_complete(out, "s")

        del os.environ["EBIRD_INPUT"]
        del os.environ["EBIRD_OUTPUT"]

    print("test_bucket_filter_and_manifest: OK")


def test_cli_help() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        capture_output=True,
        text=True,
        cwd=DATAPREP,
    )
    assert result.returncode == 0, result.stderr
    assert "--bucket" in result.stdout
    print("test_cli_help: OK")


if __name__ == "__main__":
    test_cli_help()
    test_bucket_filter_and_manifest()
    print("All tests passed.")
