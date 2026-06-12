"""Quick tests for convert_ebird_downloads (no /mnt/f data required)."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

DATAPREP = Path(__file__).resolve().parent
SCRIPT = DATAPREP / "convert_ebird_downloads.py"


def test_discover_and_convert_tsv() -> None:
    from convert_ebird_downloads import convert_tsv, source_stem
    from ebird_polars_io import (
        COMPLETE_MARKER,
        PARTITION_COL,
        discover_main_tsvs,
        list_completed_species,
        manifest_path,
        prepare_run,
        source_marker_path,
        sync_manifest,
    )

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        root = tmp_path / "ebird"
        pkg = root / "ebd_test_pkg_relApr-2026"
        pkg.mkdir(parents=True)
        tsv = pkg / "ebd_test_pkg_relApr-2026.txt"
        tsv.write_text(
            "Scientific Name\tCommon Name\tObservation Count\n"
            "Setophaga petechia\tYellow Warbler\t1\n"
            "Spiza americana\tDickcissel\t2\n",
            encoding="utf-8",
        )

        discovered = discover_main_tsvs(root)
        assert discovered == [tsv]

        out = tmp_path / "out"
        prepare_run(out, clean_incomplete=True)
        convert_tsv(tsv, out)

        stem = source_stem(tsv)
        from convert_ebird_downloads import mark_source_complete

        mark_source_complete(out, stem)

        completed = list_completed_species(out)
        assert "setophaga_petechia" in completed
        assert "spiza_americana" in completed

        sync_manifest(out, completed)
        assert manifest_path(out).is_file()
        assert source_marker_path(out, stem).is_file()

        for species in ("setophaga_petechia", "spiza_americana"):
            part = out / f"scientific_name={species}"
            assert part.is_dir()
            assert any(part.glob("*.parquet"))
            assert (part / COMPLETE_MARKER).is_file()

    print("test_discover_and_convert_tsv: OK")


def test_cli_help_and_dry_run() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        capture_output=True,
        text=True,
        cwd=DATAPREP,
    )
    assert result.returncode == 0, result.stderr
    assert "--dry-run" in result.stdout
    assert "--include-parquets" in result.stdout
    print("test_cli_help: OK")


if __name__ == "__main__":
    test_cli_help_and_dry_run()
    test_discover_and_convert_tsv()
    print("All tests passed.")
