"""Tests for gap_species pipeline helpers (stdlib unittest — no pytest required)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import polars as pl

from gap_species import (
    GAP_TEST_SPECIES,
    PipelineStage,
    classify_species,
    has_param_csvs,
    has_parquet_partition,
    seed_test_parquet,
)


def _write_parquet_partition(root: Path, species: str, n_rows: int = 3) -> None:
    part = root / f"scientific_name={species}"
    part.mkdir(parents=True)
    pl.DataFrame(
        {
            "scientific_name": [species] * n_rows,
            "common_name": ["Test"] * n_rows,
            "latitude": ["34.0"] * n_rows,
            "longitude": ["-91.0"] * n_rows,
            "observation_date": ["2020-05-01"] * n_rows,
            "protocol_name": ["Stationary"] * n_rows,
            "effort_distance_km": ["0.01"] * n_rows,
        }
    ).write_parquet(part / "0.parquet")


class GapSpeciesTests(unittest.TestCase):
    def test_gap_test_species_list_includes_agelaius(self):
        self.assertIn("agelaius_phoeniceus", GAP_TEST_SPECIES)

    def test_classify_need_gee(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pq = root / "ebird"
            param = root / "param"
            param.mkdir()
            _write_parquet_partition(pq, "anthus_spragueii")
            resolved = {"anthus_spragueii": {"speciesCode": "sprpip"}}
            status = classify_species(
                "anthus_spragueii",
                resolved=resolved,
                ebd_root=root / "ebird_input",
                ebird_parquet=pq,
                param_dir=param,
            )
            self.assertEqual(status.stage, PipelineStage.NEED_GEE)

    def test_seed_test_parquet(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_parquet_partition(root, "geothlypis_trichas", n_rows=10)
            out = seed_test_parquet("agelaius_phoeniceus", "geothlypis_trichas", root, n_rows=5)
            self.assertTrue(out.is_file())
            df = pl.read_parquet(out)
            self.assertEqual(df["scientific_name"].unique().to_list(), ["agelaius_phoeniceus"])
            self.assertTrue(has_parquet_partition(root, "agelaius_phoeniceus"))

    def test_has_param_csvs(self):
        with tempfile.TemporaryDirectory() as tmp:
            param = Path(tmp)
            (param / "sp_subset0.csv").write_text("a")
            (param / "sp_subset1.csv").write_text("b")
            self.assertTrue(has_param_csvs(param, "sp", n_subsets=2))
            self.assertFalse(has_param_csvs(param, "sp", n_subsets=3))


if __name__ == "__main__":
    unittest.main()
