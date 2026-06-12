"""Tests for species_manifest and prepare_predictors (stdlib unittest — no pytest required)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from maxent_model import prepare_predictors
from species_manifest import (
    excel_group_to_spgroup,
    get_species_entry,
    load_species_manifest,
    normalize_scientific_name,
)


class SpeciesManifestTests(unittest.TestCase):
    def test_normalize_scientific_name(self):
        self.assertEqual(normalize_scientific_name("Protonotaria citrea"), "protonotaria_citrea")

    def test_excel_group_to_spgroup(self):
        self.assertEqual(excel_group_to_spgroup("birds"), "avian")
        self.assertEqual(excel_group_to_spgroup("Birds"), "avian")
        self.assertEqual(excel_group_to_spgroup("amphibians"), "herp")
        self.assertEqual(excel_group_to_spgroup("reptiles"), "herp")
        self.assertEqual(excel_group_to_spgroup("mammals"), "mammal")

    def test_excel_group_to_spgroup_unknown_raises(self):
        with self.assertRaises(ValueError):
            excel_group_to_spgroup("fish")

    def test_load_species_manifest_from_fixture(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "species.xlsx"
            pd.DataFrame(
                {
                    "Scientific name": ["Protonotaria citrea", "Lithobates catesbeianus"],
                    "Group": ["birds", "amphibians"],
                }
            ).to_excel(path, index=False)
            manifest = load_species_manifest(path)
            self.assertEqual(len(manifest), 2)
            self.assertEqual(manifest.iloc[0]["spgroup"], "avian")
            self.assertEqual(manifest.iloc[1]["spgroup"], "herp")

    def test_get_species_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "species.xlsx"
            pd.DataFrame(
                {
                    "Scientific name": ["Agelaius phoeniceus"],
                    "Group": ["birds"],
                }
            ).to_excel(path, index=False)
            manifest = load_species_manifest(path)
            entry = get_species_entry(manifest, "Agelaius phoeniceus")
            self.assertEqual(entry["excel_group"], "birds")
            self.assertEqual(entry["spgroup"], "avian")

    def test_get_species_entry_missing_raises(self):
        manifest = pd.DataFrame(columns=["scientific_name", "excel_group", "spgroup"])
        with self.assertRaises(KeyError):
            get_species_entry(manifest, "missing_species")


class PreparePredictorsTests(unittest.TestCase):
    def _sample_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "longitude": [-91.0, -91.1],
                "latitude": [34.0, 34.1],
                "month": [5, 7],
                "season": [2, 3],
                "label": [1, 0],
                "obs_date": ["2020-05-01", "2020-07-01"],
                "year": [2020, 2020],
                "species": ["sp", "sp"],
                "temp_mean_mean": [20.0, 22.0],
            }
        )

    def test_birds_keep_month_and_season_as_categorical(self):
        X, categorical = prepare_predictors(self._sample_frame(), "birds")
        self.assertIn("month", X.columns)
        self.assertIn("season", X.columns)
        self.assertEqual(categorical, ["month", "season"])
        self.assertEqual(X["month"].dtype, int)
        self.assertEqual(X["season"].dtype, int)

    def test_non_birds_drop_month_and_season(self):
        X, categorical = prepare_predictors(self._sample_frame(), "amphibians")
        self.assertNotIn("month", X.columns)
        self.assertNotIn("season", X.columns)
        self.assertEqual(categorical, [])

    def test_mammals_drop_month_and_season(self):
        X, categorical = prepare_predictors(self._sample_frame(), "mammals")
        self.assertNotIn("month", X.columns)
        self.assertNotIn("season", X.columns)
        self.assertEqual(categorical, [])


class ManifestIntegrationTests(unittest.TestCase):
    def test_load_real_excel_manifest(self):
        manifest = load_species_manifest()
        self.assertGreater(len(manifest), 0)
        self.assertIn("scientific_name", manifest.columns)
        self.assertIn("excel_group", manifest.columns)
        self.assertIn("spgroup", manifest.columns)
        birds = manifest[manifest["excel_group"] == "birds"]
        herps = manifest[manifest["excel_group"].isin(["amphibians", "reptiles"])]
        mammals = manifest[manifest["excel_group"] == "mammals"]
        if not birds.empty:
            self.assertTrue((birds["spgroup"] == "avian").all())
        if not herps.empty:
            self.assertTrue((herps["spgroup"] == "herp").all())
        if not mammals.empty:
            self.assertTrue((mammals["spgroup"] == "mammal").all())


if __name__ == "__main__":
    unittest.main()
