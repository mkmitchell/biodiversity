"""Tests for check_occurrences (stdlib unittest — no pytest required)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from check_occurrences import (
    OccurrenceSource,
    OccurrenceStatus,
    SpeciesOccurrenceStatus,
    acquire_occurrences,
    check_ebird_species,
    check_gbif_species,
    check_manifest_occurrences,
    occurrence_source_for_group,
)
from species_manifest import load_species_manifest


class OccurrenceSourceTests(unittest.TestCase):
    def test_group_routing(self):
        self.assertEqual(occurrence_source_for_group("birds"), OccurrenceSource.EBIRD)
        self.assertEqual(occurrence_source_for_group("mammals"), OccurrenceSource.GBIF)
        self.assertEqual(occurrence_source_for_group("amphibians"), OccurrenceSource.GBIF)
        self.assertEqual(occurrence_source_for_group("reptiles"), OccurrenceSource.GBIF)


class GbifCheckTests(unittest.TestCase):
    def test_missing_gbif_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            status = check_gbif_species(
                "ursus_americanus",
                "mammals",
                Path(tmp),
                year_min=2020,
                year_max=2021,
            )
            self.assertEqual(status.status, OccurrenceStatus.MISSING)
            self.assertEqual(status.gbif_years_missing, (2020, 2021))

    def test_partial_gbif_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "ursus_americanus_2020.csv").write_text("x", encoding="utf-8")
            status = check_gbif_species(
                "ursus_americanus",
                "mammals",
                root,
                year_min=2020,
                year_max=2021,
            )
            self.assertEqual(status.status, OccurrenceStatus.PARTIAL)
            self.assertEqual(status.gbif_years_found, (2020,))
            self.assertEqual(status.gbif_years_missing, (2021,))

    def test_complete_gbif_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "ursus_americanus_2020.csv").write_text("x", encoding="utf-8")
            (root / "ursus_americanus_2021.csv").write_text("x", encoding="utf-8")
            status = check_gbif_species(
                "ursus_americanus",
                "mammals",
                root,
                year_min=2020,
                year_max=2021,
            )
            self.assertEqual(status.status, OccurrenceStatus.OK)

    def test_gbif_subfolder_and_title_case(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sub = root / "gbif_species_csvsold"
            sub.mkdir()
            (sub / "Anaxyrus_americanus_2020.csv").write_text("x", encoding="utf-8")
            (sub / "Anaxyrus_americanus_2021.csv").write_text("x", encoding="utf-8")
            status = check_gbif_species(
                "anaxyrus_americanus",
                "amphibians",
                root,
                year_min=2020,
                year_max=2021,
            )
            self.assertEqual(status.status, OccurrenceStatus.OK)

    def test_gbif_hyla_alias_for_dryophytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "hyla_squirella_2020.csv").write_text("x", encoding="utf-8")
            (root / "hyla_squirella_2021.csv").write_text("x", encoding="utf-8")
            status = check_gbif_species(
                "dryophytes_squirella",
                "amphibians",
                root,
                year_min=2020,
                year_max=2021,
            )
            self.assertEqual(status.status, OccurrenceStatus.OK)

    def test_gbif_cinerea_epithet_alias(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "hyla_cinerea_2020.csv").write_text("x", encoding="utf-8")
            status = check_gbif_species(
                "dryophytes_cinereus",
                "amphibians",
                root,
                year_min=2020,
                year_max=2021,
            )
            self.assertEqual(status.status, OccurrenceStatus.PARTIAL)
            self.assertEqual(status.gbif_years_found, (2020,))


class EbirdCheckTests(unittest.TestCase):
    def test_missing_ebd(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ebd = root / "ebd"
            pq = root / "pq"
            ebd.mkdir()
            status = check_ebird_species(
                "agelaius_phoeniceus",
                "birds",
                resolved={"agelaius_phoeniceus": {"speciesCode": "rewbla"}},
                ebd_root=ebd,
                ebird_parquet=pq,
            )
            self.assertEqual(status.status, OccurrenceStatus.MISSING)
            self.assertEqual(status.ebird_code, "rewbla")

    def test_need_convert(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ebd = root / "ebd"
            pq = root / "pq"
            folder = ebd / "ebd_US_rewbla_relApr-2026"
            folder.mkdir(parents=True)
            status = check_ebird_species(
                "agelaius_phoeniceus",
                "birds",
                resolved={"agelaius_phoeniceus": {"speciesCode": "rewbla"}},
                ebd_root=ebd,
                ebird_parquet=pq,
            )
            self.assertEqual(status.status, OccurrenceStatus.NEED_CONVERT)
            self.assertEqual(status.ebd_folders, ("ebd_US_rewbla_relApr-2026",))


class ManifestCheckTests(unittest.TestCase):
    def test_manifest_routes_by_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gbif = root / "gbif"
            gbif.mkdir()
            (gbif / "ursus_americanus_2020.csv").write_text("x", encoding="utf-8")
            manifest = load_species_manifest()
            subset = manifest[
                manifest["scientific_name"].isin(["ursus_americanus", "agelaius_phoeniceus"])
            ].head(2)
            if len(subset) < 2:
                self.skipTest("manifest missing expected species")
            statuses = check_manifest_occurrences(
                subset,
                gbif_root=gbif,
                ebd_root=root / "ebd",
                ebird_parquet=root / "pq",
                resolved={"agelaius_phoeniceus": {"speciesCode": "rewbla"}},
                year_min=2020,
                year_max=2020,
            )
            by_name = {s.scientific_name: s for s in statuses}
            self.assertEqual(by_name["ursus_americanus"].source, OccurrenceSource.GBIF)
            self.assertEqual(by_name["agelaius_phoeniceus"].source, OccurrenceSource.EBIRD)


class AcquireTests(unittest.TestCase):
    def test_acquire_dry_run_gbif(self):
        with tempfile.TemporaryDirectory() as tmp:
            excel = Path(tmp) / "species.xlsx"
            pd.DataFrame(
                {
                    "Scientific name": ["Ursus americanus"],
                    "Group": ["mammals"],
                }
            ).to_excel(excel, index=False)
            statuses = [
                SpeciesOccurrenceStatus(
                    scientific_name="ursus_americanus",
                    excel_group="mammals",
                    source=OccurrenceSource.GBIF,
                    status=OccurrenceStatus.MISSING,
                    gbif_years_missing=(2020,),
                )
            ]
            code = acquire_occurrences(
                statuses,
                excel_path=excel,
                gbif_root=Path(tmp) / "gbif",
                ebd_root=Path(tmp) / "ebd",
                ebird_parquet=Path(tmp) / "pq",
                resolved={},
                year_min=2020,
                year_max=2020,
                dry_run=True,
            )
            self.assertEqual(code, 0)


if __name__ == "__main__":
    unittest.main()
