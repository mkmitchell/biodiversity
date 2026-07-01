"""Tests for run_inference_species helpers."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "dataprep"))

from maxent_model import (  # noqa: E402
    INFERENCE_SEASONS,
    inference_stack_cache_path,
    run_inference_species,
    run_inference_species_season,
    species_output_dir,
)


class RunInferenceSpeciesSeasonTests(unittest.TestCase):
    def test_deploy_model_artifact_paths_flat_layout(self):
        from maxent_model import deploy_model_artifact_paths, load_model_config_from_deploy

        with tempfile.TemporaryDirectory() as tmp:
            models_dir = Path(tmp)
            sp = "protonotaria_citrea"
            cfg_path = models_dir / f"model_config_{sp}.json"
            cfg_path.write_text(
                '{"predictor_columns":["month","season"],"categorical_features":["month","season"]}',
                encoding="utf-8",
            )
            paths = deploy_model_artifact_paths(str(models_dir), sp)
            self.assertTrue(paths["config_path"].endswith(f"model_config_{sp}.json"))
            cfg = load_model_config_from_deploy(str(models_dir), sp)
            self.assertEqual(cfg["categorical_features"], ["month", "season"])

    def test_wires_paths_and_calls_stack_and_predict(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp)
            sp = "protonotaria_citrea"
            season = "spring"
            spdir = Path(species_output_dir(str(data_root), sp))
            spdir.mkdir(parents=True)
            (spdir / f"elapid_maxent_model_tuned_{sp}.pkl").write_bytes(b"x")
            (spdir / f"accuracy_tuned_{sp}.csv").write_text("BestThreshold,0.5\n", encoding="utf-8")
            (spdir / f"predictors_{sp}.txt").write_text("prcp_10000m\n", encoding="utf-8")
            (spdir / f"model_config_{sp}.json").write_text(
                '{"predictor_columns":["prcp_10000m"],"categorical_features":[]}',
                encoding="utf-8",
            )
            (spdir / f"convex_hull_{sp}.json").write_text("{}", encoding="utf-8")

            stack_path = inference_stack_cache_path(str(spdir), season)
            prob_path = spdir / f"predictions_prob_{sp}_{season}.tif"
            bin_path = spdir / f"predictions_binary_{sp}_{season}.tif"

            with patch("maxent_model.build_inference_covariate_stack", return_value=stack_path) as stack_mock, patch(
                "maxent_model.predict_raster_with_elapid"
            ) as predict_mock, patch(
                "maxent_model.load_model_config",
                return_value={"predictor_columns": ["prcp_10000m"], "categorical_features": []},
            ):
                result = run_inference_species_season(sp, season, str(data_root))

            stack_mock.assert_called_once()
            args, kwargs = stack_mock.call_args
            self.assertEqual(args[3], stack_path)
            self.assertFalse(kwargs["force"])

            predict_mock.assert_called_once()
            pkwargs = predict_mock.call_args.kwargs
            self.assertEqual(pkwargs["raster_path"], stack_path)
            self.assertEqual(pkwargs["season_label"], season)
            self.assertTrue(pkwargs["skip_geometry_mask"])
            self.assertEqual(result["species"], sp)
            self.assertEqual(result["season"], season)
            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["output_prob_path"], str(prob_path))
            self.assertEqual(result["output_bin_path"], str(bin_path))

    def test_run_inference_species_calls_all_seasons(self):
        with patch("maxent_model.run_inference_species_season") as season_mock:
            season_mock.side_effect = [
                {"species": "sp", "season": s, "status": "ok"} for s in INFERENCE_SEASONS
            ]
            results = run_inference_species("sp", "/tmp/data", ["spring", "summer"])
        self.assertEqual(len(results), 2)
        self.assertEqual(season_mock.call_count, 2)
        season_mock.assert_any_call("sp", "spring", "/tmp/data", force_stack=False, verbose=False)
        season_mock.assert_any_call("sp", "summer", "/tmp/data", force_stack=False, verbose=False)


if __name__ == "__main__":
    unittest.main()
