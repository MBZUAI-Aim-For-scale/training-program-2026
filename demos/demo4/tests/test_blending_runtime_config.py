"""Dashboard stage settings reach actual runtime specs without changing globals."""

import copy
import importlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from ai_weather import config
from ai_weather.blending import run_blending
from ai_weather.blending_config import runtime_spec_defaults, validate_runtime_overrides

pipeline = importlib.import_module("apps.onset_blending.predict.run_operational_pipeline")


class BlendingRuntimeConfigTests(unittest.TestCase):
    def test_defaults_follow_selected_models_and_year_without_mutation(self):
        original = copy.deepcopy(config.ONSET_FORECAST_AIFS_SPEC)
        defaults = runtime_spec_defaults("aifs", 2024)
        self.assertEqual(defaults["single"]["options"], original["options"])
        self.assertIn("2024", defaults["single"]["input"]["file_regex"])
        defaults["single"]["options"]["window"] = 99
        self.assertEqual(config.ONSET_FORECAST_AIFS_SPEC, original)
        alternate = runtime_spec_defaults("ngcm", 2024)
        self.assertEqual(
            alternate["ensemble"]["input"]["nc_folder"], original["input"]["nc_folder"]
        )
        self.assertEqual(set(alternate["combine"]["forecasts"]), {"ngcm", "aifs"})
        self.assertNotIn("sources", alternate["combine"]["forecasts"]["aifs"])

    def test_overrides_reach_every_stage_and_keep_output_wiring(self):
        specs, steps = {}, []
        overrides = {
            "single": {
                "input": {"value_col": "rain", "file_regex": r"^custom_2024\.nc$"},
                "dimensions": {"rename": {"lead": "day"}},
                "options": {"window": 5, "onset_definition": {"follow_days": 14}},
                "thresholds": {"file": "/custom/thresholds.csv"},
                "mok": {"file": "/custom/mok.csv"},
                "filter": {"max_number": 8},
            },
            "ensemble": {
                "options": {"window": 7},
                "thresholds": {"file": "/ensemble/thresholds.csv"},
            },
            "climatology": {
                "climatologies": {
                    "clim": {"train_year_min": 1991, "train_year_max": 2010, "min_onset_years": 8}
                }
            },
            "combine": {"options": {"join": "full"}, "forecasts": {"aifs": {"max_day": 44}}},
            "maps": {
                "country_boundary": "/maps/country.geojson",
                "district_boundary": "/maps/districts.geojson",
                "zoom_to_data": True,
                "onset_file": "/maps/onset.csv",
            },
        }

        def capture(spec, prefix):
            specs[prefix] = spec
            return prefix + ".yml"

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.yml"
            path.write_text(yaml.safe_dump(overrides))
            with (
                patch.object(pipeline, "write_temp_yaml", side_effect=capture),
                patch.object(pipeline, "run_step", side_effect=lambda *args: steps.append(args)),
                patch.object(pipeline, "log"),
            ):
                pipeline.main(
                    [
                        "--year",
                        "2024",
                        "--issue_date",
                        "2024-06-13",
                        "--dry_run",
                        "--work_dir",
                        directory,
                        "--gt_path",
                        "/custom/history.pkl",
                        "--runtime_config",
                        str(path),
                    ]
                )
            single = specs["aifs_2024"]
            self.assertEqual(single["input"]["value_col"], "rain")
            self.assertEqual(single["input"]["file_regex"], r"^custom_2024\.nc$")
            self.assertEqual(single["dimensions"]["rename"]["lead"], "day")
            self.assertEqual(single["options"]["onset_definition"]["follow_days"], 14)
            self.assertEqual(single["thresholds"]["file"], "/custom/thresholds.csv")
            self.assertEqual(single["filter"]["max_number"], 8)
            self.assertEqual(single["output"]["out_dir"], str(Path(directory).resolve()))
            self.assertEqual(specs["imd_clim_2024"]["input"]["gt_path"], "/custom/history.pkl")
            self.assertEqual(
                specs["combine_2024"]["input"]["ground_truth_wide_rds"], "/custom/history.pkl"
            )
            self.assertEqual(
                specs["imd_clim_2024"]["climatologies"]["clim"]["train_year_min"], 1991
            )
            self.assertEqual(
                specs["imd_clim_2024"]["climatologies"]["clim"]["test_year_min"], 2024
            )
            self.assertEqual(specs["combine_2024"]["options"]["join"], "full")
            self.assertEqual(specs["combine_2024"]["forecasts"]["aifs"]["max_day"], 44)
            self.assertEqual(
                [
                    m["rain_predictors"][0]["window"]
                    for m in specs["connect_2024"]["forecast_models"]
                ],
                [7, 5],
            )
            for flag in (
                "--country_boundary",
                "--district_boundary",
                "--onset_file",
                "--zoom_to_data",
            ):
                self.assertIn(flag, steps[7][3])

    def test_adapter_serializes_runtime_configuration_for_child_process(self):
        overrides = {"single": {"options": {"window": 5}}}
        captured = {}

        def spawn(command, *args):
            path = Path(command[command.index("--runtime_config") + 1])
            captured["path"] = path
            captured["settings"] = yaml.safe_load(path.read_text())
            return 0, False

        with patch("ai_weather.blending.spawn_process", side_effect=spawn):
            result = run_blending(
                "blending",
                "aifs",
                {"year": 2024, "issue_date": "2024-06-13", "runtime_overrides": overrides},
            )
        self.assertTrue(result["ok"])
        self.assertEqual(captured["settings"], overrides)
        self.assertFalse(captured["path"].exists())

    def test_invalid_settings_fail_before_launch(self):
        for overrides in (
            [],
            {"unknown": {}},
            {"single": []},
            {"single": {"options": {"window": 0}}},
            {"single": {"input": {"file_regex": "["}}},
            {"single": {"dimensions": {"rename": {"time": 42}}}},
            {"combine": {"forecasts": {"aifs": {"daily": ["invalid"]}}}},
            {
                "climatology": {
                    "climatologies": {"clim": {"train_year_min": 2020, "train_year_max": 2000}}
                }
            },
        ):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                validate_runtime_overrides(overrides)


if __name__ == "__main__":
    unittest.main()
