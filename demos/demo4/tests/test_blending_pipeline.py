"""Operational blending uses the requested year and trained model artifacts."""

import importlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

pipeline = importlib.import_module("apps.onset_blending.predict.run_operational_pipeline")


class OperationalPipelineTests(unittest.TestCase):
    def plan(self, *extra):
        specs, steps = {}, []

        def write(spec, prefix):
            specs[prefix] = spec
            return prefix + ".yml"

        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "not-created"
            with (
                patch.object(pipeline, "write_temp_yaml", side_effect=write),
                patch.object(pipeline, "run_step", side_effect=lambda *args: steps.append(args)),
                patch.object(pipeline, "log"),
            ):
                pipeline.main(
                    [
                        "--year",
                        "2027",
                        "--issue_date",
                        "2027-06-15",
                        "--work_dir",
                        str(target),
                        "--dry_run",
                        *extra,
                    ]
                )
            self.assertFalse(target.exists())
        return specs, steps

    def test_year_specs_and_trained_model_are_consistent(self):
        specs, steps = self.plan()
        self.assertIn("2027", specs["aifs_2027"]["input"]["file_regex"])
        self.assertEqual(specs["imd_clim_2027"]["climatologies"]["clim"]["test_year_min"], 2027)
        self.assertEqual(specs["combine_2027"]["options"]["join"], "inner")
        self.assertEqual(len(steps), 8)
        self.assertTrue(all(step[1] == 8 for step in steps))
        command = steps[5][3]
        self.assertIn("apps.onset_blending.predict.apply_blend_model", command)
        self.assertIn("--coef_dir", command)
        self.assertIn("--coef_tag", command)
        for step in steps:
            self.assertFalse(any("etagrometaix" in str(part) for part in step[3]))
        self.assertEqual(
            specs["connect_2027"]["forecast_models"][0]["rain_predictors"][0]["window"],
            specs["aifs_2027"]["options"]["window"],
        )

    def test_companion_uses_matching_input_and_combine_specs(self):
        specs, _ = self.plan("--model_single", "gencast")
        self.assertEqual(set(specs["combine_2027"]["forecasts"]), {"gencast", "aifs"})
        self.assertIn("single", specs["aifs_2027"]["input"]["nc_folder"])

    def test_bad_date_or_step_range_fails_before_writing(self):
        for args in (
            ["--year", "2027", "--issue_date", "2026-06-15"],
            ["--skip_to", "9"],
            ["--skip_to", "5", "--stop_at", "2"],
        ):
            with self.subTest(args=args), patch.object(pipeline, "write_temp_yaml") as write:
                with self.assertRaises(SystemExit):
                    pipeline.main(args)
                write.assert_not_called()

    def test_failed_step_never_accepts_existing_output(self):
        with patch.object(pipeline.subprocess, "run") as run, patch.object(pipeline, "log"):
            run.return_value.returncode = 1
            with self.assertRaises(SystemExit):
                pipeline.run_step(1, 8, "test", ["python"], __file__, False)


if __name__ == "__main__":
    unittest.main()
