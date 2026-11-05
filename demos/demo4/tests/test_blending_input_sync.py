"""Shared date selection excludes mislabeled sources and invalidates stale checks."""

import copy
import importlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import netCDF4

from ai_weather.blending_inputs import input_check_matches, inspect_blending_inputs


class BlendingInputSyncTests(unittest.TestCase):
    def make_file(self, folder, label_year, actual_year, days=(0, 1)):
        folder.mkdir(exist_ok=True)
        path = folder / f"{label_year}_adm3.nc"
        with netCDF4.Dataset(path, "w") as dataset:
            dataset.createDimension("time", len(days))
            time = dataset.createVariable("time", "i4", ("time",))
            time.units = f"days since {actual_year}-05-26"
            time[:] = days
        return path

    def settings(self, root, year=2025):
        return {
            "year": year,
            "issue_date": f"{year}-05-26",
            "model_single": "aifs",
            "aifs_nc_folder": str(root / "single"),
            "aifs_ens_nc_folder": str(root / "ensemble"),
        }

    def test_mislabeled_file_is_reported_but_never_a_synchronization_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bad = self.make_file(root / "single", 2025, 2026)
            self.make_file(root / "ensemble", 2026, 2024)
            before = bad.read_bytes()
            report = inspect_blending_inputs(self.settings(root))
            self.assertFalse(report["ok"])
            self.assertEqual(report["common_dates"], [])
            self.assertTrue(any("2025" in error and "2026" in error for error in report["errors"]))
            self.assertEqual(bad.read_bytes(), before)

    def test_only_actual_common_initializations_are_offered(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_file(root / "single", 2025, 2026)
            self.make_file(root / "single", 2026, 2026, (0, 1))
            self.make_file(root / "ensemble", 2026, 2026, (0, 2))
            settings = self.settings(root)
            initial = inspect_blending_inputs(settings)
            self.assertFalse(initial["ok"])
            self.assertEqual(initial["common_dates"], ["2026-05-26"])
            settings.update(year=2026, issue_date="2026-05-26")
            verified = inspect_blending_inputs(settings)
            self.assertTrue(verified["ok"], verified["errors"])
            self.assertTrue(input_check_matches(settings, verified))
            changed = {**settings, "issue_date": "2026-05-27"}
            self.assertFalse(input_check_matches(changed, verified))
            self.assertFalse(inspect_blending_inputs(changed)["ok"])

    def test_file_replacement_invalidates_a_successful_check(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_file(root / "single", 2025, 2025)
            target = self.make_file(root / "ensemble", 2025, 2025)
            settings = self.settings(root)
            report = inspect_blending_inputs(settings)
            self.assertTrue(report["ok"])
            target.unlink()
            self.assertFalse(input_check_matches(settings, report))

    def test_shared_dates_respect_climatology_issue_season(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("single", "ensemble"):
                self.make_file(root / name, 2025, 2025, (0, 70))
            settings = self.settings(root)
            report = inspect_blending_inputs(settings)
            self.assertEqual(report["common_dates"], ["2025-05-26"])
            settings["issue_date"] = "2025-08-04"
            self.assertFalse(inspect_blending_inputs(settings)["ok"])

    def test_cli_check_stops_before_creating_pipeline_outputs(self):
        pipeline = importlib.import_module("apps.onset_blending.predict.run_operational_pipeline")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("single", "ensemble"):
                self.make_file(root / name, 2025, 2025)
            output = root / "not-created"
            arguments = [
                "--year",
                "2025",
                "--issue_date",
                "2025-05-26",
                "--check_inputs",
                "--aifs_nc_folder",
                str(root / "single"),
                "--aifs_ens_nc_folder",
                str(root / "ensemble"),
                "--work_dir",
                str(output),
            ]
            with patch.object(pipeline, "run_step") as run, patch.object(pipeline, "log"):
                pipeline.main(arguments)
                run.assert_not_called()
                self.assertFalse(output.exists())
                self.make_file(root / "single", 2025, 2026)
                with self.assertRaises(SystemExit):
                    pipeline.main(arguments)
                run.assert_not_called()
                self.assertFalse(output.exists())

    def test_custom_regex_and_folders_are_respected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("single", "ensemble"):
                path = self.make_file(root / name, 2025, 2025)
                path.rename(path.with_name("forecast_2025_adm3.nc"))
            settings = self.settings(root)
            settings["runtime_overrides"] = {
                name: {"input": {"file_regex": r"^forecast_2025_adm3\.nc$"}}
                for name in ("single", "ensemble")
            }
            original = copy.deepcopy(settings)
            report = inspect_blending_inputs(settings)
            self.assertTrue(report["ok"], report["errors"])
            self.assertEqual(settings, original)


if __name__ == "__main__":
    unittest.main()
