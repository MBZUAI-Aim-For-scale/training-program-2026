"""Filename years cannot override decoded onset forecast dates."""

import importlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import netCDF4
import pandas as pd
import yaml

from apps.onset_blending.python.prepare_data import nc_utils
from apps.onset_blending.python.prepare_data.combine_forecasts_utils import (
    read_one_forecast_source,
)


class BlendingDateTests(unittest.TestCase):
    def write_nc(self, root, filename_year, actual_year):
        path = root / f"{filename_year}_adm3.nc"
        with netCDF4.Dataset(path, "w") as dataset:
            dataset.createDimension("time", 2)
            time = dataset.createVariable("time", "i4", ("time",))
            time.units = f"days since {actual_year}-06-13"
            time[:] = [0, 4]
        return {"input": {"nc_folder": str(root), "file_regex": rf"^{filename_year}_adm3\.nc$"}}

    def test_preflight_rejects_2024_time_in_2026_filename(self):
        with tempfile.TemporaryDirectory() as directory:
            spec = self.write_nc(Path(directory), 2026, 2024)
            with self.assertRaisesRegex(ValueError, r"2026_adm3.nc: expected year 2026.*2024"):
                nc_utils.validate_forecast_inputs(spec, "2026-06-15")

    def test_preflight_checks_selected_issue_and_lists_nearby_dates(self):
        with tempfile.TemporaryDirectory() as directory:
            spec = self.write_nc(Path(directory), 2024, 2024)
            nc_utils.validate_forecast_inputs(spec, "2024-06-13")
            with self.assertRaisesRegex(
                ValueError, r"No forecast initialization.*2024-06-15.*2024-06-13"
            ):
                nc_utils.validate_forecast_inputs(spec, "2024-06-15")

    def test_processing_does_not_overwrite_decoded_year(self):
        with tempfile.TemporaryDirectory() as directory:
            spec = {
                "type": "rainfall_forecast",
                "input": {"value_col": "tp"},
                "output": {"out_dir": directory},
            }
            frame = pd.DataFrame({"time": ["2024-06-13"], "year": [2024], "id": ["district"]})
            with (
                patch.object(nc_utils, "read_mok_dates"),
                patch.object(nc_utils, "read_thresholds"),
                patch.object(nc_utils, "read_cell_transform", return_value=None),
                patch.object(
                    nc_utils,
                    "list_nc_files_with_year",
                    return_value=pd.DataFrame({"nc_path": ["2026_adm3.nc"], "year": [2026]}),
                ),
                patch.object(nc_utils, "nc_read_forecast_wide", return_value=frame),
                patch.object(nc_utils, "process_rainfall_forecast_id") as process,
            ):
                with self.assertRaisesRegex(ValueError, "Forecast year mismatch"):
                    nc_utils.run_single_pipeline(spec)
                process.assert_not_called()
                self.assertEqual(frame.year.tolist(), [2024])

    def test_combine_rejects_stale_pickle_with_inconsistent_year(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "aifs_2026_wide.pkl"
            pd.DataFrame(
                {
                    "id": ["district"],
                    "time": ["2024-06-13"],
                    "year": [2026],
                    "predicted_prob_day_1": [0.2],
                }
            ).to_pickle(path)
            with self.assertRaisesRegex(ValueError, r"aifs_2026_wide.pkl.*2026.*2024"):
                read_one_forecast_source(
                    {"file": str(path), "years": "2026:2026"}, ["predicted_prob"], []
                )

    def test_combine_preserves_matching_forecast_keys(self):
        combine = importlib.import_module(
            "apps.onset_blending.python.pipelines.prepare_data.3_combine_datasets"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pd.DataFrame({"id": ["district"], "year": [2026]}).to_pickle(root / "truth.pkl")
            frame = pd.DataFrame(
                {
                    "id": ["district"],
                    "time": ["2026-06-13"],
                    "year": [2026],
                    "predicted_prob_day_1": [0.2],
                }
            )
            frame.to_pickle(root / "clim.pkl")
            frame.to_pickle(root / "forecast.pkl")
            spec = {
                "input": {
                    "ground_truth_wide_rds": str(root / "truth.pkl"),
                    "climatologies": {"clim": {"rds": str(root / "clim.pkl")}},
                },
                "forecasts": {
                    "aifs": {
                        "max_day": 1,
                        "sources": [{"file": str(root / "forecast.pkl")}],
                        "daily": [{"col": "predicted_prob", "out": "p_onset"}],
                    }
                },
                "options": {"join": "inner"},
                "output": {"out_dir": directory, "basename": "test"},
            }
            path = root / "combine.yml"
            path.write_text(yaml.safe_dump(spec))
            with patch("sys.argv", ["combine", "--spec_yaml", str(path)]):
                combine.main()
            combined = pd.read_pickle(root / "test_combined_wide.pkl")
            self.assertEqual(len(combined), 1)
            self.assertEqual(combined.year.tolist(), [2026])
            self.assertEqual(str(combined.time.iloc[0]), "2026-06-13")


if __name__ == "__main__":
    unittest.main()
