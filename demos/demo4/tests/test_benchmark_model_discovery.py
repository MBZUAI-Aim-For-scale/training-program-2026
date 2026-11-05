"""Forecast discovery recovers after data becomes available without restarting."""

import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import xarray as xr
from streamlit.testing.v1 import AppTest


class BenchmarkModelDiscoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from ui.pages import short_medium_benchmarking

        cls.page = short_medium_benchmarking

    def write_forecast(self, folder):
        xr.Dataset(coords={"time": pd.date_range("2024-05-01", periods=2)}).to_netcdf(
            folder / "2024.nc")

    def catalog(self, folder):
        return {"AIFS": {"dir": str(folder), "model_name": "aifs_single",
                         "probabilistic": False}}

    def finish_inventory(self, app):
        task = app.session_state.get("benchmark_inventory_task")
        if task is not None:
            self.assertTrue(task.done.wait(5))
            app.run()
        self.assertFalse(app.exception)

    def test_new_scan_does_not_reuse_an_empty_discovery(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            with patch.object(self.page, "CONFIG_BENCHMARK_MODEL_CATALOG", self.catalog(folder)):
                self.assertEqual(self.page.detect_benchmark_models(), [])
                self.write_forecast(folder)
                rows = self.page.detect_benchmark_models()
                self.assertEqual([row["_row_key"] for row in rows], ["AIFS"])
                self.assertEqual(str(rows[0]["_start_date"]), "2024-05-01")

    def test_refresh_restores_selector_from_a_cached_empty_inventory(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            with patch.object(self.page, "CONFIG_BENCHMARK_MODEL_CATALOG", self.catalog(folder)), \
                 patch.object(self.page, "available_observation_years", return_value=[]), \
                 patch.object(self.page, "ROMP_DEMO_OUT_DIR", folder / "outputs"), \
                 patch.object(self.page, "ROMP_DEMO_FIG_DIR", folder / "figures"):
                app = AppTest.from_string(
                    "from ui.pages.short_medium_benchmarking import page\npage()",
                    default_timeout=15)
                # Reproduce a previously completed, empty scan in this session.
                app.session_state["benchmark_inventory_cache"] = {
                    "CHIRPS": {"models": [], "observation_years": [],
                               "loaded_at": time.time()}}
                app.run()
                self.assertFalse(app.exception)
                self.assertTrue(any("No forecast model data was detected" in item.value
                                    for item in app.markdown))
                self.assertTrue(any(str(folder) in item.value for item in app.text))
                self.write_forecast(folder)
                app.button(key="benchmark_refresh_inventory").click().run()
                self.finish_inventory(app)
                self.assertEqual(
                    [row["_row_key"] for row in
                     app.session_state["benchmark_inventory_cache"]["CHIRPS"]["models"]],
                    ["AIFS"])
                self.assertTrue(any("Model" in table.value.columns for table in app.dataframe))
                self.assertFalse(any("No forecast model data was detected" in item.value
                                     for item in app.markdown))
