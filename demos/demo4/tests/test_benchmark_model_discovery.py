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

    def test_country_models_are_listed_without_a_folder_existence_check(self):
        from datetime import date

        for _, country in self.page.BENCHMARK_COVERAGE_COUNTRIES:
            with self.subTest(country=country):
                with patch.object(self.page.config, "BUCKET_OK", True), \
                     patch.object(self.page.config, "DATA_MODE", "bucket"):
                    catalog = self.page.config.benchmark_country_sources(country)["models"]
                catalog = {key: meta for key, meta in catalog.items()
                           if key in {"AIFS", "AIFS_ENS", "GraphCast"}}
                def listing(pattern, **kwargs):
                    folder = pattern.split("/**/")[0]
                    return [folder + "/forecast_2024.nc"]
                with patch.object(self.page.config, "exists", return_value=False) as exists, \
                     patch.object(self.page.config, "glob", side_effect=listing), \
                     patch.object(self.page, "_extract_time_range_from_nc", return_value=(date(2024, 5, 1), date(2024, 5, 2))):
                    rows = self.page.detect_benchmark_models(catalog=catalog)
                self.assertEqual(len(rows), 3)
                exists.assert_not_called()

    def test_bucket_access_error_is_not_reported_as_no_models(self):
        with patch.object(self.page.config, "glob", side_effect=PermissionError("403 access denied")), \
             self.assertRaisesRegex(RuntimeError, "Could not list forecast data.*403 access denied"):
            self.page.detect_benchmark_models(catalog=self.catalog("gs://weather-test/Peru"))

    def test_page_shows_bucket_access_error_and_retry(self):
        sources = {"models": self.catalog("gs://weather-test/Peru"),
                   "observations": {}, "available": True, "reason": ""}
        with patch.object(self.page.config, "benchmark_country_sources", return_value=sources), \
             patch.object(self.page.config, "glob", side_effect=PermissionError("403 access denied")) as listing, \
             patch.object(self.page.config, "reset_gcs_connection") as reset:
            app = AppTest.from_string(
                "from ui.pages.short_medium_benchmarking import page\npage()", default_timeout=15)
            app.run()
            self.finish_inventory(app)
            self.assertTrue(any("403 access denied" in error.value for error in app.error))
            self.assertIsNotNone(app.button(key="benchmark_retry_inventory"))
            self.assertFalse(any("Traceback" in error.value for error in app.error))
            self.assertTrue(any("Traceback" in details.value for details in app.code))
            self.assertFalse(any("No forecast model data was detected" in item.value for item in app.markdown))
            listing.side_effect = None
            listing.return_value = []
            app.button(key="benchmark_retry_inventory").click().run()
            self.finish_inventory(app)
            reset.assert_called_once_with()
            self.assertFalse(app.error)

    def test_ethiopia_inventory_keeps_recovered_country_sources(self):
        catalog = self.catalog("gs://weather-test/Forecasts_Country_0p25deg/AIFS_v2/tp/Ethiopia")
        sources = {"models": catalog, "observations": {}, "available": True}
        with patch.object(self.page.config, "benchmark_country_sources", return_value=sources), \
             patch.object(self.page, "CONFIG_BENCHMARK_MODEL_CATALOG", self.catalog("/stale/local")), \
             patch.object(self.page, "detect_benchmark_models", return_value=[]) as detect, \
             patch.object(self.page, "available_observation_years", return_value=[]):
            self.assertEqual(self.page._benchmark_sources("Ethiopia")["models"], catalog)
            self.assertEqual(self.page._load_benchmark_inventory("CHIRPS", "Ethiopia")["catalog"], catalog)
            detect.assert_called_once_with(catalog=catalog)

    def test_remote_metadata_uses_storage_reader(self):
        from datetime import date

        uri = "gs://weather-test/forecast.nc"
        with patch.object(self.page.config, "open_dataset", side_effect=lambda *args, **kwargs:
                          xr.Dataset(coords={"time": pd.date_range("2024-05-01", periods=2)})) as opened:
            self.assertEqual(self.page._extract_time_range_from_nc(uri),
                             (date(2024, 5, 1), date(2024, 5, 2)))
        self.assertTrue(opened.called)
        self.assertTrue(all(call.args[0] == uri for call in opened.call_args_list))

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
            with patch.object(self.page.config, "benchmark_country_sources", return_value={
                     "models": self.catalog(folder), "observations": {}, "available": True, "reason": ""}), \
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
