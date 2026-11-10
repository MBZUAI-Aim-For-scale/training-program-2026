"""Slow boundary downloads leave dashboard controls usable and country-isolated."""

from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

import geopandas as gpd
from shapely.geometry import box
from streamlit.testing.v1 import AppTest


class CountryBoundaryUITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from ui.pages import short_medium_benchmarking

        cls.page = short_medium_benchmarking

    def app(self):
        return AppTest.from_string(
            'import streamlit as st\n'
            'from ui.pages.short_medium_benchmarking import render_country_boundaries\n'
            'country = st.selectbox("Country", ["Peru", "Kenya", "Ethiopia"], key="country")\n'
            'st.number_input("Threshold", value=1, key="threshold")\n'
            'render_country_boundaries(country)\n', default_timeout=5)

    def test_slow_download_does_not_block_edits_or_restart_on_rerun(self):
        entered, release = threading.Event(), threading.Event()
        threads = []

        def download(country, root):
            threads.append(threading.current_thread().name)
            entered.set()
            release.wait(15)
            return {"paths": {}, "errors": {"ADM2": "Unavailable"}}

        with tempfile.TemporaryDirectory() as folder, \
             patch.object(self.page, "EXTERNAL_DATA_DIR", Path(folder)), \
             patch.object(self.page, "download_country_boundaries", side_effect=download) as worker:
            app = self.app().run()
            self.assertFalse(app.exception)
            worker.assert_not_called()
            try:
                app.button(key="benchmark_boundary_download_PER").click().run()
                self.assertFalse(app.exception)
                self.assertTrue(entered.wait(1))
                task = app.session_state["benchmark_boundary_tasks"]["Peru"]
                self.assertFalse(task.done.is_set())
                app.number_input(key="threshold").set_value(4).run()
                self.assertFalse(app.exception)
                self.assertEqual(app.number_input(key="threshold").value, 4)
                self.assertIs(app.session_state["benchmark_boundary_tasks"]["Peru"], task)
                app.selectbox(key="country").set_value("Kenya").run()
                self.assertFalse(app.exception)
                self.assertFalse(app.info)
                self.assertEqual(worker.call_count, 1)
                app.selectbox(key="country").set_value("Peru").run()
                self.assertTrue(app.info)
            finally:
                release.set()
            self.assertTrue(task.done.wait(5))
            app.run()
            self.assertFalse(app.exception)
            self.assertNotIn("Peru", app.session_state["benchmark_boundary_tasks"])
            self.assertTrue(app.warning)
            self.assertEqual(app.button(key="benchmark_boundary_download_PER").label, "Retry boundary download")
            self.assertEqual(threads, ["dashboard-task"])

    def test_ethiopia_download_control_uses_shared_country_flow(self):
        with tempfile.TemporaryDirectory() as folder, \
             patch.object(self.page, "EXTERNAL_DATA_DIR", Path(folder)), \
             patch.object(self.page, "download_country_boundaries", return_value={
                 "paths": {}, "errors": {"ADM0": "Provider unavailable"}}) as worker:
            app = self.app().run()
            app.selectbox(key="country").set_value("Ethiopia").run()
            self.assertFalse(app.exception)
            worker.assert_not_called()
            app.button(key="benchmark_boundary_download_ETH").click().run()
            task = app.session_state["benchmark_boundary_tasks"].get("Ethiopia")
            if task is not None:
                self.assertTrue(task.done.wait(5))
                app.run()
            self.assertFalse(app.exception)
            worker.assert_called_once_with("Ethiopia", Path(folder) / "boundaries")
            self.assertEqual(app.button(key="benchmark_boundary_download_ETH").label,
                             "Retry boundary download")

    def test_ethiopia_boundaries_and_subregions_never_use_preloaded_shapefile(self):
        import numpy as np
        import xarray as xr

        with tempfile.TemporaryDirectory() as folder, \
             patch.object(self.page, "EXTERNAL_DATA_DIR", Path(folder)), \
             patch.object(self.page, "_map_load_ethiopia_shapefile", create=True,
                          side_effect=AssertionError("Preloaded shapefile used")), \
             patch.object(self.page, "climatology_boundary_path",
                          side_effect=AssertionError("Preloaded outline used")):
            self.assertIsNone(self.page._country_boundary_path("Ethiopia"))
            self.assertIsNone(self.page._map_load_admin_shapefile("Ethiopia", "Admin1"))
            root = Path(folder) / "boundaries/ETH"
            root.mkdir(parents=True)
            for level, column, name in (("ADM0", "shapeName", "Ethiopia"),
                                        ("ADM1", "Admin1", "Oromia"),
                                        ("ADM2", "Admin2", "Arsi")):
                gpd.GeoDataFrame({column: [name]}, geometry=[box(38, 7, 40, 9)],
                                 crs="EPSG:4326").to_file(root / f"{level}.geojson", driver="GeoJSON")
            self.page.map_admin_options.clear()
            self.page.map_admin_bounds.clear()
            self.assertEqual(self.page._country_boundary_path("Ethiopia"), root / "ADM0.geojson")
            self.assertEqual(self.page.map_admin_options("Admin1", "Ethiopia"), ["Oromia"])
            self.assertEqual(self.page.map_admin_options("Admin2", "Ethiopia"), ["Arsi"])
            self.assertEqual(self.page.map_admin_bounds("Admin2", "Arsi", "Ethiopia"), (38, 7, 40, 9))
            self.assertIsNotNone(self.page._get_admin_subset("Admin2", "Arsi", "Ethiopia"))
            data = xr.DataArray(np.ones((4, 4)), dims=("lat", "lon"),
                                coords={"lat": [6, 7.5, 8.5, 10], "lon": [37, 38.5, 39.5, 41]},
                                attrs={"benchmark_country": "Ethiopia"})
            forecast = Path(folder) / "ethiopia_metrics.nc"
            data.to_dataset(name="mean_mae").to_netcdf(forecast)
            with patch.object(self.page, "apply_jjas_mask", side_effect=lambda value, *_: value):
                loaded = self.page.load_map_da_cached(str(forecast), "Mean Absolute Error",
                                                      "Days 1–15", country="Ethiopia")
            self.assertIsNotNone(loaded)
            self.assertTrue(np.isnan(loaded.values[0, 0]))
            self.assertEqual(loaded.values[1, 1], 1)
            figure = self.page.plot_map(data, "Ethiopia", country="Ethiopia")
            self.assertIsNotNone(figure)
            try:
                self.assertIsNotNone(figure.axes[0].images[0].get_clip_path())
                mask = np.ma.getmaskarray(figure.axes[0].images[0].get_array())
                self.assertTrue(mask[0, 0])
                self.assertFalse(mask[1, 1])
            finally:
                self.page.plt.close(figure)

    def test_country_subregion_names_bounds_and_map_clipping(self):
        import numpy as np
        import xarray as xr

        with tempfile.TemporaryDirectory() as folder, \
             patch.object(self.page, "EXTERNAL_DATA_DIR", Path(folder)):
            root = Path(folder) / "boundaries"
            for iso, name, west in (("PER", "Lima", -78), ("KEN", "Nairobi", 36)):
                (root / iso).mkdir(parents=True)
                for level in ("ADM0", "ADM1"):
                    gpd.GeoDataFrame({"Admin1": [name]}, geometry=[box(west, -13, west + 2, -11)],
                                     crs="EPSG:4326").to_file(root / iso / f"{level}.geojson", driver="GeoJSON")
            self.page.map_admin_options.clear()
            self.page.map_admin_bounds.clear()
            self.assertEqual(self.page.map_admin_options("Admin1", "Peru"), ["Lima"])
            self.assertEqual(self.page.map_admin_options("Admin1", "Kenya"), ["Nairobi"])
            self.assertEqual(self.page.map_admin_bounds("Admin1", "Lima", "Peru"), (-78, -13, -76, -11))
            self.assertIsNone(self.page._get_admin_subset("Admin1", "Lima", "Kenya"))
            self.assertEqual(self.page.map_admin_options("Admin2", "Peru"), [])
            data = xr.DataArray(np.ones((4, 4)), dims=("lat", "lon"),
                                coords={"lat": [-14, -12.5, -11.5, -10], "lon": [-79, -77.5, -76.5, -75]},
                                attrs={"benchmark_country": "Peru"})
            figure = self.page.plot_map(data, "Peru")
            self.assertIsNotNone(figure)
            try:
                self.assertIsNotNone(figure.axes[0].images[0].get_clip_path())
                mask = np.ma.getmaskarray(figure.axes[0].images[0].get_array())
                self.assertTrue(mask[0, 0])
                self.assertFalse(mask[1, 1])
                self.assertTrue(any(c.get_zorder() >= 10 for c in figure.axes[0].collections))
            finally:
                self.page.plt.close(figure)

    def test_benchmark_worker_uses_cached_country_outline(self):
        captured = {}

        def spawn(argv, env, cwd, timeout, on_line=None, **kwargs):
            captured.update(env)
            return 0, False

        for country, iso in (("Peru", "PER"), ("Ethiopia", "ETH")):
            with self.subTest(country=country), tempfile.TemporaryDirectory() as folder:
                captured.clear()
                root = Path(folder)
                runner = root / "run_momp.py"
                runner.touch()
                boundary = root / "boundaries" / iso / "ADM0.geojson"
                boundary.parent.mkdir(parents=True)
                boundary.write_text('{}')
                with patch.object(self.page, "EXTERNAL_DATA_DIR", root), \
                     patch.object(self.page, "PROJECT_ROOT", root), \
                     patch.object(self.page, "_locate_run_momp", return_value=runner), \
                     patch.object(self.page, "select_runtime_python", return_value=sys.executable), \
                     patch.object(self.page, "check_runtime_python", return_value="ready"), \
                     patch.object(self.page, "_spawn_no_fork", side_effect=spawn):
                    result = self.page.run_momp_benchmark("det", "AIFS", {"benchmark_coverage": country})
                self.assertTrue(result["ok"])
                self.assertEqual(captured["ROMP_SHAPEFILE_URI"], str(boundary.resolve()))
                self.assertEqual(captured["ROMP_CLIMATOLOGY_SHAPEFILE"], str(boundary.resolve()))
