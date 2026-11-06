"""Country selection isolates source discovery, worker settings and saved maps."""

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from streamlit.testing.v1 import AppTest
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "demos/demo4"
sys.path.insert(0, str(SOURCE_ROOT))


class BenchmarkCountryUITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from ui.pages import short_medium_benchmarking

        cls.page = short_medium_benchmarking

    def sources(self, country, ground_truth="CHIRPS"):
        folder_names = {"AIFS": ("AIFS_v2", "aifs_single", False),
                        "AIFS_ENS": ("AIFS_ENS_v2", "aifs_ens", True),
                        "GraphCast": ("graphcast", "graphcast", False)}
        models = {key: {"dir": f"gs://weather-test/Forecasts_Country_0p25deg/{folder}/tp/{country}",
                        "model_name": name, "file_pattern": f"{folder}_tp_{country}_{{}}_box025.nc",
                        "var": "tp", "unit_cvt": 1, "probabilistic": ensemble,
                        "category": "Probabilistic" if ensemble else "Deterministic"}
                  for key, (folder, name, ensemble) in folder_names.items()}
        return {"region": country, "models": models, "available": True, "reason": "",
                "observations": {"dir": f"gs://weather-test/CHIRPSv3_Country_0p25deg/{country}",
                                 "pattern": (f"CHIRPSv3_{country}_{{}}_box025.nc",), "var": "RAINFALL"}}

    def finish_background(self, app):
        for _ in range(6):
            for key in ("benchmark_inventory_task", "benchmark_coverage_task"):
                task = app.session_state.get(key)
                if task is not None:
                    self.assertTrue(task.done.wait(5))
            app.run()
            self.assertFalse(app.exception)
            if all(key not in app.session_state for key in ("benchmark_inventory_task", "benchmark_coverage_task")):
                return
        self.fail("Country discovery did not finish")

    def test_peru_selection_uses_its_catalog_without_rechecking_ethiopia(self):
        from datetime import date

        inspected = []

        def metadata(model, first, last, model_meta=None):
            inspected.append((model, model_meta["dir"]))
            return {"dates": [(year, 5, 1) for year in range(first, last + 1)],
                    "horizons": {year: 60 for year in range(first, last + 1)}, "failures": []}

        with tempfile.TemporaryDirectory() as directory, \
             patch.object(self.page.config, "region", "Ethiopia"), \
             patch.object(self.page.config, "benchmark_country_sources", side_effect=self.sources), \
             patch.object(self.page, "CONFIG_BENCHMARK_MODEL_CATALOG", self.sources("Ethiopia")["models"]), \
             patch.object(self.page, "_detect_model_data_date_range", return_value=(date(2020, 5, 1), date(2024, 7, 31))), \
             patch.object(self.page, "available_observation_years", return_value=list(range(2020, 2025))), \
             patch.object(self.page, "inspect_model_temporal_coverage", side_effect=metadata), \
             patch.object(self.page, "ROMP_DEMO_OUT_DIR", Path(directory) / "output"), \
             patch.object(self.page, "ROMP_DEMO_FIG_DIR", Path(directory) / "figures"):
            app = AppTest.from_string(
                "from ui.pages.short_medium_benchmarking import page\npage()", default_timeout=8)
            app.run()
            self.finish_background(app)
            self.assertEqual(len(inspected), 3)
            self.assertTrue(all(folder.endswith("/Ethiopia") for _, folder in inspected))
            ethiopia_inventory = app.session_state["benchmark_inventory_cache"]["CHIRPS"]

            app.selectbox(key="benchmark_country").set_value("Peru").run()
            self.finish_background(app)
            self.assertEqual(app.selectbox(key="shared_area_mask_from_config_Peru").value, "No mask")
            self.assertFalse(any("No data configured" in item.value for item in app.caption))
            self.assertFalse(any("Run Benchmark is disabled" in item.value for item in app.caption))
            self.assertFalse(app.button(key="btn_run").disabled)
            peru_inventory = app.session_state["benchmark_inventory_cache"][("Peru", "CHIRPS")]
            self.assertEqual(set(peru_inventory["catalog"]), {"AIFS", "AIFS_ENS", "GraphCast"})
            self.assertEqual(len(peru_inventory["models"]), 3)
            self.assertTrue(all(meta["dir"].endswith("/Peru") for meta in peru_inventory["catalog"].values()))
            self.assertEqual(len(inspected), 6)
            self.assertTrue(all(folder.endswith("/Peru") for _, folder in inspected[3:]))

            app.selectbox(key="benchmark_country").set_value("Ethiopia").run()
            self.finish_background(app)
            self.assertEqual(len(inspected), 6)
            self.assertEqual(app.session_state["benchmark_inventory_cache"]["CHIRPS"], ethiopia_inventory)
            self.assertEqual(app.session_state["benchmark_inventory_cache"][("Peru", "CHIRPS")], peru_inventory)

    def test_worker_receives_country_and_separate_output_paths(self):
        import os

        captured = {}

        def spawn(argv, env, cwd, timeout, on_line=None, **kwargs):
            captured.update(env)
            return 0, False

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner = root / "run_momp.py"
            runner.touch()
            with patch.object(self.page, "PROJECT_ROOT", root), \
                 patch.object(self.page, "_locate_run_momp", return_value=runner), \
                 patch.object(self.page, "select_runtime_python", return_value=sys.executable), \
                 patch.object(self.page, "check_runtime_python", return_value="runtime ready"), \
                 patch.object(self.page, "_spawn_no_fork", side_effect=spawn), \
                 patch.dict(os.environ, {"ROMP_OUT_DIR": str(root / "output"), "ROMP_FIG_DIR": str(root / "figures")}):
                result = self.page.run_momp_benchmark("det", "AIFS", {
                    "benchmark_coverage": "Peru", "ground_truth": "CHIRPS", "area_mask_file": ""},
                    progress=lambda _: None)
            self.assertTrue(result["ok"])
            self.assertEqual(captured["ROMP_REGION"], "Peru")
            self.assertEqual(captured["ROMP_OBS"], "CHIRPS")
            self.assertEqual(captured["ROMP_AREA_MASK_FILE"], "")
            self.assertEqual(captured["ROMP_OUT_DIR"], str(root / "output/countries/Peru"))
            self.assertEqual(captured["ROMP_FIG_DIR"], str(root / "figures/countries/Peru"))
            self.assertEqual(result["runtime_dir"], captured["ROMP_OUT_DIR"])

    def test_map_loading_skips_ethiopia_mask_and_isolates_country_output_folders(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ethiopia = root / "aifs_single/spatial_metrics_aifs_single_1-15.nc"
            peru = root / "countries/Peru/aifs_single/spatial_metrics_aifs_single_1-15.nc"
            for path, country, latitude, longitude in ((ethiopia, "Ethiopia", 8., 39.),
                                                       (peru, "Peru", -12., -77.)):
                path.parent.mkdir(parents=True)
                xr.Dataset({"mean_mae": (("lat", "lon"), [[1., 2.], [3., 4.]])},
                           coords={"lat": [latitude, latitude + 1], "lon": [longitude, longitude + 1]},
                           attrs={"region": country}).to_netcdf(path)
            ethiopia_outputs = self.page.discover_map_outputs(output_root=root)
            peru_outputs = self.page.discover_map_outputs(output_root=root / "countries/Peru")
            self.assertEqual([item["path"] for item in ethiopia_outputs], [str(ethiopia.resolve())])
            self.assertEqual([item["path"] for item in peru_outputs], [str(peru.resolve())])

            with patch.object(self.page, "apply_jjas_mask", side_effect=AssertionError("Ethiopia mask used")), \
                 patch.object(self.page, "clip_to_boundary", side_effect=AssertionError("Ethiopia clipping used")), \
                 patch.object(self.page, "load_boundary_frame", side_effect=AssertionError("Ethiopia zones loaded")), \
                 patch("momp.utils.land_mask.shp_mask", side_effect=lambda value, **kwargs: value):
                array = self.page.load_map_da_cached(str(peru), "Mean Absolute Error", "Days 1–15", country="Peru")
                self.assertIsNotNone(array)
                np.testing.assert_allclose(array.values, [[1., 2.], [3., 4.]])
                figure = self.page.plot_map(array, "Peru MAE")
                self.assertIsNotNone(figure)
                try:
                    self.assertLess(figure.axes[0].get_xlim()[1], 0)
                    self.assertLess(figure.axes[0].get_ylim()[1], 0)
                finally:
                    self.page.plt.close(figure)
                with xr.open_dataset(peru) as dataset, \
                     patch.object(self.page.st, "pyplot") as display, \
                     patch.object(self.page.st, "warning") as warning:
                    self.page._render_variable_plot(dataset.load(), "mean_mae", "Peru onset errors")
                    display.assert_called_once()
                    warning.assert_not_called()
