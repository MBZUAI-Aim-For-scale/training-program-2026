"""Input metadata warms before model selection and survives settings changes."""

from datetime import date
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

SOURCE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE_ROOT))


class BenchmarkMetadataWarmupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from ui.pages import short_medium_benchmarking

        cls.page = short_medium_benchmarking

    def rows(self):
        return [dict(Model=key, Category=category, Type="HISTORICAL", Coverage="AIWP",
                     **{"Date Range": "2020 to 2024"}, _row_key=key, _exec_key=key,
                     _start_date=date(2020, 5, 1), _end_date=date(2024, 7, 31),
                     _probabilistic=category == "Probabilistic")
                for key, category in (("AIFS", "Deterministic"), ("GenCast", "Probabilistic"))]

    def app(self, rows):
        app = AppTest.from_string(
            "from ui.pages.short_medium_benchmarking import page\npage()", default_timeout=5)
        # Deliberately old: input caches now live until Refresh model data.
        app.session_state["benchmark_inventory_cache"] = {
            "CHIRPS": {"models": rows, "observation_years": list(range(2020, 2025)), "loaded_at": 0}}
        return app

    def finish_metadata(self, app):
        for _ in range(3):
            task = app.session_state.get("benchmark_coverage_task")
            if task is not None:
                self.assertTrue(task.done.wait(5))
            app.run()
            self.assertFalse(app.exception)
            if "benchmark_coverage_task" not in app.session_state:
                return
        self.fail("Metadata task did not publish its results")

    def test_all_models_warm_without_selection_and_settings_reuse_metadata(self):
        release, entered = threading.Event(), threading.Event()
        calls = []
        selected = []
        rows = self.rows()
        catalog = {row["_exec_key"]: {"dir": "/test/" + row["_exec_key"], "file_pattern": "{}.nc"}
                   for row in rows}
        original_table = self.page.st.dataframe

        def table(*args, **kwargs):
            result = original_table(*args, **kwargs)
            return SimpleNamespace(selection=SimpleNamespace(rows=selected)) if kwargs.get("key") == "model_table_selection" else result

        def inspect(model, first, last, model_meta=None):
            calls.append((model, first, last, threading.current_thread().name))
            entered.set()
            release.wait(10)
            return {"dates": [(year, 5, day) for year in range(first, last + 1) for day in range(1, 29)],
                    "horizons": {year: 60 for year in range(first, last + 1)}, "failures": []}

        with patch.object(self.page, "CONFIG_BENCHMARK_MODEL_CATALOG", catalog), \
             patch.object(self.page, "inspect_model_temporal_coverage", side_effect=inspect), \
             patch.object(self.page.st, "dataframe", side_effect=table), \
             patch.object(self.page, "detect_benchmark_models", return_value=rows), \
             patch.object(self.page, "available_observation_years", return_value=list(range(2020, 2025))):
            app = self.app(rows)
            try:
                app.run()
                self.assertFalse(app.exception)
                self.assertTrue(entered.wait(1))
                self.assertEqual(app.session_state["selected_model_rows"], [])
                task = app.session_state["benchmark_coverage_task"]
                # Model controls remain usable while a slow metadata read is paused.
                selected.append(0)
                app.radio(key="model_category_filter").set_value("Probabilistic").run()
                self.assertFalse(app.exception)
                self.assertIs(app.session_state["benchmark_coverage_task"], task)
                self.assertTrue(app.button(key="btn_run").disabled)
                self.assertEqual(len(calls), 1)
            finally:
                release.set()
            self.finish_metadata(app)
            self.assertEqual([(model, first, last) for model, first, last, _ in calls],
                             [("AIFS", 2020, 2024), ("GenCast", 2020, 2024)])
            self.assertTrue(all(name == "dashboard-task" for *_, name in calls))
            for entry in app.session_state["benchmark_coverage_cache"].values():
                entry["loaded_at"] = 0
            app.date_input(key="es_GenCast").set_value(date(2021, 5, 1)).run()
            app.selectbox(key="initialization_schedule_GenCast").set_value("daily").run()
            self.assertFalse(app.exception)
            self.assertFalse(app.button(key="btn_run").disabled)
            self.assertEqual(len(calls), 2)
            self.assertNotIn("benchmark_inventory_task", app.session_state)
            app.button(key="benchmark_refresh_inventory").click().run()
            inventory_task = app.session_state.get("benchmark_inventory_task")
            if inventory_task is not None:
                self.assertTrue(inventory_task.done.wait(5))
            self.finish_metadata(app)
            self.assertEqual(len(calls), 4)

    def test_cached_coverage_is_filtered_to_selected_years(self):
        coverage = {"dates": [(2020, 5, 1), (2021, 5, 1), (2022, 5, 1)],
                    "horizons": {2020: 15, 2021: 45, 2022: 60},
                    "failures": ["2020: bad file", "2022: bad file", "connection failed"]}
        self.assertEqual(self.page._coverage_for_years(coverage, 2021, 2021), {
            "dates": [(2021, 5, 1)], "horizons": {2021: 45}, "failures": ["connection failed"]})

    def test_saved_result_refresh_does_not_invalidate_input_metadata(self):
        app = self.app([])
        cached = {("AIFS", 2020, 2024, "signature"): {"data": {}, "loaded_at": 0}}
        app.session_state["benchmark_coverage_cache"] = cached
        with patch.object(self.page, "discover_map_outputs", return_value=[]), \
             patch.object(self.page, "inspect_model_temporal_coverage") as inspect:
            app.run()
            app.toggle(key="benchmark_show_results").set_value(True).run()
            app.button(key="benchmark_refresh_results").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state["benchmark_coverage_cache"], cached)
            self.assertIn("CHIRPS", app.session_state["benchmark_inventory_cache"])
            inspect.assert_not_called()
