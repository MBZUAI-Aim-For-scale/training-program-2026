"""Cold imports must neither use an unrelated config nor block navigation."""

import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest


class BenchmarkStartupTests(unittest.TestCase):
    def run_fresh(self, source):
        result = subprocess.run(
            [sys.executable, "-c", source], cwd=Path(__file__).resolve().parents[1],
            env={**os.environ, "DEMO_DATA_SOURCE": "local"},
            capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_background_task_import_does_not_initialize_weather_config(self):
        self.run_fresh("""
import sys
from ai_weather.background_task import BackgroundTask
assert 'ai_weather.config' not in sys.modules
assert 'config' not in sys.modules
""")

    def test_unrelated_config_cannot_replace_benchmark_configuration(self):
        self.run_fresh("""
import sys, types
sys.modules['config'] = types.ModuleType('config')
from ui.pages import short_medium_benchmarking as page
from ai_weather import config
assert page.config is config
assert page.CONFIG_BENCHMARK_MODEL_CATALOG is config.BENCHMARK_MODEL_CATALOG
assert config.DATA_DIR
assert 'momp.app.onset_time_series' not in sys.modules
""")

    def test_navigation_works_while_benchmark_page_import_is_pending(self):
        import importlib
        import streamlit as st

        entered, release = threading.Event(), threading.Event()
        original = importlib.import_module

        def load(name, *args, **kwargs):
            if name == "test.slow_benchmark_page":
                entered.set()
                release.wait(10)
                return SimpleNamespace(page=lambda: st.info("Benchmark page ready"))
            return original(name, *args, **kwargs)

        script = """
import ui.app as shell
shell.PAGE_MODULE_MAPPING = {'Home': 'ui.pages.home',
    'SM_Benchmarking': 'test.slow_benchmark_page'}
shell.main()
"""
        app = AppTest.from_string(script, default_timeout=5)
        app.query_params["page"] = "SM_Benchmarking"
        with patch("importlib.import_module", side_effect=load):
            try:
                app.run()
                self.assertFalse(app.exception)
                self.assertTrue(entered.wait(1))
                self.assertTrue(any("Loading Benchmarking" in item.value for item in app.info))
                task = app.session_state["benchmark_page_import_task"]
                app.button(key="primary_nav_Home").click().run()
                self.assertFalse(app.exception)
                self.assertFalse(task.done.is_set())
                self.assertEqual(app.query_params["page"], "Home")
            finally:
                release.set()
            self.assertTrue(task.done.wait(2))
            app.button(key="primary_nav_SM_Benchmarking").click().run()
            self.assertFalse(app.exception)
            self.assertTrue(any("Benchmark page ready" in item.value for item in app.info))

    def test_opening_benchmark_does_not_load_saved_maps(self):
        from ui.pages import short_medium_benchmarking as page

        app = AppTest.from_string(
            "from ui.pages.short_medium_benchmarking import page\npage()", default_timeout=5)
        app.session_state["benchmark_inventory_cache"] = {
            "CHIRPS": {"models": [], "observation_years": [], "loaded_at": time.time()}}
        with patch.object(page, "discover_map_outputs") as discover, \
             patch.object(page, "_map_load_admin_shapefile") as geometry:
            app.run()
            self.assertFalse(app.exception)
            discover.assert_not_called()
            geometry.assert_not_called()
            self.assertFalse(app.toggle(key="benchmark_show_results").value)
