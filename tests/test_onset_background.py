"""Onset work must not occupy the script thread or download data on tab selection."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

SOURCE = Path("ui/pages/short_medium_benchmarking.py")


def load_functions(*names, **namespace):
    functions = [node for node in ast.parse(SOURCE.read_text()).body
                 if isinstance(node, ast.FunctionDef) and node.name in names]
    for function in functions:
        function.decorator_list = []
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace


class OnsetBackgroundTests(unittest.TestCase):
    def test_listing_years_does_not_prepare_or_download_observations(self):
        prepare = MagicMock(side_effect=AssertionError("Downloading during tab selection"))
        ns = load_functions("_available_years", "_raw_onset_obs_dir",
                            config=SimpleNamespace(CHIRPS_DIR="gs://observations/chirps"),
                            Path=Path, _prepare_local_obs_dir=prepare,
                            _discover_dataset_structure=lambda path: (("{}.nc",), [2022, 2023]))
        self.assertEqual(ns["_available_years"](), [2022, 2023])
        prepare.assert_not_called()

    def test_background_worker_never_calls_streamlit_ui(self):
        tree = ast.parse(SOURCE.read_text())
        worker = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                      and node.name == "_analyze_onset_streamlit")
        calls = [node.func.attr for node in ast.walk(worker) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute)
                 and isinstance(node.func.value, ast.Name) and node.func.value.id == "st"]
        self.assertEqual(calls, [])

    def test_running_task_only_renders_status_and_does_not_wait(self):
        job = SimpleNamespace(snapshot=MagicMock(return_value={"done": False}))
        state = {"onset_active_task": job}
        streamlit = SimpleNamespace(session_state=state, info=MagicMock(), rerun=MagicMock())
        ns = load_functions("render_onset_job", st=streamlit)
        ns["render_onset_job"]()
        streamlit.info.assert_called_once()
        streamlit.rerun.assert_not_called()
        self.assertIs(state["onset_active_task"], job)

    def test_finished_result_uses_submission_metadata(self):
        result = {"pngs": [b"png"], "grid_lat": 8.5, "grid_lon": 39.25,
                  "error": None, "warning": None}
        job = SimpleNamespace(snapshot=lambda: {"done": True, "result": result, "error": None})
        state = {"onset_active_task": job,
                 "onset_task_meta": {"year": 2021, "lat": 8.5, "lon": 39.2, "label": "original"}}
        streamlit = SimpleNamespace(session_state=state, rerun=MagicMock())
        ns = load_functions("render_onset_job", st=streamlit)
        ns["render_onset_job"]()
        self.assertEqual(state["onset_pngs"], [b"png"])
        self.assertEqual(state["onset_meta"]["label"], "original")
        self.assertEqual(state["onset_meta"]["grid_lon"], 39.25)
        self.assertNotIn("onset_active_task", state)
        streamlit.rerun.assert_called_once()

    def test_worker_error_clears_busy_state_and_is_retained(self):
        state = {"onset_active_task": SimpleNamespace(snapshot=lambda: {
            "done": True, "result": None, "error": "worker failed"}), "onset_task_meta": {}}
        streamlit = SimpleNamespace(session_state=state, rerun=MagicMock())
        ns = load_functions("render_onset_job", st=streamlit)
        ns["render_onset_job"]()
        self.assertEqual(state["onset_error"], "worker failed")
        self.assertNotIn("onset_active_task", state)

    def test_duplicate_submission_cannot_replace_running_task(self):
        tree = ast.parse(SOURCE.read_text())
        button_guard = next(node for node in ast.walk(tree) if isinstance(node, ast.If)
                            and any(isinstance(child, ast.Call) and child.keywords
                                    and any(keyword.arg == "key" and isinstance(keyword.value, ast.Constant)
                                            and keyword.value.value == "btn_run_ots"
                                            for keyword in child.keywords)
                                    for child in ast.walk(node.test)))
        streamlit = SimpleNamespace(button=MagicMock(return_value=True))
        task = MagicMock()
        namespace = {"st": streamlit, "onset_busy": True, "lat": 8.5, "lon": 39.2,
                     "BackgroundTask": task}
        exec(compile(ast.Module(body=[button_guard], type_ignores=[]), str(SOURCE), "exec"), namespace)
        task.assert_not_called()
        self.assertTrue(streamlit.button.call_args.kwargs["disabled"])

    def test_inventory_poll_never_runs_discovery_on_script_thread(self):
        state = {"benchmark_inventory_task": SimpleNamespace(snapshot=lambda: {"done": False})}
        streamlit = SimpleNamespace(session_state=state, info=MagicMock(), rerun=MagicMock())
        discover = MagicMock(side_effect=AssertionError("Blocking discovery"))
        ns = load_functions("render_benchmark_inventory", st=streamlit,
                            detect_benchmark_models=discover)
        ns["render_benchmark_inventory"]()
        discover.assert_not_called()
        streamlit.info.assert_called_once()
        streamlit.rerun.assert_not_called()

    def test_inventory_completion_keeps_ground_truths_separate(self):
        result = {"models": ["model"], "ground_truth": "CHIRPS", "observation_years": [2023]}
        state = {"benchmark_inventory_task": SimpleNamespace(snapshot=lambda: {
            "done": True, "result": result, "error": None}),
            "benchmark_inventory_ground_truth": "CHIRPS", "benchmark_inventory_cache": {
                "ENACTS": {"models": ["other"], "observation_years": [2022]}}}
        streamlit = SimpleNamespace(session_state=state, rerun=MagicMock())
        ns = load_functions("render_benchmark_inventory", st=streamlit)
        ns["render_benchmark_inventory"]()
        self.assertEqual(state["benchmark_inventory_cache"]["CHIRPS"], result)
        self.assertEqual(state["benchmark_inventory_cache"]["ENACTS"]["observation_years"], [2022])
        self.assertNotIn("benchmark_inventory_task", state)
        streamlit.rerun.assert_called_once()

    def test_coverage_poll_does_not_wait_or_read_netcdf(self):
        state = {"benchmark_coverage_task": SimpleNamespace(snapshot=lambda: {"done": False})}
        streamlit = SimpleNamespace(session_state=state, info=MagicMock(), rerun=MagicMock())
        inspect = MagicMock(side_effect=AssertionError("Reading NetCDF on UI thread"))
        ns = load_functions("render_benchmark_coverage", st=streamlit,
                            inspect_model_temporal_coverage=inspect)
        ns["render_benchmark_coverage"]()
        inspect.assert_not_called()
        streamlit.info.assert_called_once()
        streamlit.rerun.assert_not_called()

    def test_metadata_keys_remain_separate_and_errors_are_reported(self):
        first = ("first", 2020, 2021, "config-1")
        second = ("second", 2022, 2023, "config-2")
        inspect = MagicMock(side_effect=[{"dates": [(2020, 1, 1)], "horizons": {2020: 30}, "failures": []},
                                         ValueError("invalid coordinates")])
        ns = load_functions("_load_benchmark_coverage", inspect_model_temporal_coverage=inspect)
        results = ns["_load_benchmark_coverage"]([first, second])
        self.assertEqual(results[first]["data"]["dates"], [(2020, 1, 1)])
        self.assertIn("invalid coordinates", results[second]["data"]["failures"][0])
        self.assertEqual(inspect.call_args_list[0].args, first[:3])
        self.assertEqual(inspect.call_args_list[1].args, second[:3])

    def test_coverage_completion_merges_only_its_own_results(self):
        previous, current = ("previous", 2020, 2021, "{}"), ("current", 2022, 2023, "{}")
        saved = {"data": {"dates": []}, "loaded_at": 1}
        completed = {"data": {"dates": [(2022, 1, 1)]}, "loaded_at": 2}
        state = {"benchmark_coverage_task": SimpleNamespace(snapshot=lambda: {
            "done": True, "result": {current: completed}, "error": None}),
            "benchmark_coverage_keys": [current], "benchmark_coverage_cache": {previous: saved}}
        streamlit = SimpleNamespace(session_state=state, rerun=MagicMock())
        ns = load_functions("render_benchmark_coverage", st=streamlit)
        ns["render_benchmark_coverage"]()
        self.assertEqual(state["benchmark_coverage_cache"], {previous: saved, current: completed})
        self.assertNotIn("benchmark_coverage_task", state)
        streamlit.rerun.assert_called_once()

    def test_onset_year_listing_poll_is_nonblocking(self):
        state = {"onset_years_task": SimpleNamespace(snapshot=lambda: {"done": False})}
        streamlit = SimpleNamespace(session_state=state, info=MagicMock(), rerun=MagicMock())
        listing = MagicMock(side_effect=AssertionError("Remote listing on UI thread"))
        ns = load_functions("render_onset_years", st=streamlit, _available_years=listing)
        ns["render_onset_years"]()
        listing.assert_not_called()
        streamlit.info.assert_called_once()
        streamlit.rerun.assert_not_called()

    def test_onset_year_listing_completion_and_error_clear_busy_state(self):
        for result, error in (([2021, 2022], None), (None, "listing unavailable")):
            with self.subTest(error=error):
                state = {"onset_years_task": SimpleNamespace(snapshot=lambda: {
                    "done": True, "result": result, "error": error})}
                streamlit = SimpleNamespace(session_state=state, rerun=MagicMock())
                ns = load_functions("render_onset_years", st=streamlit)
                ns["render_onset_years"]()
                self.assertEqual(state["onset_years_result"]["years"], result)
                self.assertEqual(state["onset_years_result"]["error"], error)
                self.assertNotIn("onset_years_task", state)
                streamlit.rerun.assert_called_once()
