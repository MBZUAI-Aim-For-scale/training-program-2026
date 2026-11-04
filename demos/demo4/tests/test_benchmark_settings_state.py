"""Exercise dashboard run-state transitions without weather data or plotting imports."""
import ast
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import MagicMock

from ui.utils.benchmark_messages import cancellation_warning, execution_status


class BenchmarkSettingsStateTests(unittest.TestCase):
    def run_actions(self, *, clicked=False, selected=True, ok=True, signature="20"):
        tree = ast.parse(Path("ui/pages/short_medium_benchmarking.py").read_text())
        page = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "page")
        start = next(i for i, node in enumerate(page.body)
                     if isinstance(node, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == "previous_signature"
                             for t in node.targets))
        end = next(i for i, node in enumerate(page.body[start:], start)
                   if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                   and isinstance(node.value.func, ast.Name)
                   and node.value.func.id == "section_header")
        st = MagicMock()
        st.session_state = {"benchmark_settings_signature": signature,
                            "benchmark_last_run": {"ok": True}}
        st.spinner.return_value = nullcontext()
        runner = MagicMock(return_value={"ok": ok, "returncode": 0 if ok else 1,
                                         "stderr": "example failure" if not ok else ""})
        namespace = {"st": st, "settings_signature": "15", "run_clicked": clicked,
                         "active_rows": [{"_row_key": "a", "_exec_key": "a"}] if selected else [],
                         "model_run_windows": {}, "shared_settings": {"wet_day_threshold": 15},
                         "ground_truth": "CHIRPS", "benchmark_coverage": "Ethiopia",
                         "forecast_window": "1-15 days", "run_momp_benchmark": runner,
                         "ROMP_DEMO_OUT_DIR": Path("outputs"), "List": list, "Dict": dict, "Any": object}
        publisher = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                         and node.name == "_record_finished_benchmark")
        exec(compile(ast.Module(body=[publisher], type_ignores=[]), "job_completion", "exec"), namespace)  # noqa: S102

        class ImmediateJob:
            def __init__(self, execute, plans, signature):
                self.execute, self.plans, self.signature = execute, plans, signature

            def start(self):
                self.runs = []
                for plan in self.plans:
                    result = self.execute(plan["mode"], plan["model_key"], plan["config"])
                    self.runs.append(result)
                return self

            def snapshot(self):
                return {"status": "completed" if ok else "failed", "runs": self.runs, "done": True}

        def render_job():
            namespace["_record_finished_benchmark"](st.session_state["benchmark_active_job"])

        namespace.update(BenchmarkJob=ImmediateJob, render_benchmark_job=render_job,
                         cancellation_warning=cancellation_warning, execution_status=execution_status)
        exec(compile(ast.Module(body=page.body[start:end], type_ignores=[]),  # noqa: S102
                     "dashboard_actions", "exec"), namespace)
        return st, runner

    def test_settings_edits_preserve_cache_on_repeated_interactions(self):
        for _ in range(2):
            st, runner = self.run_actions()
            st.cache_data.clear.assert_not_called()
            st.warning.assert_called_once()
            runner.assert_not_called()
            self.assertEqual(st.session_state["benchmark_settings_signature"], "20")

    def test_successful_run_uses_new_threshold_and_refreshes_cache(self):
        st, runner = self.run_actions(clicked=True)
        self.assertEqual(runner.call_args.args[2]["wet_day_threshold"], 15)
        st.cache_data.clear.assert_called_once()
        self.assertEqual(st.session_state["benchmark_settings_signature"], "15")
        st.rerun.assert_called_once()

    def test_failed_run_does_not_mark_settings_as_computed(self):
        st, _ = self.run_actions(clicked=True, ok=False)
        self.assertEqual(st.session_state["benchmark_settings_signature"], "20")
        self.assertFalse(st.session_state["benchmark_last_run"]["ok"])
        st.error.assert_called_once()
        self.assertTrue(any("example failure" in str(call) for call in st.code.call_args_list))

    def test_run_without_selection_explains_required_action(self):
        st, runner = self.run_actions(clicked=True, selected=False)
        runner.assert_not_called()
        self.assertIn("Select at least one model", st.warning.call_args.args[0])

    def test_matching_settings_do_not_warn_or_clear_cache(self):
        st, _ = self.run_actions(signature="15")
        st.warning.assert_not_called()
        st.cache_data.clear.assert_not_called()
