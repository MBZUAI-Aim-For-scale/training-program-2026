"""Routine benchmark notices disappear instead of occupying the page."""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock

SOURCE = Path(__file__).resolve().parents[1] / "demos/demo4/ui/app.py"


def render_function(name, state):
    function = next(node for node in ast.parse(SOURCE.read_text()).body
                    if isinstance(node, ast.FunctionDef) and node.name == name)
    function.decorator_list = []
    st = SimpleNamespace(session_state=state, toast=MagicMock(), info=MagicMock(),
                         success=MagicMock(), caption=MagicMock(), error=MagicMock(),
                         expander=MagicMock(), write=MagicMock(), rerun=MagicMock())
    namespace = {"st": st}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE), "exec"), namespace)  # noqa: S102
    return namespace[name], st


class DashboardNoticeTests(unittest.TestCase):
    def test_benchmark_running_notice_is_a_single_short_toast(self):
        snapshot = {"done": False, "status": "running"}
        job = SimpleNamespace(snapshot=lambda: snapshot)
        state = {"benchmark_active_job": job, "benchmark_job_token": "first"}
        render, st = render_function("render_background_status", state)
        for _ in range(5):
            render()
        st.toast.assert_called_once()
        self.assertEqual(st.toast.call_args.kwargs["duration"], "short")
        st.info.assert_not_called()
        st.caption.assert_not_called()
        snapshot.update(done=True, status="completed")
        render()
        st.caption.assert_not_called()
        state["benchmark_job_token"] = "second"
        snapshot.update(done=False, status="running")
        render()
        self.assertEqual(st.toast.call_count, 2)

    def test_successful_configuration_notice_is_consumed(self):
        state = {"config_check_result": {"error": None, "result": []}}
        render, st = render_function("render_config_check", state)
        render()
        render()
        st.toast.assert_called_once_with("Benchmark configuration check completed.",
                                         icon="✅", duration="short")
        self.assertNotIn("config_check_result", state)
        st.success.assert_not_called()

    def test_configuration_errors_and_issues_remain_visible(self):
        state = {"config_check_result": {"error": "Cannot read model data", "result": None}}
        render, st = render_function("render_config_check", state)
        render()
        st.error.assert_called_once_with("Cannot read model data")
        self.assertIn("config_check_result", state)
        st.toast.assert_not_called()
        state["config_check_result"] = {"error": None, "result": ["Missing mask"]}
        render()
        st.expander.assert_called_once_with("Configuration issues (1)")
        st.write.assert_called_once_with("- Missing mask")
