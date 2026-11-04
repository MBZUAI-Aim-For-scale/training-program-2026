"""Stopping a job cancels its workers and skips later models."""

import ast
import os
import subprocess
import sys
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from ai_weather.benchmark_job import BenchmarkJob


class BenchmarkCancellationTests(unittest.TestCase):
    def test_stop_skips_remaining_models_and_preserves_snapshot(self):
        entered = threading.Event()
        seen = []

        def runner(mode, key, config, *, progress, cancel_event):
            seen.append((key, config["threshold"]))
            progress("started")
            entered.set()
            cancel_event.wait(1)
            return {"ok": False, "returncode": 130}

        plans = [{"mode": "det", "model_key": key, "label": key, "config": {"threshold": 20}}
                 for key in ("first", "second")]
        job = BenchmarkJob(runner, plans, "signature")
        plans[0]["config"]["threshold"] = 35
        job.start()
        self.assertTrue(entered.wait(1))
        job.cancel()
        self.assertTrue(job.done.wait(2))
        self.assertEqual(seen, [("first", 20)])
        self.assertEqual(job.snapshot()["status"], "cancelled")
        self.assertEqual(job.snapshot()["logs"], ["started"])

    def test_success_and_failure_are_recorded_without_streamlit_calls(self):
        for ok, expected in ((True, "completed"), (False, "failed")):
            with self.subTest(ok=ok):
                def runner(*args, result_ok=ok, **kwargs):
                    return {"ok": result_ok, "returncode": 0 if result_ok else 1}
                plans = [{"mode": "det", "model_key": "test", "label": "Test", "config": {}}]
                job = BenchmarkJob(runner, plans, "signature").start()
                self.assertTrue(job.done.wait(1))
                self.assertEqual(job.snapshot()["status"], expected)

    @unittest.skipUnless(hasattr(os, "posix_spawn"), "POSIX worker-group cancellation")
    @unittest.skipUnless(os.environ.get("RUN_PROCESS_CANCEL_TEST") == "1", "Opt-in process-group signaling integration test")
    def test_silent_sleeping_process_can_be_interrupted(self):
        # A harmless sleeper exercises cancellation; no weather benchmark is launched.
        source = Path("ui/pages/short_medium_benchmarking.py")
        function = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef)
                        and n.name == "_spawn_no_fork")
        ns = {"os": os, "subprocess": subprocess}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), ns)  # noqa: S102
        event = threading.Event()
        def on_line(line):
            if line == "ready":
                event.set()
        started = time.monotonic()
        code, timed_out = ns["_spawn_no_fork"](
            [sys.executable, "-u", "-c", "import time; print('ready', flush=True); time.sleep(10)"],
            os.environ.copy(), str(Path.cwd()), 15, on_line, cancel_event=event)
        self.assertEqual(code, 130)
        self.assertFalse(timed_out)
        self.assertLess(time.monotonic() - started, 3)

    @unittest.skipUnless(hasattr(os, "posix_spawn"), "POSIX worker-group cancellation")
    def test_exited_parent_is_not_signalled_again_when_group_cleanup_is_denied(self):
        source = Path("ui/pages/short_medium_benchmarking.py")
        function = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef)
                        and n.name == "_spawn_no_fork")
        ns = {"os": os, "subprocess": subprocess}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), ns)  # noqa: S102
        event = threading.Event()
        event.set()
        pid = 987654321
        for error in (PermissionError(1, "Operation not permitted"), ProcessLookupError()):
            with self.subTest(error=type(error).__name__), patch.object(os, "posix_spawn", return_value=pid), \
                    patch.object(os, "waitpid", return_value=(pid, 15)) as wait, \
                    patch.object(os, "killpg", side_effect=[None, error]) as kill:
                report = MagicMock()
                code, timed_out = ns["_spawn_no_fork"](["python", "unused"], {}, str(Path.cwd()), 15, report, event)
                self.assertEqual(code, 130)
                self.assertFalse(timed_out)
                wait.assert_called_once_with(pid, os.WNOHANG)
                self.assertEqual([call.args[1] for call in kill.call_args_list], [15, 0])
                if isinstance(error, PermissionError):
                    self.assertIn("main benchmark process exited", report.call_args.args[0])
