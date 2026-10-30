"""Workers use the project environment and diagnose broken binary dependencies."""

import ast
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from ai_weather.runtime import check_runtime_python, select_runtime_python


class ScientificRuntimeTests(unittest.TestCase):
    def test_legacy_notebook_executor_runs_outside_the_wrong_kernel(self):
        path = Path("notebooks/utils.py")
        function = next(node for node in ast.parse(path.read_text()).body
                        if isinstance(node, ast.FunctionDef) and node.name == "execute_momp_benchmark")
        worker = MagicMock(return_value={"ok": True})
        scope = {"os": os, "sys": sys, "PROJECT_ROOT": Path("/project"),
                 "select_runtime_python": lambda root: "/project/.venv/bin/python",
                 "run_momp_benchmark": worker}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), scope)  # noqa: S102
        scope["execute_momp_benchmark"]("prob")
        worker.assert_called_once_with(mode="prob", stream=True)

    def test_command_line_launchers_handoff_before_running_models(self):
        for filename in ("run_momp.py", "notebooks/run_momp.py"):
            with self.subTest(filename=filename):
                path = Path(filename).absolute()
                tree = ast.parse(path.read_text())
                function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                                and node.name == "execute_momp_benchmark")
                runner = MagicMock()
                runner.run.return_value.returncode = 7
                scope = {"__file__": str(path), "Path": Path, "os": os, "sys": sys,
                         "subprocess": runner, "PROJECT_ROOT": Path("/project"),
                         "select_runtime_python": lambda root: "/project/.venv/bin/python",
                         "check_runtime_python": MagicMock(), "select_models": MagicMock(),
                         "run_single_model": MagicMock()}
                exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), scope)  # noqa: S102
                with patch.dict(os.environ, {"ROMP_SPEC_PATH": "/project/spec.json"}):
                    code = scope["execute_momp_benchmark"]("prob", "AIFS_ENS", in_process=True)
                self.assertEqual(code, 7)
                command = runner.run.call_args.args[0]
                self.assertEqual(command[:2], ["/project/.venv/bin/python", str(path)])
                self.assertIn("AIFS_ENS", command)
                self.assertIn("--in-process", command)
                self.assertEqual(runner.run.call_args.kwargs["env"]["ROMP_SPEC_PATH"], "/project/spec.json")
                scope["select_models"].assert_not_called()
                scope["run_single_model"].assert_not_called()

    def test_command_line_dependency_failure_prevents_all_model_runs(self):
        for filename in ("run_momp.py", "notebooks/run_momp.py"):
            with self.subTest(filename=filename):
                path = Path(filename).absolute()
                function = next(node for node in ast.parse(path.read_text()).body
                                if isinstance(node, ast.FunctionDef) and node.name == "execute_momp_benchmark")
                scope = {"__file__": str(path), "Path": Path, "os": os, "sys": sys,
                         "PROJECT_ROOT": Path("/project"),
                         "select_runtime_python": lambda root: os.path.abspath(sys.executable),
                         "check_runtime_python": MagicMock(side_effect=RuntimeError("_spropack")),
                         "select_models": MagicMock()}
                exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), scope)  # noqa: S102
                with self.assertRaisesRegex(RuntimeError, "_spropack"):
                    scope["execute_momp_benchmark"]("prob")
                scope["select_models"].assert_not_called()

    def test_project_python_symlink_is_preserved(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {}, clear=True):
            python = Path(folder) / ".venv" / "bin" / "python"
            python.parent.mkdir(parents=True)
            python.symlink_to(sys.executable)
            self.assertEqual(select_runtime_python(folder), str(python))
            self.assertNotEqual(select_runtime_python(folder), str(python.resolve()))

    def test_explicit_override_is_honored_and_invalid_override_fails(self):
        with patch.dict(os.environ, {"ROMP_PYTHON": sys.executable}):
            self.assertEqual(select_runtime_python("/unused"), sys.executable)
        with patch.dict(os.environ, {"ROMP_PYTHON": "/missing/python"}), self.assertRaisesRegex(RuntimeError, "not executable"):
            select_runtime_python("/unused")

    def test_current_python_is_fallback_when_no_project_environment_exists(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {}, clear=True):
            self.assertEqual(select_runtime_python(folder), sys.executable)

    def test_failed_dependency_probe_preserves_underlying_error(self):
        def spawn(argv, env, cwd, timeout, on_line):
            self.assertEqual(argv[:2], ["/project/.venv/bin/python", "-c"])
            on_line("ImportError: cannot import name '_spropack'")
            return 1, False

        with self.assertRaisesRegex(RuntimeError, "_spropack"):
            check_runtime_python("/project/.venv/bin/python", {}, "/project", spawn)

    def test_probe_handles_timeout_and_returns_success_diagnostics(self):
        with self.assertRaisesRegex(RuntimeError, "timed out"):
            check_runtime_python("python", {}, "/project", lambda *_: (124, True))

        def spawn(argv, env, cwd, timeout, on_line):
            on_line("SciPy: working")
            return 0, False

        self.assertEqual(check_runtime_python("python", {}, "/project", spawn), "SciPy: working")
