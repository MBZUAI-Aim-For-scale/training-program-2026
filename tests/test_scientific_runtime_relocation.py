"""Relocated dashboard paths must still select the repository's scientific Python."""

import ast
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Optional
import unittest
from unittest.mock import patch

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "demos/demo4"
sys.path.insert(0, str(SOURCE_ROOT))
from ai_weather import runtime


class ScientificRuntimeRelocationTests(unittest.TestCase):
    def environment(self, root):
        python = root / ".venv/bin/python"
        python.parent.mkdir(parents=True)
        python.symlink_to(sys.executable)
        return python

    def test_nested_demo_with_its_own_data_finds_repository_environment(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            root = Path(directory)
            (root / ".git").mkdir()
            app = root / "demos/demo4"
            (app / "data").mkdir(parents=True)
            (app / "apps").mkdir()
            python = self.environment(root)
            with patch.object(runtime.sys, "executable", "/opt/anaconda3/bin/python"):
                self.assertEqual(runtime.select_runtime_python(app), str(python))

    def test_worktree_git_file_also_marks_repository_boundary(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            root = Path(directory)
            (root / ".git").write_text("gitdir: /another/worktree\n")
            app = root / "demos/demo4"
            app.mkdir(parents=True)
            python = self.environment(root)
            self.assertEqual(runtime.select_runtime_python(app), str(python))

    def test_nearest_project_boundary_preserves_flattened_container_environment(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            root = Path(directory)
            (root / ".git").mkdir()
            self.environment(root)
            app = root / "app"
            app.mkdir()
            (app / "pyproject.toml").write_text("[project]\nname='container'\n")
            self.assertEqual(runtime.select_runtime_python(app), sys.executable)

    def test_explicit_override_keeps_priority_over_ancestor_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "pyproject.toml").touch()
            self.environment(root)
            app = root / "demos/demo4"
            app.mkdir(parents=True)
            with patch.dict(os.environ, {"ROMP_PYTHON": sys.executable}):
                self.assertEqual(runtime.select_runtime_python(app), sys.executable)

    def test_unmarked_parent_environment_is_not_borrowed(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            root = Path(directory)
            self.environment(root)
            app = root / "unrelated"
            app.mkdir()
            self.assertEqual(runtime.select_runtime_python(app), sys.executable)

    def test_dashboard_finds_root_runner_from_inside_relocated_demo(self):
        source = SOURCE_ROOT / "ui/pages/short_medium_benchmarking.py"
        function = next(node for node in ast.parse(source.read_text()).body
                        if isinstance(node, ast.FunctionDef) and node.name == "_locate_run_momp")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner = root / "run_momp.py"
            runner.touch()
            app = root / "demos/demo4"
            app.mkdir(parents=True)
            scope = {"Path": Path, "PROJECT_ROOT": app, "Optional": __import__("typing").Optional}
            exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), scope)  # noqa: S102
            with patch.object(Path, "cwd", return_value=app):
                self.assertEqual(scope["_locate_run_momp"](), runner.resolve())

    def test_dashboard_worker_uses_ancestor_venv_from_anaconda_dashboard(self):
        source = SOURCE_ROOT / "ui/pages/short_medium_benchmarking.py"
        function = next(node for node in ast.parse(source.read_text()).body
                        if isinstance(node, ast.FunctionDef) and node.name == "run_momp_benchmark")
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            root = Path(directory)
            (root / ".git").mkdir()
            python = self.environment(root)
            app = root / "demos/demo4"
            app.mkdir(parents=True)
            runner = root / "run_momp.py"
            runner.touch()
            commands = []

            def spawn(argv, *args, **kwargs):
                commands.append(argv)
                return 0, False

            scope = {"os": os, "sys": sys, "Path": Path, "Any": Any, "Dict": dict, "Optional": Optional,
                     "PROJECT_ROOT": app, "ROMP_DEMO_OUT_DIR": root / "output",
                     "_locate_run_momp": lambda: runner, "select_runtime_python": runtime.select_runtime_python,
                     "check_runtime_python": lambda *args: "ready", "_spawn_no_fork": spawn}
            exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), scope)  # noqa: S102
            with patch.object(runtime.sys, "executable", "/opt/anaconda3/bin/python"):
                result = scope["run_momp_benchmark"]("prob", "AIFS_ENS", progress=lambda _: None)
            self.assertTrue(result["ok"], result.get("stderr"))
            self.assertEqual(commands[0][:2], [str(python), str(runner)])
