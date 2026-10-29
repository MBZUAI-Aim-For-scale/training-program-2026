"""Workers use the project environment and diagnose broken binary dependencies."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ai_weather.runtime import check_runtime_python, select_runtime_python


class ScientificRuntimeTests(unittest.TestCase):
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
