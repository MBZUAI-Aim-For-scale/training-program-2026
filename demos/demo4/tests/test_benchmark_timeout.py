"""Long ensemble benchmarks receive a configurable deadline and useful diagnostics."""

import ast
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Optional
import unittest
from unittest.mock import MagicMock, patch


class BenchmarkTimeoutTests(unittest.TestCase):
    def run_benchmark(self, override=None, outcome=(0, False)):
        source = Path(__file__).resolve().parents[1] / "ui/pages/short_medium_benchmarking.py"
        function = next(node for node in ast.parse(source.read_text()).body
                        if isinstance(node, ast.FunctionDef) and node.name == "run_momp_benchmark")
        spawn = MagicMock(return_value=outcome)
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {}, clear=True):
            if override is not None:
                os.environ["ROMP_BENCHMARK_TIMEOUT_SECONDS"] = override
            root = Path(folder)
            scope = {"os": os, "sys": sys, "Path": Path, "Optional": Optional,
                     "Dict": dict, "Any": Any, "COUNTRY_ISO3": {},
                     "PROJECT_ROOT": root, "ROMP_DEMO_OUT_DIR": root,
                     "_locate_run_momp": lambda: root / "run_momp.py",
                     "select_runtime_python": lambda _: sys.executable,
                     "check_runtime_python": lambda *_: "Scientific Python: OK",
                     "_spawn_no_fork": spawn}
            exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), scope)  # noqa: S102
            result = scope["run_momp_benchmark"]("prob", "aifs_ens", progress=lambda _: None)
        return result, spawn

    def test_default_allows_long_ensemble_runs(self):
        result, spawn = self.run_benchmark()
        self.assertTrue(result["ok"])
        self.assertEqual(spawn.call_args.args[3], 21600)
        self.assertFalse(result["timed_out"])

    def test_custom_deadline_is_used_and_reported_on_timeout(self):
        result, spawn = self.run_benchmark("3600", (124, True))
        self.assertEqual(spawn.call_args.args[3], 3600)
        self.assertFalse(result["ok"])
        self.assertTrue(result["timed_out"])
        self.assertEqual(result["returncode"], 124)
        self.assertIn("Timed out after 3600 seconds", result["stdout"])
        self.assertIn("ROMP_BENCHMARK_TIMEOUT_SECONDS", result["stdout"])

    def test_invalid_deadline_prevents_worker_launch(self):
        for value in ("0", "-1", "invalid"):
            with self.subTest(value=value):
                result, spawn = self.run_benchmark(value)
                self.assertFalse(result["ok"])
                self.assertIn("ValueError", result["stderr"])
                spawn.assert_not_called()
