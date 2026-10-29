"""Evaluation defaults must respect available observation years."""

import ast
import unittest
from datetime import date
from pathlib import Path
from types import SimpleNamespace


class ObservationWindowTests(unittest.TestCase):
    def test_window_and_observation_catalog(self):
        path = Path("ui/pages/short_medium_benchmarking.py")
        tree = ast.parse(path.read_text())
        functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                     and n.name in {"compute_common_model_window", "available_observation_years"}]
        config = SimpleNamespace(
            _OBS_SETS={"CHIRPS": {"dir": "obs", "pattern": ("{}.nc", "CHIRPS_{}.nc")}},
            glob=lambda _: ["obs/2023.nc", "obs/CHIRPS_2024.nc"],
            join=lambda folder, pattern: folder + "/" + pattern,
        )
        ns = dict(date=date, Path=Path, List=list, Dict=dict, Any=object,
                  BENCHMARK_OBS_CATALOG={}, config=config)
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), ns)  # noqa: S102
        years = ns["available_observation_years"]("CHIRPS")
        self.assertEqual(years, [2023, 2024])
        rows = [{"_start_date": date(2020, 1, 1), "_end_date": date(2026, 12, 31)}]
        window = ns["compute_common_model_window"]
        self.assertEqual(window(rows, years), (date(2023, 1, 1), date(2024, 12, 31)))
        self.assertEqual(window(rows, []), (None, None))
        self.assertEqual(window(rows, [2019]), (None, None))
