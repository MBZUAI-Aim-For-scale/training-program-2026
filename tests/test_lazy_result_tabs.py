"""Inactive result views must not execute file, map or onset work."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock


class LazyResultTabsTests(unittest.TestCase):
    def test_closed_result_and_map_tabs_skip_expensive_bodies(self):
        source = Path("ui/pages/short_medium_benchmarking.py")
        tree = ast.parse(source.read_text())
        guards = [n for n in ast.walk(tree) if isinstance(n, ast.If)
                  and isinstance(n.test, ast.Attribute) and n.test.attr == "open"
                  and isinstance(n.test.value, ast.Subscript)
                  and isinstance(n.test.value.value, ast.Name)
                  and n.test.value.value.id in {"res_tabs", "sub_tabs"}]
        self.assertEqual(len(guards), 11)
        expensive = {name: MagicMock() for name in ("collect_reference_figures", "load_map_da_cached",
                                                   "_render_variable_plot", "_analyze_onset_streamlit")}
        ns = {**expensive, "res_tabs": [SimpleNamespace(open=False)] * 7,
              "sub_tabs": [SimpleNamespace(open=False)] * 4,
              "tab_labels": ["Metric Map", "All Metrics", "Benchmark Outputs", "Onset Analysis",
                             "Skill vs Baseline", "Uncertainty", "Benchmark Report & Export"]}
        for guard in guards:
            exec(compile(ast.Module(body=[guard], type_ignores=[]), str(source), "exec"), ns)  # noqa: S102
        for callback in expensive.values():
            callback.assert_not_called()

    def test_figure_tab_does_not_scan_netcdf_metadata(self):
        source = Path("ui/pages/short_medium_benchmarking.py")
        tree = ast.parse(source.read_text())
        assignment = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                          and isinstance(n.value, ast.IfExp)
                          and any(isinstance(t, ast.Name) and t.id == "map_outputs" for t in n.targets))
        discover = MagicMock()
        ns = {"discover_map_outputs": discover,
              "res_tabs": [SimpleNamespace(open=False), SimpleNamespace(open=False), SimpleNamespace(open=True)],
              "tab_labels": ["Metric Map", "All Metrics", "Benchmark Outputs"]}
        exec(compile(ast.Module(body=[assignment], type_ignores=[]), str(source), "exec"), ns)  # noqa: S102
        discover.assert_not_called()
        self.assertEqual(ns["map_outputs"], [])
