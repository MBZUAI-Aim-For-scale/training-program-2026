"""Reliability uses full event windows rather than pooled five-day events."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_function(path, name, namespace):
    tree = ast.parse(Path(path).read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[function], type_ignores=[]), path, "exec"), namespace)  # noqa: S102
    return function


class ReliabilityWindowTests(unittest.TestCase):
    def setUp(self):
        self.ns = {"np": np, "pd": pd, "plt": plt}
        load_function("apps/benchmarking/ROMP/momp/io/dict.py", "extract_pd_bins", self.ns)
        self.frame = pd.DataFrame({
            "bin_start": [1, 6, 11, 1, 16], "bin_end": [5, 10, 15, 15, 30],
            "predicted_prob": [.1, .2, .3, .6, .4], "observed_onset": [0, 1, 0, 1, 0],
        })

    def test_diagram_only_counts_exact_full_window_even_when_prefiltered(self):
        load_function("apps/benchmarking/ROMP/momp/graphics/reliability.py", "plot_reliability_diagram", self.ns)
        plot = self.ns["plot_reliability_diagram"]
        for window, expected in (((1, 15), .6), ((16, 30), .4)):
            fig, ax, results = plot(self.frame, model="test", verification_window=window,
                                    day_bins=((1, 5), (6, 10), (11, 15)),
                                    save_fig=False, dir_fig="", show_plot=False, extract_bins=False)
            try:
                self.assertEqual(results.N_Forecasts.sum(), 1)
                populated = results.loc[results.N_Forecasts > 0]
                self.assertEqual(populated.Mean_Forecast_Prob.iloc[0], expected)
                self.assertIn(f"Days {window[0]}-{window[1]}", ax.get_title())
            finally:
                plt.close(fig)
        with self.assertRaises(ValueError):
            plot(self.frame, model="test", verification_window=(1, 5), day_bins=(), save_fig=False, dir_fig="")
        with self.assertRaises(ValueError):
            plot(self.frame.iloc[:3], model="test", verification_window=(1, 15), day_bins=(), save_fig=False, dir_fig="")

    def test_pair_generation_adds_full_window_without_changing_score_bins(self):
        path = "apps/benchmarking/ROMP/momp/metrics/skill.py"
        tree = ast.parse(Path(path).read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "create_score_results")
        start = next(i for i, n in enumerate(function.body) if isinstance(n, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == "pair_kwargs" for t in n.targets))
        end = next(i for i, n in enumerate(function.body) if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Subscript) and isinstance(t.slice, ast.Constant)
                           and t.slice.value == "BS" for t in n.targets))
        for parallel in (False, True):
            runner = MagicMock(return_value=self.frame)
            ns = {**self.ns, "kwargs": {"plot_reliability": True, "verification_window": (1, 15),
                                       "day_bins": ((1, 5), (6, 10), (11, 15))},
                  "day_bins": ((1, 5), (6, 10), (11, 15)), "parallel": parallel,
                  "parallel_forecast_obs_pairs": runner, "multi_year_forecast_obs_pairs": runner,
                  "probe_prob": MagicMock(), "time": SimpleNamespace(perf_counter=lambda: 0),
                  "start": 0, "results": {}}
            exec(compile(ast.Module(body=function.body[start:end], type_ignores=[]), path, "exec"), ns)  # noqa: S102
            self.assertIn((1, 15), runner.call_args.kwargs["day_bins"])
            self.assertEqual(len(ns["results"]["forecast_obs_df"]), 3)
            self.assertEqual(len(ns["results"]["reliability_forecast_obs_df"]), 1)
