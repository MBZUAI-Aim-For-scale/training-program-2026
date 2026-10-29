"""Late-window onset events must have exhaustive, window-local outcomes."""

import ast
import unittest
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd


class OnsetContingencyTests(unittest.TestCase):
    def setUp(self):
        path = Path("apps/benchmarking/ROMP/momp/stats/benchmark.py")
        tree = ast.parse(path.read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                        and n.name == "compute_onset_metrics_with_windows")
        ns = {"pd": pd, "np": np, "Tuple": tuple, "Dict": dict, "Union": Union}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), ns)  # noqa: S102
        self.compute = ns["compute_onset_metrics_with_windows"]

    @staticmethod
    def row(observed_day, model_day, year=2004):
        init = pd.Timestamp(year, 5, 1)
        return {
            "lat": 10.0, "lon": 38.0, "init_time": init,
            "obs_onset_date": init + pd.Timedelta(days=observed_day) if observed_day is not None else None,
            "onset_date": init + pd.Timedelta(days=model_day) if model_day is not None else None,
        }

    def test_every_early_in_late_or_none_pair_has_exactly_one_outcome(self):
        columns = ("true_positive", "false_positive", "false_negative", "true_negative")
        for observed in (5, 20, 40, None):
            for forecast in (5, 20, 40, None):
                with self.subTest(observed=observed, forecast=forecast):
                    metrics, summary = self.compute(pd.DataFrame([self.row(observed, forecast)]),
                                                    verification_window=(16, 30), tolerance_days=5)
                    result = metrics.iloc[0]
                    self.assertEqual(sum(result[key] for key in columns), 1)
                    expected = ("true_positive" if observed == 20 and forecast == 20 else
                                "false_positive" if forecast == 20 else
                                "false_negative" if observed == 20 else "true_negative")
                    self.assertEqual(result[expected], 1)
                    self.assertEqual(summary["overall_num_onset"], int(observed == 20))
                    self.assertEqual(summary["overall_num_no_onset"], int(observed != 20))

    def test_tolerance_applies_only_after_both_events_are_in_selected_window(self):
        for observed, forecast, expected, mae in ((20, 23, "true_positive", 3),
                                                 (16, 30, "false_positive", 14),
                                                 (16, 15, "false_negative", np.nan),
                                                 (15, 16, "false_positive", 1)):
            with self.subTest(observed=observed, forecast=forecast):
                metrics, _ = self.compute(pd.DataFrame([self.row(observed, forecast)]),
                                          verification_window=(16, 30), tolerance_days=3)
                result = metrics.iloc[0]
                self.assertEqual(result[expected], 1)
                if np.isnan(mae):
                    self.assertTrue(np.isnan(result.mae_combined))
                else:
                    self.assertEqual(result.mae_combined, mae)

    def test_each_initialization_uses_its_own_observed_date(self):
        frame = pd.DataFrame([self.row(20, 20, 2004), self.row(20, 20, 2005)])
        metrics, summary = self.compute(frame, verification_window=(16, 30))
        self.assertEqual(summary["overall_true_positive"], 2)
        self.assertEqual(summary["overall_num_onset"], 2)
        self.assertEqual(metrics.iloc[0].mae_combined, 0)

    def test_no_observed_onset_does_not_create_nan_error_samples(self):
        metrics, _ = self.compute(pd.DataFrame([self.row(None, 20)]), verification_window=(16, 30))
        self.assertEqual(metrics.iloc[0].false_positive, 1)
        self.assertEqual(metrics.iloc[0].num_fp_errors, 0)
        self.assertTrue(np.isnan(metrics.iloc[0].mae_combined))

    def test_day_one_to_fifteen_behavior_still_uses_tolerance(self):
        metrics, summary = self.compute(pd.DataFrame([self.row(5, 6), self.row(10, None)]),
                                        verification_window=(1, 15), tolerance_days=3)
        self.assertEqual(summary["overall_true_positive"], 1)
        self.assertEqual(summary["overall_false_negative"], 1)
        self.assertEqual(metrics.iloc[0].mae_combined, 1)
