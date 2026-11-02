"""Exercise probability scoring without importing data/configuration runtimes."""

import ast
import contextlib
import inspect
import io
import unittest
import warnings
from datetime import datetime
from itertools import product
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
MOMP = ROOT / "apps/benchmarking/ROMP/momp"


def load_functions(relative_path, namespace, names=None):
    path = MOMP / relative_path
    tree = ast.parse(path.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and (names is None or node.name in names)]
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)  # noqa: S102
    return functions


class ProbabilisticScoringTests(unittest.TestCase):
    def test_paired_rps_excludes_single_member_reference_from_both_scores(self):
        ns = {"np": np, "pd": pd, "warnings": warnings}
        load_functions("stats/score.py", ns)
        rows = []
        for lon, members in ((38, 1), (39, 4)):
            for index, (probability, observation) in enumerate(zip((.5, .5), (1, 0))):
                rows.append({"init_time": "2024-06-01", "lat": 10, "lon": lon,
                             "bin_index": index, "bin_label": str(index),
                             "predicted_prob": probability, "observed_onset": observation,
                             "total_members": 25, "total_members_with_onset": members})
        reference = pd.DataFrame(rows)
        forecast = reference.drop(columns="total_members_with_onset").copy()
        forecast.loc[forecast.lon == 38, "predicted_prob"] = [0, 1]
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            original = ns["calculate_rps"](reference)
            model_rps, reference_rps = ns["calculate_paired_rps"](forecast, reference)
        self.assertTrue(np.isnan(original["fair_rps"]))
        self.assertEqual(model_rps["n_forecasts"], 1)
        self.assertEqual(reference_rps["n_forecasts"], 1)
        self.assertEqual(model_rps["n_excluded_forecasts"], 1)
        self.assertAlmostEqual(model_rps["fair_rps"], .25 - .25 / 24)
        self.assertAlmostEqual(reference_rps["fair_rps"], .25 - .25 / 3)
        self.assertTrue(any("fewer than two" in str(item.message) for item in caught))

    def test_paired_rps_without_supported_reference_stays_undefined(self):
        ns = {"np": np, "pd": pd, "warnings": warnings}
        load_functions("stats/score.py", ns)
        pairs = pd.DataFrame({"init_time": ["2024-06-01"], "lat": [10], "lon": [38],
                              "bin_index": [0], "bin_label": ["1"], "predicted_prob": [1.],
                              "observed_onset": [1], "total_members": [1]})
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            model_rps, reference_rps = ns["calculate_paired_rps"](pairs, pairs)
        self.assertEqual(reference_rps["n_forecasts"], 0)
        self.assertTrue(np.isnan(model_rps["fair_rps"]))
        self.assertTrue(np.isnan(reference_rps["fair_rps"]))

    def test_rps_retains_before_and_after_events_without_reliability_overlap(self):
        ns = {"np": np, "pd": pd, "warnings": warnings}
        load_functions("io/dict.py", ns, {"extract_pd_bins"})
        load_functions("stats/bins.py", ns, {"extract_day_range"})
        load_functions("stats/score.py", ns)
        rps = MagicMock(side_effect=ns["calculate_paired_rps"])
        frames = []
        for init_time, probabilities in (("2024-06-01", [1, 0, 0, 0, 0, 0]),
                                         ("2024-06-02", [0, 0, 0, 0, 1, 0])):
            frames.append(pd.DataFrame({
                "init_time": [init_time] * 6, "lat": [10] * 6, "lon": [38] * 6,
                "bin_start": [-np.inf, 16, 21, 26, 31, 16],
                "bin_end": [15, 20, 25, 30, np.inf, 30],
                "bin_index": range(6), "total_members": [10] * 6,
                "bin_label": ["Before day 16", "Days 16-20", "Days 21-25",
                              "Days 26-30", "After day 30", "Days 16-30"],
                "predicted_prob": probabilities, "observed_onset": [0, 1, 0, 0, 0, 1],
            }))
        pairs = pd.concat(frames, ignore_index=True)
        for parallel, reference in product((False, True), ("climatology", "other_model")):
            with self.subTest(parallel=parallel, reference=reference):
                runner = MagicMock(return_value=pairs)
                brier = MagicMock(return_value={
                    "fair_brier_score": .2, "bin_fair_brier_scores": {"Days 16-20": .2}})
                auc = MagicMock()
                scope = {**ns,
                         "parallel_forecast_obs_pairs": runner, "multi_year_forecast_obs_pairs": runner,
                         "parallel_climatological_forecast_obs_pairs": runner,
                         "multi_year_climatological_forecast_obs_pairs": runner,
                         "parallel_climatological_onset_dataset": MagicMock(),
                         "compute_climatological_onset_dataset": MagicMock(),
                         "available_forecast_years": lambda years, **kwargs: years,
                         "restore_args": lambda func, kwargs, bound: {
                             **{key: value for key, value in bound.items() if key != "kwargs"}, **kwargs},
                         "probe_prob": MagicMock(),
                         "calculate_brier_score": brier, "calculate_auc": auc,
                         "calculate_brier_score_climatology": brier, "calculate_auc_climatology": auc,
                         "calculate_paired_rps": rps}
                load_functions("metrics/skill.py", scope, {"create_score_results"})
                run = scope["create_score_results"]
                arguments = {name: None for name in inspect.signature(run).parameters if name != "kwargs"}
                arguments.update(BS=True, RPS=True, AUC=True, skill_score=True,
                                 years=(2024,), ref_model=reference,
                                 day_bins=((16, 20), (21, 25), (26, 30)),
                                 plot_reliability=True, verification_window=(16, 30), parallel=parallel)
                with contextlib.redirect_stdout(io.StringIO()):
                    result = run(**arguments)
                rps_pairs = rps.call_args.args[0]
                self.assertEqual(len(rps_pairs), 10)
                self.assertNotIn("Days 16-30", set(rps_pairs.bin_label))
                self.assertIn("Before day 16", set(rps_pairs.bin_label))
                self.assertIn("After day 30", set(rps_pairs.bin_label))
                # Before-window and after-window errors have different ordered distances.
                expected = ns["calculate_rps"]
                with contextlib.redirect_stdout(io.StringIO()):
                    before = expected(rps_pairs[rps_pairs.init_time == "2024-06-01"])
                    after = expected(rps_pairs[rps_pairs.init_time == "2024-06-02"])
                self.assertEqual(before["rps"], 1)
                self.assertEqual(after["rps"], 3)
                self.assertEqual(result["RPS"]["rps"], 2)
                self.assertEqual(result["RPS_ref"]["rps"], 2)
                self.assertEqual(result["skill_results"]["fair_rps_skill_score"], 0)
                self.assertEqual(len(brier.call_args.args[0]), 6)
                self.assertEqual(len(auc.call_args.args[0]), 6)
                self.assertEqual(len(result["reliability_forecast_obs_df"]), 2)

    def test_zero_reference_scores_produce_undefined_skill_instead_of_infinity(self):
        ns = {"np": np, "pd": pd, "warnings": warnings}
        load_functions("stats/bins.py", ns, {"extract_day_range"})
        load_functions("stats/score.py", ns)
        for reference in (0.0, 0.4):
            with self.subTest(reference=reference):
                forecast = {"fair_brier_score": .2, "bin_fair_brier_scores": {"Days 1-5": .2}}
                baseline = {"fair_brier_score": reference,
                            "bin_fair_brier_scores": {"Days 1-5": reference}}
                with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    result = ns["calculate_skill_scores"](forecast, {"fair_rps": .2},
                                                         baseline, {"fair_rps": reference})
                values = [result["fair_brier_skill_score"], result["fair_rps_skill_score"],
                          result["bin_fair_brier_skill_scores"]["Days 1-5"]]
                if reference == 0:
                    self.assertTrue(all(np.isnan(value) for value in values))
                else:
                    np.testing.assert_allclose(values, [.5, .5, .5])

    def test_wholly_missing_member_is_not_a_false_no_onset_forecast(self):
        ns = {"np": np, "pd": pd, "datetime": datetime, "product": product,
              "_fit_forecast_horizon": lambda *args: (2, 0, 3),
              "restore_args": lambda func, kwargs, bound: {**kwargs, **{
                  key: value for key, value in bound.items() if key != "kwargs"}}}
        load_functions("stats/detect.py", ns, {"detect_onset", "compute_onset_for_all_members"})
        load_functions("stats/bins.py", ns, {"create_forecast_observation_pairs_with_bins"})
        values = np.full((1, 3, 3, 1, 1), np.nan)
        values[:, :2] = 5
        model = xr.DataArray(values, dims=("init_time", "member", "step", "lat", "lon"),
                             coords={"init_time": [np.datetime64("2024-06-01")],
                                     "member": [0, 1, 2], "step": [1, 2, 3],
                                     "lat": [10.], "lon": [38.]})
        observed = xr.DataArray([[np.datetime64("2024-06-10")]], dims=("lat", "lon"),
                                coords={"lat": [10.], "lon": [38.]})
        with contextlib.redirect_stdout(io.StringIO()):
            onset = ns["compute_onset_for_all_members"](model, 1, observed, members=None,
                                                       dry_extent=0, wet_spell=2)
            pairs = ns["create_forecast_observation_pairs_with_bins"](
                onset, observed, day_bins=((1, 15),), max_forecast_day=15)
        self.assertEqual(set(onset.member), {0, 1})
        self.assertTrue(onset.onset_day.notna().all())
        full_window = pairs.loc[pairs.bin_label == "Days 1-15"].iloc[0]
        self.assertEqual(full_window.total_members, 2)
        self.assertEqual(full_window.predicted_prob, 1)

    def test_climatology_fair_rps_uses_effective_member_count(self):
        ns = {"np": np, "pd": pd}
        load_functions("stats/score.py", ns)
        pairs = pd.DataFrame({
            "init_time": ["2024-06-01"] * 2, "lat": [10] * 2, "lon": [38] * 2,
            "bin_index": [0, 1], "predicted_prob": [.5, .5], "observed_onset": [1, 0],
            "total_members": [4, 4], "total_members_with_onset": [2, 2],
        })
        with contextlib.redirect_stdout(io.StringIO()):
            result = ns["calculate_rps"](pairs)
        self.assertEqual(result["rps"], .25)
        # One cumulative .25 error minus .25/(2-1) correction; final CDF error is zero.
        self.assertEqual(result["fair_rps"], 0)

    def test_no_usable_ensemble_fails_cleanly(self):
        ns = {"np": np, "pd": pd, "datetime": datetime, "product": product,
              "_fit_forecast_horizon": lambda *args: (2, 0, 3),
              "restore_args": lambda func, kwargs, bound: {**kwargs, **{
                  key: value for key, value in bound.items() if key != "kwargs"}}}
        load_functions("stats/detect.py", ns, {"detect_onset", "compute_onset_for_all_members"})
        load_functions("stats/bins.py", ns, {"create_forecast_observation_pairs_with_bins"})
        path = MOMP / "stats/bins.py"
        tree = ast.parse(path.read_text())
        error_class = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                           and node.name == "NoForecastDataError")
        exec(compile(ast.Module(body=[error_class], type_ignores=[]), str(path), "exec"), ns)  # noqa: S102
        observed = xr.DataArray([[np.datetime64("2024-06-10")]], dims=("lat", "lon"),
                                coords={"lat": [10.], "lon": [38.]})
        for valid_member_count in (0, 1):
            with self.subTest(valid_member_count=valid_member_count):
                values = np.full((1, 3, 3, 1, 1), np.nan)
                values[:, :valid_member_count] = 5
                model = xr.DataArray(values, dims=("init_time", "member", "step", "lat", "lon"),
                                     coords={"init_time": [np.datetime64("2024-06-01")],
                                             "member": [0, 1, 2], "step": [1, 2, 3],
                                             "lat": [10.], "lon": [38.]})
                with contextlib.redirect_stdout(io.StringIO()):
                    onset = ns["compute_onset_for_all_members"](model, 1, observed, members="All",
                                                               dry_extent=0, wet_spell=2)
                    with self.assertRaises(ns["NoForecastDataError"]):
                        ns["create_forecast_observation_pairs_with_bins"](
                            onset, observed, day_bins=((1, 15),), max_forecast_day=15)
