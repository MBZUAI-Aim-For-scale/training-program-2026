"""Scientific regressions for the reference ROMP ensemble/onset definitions."""

import ast
import contextlib
import io
import os
import sys
import unittest
from datetime import datetime, timedelta
from itertools import product
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'apps/benchmarking/ROMP'))
from momp.stats import benchmark, detect
from momp.utils.practical import restore_args


class ReferenceOnsetTests(unittest.TestCase):
    def members(self, days):
        init = pd.Timestamp('2024-06-01')
        return pd.DataFrame({
            'init_time': [init] * len(days), 'lat': [8.] * len(days),
            'lon': [39.] * len(days), 'member': list(range(len(days))),
            'onset_day': days,
            'onset_date': [None if day is None else str((init + pd.Timedelta(days=day)).date())
                           for day in days],
            'obs_onset_date': ['2024-06-05'] * len(days),
        })

    def test_reference_ensemble_threshold_is_applied_before_mean_timing(self):
        for days, threshold, expected in (([4, None, None], .5, None),
                                          ([2, 6, None], .5, 4),
                                          ([2, 6, None], .8, None),
                                          ([2, 3, None], 2 / 3, 2),
                                          ([None, None, None], 0, None)):
            with self.subTest(days=days, threshold=threshold):
                result = detect.aggregate_ensemble_onset(
                    self.members(days), onset_percentage_threshold=threshold)
                self.assertEqual(len(result), 1)
                actual = result.iloc[0].onset_day
                self.assertTrue(pd.isna(actual)) if expected is None else self.assertEqual(actual, expected)
                self.assertEqual(result.iloc[0].total_members, 3)

    def test_gridded_consensus_matches_supplied_reference_package(self):
        project = Path(__file__).resolve().parents[1]
        reference = Path(os.environ.get('ROMP_REFERENCE_MOMP',
                                       project.parents[2] / 'ROMP/momp')) / 'stats/detect.py'
        if not reference.is_file():
            self.skipTest('Set ROMP_REFERENCE_MOMP to compare with a local reference package')
        # Load only the reference functions, without shadowing local momp imports.
        names = {'detect_onset', 'compute_onset_for_all_members'}
        nodes = [node for node in ast.parse(reference.read_text()).body
                 if isinstance(node, ast.FunctionDef) and node.name in names]
        namespace = dict(np=np, pd=pd, xr=xr, datetime=datetime, timedelta=timedelta,
                         product=product, restore_args=restore_args)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(reference), 'exec'), namespace)
        observed = xr.DataArray(np.array([['2024-06-10']], dtype='datetime64[ns]'),
                                dims=('lat', 'lon'), coords={'lat': [8.], 'lon': [39.]})
        for days, threshold in (([4, None, None], .5), ([2, 6, None], .5),
                                ([2, 6, None], .8), ([2, 3, None], 2 / 3)):
            with self.subTest(days=days, threshold=threshold):
                values = np.zeros((1, 3, 36, 1, 1))
                for index, day in enumerate(days):
                    if day is not None:
                        values[0, index, day - 1:, 0, 0] = 5.
                model = xr.DataArray(values, dims=('init_time', 'member', 'step', 'lat', 'lon'),
                    coords={'init_time': pd.to_datetime(['2024-06-01']), 'member': [0, 1, 2],
                            'step': np.arange(1, 37), 'lat': [8.], 'lon': [39.]})
                settings = dict(wet_init=1., wet_spell=3, dry_spell=7, dry_threshold=1.,
                    dry_extent=0, members=[0, 1, 2], onset_percentage_threshold=threshold,
                    max_forecast_day=15, mok=None, end_date=(2024, 7, 31))
                with contextlib.redirect_stdout(io.StringIO()):
                    _, expected = namespace['compute_onset_for_all_members'](
                        model, 10., observed, **settings)
                    actual = detect.aggregate_ensemble_onset(
                        detect.compute_onset_for_all_members(model, 10., observed, **settings),
                        onset_percentage_threshold=threshold)
                pd.testing.assert_frame_equal(actual[expected.columns], expected, check_dtype=False)

        # Reproduce the reported 45-day forecast using the actual reference:
        # its configured default works, while an explicit 21-day check does not.
        model = model.reindex(step=np.arange(1, 46), fill_value=5.)
        settings.update(max_forecast_day=30, dry_extent=0)
        with contextlib.redirect_stdout(io.StringIO()):
            _, expected = namespace['compute_onset_for_all_members'](model, 10., observed, **settings)
            actual = detect.aggregate_ensemble_onset(
                detect.compute_onset_for_all_members(model, 10., observed, **settings),
                onset_percentage_threshold=settings['onset_percentage_threshold'])
            pd.testing.assert_frame_equal(actual[expected.columns], expected, check_dtype=False)
            settings['dry_extent'] = 21
            with self.assertRaisesRegex(ValueError, 'Not enough forecast time steps'):
                namespace['compute_onset_for_all_members'](model, 10., observed, **settings)
            with self.assertRaisesRegex(ValueError, 'day 50'):
                detect.compute_onset_for_all_members(model, 10., observed, **settings)

    def test_both_spatial_paths_score_one_ensemble_forecast_per_initialization(self):
        observed = xr.DataArray(np.array([['2024-06-05']], dtype='datetime64[ns]'),
                                dims=('lat', 'lon'), coords={'lat': [8.], 'lon': [39.]})
        for function in (benchmark.compute_metrics_multiple_years,
                         benchmark.ens_compute_metrics_multiple_years):
            for days, hits, misses in (([4, None, None], 0, 1), ([2, 6, None], 1, 0)):
                with self.subTest(function=function.__name__, days=days), \
                     patch.object(benchmark, 'load_thresh_file', return_value=10), \
                     patch.object(benchmark, 'available_forecast_years', return_value=[2024]), \
                     patch.object(benchmark, 'get_forecast_probabilistic_twice_weekly', return_value='forecast'), \
                     patch.object(benchmark, 'load_imd_rainfall', return_value='rainfall'), \
                     patch.object(benchmark, 'detect_observed_onset', return_value=observed), \
                     patch.object(benchmark, 'compute_onset_for_all_members', return_value=self.members(days)):
                    results, _ = function(years=[2024], probabilistic=True, model='ensemble',
                        onset_percentage_threshold=.5, verification_window=(1, 15))
                result = results[2024].iloc[0]
                self.assertEqual(result.total_forecasts, 1)
                self.assertEqual(result.true_positive, hits)
                self.assertEqual(result.false_negative, misses)

    def test_confirmation_window_never_changes_to_fit_short_forecast(self):
        forecast = xr.DataArray(np.ones(36), dims='step', coords={'step': np.arange(1, 37)})
        with self.assertRaisesRegex(ValueError, 'requires day 50'):
            detect._fit_forecast_horizon(forecast, 30, 21, 3, ((1, 30),))

    def test_disabled_dry_spell_only_requires_the_wet_spell_tail(self):
        forecast = xr.DataArray(np.ones(17), dims='step', coords={'step': np.arange(1, 18)})
        self.assertEqual(detect._fit_forecast_horizon(
            forecast, 15, 21, 3, ((1, 15),), dry_spell=0), (15, 21, 17))

    def test_reference_strict_accumulation_and_complete_wet_spell(self):
        self.assertFalse(detect.detect_onset(1, [5., 5., 5.], 15., dry_extent=0))
        self.assertTrue(detect.detect_onset(1, [5., 5., 5.], 14., dry_extent=0))
        self.assertFalse(detect.detect_onset(1, [10., 0., 10.], 14., wet_init=1., dry_extent=0))

    def test_reference_mistimed_prediction_has_one_false_positive_outcome(self):
        forecast = self.members([9])
        result, _ = benchmark.compute_onset_metrics_with_windows(
            forecast, tolerance_days=3, verification_window=(1, 15))
        row = result.iloc[0]
        self.assertEqual(row.total_forecasts, 1)
        self.assertEqual(row.true_positive, 0)
        self.assertEqual(row.false_positive, 1)
        self.assertEqual(row.false_negative, 0)
        self.assertEqual(row.true_negative, 0)
