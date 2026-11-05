"""Scientific ordering and arithmetic checks against the supplied ROMP reference."""

import ast
import contextlib
import io
from itertools import permutations, product
import os
from pathlib import Path
import tempfile
import unittest
import warnings

import numpy as np
import pandas as pd
from scipy import stats
import xarray as xr


PROJECT = Path(__file__).resolve().parents[1]
LOCAL = PROJECT / 'apps/benchmarking/ROMP/momp'
REFERENCE = Path(os.getenv('ROMP_REFERENCE_MOMP', PROJECT.parents[2] / 'ROMP/momp'))


def functions(path, **namespace):
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


class ReferenceScoringTests(unittest.TestCase):
    def setUp(self):
        self.scope = functions(LOCAL / 'stats/bins.py', np=np, pd=pd, product=product)
        self.scope = functions(LOCAL / 'stats/score.py', **self.scope, stats=stats, warnings=warnings)
        quiet = contextlib.redirect_stdout(io.StringIO())
        quiet.__enter__()
        self.addCleanup(quiet.__exit__, None, None, None)

    @staticmethod
    def cases(order=(1, 6, 11)):
        probabilities = {1: 1., 6: 0., 11: 0.}
        outcomes = {1: 0, 6: 0, 11: 1}
        return pd.DataFrame({
            'init_time': ['2023-06-01'] * 3, 'lat': [8.] * 3, 'lon': [39.] * 3,
            'bin_index': range(3), 'bin_label': [f'Days {start}-{start + 4}' for start in order],
            'bin_start': order, 'bin_end': [start + 4 for start in order],
            'predicted_prob': [probabilities[start] for start in order],
            'observed_onset': [outcomes[start] for start in order], 'total_members': [3] * 3})

    def test_direct_rps_rejects_nonchronological_category_indices(self):
        with self.assertRaisesRegex(ValueError, 'chronolog|order'):
            self.scope['calculate_rps'](self.cases((1, 11, 6)))
        self.assertEqual(self.scope['calculate_rps'](self.cases())['rps'], 2.)

    def test_direct_rps_rejects_gapped_or_overlapping_day_partitions(self):
        for start in (5, 7):
            with self.subTest(start=start):
                frame = self.cases()
                frame.loc[1, 'bin_start'] = start
                with self.assertRaisesRegex(ValueError, 'partition|overlap'):
                    self.scope['calculate_rps'](frame)

    def test_forecast_pair_generation_is_invariant_to_configured_bin_order(self):
        onsets = pd.DataFrame({'init_time': ['2023-06-01'] * 2, 'lat': [8.] * 2,
                               'lon': [39.] * 2, 'onset_day': [1, 1]})
        truth = xr.DataArray(np.array([['2023-06-15']], dtype='datetime64[ns]'),
                             dims=('lat', 'lon'), coords={'lat': [8.], 'lon': [39.]})
        expected = None
        for bins in permutations(((1, 5), (6, 10), (11, 15))):
            frame = self.scope['create_forecast_observation_pairs_with_bins'](
                onsets, truth, day_bins=bins, max_forecast_day=15)
            self.assertEqual(self.scope['calculate_rps'](frame)['rps'], 2.)
            if expected is None:
                expected = frame
            else:
                pd.testing.assert_frame_equal(frame, expected)

    def test_climatological_pair_generation_preserves_full_partition_under_permutation(self):
        climate = xr.DataArray(np.array(['2019-06-02', '2021-06-03', '2022-06-15'],
                                        dtype='datetime64[ns]')[:, None, None],
                               dims=('year', 'lat', 'lon'),
                               coords={'year': [2019, 2021, 2022], 'lat': [8.], 'lon': [39.]})
        truth = xr.DataArray(np.array([['2023-06-15']], dtype='datetime64[ns]'),
                             dims=('lat', 'lon'), coords={'lat': [8.], 'lon': [39.]})
        expected = None
        for bins in permutations(((1, 5), (6, 10), (11, 15))):
            frame = self.scope['create_climatological_forecast_obs_pairs'](
                climate, 2023, ['2023-06-01'], observed_onset=truth,
                day_bins=bins, max_forecast_day=15)
            self.assertEqual(frame.bin_label.tolist(),
                             ['Before day 1', 'Days 1-5', 'Days 6-10', 'Days 11-15', 'After day 15'])
            self.assertAlmostEqual(self.scope['calculate_rps'](frame)['rps'], 8 / 9)
            if expected is None:
                expected = frame
            else:
                pd.testing.assert_frame_equal(frame, expected)

    def test_overlapping_baseline_year_is_retained_but_explicitly_identified(self):
        climate = xr.DataArray(np.array(['2021-06-04', '2022-06-14'], dtype='datetime64[ns]')[:, None, None],
                               dims=('year', 'lat', 'lon'),
                               coords={'year': [2021, 2022], 'lat': [8.], 'lon': [39.]})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            frame = self.scope['create_climatological_forecast_obs_pairs'](
                climate, 2021, ['2021-06-01'], day_bins=((1, 5), (6, 10), (11, 15)),
                max_forecast_day=15)
        self.assertTrue(any('includes verification year 2021' in str(item.message) for item in caught))
        self.assertEqual(frame.loc[frame.bin_label == 'Days 1-5', 'predicted_prob'].item(), .5)
        self.assertEqual(set(frame.reference_membership), {'includes_verification_year'})

    def test_reference_missing_onset_conditioning_is_preserved_and_distinguished(self):
        climate = xr.DataArray(np.array(['2021-06-04', 'NaT'], dtype='datetime64[ns]')[:, None, None],
                               dims=('year', 'lat', 'lon'),
                               coords={'year': [2021, 2022], 'lat': [8.], 'lon': [39.]})
        truth = xr.DataArray(np.array([['2023-06-04']], dtype='datetime64[ns]'),
                             dims=('lat', 'lon'), coords={'lat': [8.], 'lon': [39.]})
        baseline = self.scope['create_climatological_forecast_obs_pairs'](
            climate, 2023, ['2023-06-01'], observed_onset=truth,
            day_bins=((1, 5),), max_forecast_day=5)
        forecast = pd.DataFrame({'init_time': ['2023-06-01'] * 2, 'lat': [8.] * 2,
                                 'lon': [39.] * 2, 'onset_day': [3., np.nan]})
        model = self.scope['create_forecast_observation_pairs_with_bins'](
            forecast, truth, day_bins=((1, 5),), max_forecast_day=5)
        self.assertEqual(baseline.loc[baseline.bin_label == 'Days 1-5', 'predicted_prob'].item(), 1.)
        self.assertEqual(model.loc[model.bin_label == 'Days 1-5', 'predicted_prob'].item(), .5)
        self.assertEqual(baseline.total_members_with_onset.iloc[0], 1)

    @unittest.skipUnless((REFERENCE / 'stats/score.py').is_file(), 'local reference checkout is absent')
    def test_valid_ordered_score_arithmetic_matches_reference(self):
        reference = functions(REFERENCE / 'stats/score.py', np=np, pd=pd, stats=stats, warnings=warnings)
        frame = self.cases()
        frame['predicted_prob'] = [.2, .3, .5]
        frame['observed_onset'] = [1, 0, 0]
        frame['total_members'] = 10
        for name, keys in (('calculate_brier_score', ('brier_score', 'fair_brier_score')),
                           ('calculate_rps', ('rps', 'fair_rps')),
                           ('calculate_auc', ('auc',))):
            expected = reference[name](frame.copy())
            actual = self.scope[name](frame.copy())
            for key in keys:
                self.assertAlmostEqual(actual[key], expected[key])

    def test_score_export_identifies_reference_membership_and_conditioning(self):
        ns = functions(LOCAL / 'io/output.py', **self.scope, os=os,
                       tuple_to_str=lambda value: '-'.join(map(str, value)))
        brier = {'fair_brier_score': .2, 'bin_fair_brier_scores': {'Days 1-5': .2}}
        result = {'BS': brier, 'BS_ref': brier, 'RPS': {'fair_rps': .3},
                  'RPS_ref': {'fair_rps': .4}, 'AUC': {'auc': .5, 'bin_auc_scores': {}},
                  'AUC_ref': {'auc': .5, 'bin_auc_scores': {}},
                  'skill_results': {'fair_brier_skill_score': 0., 'fair_rps_skill_score': .25,
                                    'bin_fair_brier_skill_scores': {}},
                  'climatology_obs_df': pd.DataFrame({
                      'reference_membership': ['includes_verification_year']}),
                  'forecast_obs_df': pd.DataFrame({'year': [2022], 'init_time': ['2022-06-01']})}
        with tempfile.TemporaryDirectory() as directory:
            _, overall = ns['save_score_results'](result, model='test', verification_window=(1, 15),
                                                  dir_out=directory, years=(2021, 2022), years_clim=(2021, 2022))
            self.assertEqual(overall['Reference_Membership'], 'includes_verification_year')
            self.assertEqual(overall['Reference_Probability_Conditioning'], 'finite_climatological_onsets')
            self.assertEqual(overall['Forecast_Probability_Conditioning'], 'all_usable_members_including_no_onset')
            self.assertEqual(overall['Evaluation_Years'], '(2021, 2022)')
            self.assertEqual(overall['Actual_Evaluation_Years'], '(2022,)')
            self.assertEqual(overall['Requested_Reference_Years'], '(2021, 2022)')
            result['forecast_obs_df'] = result['forecast_obs_df'].drop(columns='year')
            _, fallback = ns['save_score_results'](result, model='test', verification_window=(1, 15),
                                                   dir_out=directory, years=(2021, 2022), years_clim=(2021, 2022))
            self.assertEqual(fallback['Actual_Evaluation_Years'], '(2022,)')

    def test_spatial_netcdf_preserves_scientific_runtime_parameters(self):
        ns = functions(LOCAL / 'io/output.py', np=np, pd=pd, xr=xr, os=os)
        field = xr.DataArray([[1.]], dims=('lat', 'lon'), coords={'lat': [8.], 'lon': [39.]})
        with tempfile.TemporaryDirectory() as directory:
            for probabilistic in (False, True):
                settings = {'dir_out': directory, 'case_name': 'test', 'model': 'AIFS',
                            'years': (2021, 2022), 'years_clim': (2001, 2020),
                            'wet_init': 1.5, 'wet_threshold': 20., 'wet_spell': 3,
                            'dry_threshold': .7, 'dry_spell': 6, 'dry_extent': 19,
                            'onset_percentage_threshold': .6, 'members': (0, 1, 2),
                            'date_filter_year': 2024, 'init_days': (0, 3), 'init_type': 'weekly',
                            'start_date': (5, 1), 'end_date': (7, 31), 'probabilistic': probabilistic,
                            'tolerance_days': 3, 'verification_window': (1, 15),
                            'max_forecast_day': 15, 'mok': None}
                ns['save_metrics_to_netcdf']({'mae': field}, settings)
                with xr.open_dataset(Path(directory) / 'spatial_metrics_test.nc') as saved:
                    self.assertEqual(saved.attrs['dry_threshold'], .7)
                    self.assertEqual(saved.attrs['dry_extent'], 19)
                    self.assertEqual(saved.attrs['onset_percentage_threshold'], .6)
                    self.assertEqual(saved.attrs['members'], '0,1,2')
                    self.assertEqual(saved.attrs['years_clim'], '2001,2020')
                    self.assertEqual(saved.attrs['contingency_definition'], 'romp_reference_exclusive_window_v1')
                    self.assertEqual(saved.attrs['ensemble_aggregation'],
                                     'reference_mean_if_consensus' if probabilistic else 'deterministic')
