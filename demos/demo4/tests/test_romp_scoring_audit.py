"""Analytic scoring regressions for the ROMP/MOMP audit."""

import ast
import contextlib
import io
import inspect
from itertools import product
from pathlib import Path
import unittest
import warnings

import numpy as np
import pandas as pd
from scipy import stats
import xarray as xr

ROOT = Path(__file__).resolve().parents[1] / 'apps/benchmarking/ROMP/momp'


def functions(path, **namespace):
    tree = ast.parse((ROOT / path).read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


class ScoringAuditTests(unittest.TestCase):
    def setUp(self):
        self.scope = functions('stats/bins.py', np=np, pd=pd)
        self.scope = functions('stats/score.py', **self.scope, stats=stats, warnings=warnings)
        self.quiet = contextlib.redirect_stdout(io.StringIO())
        self.quiet.__enter__()
        self.addCleanup(self.quiet.__exit__, None, None, None)

    @staticmethod
    def pairs(probabilities=(.5, .5), observations=(1, 0), members=2, date='2024-06-01'):
        return pd.DataFrame({'init_time': [date] * len(probabilities), 'lat': [10.] * len(probabilities),
                             'lon': [38.] * len(probabilities), 'bin_index': range(len(probabilities)),
                             'bin_label': [f'Days {i + 1}-{i + 1}' for i in range(len(probabilities))],
                             'predicted_prob': probabilities, 'observed_onset': observations,
                             'total_members': [members] * len(probabilities)})

    def test_fair_brier_single_member_is_undefined_without_infinity(self):
        frame = self.pairs((.5,), (1,), members=1)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = self.scope['calculate_brier_score'](frame)
        self.assertAlmostEqual(result['brier_score'], .25)
        self.assertTrue(np.isnan(result['fair_brier_score']))
        self.assertTrue(np.isnan(result['bin_fair_brier_scores']['Days 1-1']))

    def test_brier_does_not_mutate_pair_table(self):
        frame = self.pairs()
        original = frame.copy(deep=True)
        result = self.scope['calculate_brier_score'](frame)
        self.assertEqual(result['brier_score'], .25)
        self.assertEqual(result['fair_brier_score'], 0)
        pd.testing.assert_frame_equal(frame, original)

    def test_scores_reject_nonfinite_or_out_of_range_probability_and_nonbinary_truth(self):
        for probability, truth in ((1.1, 1), (-.1, 0), (np.inf, 0), (np.nan, 1), (.5, 2), (.5, np.nan)):
            for name in ('calculate_brier_score', 'calculate_auc', 'calculate_rps'):
                with self.subTest(probability=probability, truth=truth, scorer=name):
                    with self.assertRaisesRegex(ValueError, 'probab|observ'):
                        self.scope[name](self.pairs((probability,), (truth,)))

    def test_auc_ties_are_half_credit_and_single_class_undefined(self):
        result = self.scope['calculate_auc'](self.pairs((.5, .5), (1, 0)))
        self.assertEqual(result['auc'], .5)
        result = self.scope['calculate_auc'](self.pairs((.5, .5), (1, 1)))
        self.assertTrue(np.isnan(result['auc']))

    def test_rps_rejects_incomplete_or_overlapping_probability_partition(self):
        for probabilities, observations in (((.2, .2), (1, 0)), ((.7, .7), (1, 0)), ((.5, .5), (1, 1))):
            with self.subTest(probabilities=probabilities, observations=observations):
                with self.assertRaisesRegex(ValueError, 'partition|sum|categor'):
                    self.scope['calculate_rps'](self.pairs(probabilities, observations))

    def test_rps_rejects_inconsistent_member_count_within_one_forecast(self):
        frame = self.pairs()
        frame['total_members'] = [2, 3]
        with self.assertRaisesRegex(ValueError, 'member'):
            self.scope['calculate_rps'](frame)

    def test_match_rejects_different_observations_for_identical_case(self):
        forecast = self.pairs()
        reference = self.pairs(observations=(0, 1))
        with self.assertRaisesRegex(ValueError, 'observ'):
            self.scope['match_samples'](forecast, reference)

    def test_sample_matching_rejects_missing_initialization_or_coordinates(self):
        for column, value in (('init_time', None), ('lat', np.nan), ('lon', np.inf)):
            frame = self.pairs()
            frame[column] = value
            with self.subTest(column=column):
                with self.assertRaisesRegex(ValueError, 'sample keys'):
                    self.scope['match_samples'](frame, frame)

    def test_rps_accepts_numeric_member_metadata_and_sorts_indices_numerically(self):
        frame = self.pairs((.2, .3, .5), (1, 0, 0), members=10)
        frame['total_members'] = ['10', '10', '10']
        frame['bin_index'] = ['2', '10', '11']
        self.assertAlmostEqual(self.scope['calculate_rps'](frame)['rps'], .89)

    def test_paired_brier_excludes_unsupported_reference_on_both_sides(self):
        forecast = pd.concat([self.pairs(), self.pairs((0., 1.), (1, 0), date='2024-06-02')], ignore_index=True)
        reference = forecast.copy()
        reference['total_members_with_onset'] = [2, 2, 1, 1]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model, baseline = self.scope['calculate_paired_brier'](forecast, reference)
        self.assertEqual(model['brier_score'], .25)
        self.assertEqual(baseline['brier_score'], .25)
        self.assertEqual(model['n_excluded_samples'], 2)
        self.assertEqual(baseline['n_excluded_samples'], 2)

    def test_disabled_score_families_do_not_crash_skill_computation(self):
        brier = {'fair_brier_score': .2, 'bin_fair_brier_scores': {'Days 1-5': .2}}
        result = self.scope['calculate_skill_scores'](brier, None, brier, None)
        self.assertEqual(result['fair_brier_skill_score'], 0)
        self.assertTrue(np.isnan(result['fair_rps_skill_score']))

    def test_pipeline_brier_and_auc_compare_only_common_reference_cases(self):
        correct = self.pairs((1., 0.), (1, 0))
        wrong = self.pairs((0., 1.), (1, 0), date='2024-06-02')
        forecast = pd.concat([correct, wrong], ignore_index=True)
        reference = self.pairs((.5, .5), (1, 0), date='2024-06-02')
        for frame in (forecast, reference):
            frame['bin_start'] = [1, 6] * (len(frame) // 2)
            frame['bin_end'] = [5, 10] * (len(frame) // 2)
            frame['bin_label'] = ['Days 1-5', 'Days 6-10'] * (len(frame) // 2)
        runner_frames = iter((forecast, reference))
        ns = functions('io/dict.py', **self.scope)
        ns.update(available_forecast_years=lambda years, **kwargs: years,
                  multi_year_forecast_obs_pairs=lambda **kwargs: next(runner_frames),
                  restore_args=lambda function, kwargs, bound: {
                      **{key: value for key, value in bound.items() if key != 'kwargs'}, **kwargs})
        ns = functions('metrics/skill.py', **ns)
        run = ns['create_score_results']
        args = {name: None for name in inspect.signature(run).parameters if name != 'kwargs'}
        args.update(BS=True, RPS=False, AUC=True, skill_score=False, years=(2024,),
                    ref_model='other', day_bins=((1, 5), (6, 10)), parallel=False)
        result = run(**args)
        self.assertEqual(result['BS']['brier_score'], 1.)
        self.assertEqual(result['BS_ref']['brier_score'], .25)
        self.assertEqual(result['AUC']['auc'], 0.)
        self.assertEqual(result['AUC_ref']['auc'], .5)
        self.assertEqual(len(result['forecast_obs_df']), 2)

    def test_paired_rps_cannot_drop_zero_mass_categories_from_one_side(self):
        forecast = self.pairs((1., 0.), (1, 0))
        reference = forecast.iloc[:1].copy()
        with self.assertRaisesRegex(ValueError, 'categor|partition|bins'):
            self.scope['calculate_paired_rps'](forecast, reference)

    def test_paired_rps_normalizes_equivalent_date_and_coordinate_keys(self):
        forecast = self.pairs()
        forecast['init_time'] = pd.to_datetime(forecast['init_time'])
        reference = self.pairs()
        model, baseline = self.scope['calculate_paired_rps'](forecast, reference)
        self.assertEqual(model['n_forecasts'], 1)
        self.assertEqual(baseline['n_forecasts'], 1)
        self.assertEqual(model['rps'], baseline['rps'])
        reference = reference.iloc[:1].copy()
        reference['lat'] = '10.0000000001'
        reference['lon'] = '38'
        with self.assertRaisesRegex(ValueError, 'categor|partition|bins'):
            self.scope['calculate_paired_rps'](forecast, reference)



class ClimatologyBinAuditTests(unittest.TestCase):
    def test_overlapping_reliability_bin_does_not_change_effective_members_or_probability(self):
        scope = functions('stats/bins.py', pd=pd, np=np, product=product)
        climate = xr.DataArray(np.array(['2021-06-04', '2022-06-09', '2023-06-14'], dtype='datetime64[ns]')[:, None, None],
                               dims=('year', 'lat', 'lon'), coords={'year': [2021, 2022, 2023], 'lat': [10.], 'lon': [38.]})
        with contextlib.redirect_stdout(io.StringIO()):
            result = scope['create_climatological_forecast_obs_pairs'](
                climate, 2021, ['2021-06-01'], day_bins=((1, 5), (6, 10), (11, 15), (1, 15)), max_forecast_day=15)
        self.assertTrue((result.total_members_with_onset == 3).all())
        self.assertEqual(result.loc[result.bin_label == 'Days 1-15', 'predicted_prob'].item(), 1.)
        self.assertEqual(result.loc[result.bin_label == 'Days 1-5', 'predicted_prob'].item(), 1/3)


    def test_climatology_diagnostics_are_local_to_each_initialization(self):
        scope = functions('stats/bins.py', pd=pd, np=np, product=product)
        climate = xr.DataArray(np.array(['2021-06-24', '2022-06-04', '2023-06-14'], dtype='datetime64[ns]')[:, None, None],
                               dims=('year', 'lat', 'lon'), coords={'year': [2021, 2022, 2023], 'lat': [10.], 'lon': [38.]})
        with contextlib.redirect_stdout(io.StringIO()):
            result = scope['create_climatological_forecast_obs_pairs'](
                climate, 2021, ['2021-06-01', '2021-06-02'], day_bins=((16, 30),), max_forecast_day=30)
        before = result[result.bin_label == 'Before day 16']
        self.assertEqual(before.obs_days_from_init_doy.tolist(), [23, 22])
        self.assertTrue((before.members_with_onset == before.n_contributing_years).all())

    def test_disjoint_baseline_years_use_separate_target_observation_without_adding_member(self):
        scope = functions('stats/bins.py', pd=pd, np=np, product=product)
        climate = xr.DataArray(np.array(['2022-06-04', '2023-06-14'], dtype='datetime64[ns]')[:, None, None],
                               dims=('year', 'lat', 'lon'), coords={'year': [2022, 2023], 'lat': [10.], 'lon': [38.]})
        truth = xr.DataArray(np.array([['2024-06-09']], dtype='datetime64[ns]'),
                            dims=('lat', 'lon'), coords={'lat': [10.], 'lon': [38.]})
        with contextlib.redirect_stdout(io.StringIO()):
            result = scope['create_climatological_forecast_obs_pairs'](
                climate, 2024, ['2024-06-01'], observed_onset=truth,
                day_bins=((1, 5), (6, 10), (11, 15)), max_forecast_day=15)
        self.assertTrue((result.total_members == 2).all())
        self.assertTrue((result.total_members_with_onset == 2).all())
        self.assertEqual(result.loc[result.bin_label == 'Days 6-10', 'observed_onset'].item(), 1)
        self.assertEqual(result.loc[result.bin_label == 'Days 1-5', 'predicted_prob'].item(), .5)
        self.assertEqual(result.loc[result.bin_label == 'Days 11-15', 'predicted_prob'].item(), .5)

    def test_multiyear_baseline_loads_truth_for_evaluation_year_absent_from_baseline(self):
        climate = xr.DataArray(np.array(['2022-06-04', '2023-06-14'], dtype='datetime64[ns]')[:, None, None],
                               dims=('year', 'lat', 'lon'), coords={'year': [2022, 2023], 'lat': [10.], 'lon': [38.]})
        truth = xr.DataArray(np.array([['2024-06-09']], dtype='datetime64[ns]'),
                            dims=('lat', 'lon'), coords={'lat': [10.], 'lon': [38.]})
        loaded = []
        scope = functions('stats/bins.py', pd=pd, np=np, product=product,
                          restore_args=lambda function, kwargs, bound: {
                              **{key: value for key, value in bound.items() if key != 'kwargs'}, **kwargs},
                          get_initialization_dates=lambda year, **kwargs: ['2024-06-01'],
                          load_thresh_file=lambda **kwargs: 1.,
                          load_imd_rainfall=lambda year, **kwargs: loaded.append(year) or 'rainfall',
                          detect_observed_onset=lambda rainfall, threshold, year, **kwargs: truth)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = scope['multi_year_climatological_forecast_obs_pairs'](
                climate, years=(2024,), day_bins=((1, 5), (6, 10), (11, 15)),
                max_forecast_day=15, date_filter_year=2024, init_days=(0,), start_date='06-01', end_date='06-30')
        self.assertEqual(loaded, [2024])
        self.assertEqual(set(result.year), {2024})
        self.assertTrue((result.total_members == 2).all())
        self.assertNotIn('Error processing target year', output.getvalue())
        self.assertNotIn('not found in climatological dataset', output.getvalue())
