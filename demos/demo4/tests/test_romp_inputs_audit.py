"""Analytic regressions for shared onset and daily input handling."""

import sys
import contextlib
import inspect
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'apps/benchmarking/ROMP'))
from momp.io import input as inputs
from momp.stats import detect
from momp.stats import bins
from momp.utils.standard import dim_fmt, dim_fmt_model


class OnsetInputAuditTests(unittest.TestCase):
    def test_default_30_day_criteria_fit_45_days_but_explicit_21_requires_50(self):
        forecast = xr.DataArray(np.full((1, 45, 1, 1), 10.),
            dims=('init_time', 'step', 'lat', 'lon'),
            coords={'init_time': pd.to_datetime(['2024-05-01']), 'step': np.arange(1, 46),
                    'lat': [8.], 'lon': [39.]})
        truth = xr.DataArray(np.array([['2024-05-10']], dtype='datetime64[ns]'),
                            dims=('lat', 'lon'), coords={'lat': [8.], 'lon': [39.]})
        for ensemble in (False, True):
            compute = detect.compute_onset_for_all_members if ensemble else detect.compute_onset_for_deterministic_model
            model = forecast.expand_dims(member=[0, 1]) if ensemble else forecast
            if ensemble:
                model = model.transpose('init_time', 'member', 'step', 'lat', 'lon')
            with self.subTest(ensemble=ensemble):
                result = compute(model, 20., truth, max_forecast_day=30)
                self.assertTrue(result.onset_day.eq(1).all())
                with self.assertRaisesRegex(ValueError, 'day 50'):
                    compute(model, 20., truth, max_forecast_day=30, dry_extent=21)

    def test_insufficient_ensemble_horizon_fails_before_loading_observations(self):
        arguments = {name: None for name, parameter in
                     inspect.signature(bins.multi_year_forecast_obs_pairs).parameters.items()
                     if parameter.default is inspect.Parameter.empty and name != 'kwargs'}
        arguments.update(years=(2003,), max_forecast_day=30, wet_spell=3,
                         dry_extent=21, dry_spell=7, day_bins=((1, 15), (16, 30)))
        forecast = xr.DataArray(np.ones(45), dims='step', coords={'step': np.arange(1, 46)})
        with patch.object(bins, 'load_thresh_file', return_value=20.), \
             patch.object(bins, 'get_forecast_probabilistic_twice_weekly', return_value=forecast), \
             patch.object(bins, 'load_imd_rainfall') as observations, \
             contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, 'day 50.*latest verifiable onset day.*25'):
                bins.multi_year_forecast_obs_pairs(**arguments)
        observations.assert_not_called()

    def rain(self, values, dates=None):
        return xr.DataArray(np.asarray(values, dtype=float)[:, None, None],
                            dims=('time', 'lat', 'lon'),
                            coords={'time': pd.date_range('2024-05-01', periods=len(values)) if dates is None else dates,
                                    'lat': [8.], 'lon': [39.]}, attrs={'units': 'mm'})

    def observed(self, values, **kwargs):
        return detect.detect_observed_onset(self.rain(values), 10., 2024,
                    start_date=(5, 1), end_date=(5, 31), extend_end_day=0, **kwargs).item()

    def test_forecast_wrapper_accepts_explicit_onset_parameters(self):
        self.assertTrue(detect.detect_forecast_onset(1, [5., 5., 5.], 10., wet_spell=3, dry_extent=0))
        self.assertEqual(detect.detect_forecast_onset([5., 5., 5.], thresh_val=10.,
                                                     wet_spell=3, dry_extent=0, max_forecast_day=1), 1)

    def test_dry_threshold_is_independent_of_wet_day_threshold(self):
        series = [5., 5., 5.] + [2.] * 7
        self.assertTrue(detect.detect_onset(1, series, 10., wet_init=3., dry_threshold=1.,
                                           dry_spell=2, dry_extent=10))
        self.assertEqual(pd.Timestamp(self.observed(series, wet_init=3., dry_threshold=1.,
                                      dry_spell=2, dry_extent=10)), pd.Timestamp('2024-05-01'))

    def test_missing_or_incomplete_dry_confirmation_cannot_confirm_onset(self):
        for values in ([5., 5., 5., np.nan, np.nan, np.nan], [5., 5., 5., 2., 2.]):
            with self.subTest(values=values):
                self.assertFalse(detect.detect_onset(1, values, 10., dry_spell=2, dry_extent=6))
                self.assertTrue(pd.isna(self.observed(values, dry_spell=2, dry_extent=6)))

    def test_observed_dry_window_matches_forecast_exclusive_endpoint(self):
        values = [5., 5., 5., 2., 2., 0., 0., 2., 2., 2.]
        self.assertTrue(detect.detect_onset(1, values, 10., dry_spell=2, dry_extent=6))
        self.assertEqual(pd.Timestamp(self.observed(values, dry_spell=2, dry_extent=6)),
                         pd.Timestamp('2024-05-01'))

    def test_no_dates_in_observation_search_returns_missing_onset(self):
        rain = self.rain([5., 5., 5.], pd.date_range('2023-05-01', periods=3))
        result = detect.detect_observed_onset(rain, 10., 2024, fallback_date=None)
        self.assertTrue(result.isnull().all())

    def test_non_daily_observation_samples_are_rejected(self):
        rain = self.rain([5., 5., 5.], pd.to_datetime(['2024-05-01', '2024-05-03', '2024-05-04']))
        with self.assertRaisesRegex(ValueError, 'daily'):
            detect.detect_observed_onset(rain, 10., 2024, dry_extent=0)

    def test_cross_year_observation_season_is_rejected_instead_of_scoring_no_onset(self):
        rain = self.rain(np.full(366, 10.), pd.date_range('2024-01-01', periods=366))
        with self.assertRaisesRegex(ValueError, 'Cross-year evaluation seasons'):
            detect.detect_observed_onset(rain, 20., 2024, start_date=(2024, 11, 1),
                                         end_date=(2025, 3, 31), dry_extent=0)

    def test_dashboard_preflight_rejects_unsupported_cross_year_season(self):
        from datetime import date

        from ui.utils.verification_settings import temporal_settings_errors

        settings = {'eval_start': date(2024, 11, 1), 'eval_end': date(2025, 3, 31),
                    'baseline_start_year': 2021, 'baseline_end_year': 2023,
                    'init_type': 'daily', 'initialization_days': []}
        errors = temporal_settings_errors(settings, range(2021, 2026))
        self.assertTrue(any('Cross-year evaluation seasons' in error for error in errors))

    def test_partial_spatial_aliases_normalize_independently(self):
        for names in (('lat', 'longitude'), ('latitude', 'lon')):
            ds = xr.Dataset({'rain': (names, [[1.]])}, coords={names[0]: [8.], names[1]: [39.]})
            actual = dim_fmt(ds)
            self.assertEqual(set(actual['rain'].dims), {'lat', 'lon'})

    def test_hourly_leads_normalize_to_daily_and_subdaily_or_gaps_reject(self):
        ds = xr.Dataset(coords={'init_time': pd.to_datetime(['2024-05-01']),
                                'step': xr.DataArray([0, 24, 48, 72], dims='step', attrs={'units': 'hours'})})
        np.testing.assert_equal(dim_fmt_model(ds).step.values, [0, 1, 2, 3])
        for steps in (pd.to_timedelta([0, 12, 24], unit='h'), [1, 3, 4]):
            with self.subTest(steps=steps), self.assertRaisesRegex(ValueError, 'daily|consecutive'):
                dim_fmt_model(xr.Dataset(coords={'step': steps}))

    def test_observation_units_use_metadata_and_literal_filename(self):
        with tempfile.TemporaryDirectory() as folder:
            rainfall = self.rain([.005, .005, .005]).rename('rain')
            rainfall.attrs['units'] = 'm'
            rainfall.to_dataset().to_netcdf(Path(folder) / 'observations.nc')
            with patch.object(inputs, 'region_select', side_effect=lambda ds, **kwargs: ds):
                actual = inputs.load_imd_rainfall(2024, obs_dir=folder,
                    obs_file_pattern='observations.nc', obs_var='rain', obs_unit_cvt=1, region='test')
            np.testing.assert_allclose(actual.values.ravel(), [5., 5., 5.])
            self.assertEqual(actual.attrs['units'], 'mm')

    def test_nonfinite_wet_days_cannot_confirm_observed_onset(self):
        self.assertTrue(pd.isna(self.observed([np.inf, 5., 5.], dry_extent=0)))

    def test_daily_observation_rate_is_integrated_without_catalog_scaling(self):
        with tempfile.TemporaryDirectory() as folder:
            rainfall = self.rain([5., 10., 15.]).rename('rain')
            rainfall.attrs['units'] = 'mm/day'
            path = Path(folder) / '2024.nc'
            rainfall.to_dataset().to_netcdf(path)
            with patch.object(inputs, 'region_select', side_effect=lambda ds, **kwargs: ds):
                actual = inputs.load_imd_rainfall(2024, obs_dir=folder,
                    obs_file_pattern='{}.nc', obs_var='rain', obs_unit_cvt=1000, region='test')
            np.testing.assert_allclose(actual.values.ravel(), [5., 10., 15.])
            self.assertEqual(actual.attrs['units'], 'mm')
            for dates in (pd.date_range('2024-05-01', periods=3, freq='12h'),
                          pd.to_datetime(['2024-05-01', '2024-05-03', '2024-05-04'])):
                rainfall.assign_coords(time=dates).to_dataset().to_netcdf(path)
                with patch.object(inputs, 'region_select', side_effect=lambda ds, **kwargs: ds):
                    with self.assertRaisesRegex(ValueError, 'daily'):
                        inputs.load_imd_rainfall(2024, obs_dir=folder,
                            obs_file_pattern='{}.nc', obs_var='rain', region='test')

    def test_precipitation_variable_with_time_in_name_remains_data(self):
        ds = self.rain([1., 2., 3.]).rename('rainfall_time_series').to_dataset()
        self.assertIn('rainfall_time_series', dim_fmt(ds).data_vars)

    def test_literal_forecast_filename_is_respected_in_year_discovery(self):
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / 'forecast.nc').touch()
            self.assertEqual(inputs.available_forecast_years([2024], model_dir=folder,
                                                            file_pattern='forecast.nc'), (2024,))

    def test_climatology_dataset_rejects_positional_grid_mismatch(self):
        from momp.stats import climatology
        first = xr.DataArray(np.array([['2020-05-01']], dtype='datetime64[ns]'),
                             dims=('lat', 'lon'), coords={'lat': [8.], 'lon': [39.]})
        second = first.assign_coords(lat=[9.])
        with patch.object(climatology, 'load_thresh_file', return_value=10.), \
             patch.object(climatology, 'load_imd_rainfall', side_effect=[self.rain([1.]), self.rain([1.])]), \
             patch.object(climatology, 'detect_observed_onset', side_effect=[first, second]):
            with self.assertRaisesRegex(ValueError, 'align|index|grid'):
                climatology.compute_climatological_onset_dataset(years_clim=[2020, 2021])

    def test_missing_deterministic_rainfall_is_excluded_not_scored_as_no_onset(self):
        model = xr.DataArray(np.array([5., np.nan, 5.]).reshape(1, 3, 1, 1),
                             dims=('init_time', 'step', 'lat', 'lon'),
                             coords={'init_time': pd.to_datetime(['2024-06-01']),
                                     'step': [1, 2, 3], 'lat': [8.], 'lon': [39.]})
        observed = xr.DataArray(np.array([['2024-06-10']], dtype='datetime64[ns]'),
                                dims=('lat', 'lon'), coords={'lat': [8.], 'lon': [39.]})
        result = detect.compute_onset_for_deterministic_model(model, 10., observed,
                    max_forecast_day=2, wet_spell=2, dry_extent=0)
        self.assertTrue(result.empty)
        self.assertIn('onset_date', result.columns)

    def test_invalid_climatology_year_is_not_silently_omitted(self):
        from momp.stats import climatology
        valid = xr.DataArray(np.array([['2021-05-01']], dtype='datetime64[ns]'),
                             dims=('lat', 'lon'), coords={'lat': [8.], 'lon': [39.]})
        for function in (climatology.compute_climatological_onset,
                         climatology.compute_climatological_onset_dataset):
            with self.subTest(function=function.__name__), \
                 patch.object(climatology, 'load_thresh_file', return_value=10.), \
                 patch.object(climatology, 'load_imd_rainfall', side_effect=[self.rain([1.]), self.rain([1.])]), \
                 patch.object(climatology, 'detect_observed_onset', side_effect=[ValueError('invalid daily input'), valid]):
                with self.assertRaisesRegex(ValueError, '2020.*invalid daily input'):
                    function(years_clim=[2020, 2021])

    def test_forecast_sources_close_when_coordinate_validation_fails(self):
        from unittest.mock import Mock
        for loader, formatter in ((inputs.get_forecast_deterministic_twice_weekly, 'dim_fmt_model'),
                                  (inputs.get_forecast_probabilistic_twice_weekly, 'dim_fmt_model_ensemble')):
            source = xr.Dataset()
            close = Mock()
            source.set_close(close)
            with self.subTest(loader=loader.__name__), \
                 patch.object(inputs.storage, 'exists', return_value=True), \
                 patch.object(inputs.storage, 'open_dataset', return_value=source), \
                 patch.object(inputs, 'get_initialization_dates', return_value=pd.date_range('2024-05-01', periods=1)), \
                 patch.object(inputs, formatter, side_effect=ValueError('invalid daily leads')):
                with self.assertRaisesRegex(ValueError, 'invalid daily leads'):
                    loader(2024, model_dir='/test', model_var='tp', file_pattern='{}.nc')
                close.assert_called_once()

    def test_forecast_horizon_counts_candidate_day_inside_confirmation_extent(self):
        model = xr.DataArray(np.ones(35), dims='step', coords={'step': np.arange(1, 36)})
        self.assertEqual(detect._fit_forecast_horizon(model, 15, 21, 3, ((1, 15),)),
                         (15, 21, 35))
        truncated = model.sel(step=slice(1, 30))
        with self.assertRaisesRegex(ValueError, 'requires day 35'):
            detect._fit_forecast_horizon(truncated, 15, 21, 3, ((1, 15),))
        self.assertEqual(detect._fit_forecast_horizon(model, 15, 0, 3, ((1, 15),)),
                         (15, 0, 17))

    def test_final_available_day_can_complete_rejecting_dry_spell(self):
        values = np.zeros(35)
        values[14:17] = 5.
        values[17:28] = 2.
        model = xr.DataArray(values.reshape(1, 35, 1, 1),
                             dims=('init_time', 'step', 'lat', 'lon'),
                             coords={'init_time': pd.to_datetime(['2024-05-01']),
                                     'step': np.arange(1, 36), 'lat': [8.], 'lon': [39.]})
        observed = xr.DataArray(np.array([['2024-06-10']], dtype='datetime64[ns]'),
                                dims=('lat', 'lon'), coords={'lat': [8.], 'lon': [39.]})
        result = detect.compute_onset_for_deterministic_model(model, 10., observed,
                    max_forecast_day=15, wet_spell=3, dry_spell=7, dry_extent=21,
                    day_bins=((1, 15),))
        self.assertTrue(result.onset_day.isna().all())
