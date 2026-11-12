"""Spatial climatology forecasts must not learn from their verification year."""

import ast
import contextlib
import io
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import xarray as xr

SOURCE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE_ROOT))
sys.path.insert(0, str(SOURCE_ROOT / 'apps/benchmarking/ROMP'))
from momp.stats import benchmark, climatology


class ClimatologyLeaveOneOutTests(unittest.TestCase):
    def climate(self):
        return xr.DataArray(
            np.array(['2019-06-02', '2020-06-20', '2021-06-30'],
                     dtype='datetime64[ns]')[:, None, None],
            dims=('year', 'lat', 'lon'),
            coords={'year': [2019, 2020, 2021], 'lat': [8.], 'lon': [39.]})

    def test_target_onset_cannot_change_its_reference_prediction(self):
        source = self.climate()
        original = source.copy(deep=True)
        first = climatology.climatological_onset_for_verification(source, 2019)
        xr.testing.assert_identical(source, original)
        source.loc[{'year': 2019}] = np.datetime64('2019-07-10', 'ns')
        changed = climatology.climatological_onset_for_verification(source, 2019)
        xr.testing.assert_identical(first, changed)
        self.assertEqual(first.attrs['reference_years'], (2020, 2021))
        self.assertEqual(first.attrs['reference_membership'], 'leave_one_year_out')
        self.assertEqual(first.item(), np.round(original.sel(year=[2020, 2021]).dt.dayofyear.mean()).item())

    def test_disjoint_target_keeps_the_full_historical_mean(self):
        source = self.climate()
        actual = climatology.climatological_onset_for_verification(source, 2024)
        expected = np.round(source.dt.dayofyear.mean('year'))
        np.testing.assert_equal(actual.values, expected.values)
        self.assertEqual(actual.attrs['reference_years'], (2019, 2020, 2021))
        self.assertEqual(actual.attrs['reference_membership'], 'held_out_verification_year')

    def test_only_target_year_is_rejected_instead_of_reintroduced(self):
        with self.assertRaisesRegex(ValueError, 'No baseline years remain.*2019'):
            climatology.climatological_onset_for_verification(self.climate().sel(year=[2019]), 2019)

    def test_missing_remaining_onsets_do_not_fall_back_to_the_target(self):
        source = self.climate()
        source.loc[{'year': [2020, 2021]}] = np.datetime64('NaT', 'ns')
        actual = climatology.climatological_onset_for_verification(source, 2019)
        self.assertTrue(actual.isnull().all())

    def test_export_retains_each_target_and_its_actual_reference_years(self):
        source = self.climate()
        fields = {year: climatology.climatological_onset_for_verification(source, year)
                  for year in (2019, 2021)}
        exported = climatology.climatology_reference_dataset(fields)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'references.nc'
            exported.to_netcdf(path)
            with xr.open_dataarray(path) as saved:
                self.assertEqual(saved.verification_year.values.tolist(), [2019, 2021])
                self.assertEqual(saved.reference_years.values.tolist(), ['2020,2021', '2019,2020'])
                self.assertEqual(saved.reference_membership.values.tolist(),
                                 ['leave_one_year_out', 'leave_one_year_out'])
                self.assertEqual(saved.attrs['reference_policy'], 'leave_one_year_out')
                self.assertNotIn('reference_years', saved.attrs)
                for year in (2019, 2021):
                    np.testing.assert_equal(saved.sel(verification_year=year).values, fields[year].values)

    def test_both_application_exports_preserve_all_held_out_fields(self):
        source = self.climate()
        fields = {year: climatology.climatological_onset_for_verification(source, year)
                  for year in (2019, 2021)}
        for filename in ('spatial_far_mr_mae.py', 'ens_spatial_far_mr_mae.py'):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as directory:
                path = SOURCE_ROOT / 'apps/benchmarking/ROMP/momp/app' / filename
                tree = ast.parse(path.read_text())
                block = next(node for node in ast.walk(tree) if isinstance(node, ast.If)
                             and node.body and isinstance(node.body[0], ast.Assign)
                             and any(isinstance(target, ast.Name) and target.id == 'fout'
                                     for target in node.body[0].targets))
                namespace = {'os': os, 'onset_da_dict': fields,
                             'case': SimpleNamespace(model='climatology', years_clim=(2019, 2020, 2021)),
                             'case_cfg': {'save_nc_climatology': True},
                             'case_cfg_ref': {'dir_out': directory, 'region': 'Peru',
                                              'years_clim': (2019, 2020, 2021),
                                              'save_nc_climatology': True},
                             'tuple_to_str_range': lambda years: f'{min(years)}-{max(years)}',
                             'clip_to_boundary': lambda data, boundary: data,
                             'climatology_boundary_path': lambda config: None,
                             'weather_config': None}
                exec(compile(ast.Module(body=[block], type_ignores=[]), str(path), 'exec'), namespace)  # noqa: S102
                with xr.open_dataarray(namespace['fout']) as saved:
                    self.assertEqual(saved.verification_year.values.tolist(), [2019, 2021])
                    self.assertEqual(saved.reference_years.values.tolist(), ['2020,2021', '2019,2020'])

    def test_reference_map_labels_the_descriptive_average_of_target_specific_fields(self):
        from momp.graphics import onset_map

        source = self.climate()
        fields = {year: climatology.climatological_onset_for_verification(source, year)
                  for year in (2019, 2021)}
        with patch.object(onset_map, 'get_region_outline', return_value=[]), \
             patch.object(onset_map, 'get_shp', return_value=None), \
             patch.object(onset_map, 'set_basemap'), \
             patch.object(onset_map, 'apply_jjas_mask', side_effect=lambda data, mask: data):
            figure, axis, _ = onset_map.plot_spatial_climatology_onset(
                fields, years_clim=(2019, 2020, 2021), shpfile_dir='', polygon=False,
                dir_fig=None, region='fixture', show_plot=False)
        try:
            self.assertIn('Descriptive average of held-out reference fields', axis.get_title())
            self.assertIn('2019-2021', axis.get_title())
            expected = np.mean([field.item() for field in fields.values()])
            np.testing.assert_equal(np.asarray(axis.collections[0].get_array()).ravel(), [expected])
        finally:
            onset_map.plt.close(figure)

    def test_both_spatial_paths_compute_baseline_once_and_hold_out_each_target(self):
        source = self.climate()
        for function in (benchmark.compute_metrics_multiple_years,
                         benchmark.ens_compute_metrics_multiple_years):
            predictions = []
            truths = []

            def as_forecast(prediction, year, dates, truth, *,
                            recorded_predictions=predictions, recorded_truths=truths, **kwargs):
                recorded_predictions.append(prediction.copy(deep=True))
                recorded_truths.append(truth.copy(deep=True))
                return climatology.compute_climatology_as_forecast(prediction, year, dates, truth, **kwargs)

            with self.subTest(function=function.__name__), contextlib.redirect_stdout(io.StringIO()), \
                 patch.object(benchmark, 'load_thresh_file', return_value=20.), \
                 patch.object(benchmark, 'compute_climatological_onset_dataset', return_value=source) as build, \
                 patch.object(benchmark, 'load_imd_rainfall', side_effect=lambda year, **kwargs: year), \
                 patch.object(benchmark, 'detect_observed_onset',
                              side_effect=lambda rain, threshold, year, **kwargs: source.sel(year=year)), \
                 patch.object(benchmark, 'get_initialization_dates',
                              side_effect=lambda year, **kwargs: pd.to_datetime([f'{year}-06-01'])), \
                 patch.object(benchmark, 'compute_climatology_as_forecast', side_effect=as_forecast):
                results, onsets = function(
                    model='climatology', years=[2019, 2021], years_clim=[2019, 2020, 2021],
                    max_forecast_day=45, verification_window=(1, 45))
            build.assert_called_once()
            self.assertEqual(set(results), {2019, 2021})
            for index, year in enumerate((2019, 2021)):
                expected = climatology.climatological_onset_for_verification(source, year)
                xr.testing.assert_identical(predictions[index], expected)
                xr.testing.assert_identical(onsets[year], expected)
                xr.testing.assert_identical(truths[index], source.sel(year=year))
