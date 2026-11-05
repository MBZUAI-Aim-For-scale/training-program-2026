"""Analytic regression cases for ROMP's exported figures and diagnostics."""

import ast
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path('apps/benchmarking/ROMP/momp')


def function_namespace(relative, name, **namespace):
    path = ROOT / relative
    function = next(node for node in ast.parse(path.read_text()).body
                    if isinstance(node, ast.FunctionDef) and node.name == name)
    function.decorator_list = []
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


class ReliabilityAuditTests(unittest.TestCase):
    def plot(self, frame):
        ns = function_namespace('io/dict.py', 'extract_pd_bins', pd=pd)
        ns.update(np=np, plt=plt, os=__import__('os'), tuple_to_str=lambda value: '-'.join(map(str, value)))
        ns = function_namespace('graphics/reliability.py', 'plot_reliability_diagram', **ns)
        return ns['plot_reliability_diagram'](
            frame, model='test', verification_window=(1, 15), day_bins=((1, 15),),
            save_fig=False, dir_fig='', show_plot=False)

    def frame(self, probabilities, outcomes):
        return pd.DataFrame({'bin_start': [1] * len(probabilities), 'bin_end': [15] * len(probabilities),
                             'predicted_prob': probabilities, 'observed_onset': outcomes})

    def test_missing_pairs_are_excluded_from_counts_and_frequency(self):
        fig, _, result = self.plot(self.frame([.25, .25, .75, np.nan], [1, np.nan, 0, 1]))
        try:
            populated = result.loc[result.N_Forecasts > 0]
            self.assertEqual(result.N_Forecasts.sum(), 2)
            self.assertEqual(list(populated.N_Forecasts), [1, 1])
            np.testing.assert_allclose(populated.Frequency, [.5, .5])
            np.testing.assert_allclose(populated.Observed_Frequency, [1, 0])
        finally:
            plt.close(fig)

    def test_invalid_probabilities_and_nonbinary_outcomes_are_rejected(self):
        for probabilities, outcomes in (([1.2], [1]), ([np.inf], [0]), ([.5], [2])):
            with self.subTest(probabilities=probabilities, outcomes=outcomes):
                with self.assertRaisesRegex(ValueError, 'probabilit|binary'):
                    self.plot(self.frame(probabilities, outcomes))

    def test_all_missing_pairs_fail_before_rendering(self):
        with self.assertRaisesRegex(ValueError, 'valid|pairs'):
            self.plot(self.frame([np.nan, .5], [1, np.nan]))


class OnsetSeriesAuditTests(unittest.TestCase):
    def test_missing_climatological_onset_remains_missing(self):
        time = pd.date_range('2024-05-01', periods=3)
        rain = xr.DataArray(np.ones((3, 1, 1)), dims=('time', 'lat', 'lon'),
                            coords={'time': time, 'lat': [8.], 'lon': [39.]})
        clim = xr.DataArray([1., 2., 3.], dims='doy', coords={'doy': time.dayofyear})
        draw = MagicMock()
        ns = function_namespace(
            'app/onset_time_series.py', 'obs_onset_analysis',
            restore_args=lambda function, kwargs, local: {**kwargs, **{
                key: local[key] for key in ('year', 'lat_select', 'lon_select', 'incl_clim')}},
            years_tuple_clim=lambda start, end: tuple(range(start, end + 1)),
            load_thresh_file=lambda **kwargs: 20,
            load_imd_rainfall=lambda *args, **kwargs: rain,
            detect_observed_onset=lambda *args, **kwargs: rain.isel(time=0),
            compute_climatological_onset=lambda **kwargs: (xr.DataArray(np.nan), clim),
            plot_rainfall_timeseries_with_onset_and_wetspell=draw,
            np=np, pd=pd, os=__import__('os'))
        ns['obs_onset_analysis'](2024, lat_select=8., lon_select=39., incl_clim=True,
                                 start_year_clim=2000, end_year_clim=2020,
                                 start_date=(2024, 5, 1), end_date=(2024, 5, 3), dir_fig='/tmp')
        self.assertIsNone(draw.call_args.kwargs['onset_clim'])
        np.testing.assert_allclose(draw.call_args.kwargs['pr_clim'], [1., 2., 3.])


class RainfallTraceAuditTests(unittest.TestCase):
    def render(self, data=None):
        import plotly.graph_objects as go
        ns = function_namespace('graphics/rainfall_time_series.py', 'rainfall_time_series',
                                go=go, pd=pd, xr=xr, np=np)
        return ns['rainfall_time_series'](data)

    def test_missing_input_never_fabricates_observations_or_changes_random_state(self):
        expected = np.random.get_state()
        figure = self.render(None)
        actual = np.random.get_state()
        self.assertEqual(len(figure.data), 0)
        self.assertTrue(any('No rainfall data' in annotation.text for annotation in figure.layout.annotations))
        self.assertEqual(expected[0], actual[0])
        np.testing.assert_array_equal(expected[1], actual[1])
        self.assertEqual(expected[2:], actual[2:])

    def test_dataset_uses_actual_rainfall_values(self):
        times = pd.date_range('2024-05-01', periods=2)
        data = xr.Dataset({'Observed': ('time', [1., 3.]),
                           'Model Forecast': ('time', [2., 4.])}, coords={'time': times})
        figure = self.render(data)
        np.testing.assert_allclose(figure.data[0].y, [1., 3.])
        np.testing.assert_allclose(figure.data[1].y, [2., 4.])
        self.assertEqual(figure.data[1].name, 'Model Forecast')

    def test_ambiguous_inputs_fail_explicitly(self):
        with self.assertRaisesRegex((TypeError, ValueError), 'DataFrame|Dataset|rainfall'):
            self.render(np.array([1., 2., 3.]))


class MissingReferencePlotTests(unittest.TestCase):
    def test_daily_series_plots_without_an_invented_reference_onset(self):
        import matplotlib.dates as mdates
        times = pd.date_range('2024-05-01', periods=3)
        rain = xr.DataArray(np.ones((3, 1, 1)), dims=('time', 'lat', 'lon'),
                            coords={'time': times, 'lat': [8.], 'lon': [39.]})
        onset = xr.DataArray([[np.datetime64('NaT', 'ns')]], dims=('lat', 'lon'),
                             coords={'lat': [8.], 'lon': [39.]})
        ns = function_namespace('graphics/rainfall_time_series.py',
                                'plot_rainfall_timeseries_with_onset_and_wetspell',
                                plt=plt, pd=pd, mdates=mdates)
        ns['plot_rainfall_timeseries_with_onset_and_wetspell'](
            rain, onset, None, 8., 39., year_select=2024,
            incl_clim=True, pr_clim=rain.sel(lat=8., lon=39.), onset_clim=None)
        figure = plt.gcf()
        try:
            self.assertEqual([line.get_label() for line in figure.axes[0].lines],
                             ['Daily rainfall', 'Climatology'])
        finally:
            plt.close(figure)
