"""Scientific comparisons must pair the same physical grid cells."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'apps/benchmarking/ROMP'))
from momp.stats import detect
from momp.io import input as inputs
from momp.utils import standard


class GridIntegrityTests(unittest.TestCase):
    def test_a_single_unrelated_variable_cannot_be_assumed_to_be_rainfall(self):
        dataset = xr.Dataset({'temperature': ('time', [280.])})
        with self.assertRaisesRegex(KeyError, 'rainfall variable'):
            inputs._resolve_data_variable(dataset, 'precip', 'temperature.nc')
        self.assertEqual(inputs._resolve_data_variable(
            xr.Dataset({'rain': ('time', [5.])}), 'precip', 'rain.nc'), 'rain')

    def model(self):
        return xr.DataArray(np.full((1, 3, 2, 1), 5.),
                            dims=('init_time', 'step', 'lat', 'lon'),
                            coords={'init_time': pd.to_datetime(['2024-05-01']),
                                    'step': [1, 2, 3], 'lat': [8., 9.], 'lon': [39.]})

    def truth(self):
        return xr.DataArray(np.array([['2024-05-20'], ['2024-05-10']], dtype='datetime64[ns]'),
                            dims=('lat', 'lon'), coords={'lat': [9., 8.], 'lon': [39.]})

    def test_coordinate_order_is_restored_without_interpolation(self):
        result = standard.align_spatial_grid(self.truth(), self.model(), label='observation')
        self.assertEqual(result.lat.values.tolist(), [8., 9.])
        self.assertEqual(pd.Timestamp(result.sel(lat=8., lon=39.).item()), pd.Timestamp('2024-05-10'))
        self.assertEqual(self.truth().lat.values.tolist(), [9., 8.])

    def test_different_duplicate_or_nonfinite_coordinates_are_rejected(self):
        for latitudes in ([8., 10.], [8., 8.], [8., np.nan]):
            with self.subTest(latitudes=latitudes), self.assertRaisesRegex(ValueError, 'grid|coordinate'):
                standard.align_spatial_grid(self.truth().assign_coords(lat=latitudes), self.model())

    def test_deterministic_and_member_onsets_use_truth_at_the_same_location(self):
        for ensemble in (False, True):
            model = self.model()
            if ensemble:
                model = model.expand_dims(member=[0, 1]).transpose('init_time', 'member', 'step', 'lat', 'lon')
            compute = detect.compute_onset_for_all_members if ensemble else detect.compute_onset_for_deterministic_model
            result = compute(model, 1., self.truth(), max_forecast_day=1, wet_spell=1, dry_extent=0)
            observed = result.loc[result.lat == 8., 'obs_onset_date']
            self.assertTrue(observed.eq('2024-05-10').all())
            with self.assertRaisesRegex(ValueError, 'grid|coordinate'):
                compute(model, 1., self.truth().assign_coords(lat=[9., 7.]),
                        max_forecast_day=1, wet_spell=1, dry_extent=0)

    def test_threshold_grid_cannot_silently_trim_the_observed_domain(self):
        rain = self.model().isel(init_time=0, drop=True).rename(step='time').assign_coords(
            time=pd.date_range('2024-05-01', periods=3))
        threshold = xr.DataArray([[1.]], dims=('lat', 'lon'), coords={'lat': [8.], 'lon': [39.]})
        with self.assertRaisesRegex(ValueError, 'grid|coordinate'):
            detect.detect_observed_onset(rain, threshold, 2024, start_date=(5, 1),
                                         end_date=(5, 3), extend_end_day=0, wet_spell=1, dry_extent=0)
