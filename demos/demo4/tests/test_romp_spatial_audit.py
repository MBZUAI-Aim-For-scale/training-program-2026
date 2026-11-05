"""Analytic spatial and storage regressions from the ROMP audit."""

import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import xarray as xr

ROMP = Path(__file__).resolve().parents[1] / 'apps/benchmarking/ROMP'
if str(ROMP) not in sys.path:
    sys.path.insert(0, str(ROMP))
from momp.io.dict import extract_pd_bins
from momp.utils import land_mask, region, storage


class SpatialAuditTests(unittest.TestCase):
    def grid(self):
        return xr.DataArray(np.ones((2, 2)), dims=('lat', 'lon'),
                            coords={'lat': [0., 1.], 'lon': [0., 1.]}, name='rain')

    def test_mask_supports_dataarray_without_extrapolating_outside_mask(self):
        data = self.grid().reindex(lat=[-1., 0., 1., 2.], fill_value=1.)
        mask = self.grid().copy(data=[[1, 0], [1, 0]])
        result = land_mask.apply_nc_mask(data, mask)
        self.assertTrue(result.sel(lat=-1).isnull().all())
        self.assertTrue(result.sel(lat=2).isnull().all())
        self.assertTrue(result.sel(lon=1).isnull().all())
        self.assertEqual(float(result.sel(lat=0, lon=0)), 1.)

    def test_descending_longitude_is_sliced_in_the_requested_domain(self):
        data = self.grid().to_dataset().sortby('lon', ascending=False)
        with patch.object(region, 'domain', return_value=(0, 1, 0, 1)):
            result = region.region_select(data, region='test', nc_mask='')
        self.assertEqual(result.sizes, {'lat': 2, 'lon': 2})
        np.testing.assert_array_equal(result.lon, [0, 1])

    def test_requested_failed_mask_cannot_silently_evaluate_unmasked_data(self):
        with patch.object(region, 'domain', return_value=(0, 1, 0, 1)), \
                patch.object(region, 'apply_nc_mask', side_effect=OSError('unreadable mask')):
            with self.assertRaisesRegex(ValueError, 'mask'):
                region.region_select(self.grid().to_dataset(), region='test', nc_mask='missing.nc')

    def test_bin_selection_preserves_nondefault_index(self):
        frame = pd.DataFrame({'bin_start': [1, 6], 'bin_end': [5, 10], 'value': [7, 8]},
                             index=[10, 20])
        selected = extract_pd_bins(frame, ((6, 10),), method='series')
        self.assertEqual(selected.index.tolist(), [20])
        self.assertEqual(selected.value.tolist(), [8])

    def test_duplicate_requested_bins_do_not_duplicate_forecast_samples(self):
        frame = pd.DataFrame({'bin_start': [1], 'bin_end': [5], 'value': [7]})
        self.assertEqual(len(extract_pd_bins(frame, ((1, 5), (1, 5)))), 1)


class StorageAuditTests(unittest.TestCase):
    def test_failed_remote_open_closes_its_stream(self):
        stream = io.BytesIO(b'not netcdf')
        class FS:
            def open(self, *args):
                return stream
        with patch.object(storage, 'gcs_fs', return_value=FS()), \
                patch.object(storage.xr, 'open_dataset', side_effect=ValueError('invalid file')):
            with self.assertRaisesRegex(RuntimeError, 'Failed to open'):
                storage.open_dataset('gs://bucket/bad.nc')
        self.assertTrue(stream.closed)

    def test_explicit_backend_is_not_overridden_on_failure(self):
        with patch.object(storage.xr, 'open_dataset', side_effect=FileNotFoundError('missing')) as opened:
            with self.assertRaises(FileNotFoundError):
                storage.open_dataset('/missing.nc', engine='scipy')
        self.assertEqual(opened.call_count, 1)

    def test_successful_remote_open_owns_stream_until_dataset_close(self):
        stream = io.BytesIO(b'input bytes')
        dataset = xr.Dataset({'rain': ('x', [1.])})
        class FS:
            def open(self, *args):
                return stream
        with patch.object(storage, 'gcs_fs', return_value=FS()), \
                patch.object(storage.xr, 'open_dataset', return_value=dataset):
            opened = storage.open_dataset('gs://bucket/rain.nc')
            self.assertFalse(stream.closed)
            opened.close()
            self.assertTrue(stream.closed)
