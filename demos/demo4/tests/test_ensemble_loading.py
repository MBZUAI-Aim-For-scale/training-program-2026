"""Annual ensembles must be filtered before rainfall is read into memory."""

import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import xarray as xr

SOURCE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE_ROOT))
sys.path.insert(0, str(SOURCE_ROOT / "apps/benchmarking/ROMP"))
with patch.dict(os.environ, {"DEMO_DATA_SOURCE": "local"}):
    from momp.io import input as inputs


class EnsembleLoadingTests(unittest.TestCase):
    def test_selected_rainfall_loads_without_materializing_annual_dataset(self):
        dates = pd.date_range("2024-05-01", periods=3)
        values = np.arange(3 * 4 * 3 * 2 * 2, dtype=float).reshape(3, 4, 3, 2, 2)
        rainfall = xr.DataArray(
            values, dims=("time", "number", "step", "latitude", "longitude"),
            coords={"time": dates, "number": [0, 1, 2, 3], "step": [0, 1, 2],
                    "latitude": [8., 9.], "longitude": [39., 40.]},
            attrs={"units": "mm"}, name="tp")
        source = rainfall.to_dataset()
        source["unused"] = rainfall * 2

        def select_region(dataset, **kwargs):
            self.assertEqual(dataset.sizes["init_time"], 1)
            self.assertEqual(dataset.sizes["member"], 2)
            return dataset.sel(lat=[8.], lon=[39.])

        dataset_load = xr.Dataset.load
        loaded_sizes = []

        def bounded_load(dataset, **kwargs):
            self.assertEqual(dict(dataset.sizes), {
                "init_time": 1, "member": 2, "step": 2, "lat": 1, "lon": 1})
            self.assertNotIn("unused", dataset.data_vars)
            loaded_sizes.append(dict(dataset.sizes))
            return dataset_load(dataset, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            source.to_netcdf(Path(directory) / "2024.nc")
            with patch.object(inputs, "get_initialization_dates", return_value=dates[1:2]), \
                 patch.object(inputs, "region_select", side_effect=select_region), \
                 patch.object(xr.Dataset, "load", new=bounded_load):
                result = inputs.get_forecast_probabilistic_twice_weekly(
                    2024, model_dir=directory, model_var="tp", file_pattern="{}.nc",
                    members=(1, 3), region="test", unit_cvt=1000)

        # Results remain available after the source file has been closed/deleted.
        self.assertTrue(loaded_sizes)
        self.assertIsInstance(result.data, np.ndarray)
        np.testing.assert_array_equal(result.values, values[1:2, [1, 3], 1:, :1, :1])
        np.testing.assert_array_equal(result.init_time.values, dates[1:2].values)
        np.testing.assert_array_equal(result.member.values, [1, 3])
        self.assertEqual(result.attrs["units"], "mm")
