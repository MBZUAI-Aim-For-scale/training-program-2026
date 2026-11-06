"""Daily forecast rates integrate only after the lead cadence is validated."""

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
    from ai_weather.precipitation import precipitation_to_mm
    from momp.io import input as inputs


class ForecastRateInputTests(unittest.TestCase):
    def load(self, *, ensemble=False, units="mm/day", steps=(0, 1, 2), lead_units="days", values=(99., 2., 4.),
             named_leads=True):
        dimensions = ("init_time", "step" if named_leads else "forecast_axis", "lat", "lon")
        coordinates = {"init_time": pd.to_datetime(["2024-05-01"]), "lat": [8.], "lon": [39.]}
        if named_leads:
            coordinates["step"] = xr.DataArray(list(steps), dims="step", attrs={"units": lead_units})
        rainfall = xr.DataArray(np.asarray(values).reshape(1, 3, 1, 1), dims=dimensions,
                                coords=coordinates, attrs={"units": units})
        if ensemble:
            rainfall = rainfall.expand_dims(member=[0, 1]).transpose("init_time", "member", dimensions[1], "lat", "lon")
        loader = (inputs.get_forecast_probabilistic_twice_weekly if ensemble
                  else inputs.get_forecast_deterministic_twice_weekly)
        with tempfile.TemporaryDirectory() as directory:
            rainfall.rename("tp").to_dataset().to_netcdf(Path(directory) / "2024.nc")
            with patch.object(inputs, "get_initialization_dates", return_value=coordinates["init_time"]), \
                 patch.object(inputs, "region_select", side_effect=lambda dataset, **kwargs: dataset):
                return loader(2024, model_dir=directory, model_var="tp", file_pattern="{}.nc",
                              unit_cvt=1000, region="test")

    def test_daily_rate_forecasts_keep_one_day_amounts_in_both_loaders(self):
        for ensemble in (False, True):
            for steps, lead_units in (((0, 1, 2), "days"), ((0, 24, 48), "hours")):
                with self.subTest(ensemble=ensemble, lead_units=lead_units):
                    result = self.load(ensemble=ensemble, steps=steps, lead_units=lead_units)
                    np.testing.assert_allclose(result.values.reshape(-1, 2), [[2., 4.]] * (2 if ensemble else 1))
                    self.assertEqual(result.attrs["units"], "mm")
                    np.testing.assert_equal(result.step.values, [1, 2])
                    np.testing.assert_allclose(precipitation_to_mm(result, 1000), result)

    def test_amounts_still_follow_units_rather_than_catalog_multiplier(self):
        for ensemble in (False, True):
            for units, values in (("mm", (99., 2., 4.)), ("m", (.099, .002, .004))):
                with self.subTest(ensemble=ensemble, units=units):
                    result = self.load(ensemble=ensemble, units=units, values=values)
                    np.testing.assert_allclose(result.values.reshape(-1, 2), [[2., 4.]] * (2 if ensemble else 1))

    def test_subdaily_or_gapped_rates_are_rejected_in_both_loaders(self):
        for ensemble in (False, True):
            for steps in ((0, 12, 24), (0, 24, 72)):
                with self.subTest(ensemble=ensemble, steps=steps), self.assertRaisesRegex(ValueError, "daily|aggregation"):
                    self.load(ensemble=ensemble, steps=steps, lead_units="hours")

    def test_unproven_forecast_duration_does_not_relabel_rates_as_amounts(self):
        for ensemble in (False, True):
            with self.subTest(ensemble=ensemble), self.assertRaisesRegex(ValueError, "sample duration"):
                self.load(ensemble=ensemble, named_leads=False)
        with self.assertRaisesRegex(ValueError, "sample duration"):
            precipitation_to_mm(xr.DataArray([2.], attrs={"units": "mm/day"}), 1000)

    def test_bare_lead_indices_survive_removing_day_zero_for_amounts(self):
        for ensemble in (False, True):
            with self.subTest(ensemble=ensemble):
                result = self.load(ensemble=ensemble, units="mm", named_leads=False)
                np.testing.assert_equal(result.step.values, [1, 2])
                np.testing.assert_allclose(result.values.reshape(-1, 2), [[2., 4.]] * (2 if ensemble else 1))
