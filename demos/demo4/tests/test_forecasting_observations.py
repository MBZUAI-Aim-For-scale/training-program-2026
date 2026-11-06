"""Unavailable CHIRPS observations must not crash the forecast page."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import xarray as xr
from streamlit.testing.v1 import AppTest

from apps.forecasting import compare, obs, onset, onset_view
from apps.forecasting.regions import Box


class ForecastObservationTests(unittest.TestCase):
    def setUp(self):
        self.box = Box(9, 11, 37, 39)
        self.rain = xr.DataArray(np.ones((2, 2, 2)), dims=("time", "lat", "lon"),
                                 coords={"time": pd.date_range("2025-07-15", periods=2),
                                         "lat": [9., 11.], "lon": [37., 39.]})

    def test_missing_dates_are_unavailable_observations_before_any_file_access(self):
        for start, end in ((None, None), (pd.NaT, pd.Timestamp("2025-08-03")),
                           (pd.Timestamp("2025-05-01"), pd.NaT)):
            for region in (None, "Ethiopia"):
                with self.subTest(start=start, end=end, region=region), \
                     patch.object(obs, "_country_file") as country, patch.object(obs, "_tif_days") as days:
                    with self.assertRaises(FileNotFoundError):
                        obs.daily(self.box, start, end, region)
                    country.assert_not_called()
                    days.assert_not_called()

    def test_empty_selected_dates_are_unavailable_not_an_empty_template(self):
        # A country file that has other days only (here July), and daily files for July only
        dataset = self.rain.to_dataset(name="precip")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "CHIRPSv3_Ethiopia_2025_box025.nc"
            dataset.to_netcdf(path)
            with patch.object(obs, "_country_file", return_value=path), \
                 patch.object(obs, "_tif_days", return_value=list(pd.date_range("2025-07-15", periods=2))), \
                 patch.object(obs, "_tif") as tif, self.assertRaises(FileNotFoundError):
                obs.daily(self.box, "2025-08-01", "2025-08-03", "Ethiopia")
            tif.assert_not_called()
            with patch.object(obs, "_tif_days", return_value=list(pd.date_range("2025-07-15", periods=2))), \
                 self.assertRaises(FileNotFoundError):
                obs.daily(self.box, "2025-08-01", "2025-08-03")

    def test_last_day_skips_empty_or_invalid_calendar_and_uses_previous_file(self):
        for invalid in ([], [pd.NaT]):
            with self.subTest(invalid=invalid):
                bad = xr.Dataset(coords={"time": pd.DatetimeIndex(invalid)})
                good = xr.Dataset(coords={"time": pd.to_datetime(["2024-12-30", "2024-12-31"])})
                with patch.object(obs, "_country_file", return_value="unused.nc"), \
                     patch.object(obs, "_tif_days", return_value=[]), \
                     patch.object(obs.xr, "open_dataset", side_effect=[bad, good]):
                    self.assertEqual(obs.last_day("Ethiopia"), pd.Timestamp("2024-12-31"))

    def test_missing_observations_preserve_original_date_range_error(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "forecast.nc"
            self.rain.to_dataset(name="tp").to_netcdf(path)
            for last in (None, pd.NaT):
                with self.subTest(last=last), \
                     patch.object(onset_view, "crop", return_value=self.rain.to_dataset(name="tp")), \
                     patch.object(onset_view.onset, "forecast_daily", return_value=self.rain), \
                     patch.object(obs, "last_day", return_value=last), \
                     patch.object(obs, "_country_file", side_effect=OSError("offline")), \
                     patch.object(obs, "_tif_days", return_value=[]), \
                     self.assertRaisesRegex(FileNotFoundError, "2025-05-01"):
                    onset_view.compute.__wrapped__(str(path), 0, (9, 11, 37, 39),
                                                   pd.Timestamp("2025-07-15"), 2,
                                                   onset.DEFINITIONS["icpac"],
                                                   pd.Timestamp("2025-05-01"))

    def test_cached_view_handles_nat_and_keeps_the_rainfall_chart(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "forecast.nc"
            self.rain.to_dataset(name="tp").to_netcdf(path)
            source = '''
from types import SimpleNamespace
from unittest.mock import patch
import pandas as pd
import xarray as xr
from apps.forecasting import catalog, compare, obs, onset, onset_view
from apps.forecasting.regions import Box
path = PATH_VALUE
with xr.open_dataset(path) as opened:
    dataset = opened.load()
model = SimpleNamespace(key='observation_test', name='Test model')
with patch.object(obs, 'last_day', return_value=pd.NaT), patch.object(obs, '_country_file', side_effect=OSError('offline')), patch.object(obs, '_tif_days', return_value=[]), patch.object(onset, 'forecast_daily', return_value=dataset.tp), patch.object(onset_view, 'crop', return_value=dataset), patch.object(compare, 'crop', return_value=dataset), patch.object(onset_view, 'usable', return_value=None), patch.dict(catalog.BY_KEY, {'observation_test': model}):
    onset_view.view(path, dataset, model, pd.Timestamp('2025-07-15'), Box(9, 11, 37, 39), 'Ethiopia', 2, onset.DEFINITIONS['icpac'], pd.Timestamp('2025-05-03'))
    compare.chart({'observation_test': path}, ['observation_test'], pd.Timestamp('2025-07-15'), 2, Box(9, 11, 37, 39), 'Ethiopia', 'tp')
'''.replace('PATH_VALUE', repr(str(path)))
            app = AppTest.from_string(source, default_timeout=30).run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.error), 1)
            self.assertIn("CHIRPS", app.error[0].value)
            self.assertIn("2025-05-03", app.error[0].value)
            self.assertEqual(len(app.get("plotly_chart")), 1)
            app.run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.get("plotly_chart")), 1)

    def test_available_requested_observations_work_without_latest_year_files(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "forecast.nc"
            self.rain.to_dataset(name="tp").to_netcdf(path)
            with patch.object(onset_view, "crop", return_value=self.rain.to_dataset(name="tp")), \
                 patch.object(onset, "forecast_daily", return_value=self.rain), \
                 patch.object(onset, "to_grid", return_value=self.rain), \
                 patch.object(obs, "last_day", return_value=None), \
                 patch.object(obs, "daily", return_value=self.rain) as daily:
                result = onset_view.compute.__wrapped__(str(path), 0, (9, 11, 37, 39),
                                                       pd.Timestamp("2025-07-15"), 2,
                                                       onset.DEFINITIONS["icpac"],
                                                       pd.Timestamp("2025-05-01"))
            self.assertEqual(result["obs_last"], pd.Timestamp("2025-07-16"))
            daily.assert_called_once()

    def test_comparison_uses_forecast_grid_when_observations_are_unavailable(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "forecast.nc"
            self.rain.to_dataset(name="tp").to_netcdf(path)
            with patch.object(compare, "crop", return_value=self.rain.to_dataset(name="tp")), \
                 patch.object(onset, "forecast_daily", return_value=self.rain), \
                 patch.object(obs, "last_day", return_value=None), \
                 patch.object(obs, "_country_file", side_effect=OSError("offline")), \
                 patch.object(obs, "_tif_days", return_value=[]):
                series = compare._area_series.__wrapped__(str(path), 0, (9, 11, 37, 39),
                                                         pd.Timestamp("2025-07-15"), 2, "tp")
            self.assertEqual(series.sizes["time"], 2)
            np.testing.assert_allclose(series, 1.)


if __name__ == "__main__":
    unittest.main()
