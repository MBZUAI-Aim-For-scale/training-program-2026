"""Rainfall conversions follow metadata and cannot be applied twice."""

import ast
import unittest
import warnings
from pathlib import Path

import numpy as np
import xarray as xr

from ai_weather.precipitation import precipitation_to_mm


class PrecipitationUnitTests(unittest.TestCase):
    def test_declared_units_override_catalog(self):
        for units, expected in (("m", 1000), (" Meters ", 1000), ("cm", 10),
                                ("mm", 1), ("millimetre", 1), ("MM", 1),
                                ("kg m-2", 1), ("kg m**-2", 1), ("kg/m²", 1)):
            with self.subTest(units=units):
                data = xr.DataArray([.0001, .025, np.nan], name="tp", attrs={"units": units})
                with warnings.catch_warnings(record=True) as captured:
                    warnings.simplefilter("always")
                    actual = precipitation_to_mm(data, 1000 if expected == 1 else 1)
                self.assertEqual(captured, [])
                np.testing.assert_allclose(actual, data * expected)
                self.assertEqual(actual.attrs["units"], "mm")
                np.testing.assert_allclose(data, [.0001, .025, np.nan])

    def test_repeated_conversion_does_not_inflate(self):
        data = xr.DataArray([.0001, .025], attrs={"units": "m"})
        converted = precipitation_to_mm(data, 1000)
        repeated = precipitation_to_mm(converted, 1000)
        np.testing.assert_allclose(repeated, [.1, 25])
        self.assertEqual(repeated.attrs["units"], "mm")

    def test_unlabeled_data_requires_explicit_valid_fallback(self):
        data = xr.DataArray([2.7, 267.7])
        for factor in (None, 0, -1, np.nan, np.inf):
            with self.subTest(factor=factor), self.assertRaises(ValueError):
                precipitation_to_mm(data, factor)
        with self.assertLogs("ai_weather.precipitation", level="INFO"), warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            actual = precipitation_to_mm(data, 1)
        self.assertEqual(captured, [])
        np.testing.assert_allclose(actual, data)
        with self.assertWarns(RuntimeWarning):
            scaled = precipitation_to_mm(data, 1000)
        np.testing.assert_allclose(scaled, data * 1000)

    def test_unknown_units_and_rates_are_not_relabelled_as_amounts(self):
        for units in ("mm/day", "m/s", "kg m-2 s-1", "unknown"):
            with self.subTest(units=units), self.assertRaises(ValueError):
                precipitation_to_mm(xr.DataArray([1.0], attrs={"units": units}), 1000)

    def test_ui_small_mm_values_stay_mm_and_conversion_is_idempotent(self):
        path = Path("ui/pages/short_medium_benchmarking.py")
        tree = ast.parse(path.read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                        and n.name == "normalize_aifs_precipitation_dataset")
        ns = {"xr": xr, "precipitation_to_mm": precipitation_to_mm}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), ns)  # noqa: S102
        normalize = ns["normalize_aifs_precipitation_dataset"]
        for units, values, expected in (("mm", [.1, .2], [.1, .2]),
                                        ("m", [.0001, .0002], [.1, .2])):
            with self.subTest(units=units):
                source = xr.Dataset({"tp": xr.DataArray(values, dims="time", attrs={"units": units})})
                once = normalize(source, 1000)
                twice = normalize(once, 1000)
                np.testing.assert_allclose(twice.RAINFALL, expected)
                np.testing.assert_allclose(source.tp, values)
