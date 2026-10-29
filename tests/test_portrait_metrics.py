"""Portrait reductions preserve valid values and honestly mark unavailable metrics."""

import ast
import unittest
import warnings
from pathlib import Path

import numpy as np
import xarray as xr


def load_functions(path, names, namespace):
    tree = ast.parse(Path(path).read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)  # noqa: S102


class PortraitMetricTests(unittest.TestCase):
    def setUp(self):
        self.ns = {"np": np}
        load_functions("apps/benchmarking/ROMP/momp/io/output.py", {"nested_dict_to_array"}, self.ns)
        load_functions("apps/benchmarking/ROMP/momp/graphics/panel_portrait_error.py",
                       {"_portrait_metric_data", "_symmetric_range", "_format_portrait_value"}, self.ns)

    def test_spatial_reduction_uses_finite_values_for_arrays_and_scalars(self):
        results = {"model": {(1, 15): {"mae": np.array([2, np.nan, 4, np.inf])},
                             (16, 30): {"mae": xr.DataArray([5, np.nan])}},
                   "climatology": {(1, 15): {"mae": 2.0}, (16, 30): {"mae": 3.0}}}
        values, rows, columns = self.ns["nested_dict_to_array"](results, "mae")
        np.testing.assert_allclose(values, [[3, 5], [2, 3]])
        self.assertEqual(rows, ["model", "climatology"])
        self.assertEqual(columns, [(1, 15), (16, 30)])

    def test_union_windows_and_missing_cells_are_retained(self):
        results = {"model": {(1, 15): {"mae": 2.0}},
                   "climatology": {(1, 15): {"mae": 1.0}, (16, 30): {"mae": 3.0}}}
        values, rows, columns = self.ns["nested_dict_to_array"](results, "mae")
        self.assertEqual(values.shape, (2, 2))
        self.assertEqual(rows, ["model", "climatology"])
        self.assertEqual(columns, [(1, 15), (16, 30)])
        self.assertTrue(np.isnan(values[0, 1]))
        anomaly, labels, _, has_baseline = self.ns["_portrait_metric_data"](results, "mae")
        self.assertTrue(has_baseline)
        self.assertEqual(labels, ["model"])
        self.assertEqual(anomaly[0, 0], 1)
        self.assertTrue(np.isnan(anomaly[0, 1]))

    def test_all_missing_metrics_stay_undefined_without_runtime_warnings(self):
        results = {"model": {(16, 30): {"mae": np.array([np.nan, np.inf])}}}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            values, _, _ = self.ns["nested_dict_to_array"](results, "mae")
        self.assertEqual(caught, [])
        self.assertTrue(np.isnan(values).all())
        self.assertEqual(self.ns["_symmetric_range"](values), (-1.0, 1.0))
        self.assertEqual(self.ns["_format_portrait_value"](np.nan), "N/A")

    def test_baseline_only_is_shown_as_raw_metrics_not_empty_anomaly(self):
        values, labels, _, has_baseline = self.ns["_portrait_metric_data"](
            {"climatology": {(1, 15): {"mae": 3.0}}}, "mae")
        np.testing.assert_allclose(values, [[3]])
        self.assertEqual(labels, ["climatology"])
        self.assertFalse(has_baseline)
