"""Preflight reads source forecast coordinates and reports metadata failures."""

import ast
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import xarray as xr


class ForecastTemporalCoverageTests(unittest.TestCase):
    def inspector(self, folder):
        source = Path("ui/pages/short_medium_benchmarking.py")
        function = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef)
                        and n.name == "inspect_model_temporal_coverage")
        function.decorator_list = []
        ns = {"np": np, "pd": pd, "config": SimpleNamespace(
            join=lambda root, name: str(Path(root) / name), exists=lambda path: Path(path).exists(),
            open_dataset=xr.open_dataset),
            "CONFIG_BENCHMARK_MODEL_CATALOG": {"test": {"dir": folder, "file_pattern": "{}.nc"}}}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), ns)  # noqa: S102
        return ns["inspect_model_temporal_coverage"]

    def test_actual_coordinates_and_hourly_lead_units_are_read(self):
        with tempfile.TemporaryDirectory() as folder:
            for year, last_hour in ((2020, 720), (2021, 360)):
                ds = xr.Dataset(coords={"time": pd.to_datetime([f"{year}-05-02", f"{year}-05-06"]),
                                        "step": xr.DataArray([0, 24, last_hour], dims="step", attrs={"units": "hours"})})
                ds.to_netcdf(Path(folder) / f"{year}.nc")
            coverage = self.inspector(folder)("test", 2019, 2021)
            self.assertEqual(coverage["failures"], [])
            self.assertEqual(coverage["horizons"], {2020: 30, 2021: 15})
            self.assertEqual(coverage["dates"], [(2020, 5, 2), (2020, 5, 6), (2021, 5, 2), (2021, 5, 6)])

    def test_unreadable_data_is_reported_before_worker_launch(self):
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "2020.nc").write_text("not a NetCDF file")
            coverage = self.inspector(folder)("test", 2020, 2020)
            self.assertEqual(coverage["dates"], [])
            self.assertEqual(len(coverage["failures"]), 1)
            self.assertTrue(coverage["failures"][0].startswith("2020:"))
