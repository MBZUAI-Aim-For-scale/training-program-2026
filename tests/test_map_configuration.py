"""Map resources follow canonical config fields, including older loaded modules."""

import ast
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import xarray as xr

from ai_weather.spatial import apply_jjas_mask, climatology_boundary_path


class MapConfigurationTests(unittest.TestCase):
    def test_dashboard_mask_defaults_follow_config_instead_of_disabling_it(self):
        path = Path("ui/pages/short_medium_benchmarking.py")
        tree = ast.parse(path.read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "render_shared_settings")
        for mask in (None, "/configured/external/jjas_100mm_rainfall_mask_0p25.nc", "/custom/jjas.nc"):
            with self.subTest(mask=mask):
                st, column = MagicMock(), MagicMock()
                st.columns.side_effect = lambda count, target=column: [target] * count
                column.number_input.side_effect = lambda label, *, value, **_: value
                column.selectbox.side_effect = lambda label, options, index=0, **_: options[index]
                column.text_input.side_effect = lambda label, *, value, **_: value
                ns = {"st": st, "section_header": MagicMock(), "config": SimpleNamespace(nc_mask=mask), "Dict": dict, "Any": object,
                      "EXTERNAL_DATA_DIR": Path("/configured/external"), "ONSET_REFERENCE_DIR": Path("/reference")}
                exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), ns)  # noqa: S102
                self.assertEqual(ns["render_shared_settings"]()["area_mask_file"], mask or "")

    def test_boundary_is_derived_from_shpfile_dir_without_new_attribute(self):
        config = SimpleNamespace(shpfile_dir="/configured/shapefile/")
        self.assertEqual(climatology_boundary_path(config), "/configured/shapefile/ETH_Zones.geojson")
        config.CLIMATOLOGY_SHAPEFILE = "/stale/incorrect.geojson"
        self.assertEqual(climatology_boundary_path(config), "/configured/shapefile/ETH_Zones.geojson")
        config.shpfile_dir = "gs://bucket/shapefile"
        self.assertEqual(climatology_boundary_path(config), "gs://bucket/shapefile/ETH_Zones.geojson")

    def test_jjas_keep_flags_and_coordinates_follow_existing_mask_convention(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "jjas.nc"
            mask = xr.DataArray([[False, True], [True, False]], dims=("latitude", "longitude"),
                                coords={"latitude": [2, 1], "longitude": [30, 31]}, name="low_rainfall_mask")
            mask.to_dataset().to_netcdf(path)
            data = xr.DataArray([[1., 2.], [3., 4.]], dims=("lat", "lon"),
                                coords={"lat": [1, 2], "lon": [30, 31]})
            actual = apply_jjas_mask(data, path)
            np.testing.assert_allclose(actual, [[1, np.nan], [np.nan, 4]])
            np.testing.assert_allclose(data, [[1, 2], [3, 4]])
            self.assertIs(apply_jjas_mask(data, None), data)

    def test_standard_keep_mask_does_not_extrapolate_outside_coverage(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "keep.nc"
            xr.DataArray([[0, 1], [1, 0]], dims=("lat", "lon"),
                         coords={"lat": [1, 2], "lon": [30, 31]}, name="wet_area").to_dataset().to_netcdf(path)
            data = xr.DataArray(np.ones((3, 2)), dims=("lat", "lon"),
                                coords={"lat": [1, 2, 3], "lon": [30, 31]})
            actual = apply_jjas_mask(data, path)
            np.testing.assert_allclose(actual, [[np.nan, 1], [1, np.nan], [np.nan, np.nan]])

    def test_pipeline_mask_honors_keep_flags_and_explicit_override(self):
        path = Path("apps/benchmarking/ROMP/momp/utils/land_mask.py")
        tree = ast.parse(path.read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "apply_nc_mask")
        ns = {"np": np, "xr": xr, "Path": Path, "_sanitize_dataset_coords": lambda value: value,
              "_dedup_coords": lambda value: value,
              "_clean_coord_array": lambda value: np.asarray(value.values, dtype=float)}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), ns)  # noqa: S102
        mask = xr.DataArray([[False, True], [True, False]], dims=("lat", "lon"),
                            coords={"lat": [1, 2], "lon": [30, 31]}, name="low_rainfall_mask")
        data = xr.Dataset({"rainfall": xr.ones_like(mask, dtype=float)})
        np.testing.assert_allclose(ns["apply_nc_mask"](data, mask).rainfall, [[np.nan, 1], [1, np.nan]])
        np.testing.assert_allclose(ns["apply_nc_mask"](data, mask, keep_value=0).rainfall, [[1, np.nan], [np.nan, 1]])
