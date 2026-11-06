"""Country-box coordinates and exports must not inherit Ethiopian geography."""

import ast
import copy
from dataclasses import asdict, dataclass
from itertools import product
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import xarray as xr

MOMP = Path(__file__).resolve().parents[1] / "demos/demo4/apps/benchmarking/ROMP/momp"


def functions(relative, names, namespace):
    path = MOMP / relative
    nodes = [node for node in ast.parse(path.read_text()).body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)  # noqa: S102 -- Load isolated scientific functions.
    return namespace


class CountryBoxSpatialTests(unittest.TestCase):
    def region_functions(self):
        namespace = {"np": np, "xr": xr, "params": SimpleNamespace(region="Ethiopia", nc_mask="ethiopian-mask.nc"),
                     "_dedup_coords": lambda value: value,
                     "apply_nc_mask": MagicMock(side_effect=AssertionError("Unexpected Ethiopian mask"))}
        functions("params/region_def.py", {"domain"}, namespace)
        functions("utils/region.py", {"swap_lon_axis", "lon_swap", "lat_swap", "coords_fmt", "region_select"}, namespace)
        return namespace

    def test_peru_preserves_country_box_coordinates_without_an_ethiopian_mask(self):
        namespace = self.region_functions()
        data = xr.Dataset({"tp": (("lat", "lon"), np.arange(6).reshape(2, 3))},
                          coords={"lat": [-4., -12.], "lon": [-81., -75., -69.]})
        actual = namespace["region_select"](data, region="Peru", nc_mask=None)
        xr.testing.assert_equal(actual, data.sortby("lat"))
        namespace["apply_nc_mask"].assert_not_called()

    def test_country_box_extent_works_after_coordinate_alias_normalization(self):
        namespace = self.region_functions()
        data = xr.Dataset({"tp": (("latitude", "longitude"), [[1., 2.]])},
                          coords={"latitude": [-10.], "longitude": [-78., -77.75]})
        south, north, west, east, normalized = namespace["lon_swap"](data, region="Peru")
        self.assertEqual((south, north, west, east), (-10., -10., -78., -77.75))
        self.assertEqual(set(normalized.coords), {"lat", "lon"})

    def test_known_regions_keep_their_configured_bounds(self):
        namespace = self.region_functions()
        for name, expected in (("Ethiopia", (3, 15, 33, 48)),
                               ("Sub_Ethiopia", (7.5, 13.5, 37.5, 40)),
                               ("India", (6.46, 35.51, 68.11, 91.4))):
            self.assertEqual(namespace["domain"](name, coordinate_bounds=(-12, -4, -81, -69)), expected)
        with self.assertRaisesRegex(ValueError, "Peru.*coordinate_bounds"):
            namespace["domain"]("Peru")

    def test_explicit_antimeridian_bounds_preserve_wrapped_cells(self):
        namespace = self.region_functions()
        data = xr.Dataset({"tp": (("lat", "lon"), [[1., 2., 3., 4.]])},
                          coords={"lat": [0.], "lon": [-175., -170., 170., 175.]})
        actual = namespace["region_select"](
            data, region="country_box", coordinate_bounds=(-1., 1., 170., -170.))
        np.testing.assert_equal(actual.lon.values, [170., 175., 185., 190.])
        self.assertEqual(actual.sizes["lon"], 4)

    def test_climatology_export_clips_ethiopia_only(self):
        @dataclass
        class Case:
            model: str
            region: str
            case_name: str = "case"
            verification_window: tuple = (1, 15)
            model_dir: str = "forecast"
            model_var: str = "tp"
            file_pattern: str = "{}.nc"
            unit_cvt: float = 1.
            years: tuple = (2024,)
            years_clim: tuple = (2020, 2021)

        @dataclass
        class Setting:
            dir_out: str
            dir_fig: str
            region: str
            parallel: bool = False
            save_nc_spatial_far_mr_mae: bool = True
            save_nc_climatology: bool = True
            plot_spatial_far_mr_mae: bool = False
            plot_climatology_onset: bool = False
            plot_panel_heatmap_error: bool = False
            ref_model_dir: str = "observations"
            ref_model_var: str = "rainfall"
            ref_model_file_pattern: str = "{}.nc"
            ref_model_unit_cvt: float = 1.

        grid = xr.DataArray([[4.]], dims=("lat", "lon"), coords={"lat": [-10.], "lon": [-76.]})
        for country in ("Peru", "Ethiopia"):
            for parallel in (False, True):
                with self.subTest(country=country, parallel=parallel), tempfile.TemporaryDirectory() as directory:
                    clip = MagicMock(side_effect=lambda value, boundary: value + 100)
                    geography = ModuleType("ai_weather.spatial")
                    geography.clip_to_boundary = clip
                    weather = ModuleType("ai_weather")
                    weather.config = SimpleNamespace()
                    cfg = SimpleNamespace(model_list=("aifs_ens",), ref_model="climatology", region=country)
                    setting = Setting(directory, directory, country, parallel=parallel)
                    compute = MagicMock(return_value=({2024: "metrics"}, {2024: grid}))
                    save = MagicMock()
                    namespace = {"np": np, "xr": xr, "os": os, "Path": Path, "copy": copy, "product": product,
                                     "cfg": cfg, "setting": setting, "asdict": asdict, "Case": Case,
                                     "iter_list": lambda values: [values["model_list"]],
                                     "make_case": lambda cls, combi, values, selected_country=country: cls(combi[0], selected_country),
                                     "compute_metrics_multiple_years": compute,
                                     "parallel_metrics_multiple_years": compute,
                                     "create_spatial_far_mr_mae": lambda *_: {"mean_mae": grid},
                                     "save_metrics_to_netcdf": save,
                                     "climatology_boundary_path": lambda _: "ETH_Zones.geojson",
                                     "tuple_to_str_range": lambda values: f"{min(values)}-{max(values)}"}
                    functions("io/output.py", {"set_nested"}, namespace)
                    functions("app/spatial_far_mr_mae.py", {"get_canonical_model_key", "spatial_far_mr_mae_map"}, namespace)
                    with patch.dict(sys.modules, {"ai_weather": weather, "ai_weather.spatial": geography}):
                        namespace["spatial_far_mr_mae_map"](cfg=cfg, setting=setting)
                    self.assertEqual(compute.call_count, 2)
                    self.assertTrue(all(call.kwargs["region"] == country for call in compute.call_args_list))
                    saved = Path(directory) / "climatology/climatology_onset_doy_2020-2021.nc"
                    with xr.open_dataarray(saved) as onset:
                        np.testing.assert_allclose(onset, 104. if country == "Ethiopia" else 4.)
                    self.assertEqual(clip.call_count, 2 if country == "Ethiopia" else 0)
                    expected = grid + 100 if country == "Ethiopia" else grid
                    xr.testing.assert_equal(save.call_args.args[0]["mean_mae"], expected)
