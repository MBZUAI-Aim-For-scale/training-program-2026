"""Climatology onset plots frame the selected boundary, not the source grid."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import cartopy.crs as ccrs
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib import colors as mcolors

from ai_weather.map_extent import geographic_extent
from ai_weather.spatial import apply_jjas_mask, climatology_boundary_path


class ClimatologyFramingTests(unittest.TestCase):
    def setUp(self):
        path = Path("apps/benchmarking/ROMP/momp/graphics/onset_map.py")
        tree = ast.parse(path.read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                        and n.name == "plot_spatial_climatology_onset")
        self.outlines = MagicMock(return_value=[([35, 42, 42, 35, 35], [5, 5, 14, 14, 5])])
        self.geometry = MagicMock(return_value=None)
        self.config = SimpleNamespace(CLIMATOLOGY_SHAPEFILE="ETH_Zones.geojson", exists=lambda _: True)

        def basemap(ax, *args, map_extent, **kwargs):
            ax.set_extent(map_extent, crs=ccrs.PlateCarree())

        self.basemap = MagicMock(side_effect=basemap)
        ns = {"plt": plt, "np": np, "ccrs": ccrs, "mcolors": mcolors, "weather_config": self.config,
              "get_region_outline": self.outlines, "get_shp": self.geometry,
              "clip_to_boundary": lambda data, _: data, "clip_map_artist": lambda *_: None,
              "apply_jjas_mask": apply_jjas_mask, "climatology_boundary_path": climatology_boundary_path,
              "geographic_extent": geographic_extent, "set_basemap": self.basemap,
              "doy_to_mmm_dd": str}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), ns)  # noqa: S102
        self.plot = ns["plot_spatial_climatology_onset"]
        self.data = xr.DataArray(np.full((49, 61), 180.0), dims=("lat", "lon"),
                                 coords={"lat": np.linspace(3, 15, 49), "lon": np.linspace(33, 48, 61)})
        self.addCleanup(plt.close, "all")

    def render(self, **kwargs):
        return self.plot({"onset": self.data}, years_clim=(2000, 2024),
                         shpfile_dir="region.shp", polygon=False, dir_fig=None,
                         region=kwargs.pop("region", "Selected region"), show_plot=False, **kwargs)

    def test_region_and_explicit_extents_survive_plotting(self):
        _, ax, _ = self.render()
        expected = geographic_extent(self.data.lon, self.data.lat, region_bounds=(35, 42, 5, 14))
        np.testing.assert_allclose(ax.get_extent(), expected)
        self.outlines.assert_called_with("region.shp", region="Selected region")
        self.assertEqual(float(self.data.min()), 180.0)
        explicit = (34, 43, 4, 15)
        _, ax, _ = self.render(map_extent=explicit)
        np.testing.assert_allclose(ax.get_extent(), explicit)

    def test_data_extent_is_fallback_when_boundary_is_unavailable(self):
        self.outlines.return_value = []
        self.geometry.side_effect = OSError("No local boundary available")
        _, ax, _ = self.render()
        expected = geographic_extent(self.data.lon, self.data.lat, self.data.values)
        np.testing.assert_allclose(ax.get_extent(), expected)

    def test_ethiopia_uses_zones_for_both_extent_and_overlay(self):
        _, ax, _ = self.render(region="Ethiopia")
        self.outlines.assert_called_with("ETH_Zones.geojson", region="Ethiopia")
        self.assertEqual(self.basemap.call_args.args[2], "ETH_Zones.geojson")
        expected = geographic_extent(self.data.lon, self.data.lat, region_bounds=(35, 42, 5, 14))
        np.testing.assert_allclose(ax.get_extent(), expected)

    def test_ethiopia_does_not_silently_substitute_an_unrelated_boundary(self):
        self.config.exists = lambda _: False
        with self.assertRaises(FileNotFoundError):
            self.render(region="Ethiopia")
        self.config.exists = lambda _: True
        self.outlines.return_value = []
        with self.assertRaisesRegex(ValueError, "Could not load Ethiopia zones"):
            self.render(region="Ethiopia")
