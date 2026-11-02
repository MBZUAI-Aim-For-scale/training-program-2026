"""All model maps keep configured region bounds and overlays above rasters."""

import ast
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import cartopy.crs as ccrs
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib import colormaps
from matplotlib import colors as mcolors

from ai_weather import config
from ai_weather.map_extent import geographic_extent
from ai_weather.spatial import (
    apply_jjas_mask,
    boundary_geometry,
    climatology_boundary_path,
    clip_map_artist,
    clip_to_boundary,
)


def load_function(relative_path, name, namespace):
    path = Path("apps/benchmarking/ROMP/momp") / relative_path
    tree = ast.parse(path.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)  # noqa: S102
    return namespace[name]


class BasemapOverlayTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.folder = Path(folder.name)
        self.source = self.folder / "ETH_Zones.geojson"
        self.ring = ([35, 42, 42, 35, 35], [5, 5, 14, 14, 5])
        self.source.write_text(json.dumps({"type": "FeatureCollection", "features": [
            {"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [
                list(map(list, zip(*self.ring, strict=True)))]}}]}))
        self.outline = MagicMock(return_value=[self.ring])
        modules = {name: types.ModuleType(name) for name in (
            "momp", "momp.utils", "momp.params", "momp.utils.land_mask", "momp.params.region_def")}
        modules["momp.utils.land_mask"].get_india_outline = self.outline
        modules["momp.utils.land_mask"].shp_outline = MagicMock()
        modules["momp.params.region_def"].domain = lambda _: (3, 15, 33, 48)
        self.addCleanup(patch.stopall)
        patch.dict(sys.modules, modules).start()
        patch.object(config, "shpfile_dir", str(self.folder)).start()
        self.basemap = load_function("utils/visual.py", "set_basemap", {})
        self.addCleanup(plt.close, "all")

    def test_boundary_transform_and_zorder_are_correct_above_raster(self):
        fig = plt.figure()
        ax = fig.add_subplot(projection=ccrs.Mercator())
        self.basemap(ax, "Ethiopia", "wrong-subset.shp", False, grid=False)
        image = ax.pcolormesh([33, 48], [3, 15], np.ones((2, 2)), transform=ccrs.PlateCarree())
        self.outline.assert_called_with(str(self.source), region="Ethiopia")
        line = ax.lines[0]
        self.assertGreaterEqual(line.get_zorder(), 10)
        self.assertGreater(line.get_zorder(), image.get_zorder())
        expected_position = ax.transData.transform(ax.projection.transform_point(35, 5, ccrs.PlateCarree()))
        np.testing.assert_allclose(line.get_transform().transform((35, 5)), expected_position)
        expected = geographic_extent([], [], region_bounds=(35, 42, 5, 14))
        np.testing.assert_allclose(ax.get_extent(ccrs.PlateCarree()), expected, atol=1e-6)
        explicit = (34, 44, 4, 15)
        self.basemap(ax, "Ethiopia", "wrong-subset.shp", False, grid=False, map_extent=explicit)
        np.testing.assert_allclose(ax.get_extent(ccrs.PlateCarree()), explicit, atol=1e-6)

    def test_every_model_uses_same_region_frame_and_configured_jjas_mask(self):
        lat, lon = np.linspace(3, 15, 49), np.linspace(33, 48, 61)
        mask_values = np.zeros((49, 61), dtype=bool)
        mask_values[24, 24] = True
        mask = xr.DataArray(mask_values, dims=("lat", "lon"), coords={"lat": lat, "lon": lon}, name="keep")
        mask_path = self.folder / "jjas.nc"
        mask.to_dataset().to_netcdf(mask_path)
        namespace = {"np": np, "plt": plt, "os": os, "ccrs": ccrs,
                     "mcolors": mcolors, "colormaps": colormaps, "weather_config": config,
                     "apply_jjas_mask": apply_jjas_mask, "boundary_geometry": boundary_geometry,
                     "climatology_boundary_path": climatology_boundary_path,
                     "clip_to_boundary": clip_to_boundary, "clip_map_artist": clip_map_artist,
                     "geographic_extent": geographic_extent, "set_basemap": self.basemap,
                     "tuple_to_str": lambda value: "-".join(map(str, value))}
        plot = load_function("graphics/func_map.py", "spatial_metrics_map", namespace)
        expected = geographic_extent(lon, lat, region_bounds=(35, 42, 5, 14))
        for model in ("AIFS", "GenCast", "climatology", "empty"):
            data = xr.DataArray(np.full((49, 61), np.nan if model == "empty" else 180.0), dims=("lat", "lon"),
                                coords={"lat": lat, "lon": lon}, name="metric")
            _, ax, image, _ = plot(data, model, years=(2024,), shpfile_dir="wrong-subset.shp",
                                    polygon=False, dir_fig=str(self.folder), region="Ethiopia",
                                    nc_mask=str(mask_path), n_colors=0, panel=True,
                                    verification_window=(1, 15))
            np.testing.assert_allclose(ax.get_extent(), expected)
            self.assertEqual(np.isfinite(np.asarray(image.get_array())).sum(), 0 if model == "empty" else 1)
            if model == "empty":
                self.assertTrue(any(text.get_text() == "No valid data" for text in ax.texts))
            self.assertIsNotNone(image.get_clip_path())
            self.assertTrue(ax.lines)
            self.assertGreater(ax.lines[0].get_zorder(), image.get_zorder())
