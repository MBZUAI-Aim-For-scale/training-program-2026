"""Every model panel shares shapefile bounds, JJAS masking and visible zones."""

import ast
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from ai_weather.map_extent import geographic_extent
from ai_weather.spatial import (
    apply_jjas_mask,
    boundary_geometry,
    climatology_boundary_path,
    clip_map_artist,
    clip_to_boundary,
    load_boundary_frame,
)


class FixedSpatialPanelTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.addCleanup(plt.close, "all")
        root = Path(self.folder.name)
        self.boundary = root / "ETH_Zones.geojson"
        zones = []
        for west, east in ((30, 32), (32, 34)):
            zones.append({"type": "Feature", "properties": {}, "geometry": {
                "type": "Polygon", "coordinates": [[[west, 5], [east, 5], [east, 8], [west, 8], [west, 5]]]}})
        self.boundary.write_text(json.dumps({"type": "FeatureCollection", "features": zones}))
        self.mask_path = root / "jjas.nc"
        mask = xr.DataArray(np.zeros((12, 14), dtype=bool), dims=("lat", "lon"),
                            coords={"lat": np.arange(4, 10, .5), "lon": np.arange(29, 36, .5)}, name="keep_area")
        mask.loc[{"lat": 6, "lon": 31}] = True
        mask.to_dataset().to_netcdf(self.mask_path)
        path = Path("apps/benchmarking/ROMP/momp/graphics/maps.py")
        tree = ast.parse(path.read_text())
        names = {"plot_spatial_metrics", "_coordinate_spacing", "_geometry_boundaries", "calculate_cmz_averages"}
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
        ns = {"np": np, "plt": plt, "xr": xr, "os": os,
              "weather_config": SimpleNamespace(shpfile_dir=str(root), nc_mask=str(self.mask_path)),
              "climatology_boundary_path": climatology_boundary_path, "apply_jjas_mask": apply_jjas_mask,
              "boundary_geometry": boundary_geometry, "clip_map_artist": clip_map_artist,
              "clip_to_boundary": clip_to_boundary, "load_boundary_frame": load_boundary_frame}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), ns)  # noqa: S102
        self.plot = ns["plot_spatial_metrics"]

    def metrics(self, step, empty=False):
        lat, lon = np.arange(4, 10, step), np.arange(29, 36, step)
        values = np.full((len(lat), len(lon)), np.nan if empty else 2.)
        da = xr.DataArray(values, dims=("lat", "lon"), coords={"lat": lat, "lon": lon})
        return {"mean_mae": da, "false_alarm_rate": da / 10, "miss_rate": da / 10}

    def render(self, model, step=.5, empty=False, **kwargs):
        return self.plot(self.metrics(step, empty), model=model, verification_window=(1, 15),
                         region="Ethiopia", shpfile_dir="unrelated_directory", rect_box=False,
                         dir_fig=self.folder.name, show_plot=False, **kwargs)

    def test_models_share_bounds_despite_resolution_and_sparse_mask(self):
        bounds = (30, 34, 5, 8)
        expected = None
        actual_bounds = []
        for model, step in (("aifs_single", .5), ("gencast", 1)):
            with self.subTest(model=model):
                fig, axes = self.render(model, step)
                metric = self.metrics(step)["mean_mae"]
                extent = geographic_extent(metric.lon, metric.lat, region_bounds=bounds)
                expected = extent
                for ax in axes:
                    np.testing.assert_allclose(ax.get_xlim(), extent[:2])
                    np.testing.assert_allclose(ax.get_ylim(), extent[2:])
                    self.assertEqual(np.ma.count(ax.collections[0].get_array()), 1)
                    self.assertIsNotNone(ax.collections[0].get_clip_path())
                    self.assertEqual(len(ax.lines), 2)
                    self.assertTrue(all(line.get_zorder() >= 10 for line in ax.lines))
                actual_bounds.append((*axes[0].get_xlim(), *axes[0].get_ylim()))
                plt.close(fig)
        np.testing.assert_allclose(actual_bounds[0], actual_bounds[1])
        self.assertLess(expected[0], bounds[0])
        self.assertGreater(expected[1], bounds[1])

    def test_empty_metric_fields_still_show_full_region_and_zones(self):
        fig, axes = self.render("fuxi", empty=True)
        expected = geographic_extent(np.arange(29, 36, .5), np.arange(4, 10, .5), region_bounds=(30, 34, 5, 8))
        for ax in axes:
            np.testing.assert_allclose((*ax.get_xlim(), *ax.get_ylim()), expected)
            self.assertEqual(len(ax.lines), 2)
        plt.close(fig)

    def test_explicit_map_extent_remains_supported(self):
        override = (28, 36, 3, 10)
        fig, axes = self.render("graphcast", map_extent=override)
        for ax in axes:
            np.testing.assert_allclose((*ax.get_xlim(), *ax.get_ylim()), override)
        plt.close(fig)
