"""Climatology grids and rendered cells respect polygon edges and holes."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from ai_weather.spatial import (
    boundary_geometry,
    clip_map_artist,
    clip_to_boundary,
    load_boundary_frame,
    to_boundary_longitudes,
)


class SpatialClippingTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name) / "zones.geojson"
        polygon = {"type": "Polygon", "coordinates": [
            [[0, 0], [3, 0], [3, 3], [0, 3], [0, 0]],
            [[1, 1], [1, 2], [2, 2], [2, 1], [1, 1]]]
        }
        self.path.write_text(json.dumps({"type": "FeatureCollection", "features": [
            {"type": "Feature", "properties": {}, "geometry": polygon}]}))

    def test_mask_excludes_outside_and_holes_without_changing_input(self):
        data = xr.DataArray(np.ones((2, 5, 5)), dims=("year", "lat", "lon"),
                            coords={"year": [2023, 2024], "lat": [-.5, .5, 1.5, 2.5, 3.5],
                                    "lon": [-.5, .5, 1.5, 2.5, 3.5]})
        masked = clip_to_boundary(data, self.path)
        self.assertTrue(masked.sel(lat=-.5).isnull().all())
        self.assertTrue(masked.sel(lat=1.5, lon=1.5).isnull().all())
        self.assertTrue((masked.sel(lat=.5, lon=.5) == 1).all())
        self.assertTrue((data == 1).all())

    def test_exact_artist_clip_preserves_polygon_holes(self):
        fig, ax = plt.subplots(figsize=(3, 3), dpi=100)
        self.addCleanup(plt.close, fig)
        image = ax.imshow(np.ones((4, 4)), extent=(-.5, 3.5, -.5, 3.5), origin="lower", cmap="viridis")
        clip_map_artist(image, ax, self.path)
        ax.set_axis_off()
        fig.canvas.draw()
        pixels = np.asarray(fig.canvas.buffer_rgba())

        def color(x, y):
            px, py = ax.transData.transform((x, y))
            return pixels[pixels.shape[0] - 1 - int(py), int(px), :3]

        np.testing.assert_array_equal(color(1.5, 1.5), [255, 255, 255])
        np.testing.assert_array_equal(color(3.25, 2.5), [255, 255, 255])
        self.assertFalse(np.all(color(.5, .5) == 255))

    def test_boundary_reads_are_cached_until_file_changes(self):
        import geopandas as gpd

        with patch.object(gpd, "read_file", wraps=gpd.read_file) as read:
            first = load_boundary_frame(self.path)
            self.assertIs(first, load_boundary_frame(self.path))
            geometry = boundary_geometry(self.path)
            self.assertIs(geometry, boundary_geometry(self.path))
            self.assertEqual(read.call_count, 1)
            self.path.write_text(self.path.read_text() + "\n")
            self.assertIsNot(first, load_boundary_frame(self.path))
            self.assertIsNot(geometry, boundary_geometry(self.path))
            self.assertEqual(read.call_count, 2)


class WestOfGreenwichTests(unittest.TestCase):
    """Country data west of Greenwich is stored 0-360 (Senegal at 342-349); boundaries are -180-180."""

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name) / "senegal.geojson"
        box = [[-17.5, 12.3], [-11.3, 12.3], [-11.3, 16.7], [-17.5, 16.7], [-17.5, 12.3]]
        self.path.write_text(json.dumps({"type": "FeatureCollection", "features": [
            {"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [box]}}]}))
        lon = np.arange(342.0, 349.01, 0.25)
        lat = np.arange(11.75, 17.26, 0.25)
        self.data = xr.DataArray(np.ones((lat.size, lon.size)), dims=("lat", "lon"), coords={"lat": lat, "lon": lon})

    def test_clip_keeps_the_country_and_returns_minus_180_to_180(self):
        masked = clip_to_boundary(self.data, self.path)
        self.assertTrue(np.isfinite(masked.values).any())                   # (was all NaN: "No valid data")
        self.assertAlmostEqual(float(masked.lon.min()), -18.0)
        self.assertAlmostEqual(float(masked.lon.max()), -11.0)
        self.assertTrue(bool((masked.lon.diff("lon") > 0).all()))           # (sorted)
        self.assertEqual(float(masked.sel(lat=14.0, lon=-14.0)), 1.0)       # (inside)
        self.assertTrue(np.isnan(float(masked.sel(lat=14.0, lon=-18.0))))   # (outside, west)
        self.assertEqual(float(self.data.lon.min()), 342.0)                 # (input unchanged)

    def test_east_of_greenwich_is_unchanged(self):
        geometry = boundary_geometry(self.path)
        nigeria = self.data.assign_coords(lon=np.linspace(2.25, 15.25, self.data.lon.size))
        self.assertIs(to_boundary_longitudes(nigeria, geometry), nigeria)
