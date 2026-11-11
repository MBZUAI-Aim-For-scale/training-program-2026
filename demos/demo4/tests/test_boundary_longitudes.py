"""Bucket longitudes and WGS84 country boundaries share the same map coordinates."""

from pathlib import Path
import tempfile
import unittest

import geopandas as gpd
import numpy as np
from shapely.geometry import box
import xarray as xr

from ai_weather.spatial import clip_to_boundary


class BoundaryLongitudeTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)

    def boundary(self, bounds):
        path = self.root / "ADM0.geojson"
        gpd.GeoDataFrame(geometry=[box(*bounds)], crs="EPSG:4326").to_file(path, driver="GeoJSON")
        return path

    def test_western_country_grid_wraps_without_reordering_values_or_mutating_input(self):
        path = self.boundary((-72, -35, -69, -32))
        source = xr.DataArray([[1., 2., 3., 4.], [5., 6., 7., 8.]], dims=("lat", "lon"),
                              coords={"lat": [-34., -33.], "lon": [287., 288., 289., 290.]})
        source.lon.attrs["units"] = "degrees_east"
        original = source.copy(deep=True)
        clipped = clip_to_boundary(source, path)
        np.testing.assert_array_equal(clipped.lon, [-73., -72., -71., -70.])
        np.testing.assert_allclose(clipped.values, [[np.nan, 2., 3., 4.], [np.nan, 6., 7., 8.]])
        self.assertEqual(clipped.lon.attrs["units"], "degrees_east")
        xr.testing.assert_identical(source, original)

    def test_eastern_ethiopia_grid_preserves_coordinates_and_values(self):
        path = self.boundary((33, 3, 48, 15))
        source = xr.DataArray([[1., 2.], [3., 4.]], dims=("latitude", "longitude"),
                              coords={"latitude": [8., 9.], "longitude": [38., 39.]})
        xr.testing.assert_identical(clip_to_boundary(source, path), source)

    def test_greenwich_crossing_grid_keeps_values_attached_to_wrapped_coordinates(self):
        path = self.boundary((-1.5, -1, 1.5, 1))
        source = xr.DataArray([[10., 11., 12., 13.]], dims=("lat", "lon"),
                              coords={"lat": [0.], "lon": [0., 1., 358., 359.]})
        clipped = clip_to_boundary(source, path)
        np.testing.assert_array_equal(clipped.lon, [-2., -1., 0., 1.])
        np.testing.assert_allclose(clipped.values, [[np.nan, 13., 10., 11.]])
        np.testing.assert_array_equal(source.lon, [0., 1., 358., 359.])
        np.testing.assert_allclose(source.values, [[10., 11., 12., 13.]])
