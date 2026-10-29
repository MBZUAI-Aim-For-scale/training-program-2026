"""Check map bounds independently of boundary downloads and plot styling."""

import ast
import unittest
from pathlib import Path

import numpy as np

from ai_weather.map_extent import geographic_extent


class MapExtentTests(unittest.TestCase):
    def test_full_cells_and_symmetric_margin(self):
        self.assertEqual(geographic_extent([30, 31, 32], [5, 6], padding=0),
                         (29.5, 32.5, 4.5, 6.5))
        np.testing.assert_allclose(geographic_extent([30, 31, 32], [5, 6]),
                                   [29.26, 32.74, 4.34, 6.66])

    def test_masked_borders_and_descending_latitude(self):
        values = np.full((5, 6), np.nan)
        values[1:4, 2:4] = 1
        extent = geographic_extent(np.arange(30, 36), np.arange(10, 5, -1), values, padding=0)
        self.assertEqual(extent, (31.5, 33.5, 6.5, 9.5))

    def test_singleton_all_missing_and_irregular_coordinates(self):
        self.assertEqual(geographic_extent([35], [10], [[np.nan]], padding=0),
                         (34.5, 35.5, 9.5, 10.5))
        self.assertEqual(geographic_extent([30, 31, 34], [5, 7], padding=0),
                         (29.5, 35.5, 4, 8))

    def test_eastern_hemisphere_longitudes_are_preserved(self):
        west, east, _, _ = geographic_extent([200, 201], [0, 1])
        self.assertLess(west, east)
        self.assertGreater(west, 180)

    def test_invalid_coordinates_report_error(self):
        with self.assertRaises(ValueError):
            geographic_extent([np.nan], [1])

    def test_region_boundary_is_centered_instead_of_wider_source_grid(self):
        extent = geographic_extent(np.arange(30, 49, .25), np.arange(0, 16, .25),
                                   region_bounds=(35, 42, 5, 14))
        np.testing.assert_allclose(extent, (34.44, 42.56, 4.28, 14.72))
        self.assertAlmostEqual((extent[0] + extent[1]) / 2, 38.5)
        self.assertAlmostEqual((extent[2] + extent[3]) / 2, 9.5)

    def test_region_padding_is_fixed_across_model_resolutions(self):
        extent = geographic_extent([40, 38, 36], [14, 12, 10],
                                   region_bounds=(36, 40, 10, 14))
        np.testing.assert_allclose(extent, (35.68, 40.32, 9.68, 14.32))
        fine = geographic_extent(np.arange(36, 40, .25), np.arange(10, 14, .25),
                                 region_bounds=(36, 40, 10, 14))
        self.assertEqual(extent, fine)

    def test_dashboard_renderer_uses_configured_boundary_bounds(self):
        import json
        import tempfile
        from types import SimpleNamespace
        from unittest.mock import MagicMock

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import xarray as xr

        from ai_weather.spatial import (
            apply_jjas_mask,
            climatology_boundary_path,
            clip_map_artist,
            clip_to_boundary,
            load_boundary_frame,
        )

        path = Path("ui/pages/short_medium_benchmarking.py")
        tree = ast.parse(path.read_text())
        functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                     and n.name in {"plot_map", "_map_lon_lat_names"}]
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        boundary = Path(folder.name) / "ETH_Zones.geojson"
        boundary.write_text(json.dumps({"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {},
            "geometry": {"type": "Polygon", "coordinates": [[[30.5, 6.5], [34.5, 6.5], [34.5, 9.5], [30.5, 9.5], [30.5, 6.5]]]}}]}))
        ns = {"np": np, "plt": plt, "xr": xr, "geographic_extent": geographic_extent,
              "config": SimpleNamespace(shpfile_dir=folder.name, nc_mask=None), "st": MagicMock(),
              "climatology_boundary_path": climatology_boundary_path, "clip_to_boundary": clip_to_boundary,
              "apply_jjas_mask": apply_jjas_mask, "clip_map_artist": clip_map_artist, "load_boundary_frame": load_boundary_frame}
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), ns)  # noqa: S102
        values = np.full((5, 6), np.nan)
        values[1:4, 2:4] = 1
        da = xr.DataArray(values, dims=("lat", "lon"),
                          coords={"lat": np.arange(10, 5, -1), "lon": np.arange(30, 36)})
        fig = ns["plot_map"](da, "Test")
        self.assertIsNotNone(fig)
        try:
            expected = geographic_extent(da.lon, da.lat, region_bounds=(30.5, 34.5, 6.5, 9.5))
            np.testing.assert_allclose(fig.axes[0].get_xlim(), expected[:2])
            np.testing.assert_allclose(fig.axes[0].get_ylim(), expected[2:])
        finally:
            plt.close(fig)
