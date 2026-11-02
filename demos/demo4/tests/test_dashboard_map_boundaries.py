"""Every model map shows the configured overlay and shares its padded frame."""

import ast
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from ai_weather.map_extent import geographic_extent
from ai_weather.spatial import (
    apply_jjas_mask,
    climatology_boundary_path,
    clip_map_artist,
    clip_to_boundary,
    load_boundary_frame,
)


class DashboardMapBoundaryTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        root = Path(folder.name)
        boundary = root / "ETH_Zones.geojson"
        boundary.write_text(json.dumps({"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {},
            "geometry": {"type": "Polygon", "coordinates": [[[30.5, 5.5], [32.5, 5.5], [32.5, 7.5], [30.5, 7.5], [30.5, 5.5]]]}}]}))
        self.data = xr.DataArray(np.ones((4, 4)), dims=("lat", "lon"),
                                 coords={"lat": [5, 6, 7, 8], "lon": [30, 31, 32, 33]})
        mask = xr.ones_like(self.data, dtype=bool).rename("keep_mask")
        mask.loc[{"lat": 6, "lon": 31}] = False
        mask.to_dataset().to_netcdf(root / "jjas.nc")
        source = Path("ui/pages/short_medium_benchmarking.py")
        tree = ast.parse(source.read_text())
        functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                     and n.name in {"plot_map", "_map_lon_lat_names"}]
        self.st = MagicMock()
        ns = {"np": np, "plt": plt, "xr": xr, "st": self.st,
              "config": SimpleNamespace(shpfile_dir=str(root), nc_mask=str(root / "jjas.nc")),
              "geographic_extent": geographic_extent, "climatology_boundary_path": climatology_boundary_path,
              "apply_jjas_mask": apply_jjas_mask, "clip_to_boundary": clip_to_boundary,
              "clip_map_artist": clip_map_artist, "load_boundary_frame": load_boundary_frame}
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), ns)  # noqa: S102
        self.plot = ns["plot_map"]
        self.addCleanup(plt.close, "all")

    def test_models_share_boundary_frame_and_visible_overlay_despite_missing_data(self):
        expected = geographic_extent(self.data.lon, self.data.lat, region_bounds=(30.5, 32.5, 5.5, 7.5))
        sparse = self.data.where(self.data.lon == 32)
        for model, data in (("AIFS", self.data), ("GenCast", sparse), ("Climatology", self.data)):
            with self.subTest(model=model):
                fig = self.plot(data, f"MAE - {model}")
                self.assertIsNotNone(fig)
                ax = fig.axes[0]
                np.testing.assert_allclose(ax.get_xlim(), expected[:2])
                np.testing.assert_allclose(ax.get_ylim(), expected[2:])
                self.assertTrue(any(collection.get_zorder() >= 10 for collection in ax.collections))
                self.assertIsNotNone(ax.images[0].get_clip_path())
                mask = np.ma.getmaskarray(ax.images[0].get_array())
                self.assertTrue(mask[0, 0])
                self.assertTrue(mask[1, 1])
        self.st.warning.assert_not_called()

    def test_all_missing_map_still_shows_boundary_and_explanation(self):
        fig = self.plot(self.data * np.nan, "MAE - AIFS")
        self.assertIsNotNone(fig)
        ax = fig.axes[0]
        self.assertTrue(any(collection.get_zorder() >= 10 for collection in ax.collections))
        self.assertTrue(any("No valid metric values" in text.get_text() for text in ax.texts))
        self.assertEqual(len(fig.axes), 1)
        self.st.warning.assert_not_called()
