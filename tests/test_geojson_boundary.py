"""Explicit GeoJSON boundaries take precedence over cached shapefiles."""

import ast
import glob
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

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


class GeojsonBoundaryTests(unittest.TestCase):
    def test_explicit_geojson_is_not_replaced_by_cached_shapefile(self):
        source = Path("apps/benchmarking/ROMP/momp/utils/land_mask.py")
        tree = ast.parse(source.read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                        and n.name == "ensure_local_shapefile")
        ns = {"os": os, "glob": glob, "Path": Path}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), ns)  # noqa: S102
        with tempfile.TemporaryDirectory() as folder:
            for name in ("ETH_Zones.geojson", "ETH_Zones.GEOJSON", "ETH_Zones.json"):
                path = Path(folder) / name
                path.write_text('{"type":"FeatureCollection","features":[]}')
                with self.subTest(name=name), patch.object(glob, "glob", return_value=["/tmp/stale_woreda.shp"]):
                    self.assertEqual(ns["ensure_local_shapefile"](path), str(path))

    def test_netcdf_climatology_doy_explorer_uses_zones_overlay_and_extent(self):
        source = Path("ui/pages/short_medium_benchmarking.py")
        tree = ast.parse(source.read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                        and n.name == "_render_variable_plot")
        feature = {"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {},
                   "geometry": {"type": "Polygon", "coordinates": [[[33, 4], [48, 4], [48, 15], [33, 15], [33, 4]]]}}]}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "ETH_Zones.geojson"
            path.write_text(json.dumps(feature))
            st = MagicMock()
            ns = {"plt": plt, "xr": xr, "st": st, "geographic_extent": geographic_extent,
                  "clip_map_artist": clip_map_artist, "clip_to_boundary": clip_to_boundary,
                  "apply_jjas_mask": apply_jjas_mask, "climatology_boundary_path": climatology_boundary_path,
                  "load_boundary_frame": load_boundary_frame,
                  "config": SimpleNamespace(shpfile_dir=str(path.parent), nc_mask=None)}
            exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), ns)  # noqa: S102
            ds = xr.Dataset({"dayofyear": xr.DataArray(np.full((49, 61), 180.0), dims=("lat", "lon"),
                            coords={"lat": np.linspace(3, 15, 49), "lon": np.linspace(33, 48, 61)})})
            ns["_render_variable_plot"](ds, "dayofyear", "Climatology Onset Doy 2000-2024")
            st.warning.assert_not_called()
            fig = st.pyplot.call_args.args[0]
            expected = geographic_extent(ds.lon, ds.lat, region_bounds=(33, 48, 4, 15))
            np.testing.assert_allclose(fig.axes[0].get_xlim(), expected[:2])
            np.testing.assert_allclose(fig.axes[0].get_ylim(), expected[2:])
            self.assertGreaterEqual(len(fig.axes[0].collections), 2)
