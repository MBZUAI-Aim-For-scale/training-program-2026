"""Downloaded country boundary files override legacy Ethiopian boundary folders."""

import ast
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import warnings

from ai_weather.spatial import climatology_boundary_path


class BoundaryFileOverrideTests(unittest.TestCase):
    def configuration(self, root, boundary, mode, climatology=None):
        path = Path(__file__).resolve().parents[1] / "ai_weather/config.py"
        tree = ast.parse(path.read_text())
        tree.body = [node for node in tree.body if not (
            isinstance(node, ast.FunctionDef) and node.name == "_remote_has_data")]
        namespace = {"__file__": str(path), "__name__": "boundary_override_config",
                     "_remote_has_data": lambda _: False}
        environment = {"DEMO_DATA_SOURCE": mode, "DEMO_BUCKET": "weather-test-data",
                       "DEMO_DATA_DIR": str(root), "ROMP_REGION": "Ethiopia", "ROMP_OBS": "CHIRPS",
                       "ROMP_OUT_DIR": str(root / "output"), "ROMP_FIG_DIR": str(root / "figures"),
                       "ROMP_MASK_URI": "", "ROMP_SHAPEFILE_URI": str(boundary),
                       "ROMP_CLIMATOLOGY_SHAPEFILE": str(climatology or ""),
                       "DEMO_COUNTRY_DATA": str(root / "no-country-data")}
        with patch.dict(os.environ, environment), patch("dotenv.load_dotenv"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            exec(compile(tree, str(path), "exec"), namespace)  # noqa: S102 -- evaluate config with cloud probing stubbed
        return SimpleNamespace(**namespace)

    def test_downloaded_ethiopia_file_wins_over_preloaded_folder_in_all_modes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            legacy = root / "external/shapefile"
            legacy.mkdir(parents=True)
            (legacy / "ETH_Zones.geojson").write_text("{}")
            boundary = root / "external/boundaries/ETH/ADM0.geojson"
            boundary.parent.mkdir(parents=True)
            boundary.write_text("{}")
            for mode in ("auto", "local", "bucket"):
                with self.subTest(mode=mode):
                    cfg = self.configuration(root, boundary, mode, climatology=boundary)
                    self.assertEqual(cfg.shpfile_dir, str(boundary))
                    self.assertEqual(cfg.CLIMATOLOGY_SHAPEFILE, str(boundary))
                    self.assertEqual(climatology_boundary_path(cfg), str(boundary))
                    self.assertEqual(cfg.DATA_SOURCES["administrative boundary"], "local")

    def test_explicit_file_is_also_climatology_default_without_separate_override(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            boundary = "gs://weather-test-data/boundaries/ETH/ADM0.geojson"
            cfg = self.configuration(root, boundary, "bucket")
            self.assertEqual(cfg.shpfile_dir, boundary)
            self.assertEqual(cfg.CLIMATOLOGY_SHAPEFILE, boundary)
            self.assertEqual(climatology_boundary_path(cfg), boundary)
            self.assertEqual(cfg.DATA_SOURCES["administrative boundary"], "bucket")

    def test_legacy_folder_keeps_existing_zone_boundary_default(self):
        cfg = SimpleNamespace(shpfile_dir="/legacy/shapefile", CLIMATOLOGY_SHAPEFILE="/stale/file.geojson")
        self.assertEqual(climatology_boundary_path(cfg), "/legacy/shapefile/ETH_Zones.geojson")
