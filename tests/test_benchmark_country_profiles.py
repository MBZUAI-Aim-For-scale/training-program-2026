"""Country source profiles select independent files and safe spatial defaults."""

import ast
import copy
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import warnings

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "demos/demo4/ai_weather/config.py"


def configuration(root, *, country="Ethiopia", mode="auto", remote=False, overrides=None):
    tree = ast.parse(CONFIG.read_text())
    tree.body = [node for node in tree.body if not (
        isinstance(node, ast.FunctionDef) and node.name == "_remote_has_data")]
    namespace = {"__file__": str(CONFIG), "__name__": "country_profile_test_config",
                 "_remote_has_data": lambda _: remote}
    environment = {"DEMO_DATA_SOURCE": mode, "DEMO_BUCKET": "weather-test-data",
                   "DEMO_DATA_DIR": str(root), "ROMP_REGION": country, "ROMP_OBS": "CHIRPS",
                   "ROMP_OUT_DIR": str(root / "output"), "ROMP_FIG_DIR": str(root / "figures"),
                   "ROMP_MASK_URI": "", "ROMP_SHAPEFILE_URI": "", "ROMP_CLIMATOLOGY_SHAPEFILE": "",
                   "DEMO_COUNTRY_DATA": str(root / "country-copy"), **(overrides or {})}
    with patch.dict(os.environ, environment), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exec(compile(tree, str(CONFIG), "exec"), namespace)  # noqa: S102 -- Evaluate config with the cloud probe stubbed.
    return SimpleNamespace(**namespace), namespace


class BenchmarkCountryProfileTests(unittest.TestCase):
    def test_peru_profile_uses_aligned_chirps_and_three_country_models(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg, namespace = configuration(Path(directory), remote=True)
            before_catalog = copy.deepcopy(cfg.BENCHMARK_MODEL_CATALOG)
            before_sources = dict(cfg.DATA_SOURCES)

            def unexpected_cloud_call(*args, **kwargs):
                raise AssertionError("Profile construction must not contact cloud storage")

            namespace.update(_remote_has_data=unexpected_cloud_call, gcs_fs=unexpected_cloud_call)
            profile = cfg.benchmark_country_sources("Peru")
            self.assertTrue(profile["available"])
            self.assertEqual(profile["region"], "Peru")
            self.assertEqual(set(profile["models"]), {"AIFS", "AIFS_ENS", "GraphCast"})
            self.assertEqual(profile["observations"]["dir"],
                             "gs://weather-test-data/CHIRPSv3_Country_0p25deg/Peru")
            self.assertEqual(profile["observations"]["pattern"], ("CHIRPSv3_Peru_{}_box025.nc",))
            self.assertEqual(profile["observations"]["pattern"], profile["observations"]["file_pattern"])
            for key, folder in (("AIFS", "AIFS_v2"), ("AIFS_ENS", "AIFS_ENS_v2"),
                                ("GraphCast", "graphcast")):
                metadata = profile["models"][key]
                self.assertEqual(metadata["dir"],
                                 f"gs://weather-test-data/Forecasts_Country_0p25deg/{folder}/tp/Peru")
                self.assertEqual(metadata["file_pattern"].format(2000), f"{folder}_tp_Peru_2000_box025.nc")
            profile["models"]["AIFS"]["unit_cvt"] = 123
            self.assertEqual(cfg.BENCHMARK_MODEL_CATALOG, before_catalog)
            self.assertEqual(cfg.DATA_SOURCES, before_sources)
            self.assertNotEqual(cfg.benchmark_country_sources("Peru")["models"]["AIFS"]["unit_cvt"], 123)
            self.assertEqual(cfg.region, "Ethiopia")

    def test_ethiopia_profile_preserves_configured_catalog(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg, _ = configuration(Path(directory), remote=True)
            profile = cfg.benchmark_country_sources("Ethiopia")
            self.assertEqual(profile["models"], cfg.BENCHMARK_MODEL_CATALOG)
            self.assertIsNot(profile["models"], cfg.BENCHMARK_MODEL_CATALOG)
            self.assertEqual(profile["observations"]["dir"], cfg.OBS_DIR)

    def test_multiword_country_uses_downloaded_country_token(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg, _ = configuration(Path(directory), remote=True)
            profile = cfg.benchmark_country_sources(" Sierra   Leone ")
            self.assertEqual(profile["region"], "Sierra_Leone")
            self.assertTrue(profile["observations"]["dir"].endswith("/Sierra_Leone"))
            self.assertEqual(profile["observations"]["pattern"], ("CHIRPSv3_Sierra_Leone_{}_box025.nc",))

    def test_other_country_never_falls_back_to_flat_ethiopian_files_or_masks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for folder in ("aifs_single", "aifs_ens", "graphcast", "CHIRPS"):
                target = root / "external" / folder
                target.mkdir(parents=True)
                (target / "2024.nc").touch()
            cfg, _ = configuration(root, country="Peru", mode="local")
            self.assertIn("/Peru", cfg.OBS_DIR)
            for key in ("AIFS", "AIFS_ENS", "GraphCast"):
                self.assertIn("/Peru", cfg.BENCHMARK_MODEL_CATALOG[key]["dir"])
            self.assertIsNone(cfg.nc_mask)
            self.assertIsNone(cfg.shpfile_dir)
            self.assertIsNone(cfg.CLIMATOLOGY_SHAPEFILE)
            self.assertFalse(cfg.benchmark_country_sources("Peru")["available"])

    def test_local_country_files_are_supported_without_a_bucket(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            obs = root / "external/CHIRPSv3_Country_0p25deg/Peru"
            model = root / "external/Forecasts_Country_0p25deg/AIFS_v2/tp/Peru"
            for folder, filename in ((obs, "CHIRPSv3_Peru_2024_box025.nc"),
                                     (model, "AIFS_v2_tp_Peru_2024_box025.nc")):
                folder.mkdir(parents=True)
                (folder / filename).touch()
            cfg, _ = configuration(root, mode="local")
            profile = cfg.benchmark_country_sources("Peru")
            self.assertTrue(profile["available"])
            self.assertEqual(profile["observations"]["dir"], str(obs.resolve()))
            self.assertEqual(set(profile["models"]), {"AIFS"})

    def test_explicit_spatial_overrides_are_retained_and_unsupported_sources_are_clear(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg, _ = configuration(Path(directory), country="Peru", remote=True, overrides={
                "ROMP_MASK_URI": "/custom/peru-mask.nc", "ROMP_SHAPEFILE_URI": "/custom/peru.geojson",
                "ROMP_CLIMATOLOGY_SHAPEFILE": "/custom/peru-climatology.geojson"})
            self.assertEqual(cfg.nc_mask, "/custom/peru-mask.nc")
            self.assertEqual(cfg.shpfile_dir, "/custom/peru.geojson")
            self.assertEqual(cfg.CLIMATOLOGY_SHAPEFILE, "/custom/peru-climatology.geojson")
            profile = cfg.benchmark_country_sources("Peru", "ENACTS")
            self.assertFalse(profile["available"])
            self.assertIn("ENACTS", profile["reason"])
            self.assertFalse(cfg.benchmark_country_sources("../Ethiopia")["available"])
