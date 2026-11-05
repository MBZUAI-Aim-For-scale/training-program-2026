"""Country-box storage names reach the catalogs and year discovery unchanged."""

import ast
from datetime import date
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import warnings


ROOT = Path(__file__).resolve().parents[1]


class CountryBoxBucketPathTests(unittest.TestCase):
    def configuration(self, data_root, country="Ethiopia", remote=True):
        path = ROOT / "ai_weather/config.py"
        tree = ast.parse(path.read_text())
        # Exercise the entire configuration without credentials or cloud I/O.
        tree.body = [node for node in tree.body if not (
            isinstance(node, ast.FunctionDef) and node.name == "_remote_has_data")]
        probes = []

        def probe(uri):
            probes.append(uri)
            return remote

        namespace = {"__file__": str(path), "__name__": "country_box_test_config",
                     "_remote_has_data": probe}
        environment = {"DEMO_DATA_SOURCE": "bucket" if remote else "auto",
                       "DEMO_BUCKET": "weather-test-data", "DEMO_BUCKET_PREFIX": "custom-demo",
                       "DEMO_DATA_DIR": str(data_root), "ROMP_REGION": country, "ROMP_OBS": "CHIRPS",
                       "ROMP_OUT_DIR": str(data_root / "output"),
                       "ROMP_FIG_DIR": str(data_root / "figures")}
        with patch.dict(os.environ, environment), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            exec(compile(tree, str(path), "exec"), namespace)
        return SimpleNamespace(**namespace), probes

    def test_bucket_catalog_uses_country_box_names_without_demo_prefix(self):
        for country in ("Ethiopia", "Kenya"):
            with self.subTest(country=country), tempfile.TemporaryDirectory() as directory:
                cfg, probes = self.configuration(Path(directory), country)
                observations = f"gs://weather-test-data/CHIRPSv3_Country_Boxes_025deg/{country}"
                self.assertEqual(cfg.OBS_URI, observations)
                self.assertEqual(cfg.obs_dir, observations)
                self.assertIn(observations, probes)
                pattern = f"CHIRPSv3_{country}_{{}}_box025.nc"
                self.assertIn(pattern, cfg.obs_file_pattern)
                self.assertIn(pattern, cfg.ref_model_file_pattern)
                for key, model in (("AIFS", "AIFS_v2"), ("AIFS_ENS", "AIFS_ENS_v2"),
                                   ("GraphCast", "graphcast")):
                    entry = cfg.BENCHMARK_MODEL_CATALOG[key]
                    folder = f"gs://weather-test-data/Forecasts_Country_0p25deg/{model}/tp/{country}"
                    self.assertEqual(entry["dir"], folder)
                    self.assertEqual(cfg._model_uri(key), folder)
                    self.assertIn(folder, probes)
                    for year in (2000, 2025):
                        self.assertEqual(entry["file_pattern"].format(year),
                                         f"{model}_tp_{country}_{year}_box025.nc")
                self.assertEqual(cfg.BENCHMARK_MODEL_CATALOG["FuXi"]["file_pattern"], "{}.nc")
                self.assertIn("/custom-demo/MODEL/", cfg.BENCHMARK_MODEL_CATALOG["FuXi"]["dir"])

    def test_local_fallback_keeps_existing_model_filenames(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for model in ("aifs_single", "aifs_ens", "graphcast", "CHIRPS"):
                folder = root / "external" / model
                folder.mkdir(parents=True)
                (folder / "2024.nc").touch()
            cfg, _ = self.configuration(root, remote=False)
            for key in ("AIFS", "AIFS_ENS", "GraphCast"):
                self.assertFalse(cfg.is_uri(cfg.BENCHMARK_MODEL_CATALOG[key]["dir"]))
                self.assertEqual(cfg.BENCHMARK_MODEL_CATALOG[key]["file_pattern"], "{}.nc")
            self.assertEqual(cfg.obs_dir, str((root / "external/CHIRPS").resolve()))
            self.assertIn("{}.nc", cfg.obs_file_pattern)

    def test_observation_inventory_discovers_1998_and_2025_from_new_names(self):
        path = ROOT / "ui/pages/short_medium_benchmarking.py"
        function = next(node for node in ast.parse(path.read_text()).body
                        if isinstance(node, ast.FunctionDef) and node.name == "available_observation_years")
        folder = "gs://weather-test-data/CHIRPSv3_Country_Boxes_025deg/Ethiopia"
        config = SimpleNamespace(
            _OBS_SETS={"CHIRPS": {"dir": folder, "pattern": ("CHIRPSv3_Ethiopia_{}_box025.nc",)}},
            join=lambda base, name: base + "/" + name,
            glob=lambda _: [f"{folder}/CHIRPSv3_Ethiopia_{year}_box025.nc" for year in (1998, 2025)])
        namespace = dict(config=config, BENCHMARK_OBS_CATALOG={}, Path=Path, date=date, List=list)
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
        self.assertEqual(namespace["available_observation_years"]("CHIRPS"), [1998, 2025])

    def test_forecast_inventory_formats_model_years_with_new_patterns(self):
        path = ROOT / "apps/benchmarking/ROMP/momp/io/input.py"
        function = next(node for node in ast.parse(path.read_text()).body
                        if isinstance(node, ast.FunctionDef) and node.name == "available_forecast_years")
        for model in ("AIFS_v2", "AIFS_ENS_v2", "graphcast"):
            folder = f"gs://weather-test-data/Forecasts_Country_0p25deg/{model}/tp/Ethiopia"
            existing = {f"{folder}/{model}_tp_Ethiopia_{year}_box025.nc" for year in (2000, 2025)}
            storage = SimpleNamespace(join=lambda base, name: base + "/" + name,
                                      exists=lambda name: name in existing)
            namespace = {"storage": storage}
            exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
            self.assertEqual(namespace["available_forecast_years"](
                (2000, 2025), model_dir=folder, file_pattern=f"{model}_tp_Ethiopia_{{}}_box025.nc"),
                (2000, 2025))
