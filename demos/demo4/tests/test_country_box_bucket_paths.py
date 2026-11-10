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
    def configuration(self, data_root, country="Ethiopia", remote=True, country_data=None, mode=None):
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
        environment = {"DEMO_DATA_SOURCE": mode or ("bucket" if remote else "auto"),
                       "DEMO_COUNTRY_DATA": str(country_data or data_root / "no-country-data"),
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
                # (on the forecasts' 0.25° grid, not CHIRPS's own: see config.OBS_URI)
                observations = f"gs://weather-test-data/CHIRPSv3_Country_0p25deg/{country}"
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

    def test_observation_inventory_discovers_2000_and_2025_from_new_names(self):
        path = ROOT / "ui/pages/short_medium_benchmarking.py"
        function = next(node for node in ast.parse(path.read_text()).body
                        if isinstance(node, ast.FunctionDef) and node.name == "available_observation_years")
        folder = "gs://weather-test-data/CHIRPSv3_Country_0p25deg/Ethiopia"
        config = SimpleNamespace(
            _OBS_SETS={"CHIRPS": {"dir": folder, "pattern": ("CHIRPSv3_Ethiopia_{}_box025.nc",)}},
            join=lambda base, name: base + "/" + name,
            glob=lambda _: [f"{folder}/CHIRPSv3_Ethiopia_{year}_box025.nc" for year in (2000, 2025)])
        namespace = dict(config=config, BENCHMARK_OBS_CATALOG={}, Path=Path, date=date, List=list)
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
        self.assertEqual(namespace["available_observation_years"]("CHIRPS"), [2000, 2025])

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


class CountryDataFirstTests(unittest.TestCase):
    """Bucket files this Spark has (Demo 1's country data, the bucket's paths) are read from there."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.country = root / "country_data"
        obs = self.country / "CHIRPSv3_Country_0p25deg/Kenya"
        obs.mkdir(parents=True)
        self.local_file = obs / "CHIRPSv3_Kenya_2020_box025.nc"
        self.local_file.write_bytes(b"x")
        self.cfg, _ = CountryBoxBucketPathTests().configuration(root, "Kenya", remote=False,
                                                                country_data=self.country, mode="auto")
        self.uri = "gs://weather-test-data/CHIRPSv3_Country_0p25deg/Kenya/CHIRPSv3_Kenya_2020_box025.nc"

    def tearDown(self):
        self.tmp.cleanup()

    def test_local_copy_maps_bucket_paths(self):
        self.assertEqual(self.cfg.local_copy(self.uri), str(self.local_file))
        self.assertIsNone(self.cfg.local_copy(self.uri.replace("2020", "2021")))  # not here: the bucket
        self.assertIsNone(self.cfg.local_copy(self.uri.replace("weather-test-data", "other-bucket")))
        self.assertIsNone(self.cfg.local_copy("gs://weather-test-data/../etc/passwd"))
        self.assertIsNone(self.cfg.local_copy(str(self.local_file)))  # (a local path is no bucket file)

    def test_exists_and_open_use_the_copy_without_the_bucket(self):
        opened = []
        self.cfg.open_dataset.__globals__["gcs_fs"] = lambda: (_ for _ in ()).throw(AssertionError("the bucket was asked"))
        import xarray as xr
        with patch.object(xr, "open_dataset", lambda path, **kw: opened.append(path) or "ds"):
            self.assertTrue(self.cfg.exists(self.uri))
            self.assertEqual(self.cfg.open_dataset(self.uri), "ds")
        self.assertEqual(opened, [str(self.local_file)])

    def test_glob_lists_the_bucket_and_the_copy_only_without_it(self):
        listed = ["weather-test-data/CHIRPSv3_Country_0p25deg/Kenya/CHIRPSv3_Kenya_%d_box025.nc" % y
                  for y in (2019, 2020)]
        self.cfg.glob.__globals__["gcs_fs"] = lambda: SimpleNamespace(glob=lambda pattern: listed)
        pattern = "gs://weather-test-data/CHIRPSv3_Country_0p25deg/Kenya/*.nc"
        self.assertEqual(self.cfg.glob(pattern), ["gs://" + n for n in listed])  # (the bucket decides)

        def unreachable():
            raise OSError("no network")
        self.cfg.glob.__globals__["gcs_fs"] = unreachable
        self.assertEqual(self.cfg.glob(pattern), [self.uri])

    def test_runtime_listing_recovers_after_startup_timeout(self):
        namespace = self.cfg.glob.__globals__
        namespace["_REMOTE_PROBE_UNAVAILABLE"] = True
        listed = self.uri.removeprefix("gs://")
        namespace["gcs_fs"] = lambda: SimpleNamespace(glob=lambda pattern: [listed], exists=lambda path: True)
        self.assertEqual(self.cfg.glob(self.uri.rsplit("/", 1)[0] + "/*.nc"), [self.uri])
        self.assertTrue(self.cfg.exists(self.uri.replace("2020", "2021")))

    def test_strict_listing_preserves_bucket_errors_without_a_local_copy(self):
        def denied():
            raise PermissionError("403 access denied")
        self.cfg.glob.__globals__["gcs_fs"] = denied
        with self.assertRaisesRegex(PermissionError, "403 access denied"):
            self.cfg.glob(self.uri.replace("Kenya", "Peru"), raise_errors=True)
        self.assertEqual(self.cfg.glob(self.uri.rsplit("/", 1)[0] + "/*.nc", raise_errors=True), [self.uri])

    def test_recursive_listing_finds_country_copy_after_startup_timeout(self):
        namespace = self.cfg.glob.__globals__
        namespace["_REMOTE_PROBE_UNAVAILABLE"] = True
        def unreachable():
            raise OSError("no network")
        namespace["gcs_fs"] = unreachable
        self.assertEqual(self.cfg.glob(self.uri.rsplit("/", 1)[0] + "/**/*.nc"), [self.uri])

    def test_folder_counts_as_available_from_the_copy_when_the_bucket_is_down(self):
        self.assertEqual(self.cfg.DATA_SOURCES.get("observations"), "country data")
        self.assertEqual(self.cfg.OBS_DIR, "gs://weather-test-data/CHIRPSv3_Country_0p25deg/Kenya")


class SnapToGridLongitudeTests(unittest.TestCase):
    """A typed point (-180..180) is snapped to the data's own grid, whatever its convention."""

    def snap(self, data_lons, lat, lon):
        import numpy as np
        import xarray as xr
        path = ROOT / "ui/pages/short_medium_benchmarking.py"
        function = next(node for node in ast.parse(path.read_text()).body
                        if isinstance(node, ast.FunctionDef) and node.name == "_snap_to_grid")
        ds = xr.Dataset(coords={"lat": np.arange(-60, 60.01, 0.25), "lon": np.asarray(data_lons, float)})
        config = SimpleNamespace(glob=lambda pattern: ["gs://b/x.nc"], open_dataset=lambda path: ds)
        # (it's cached on the page: a pass-through st.cache_data here)
        st = SimpleNamespace(cache_data=lambda *a, **k: a[0] if a and callable(a[0]) else (lambda f: f))
        namespace = {"config": config, "st": st}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
        return namespace["_snap_to_grid"](lat, lon, "gs://b/CHIRPSv3_Country_0p25deg/Chile")

    def test_west_of_greenwich_on_0_360_data(self):
        import numpy as np
        self.assertEqual(self.snap(np.arange(250, 294.01, 0.25), -33.45, -70.66), (-33.5, 289.25))  # Santiago

    def test_already_matching_conventions(self):
        import numpy as np
        self.assertEqual(self.snap(np.arange(-110, -65.99, 0.25), -33.45, -70.66), (-33.5, -70.75))
        self.assertEqual(self.snap(np.arange(33.5, 42.51, 0.25), -1.29, 36.82), (-1.25, 36.75))  # Nairobi-like
