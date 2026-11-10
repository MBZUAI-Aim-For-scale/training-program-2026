"""Runtime file checks must recover from failed startup discovery in fresh workers."""

import ast
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[1] / "apps/benchmarking/ROMP/momp"


def functions(path, names, namespace):
    nodes = [node for node in ast.parse(path.read_text()).body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)  # noqa: S102
    return namespace


def inventory_in_fresh_worker():
    fs, _, _, inventory = RuntimeStorageExistenceTests().runtime()
    fs.exists.side_effect = lambda path: "_2000_" in path
    years = inventory((1999, 2000), model_dir=RuntimeStorageExistenceTests.folder,
                      file_pattern=RuntimeStorageExistenceTests.pattern)
    return os.getpid(), years


class RuntimeStorageExistenceTests(unittest.TestCase):
    folder = "gs://aim4scale_training_2026/Forecasts_Country_0p25deg/AIFS_ENS_v2/tp/Ethiopia"
    pattern = "AIFS_ENS_v2_tp_Ethiopia_{}_box025.nc"

    def runtime(self, *, local=None):
        fs = MagicMock()
        config = SimpleNamespace(
            _REMOTE_PROBE_UNAVAILABLE=True,
            exists=MagicMock(return_value=False),
            local_copy=MagicMock(return_value=local),
        )
        discovery = functions(Path(__file__).resolve().parents[1] / "ai_weather/config.py",
                              {"is_uri", "_strip", "exists"},
                              {"Any": object, "Optional": object, "Path": Path,
                               "local_copy": config.local_copy, "gcs_fs": lambda: fs,
                               "_REMOTE_PROBE_UNAVAILABLE": True})
        config.exists = MagicMock(wraps=discovery["exists"])
        runtime = functions(ROOT / "utils/storage.py", {"is_uri", "_strip", "join", "exists"},
                            {"config": config, "gcs_fs": lambda: fs, "Path": Path,
                             "Any": object, "os": os})
        storage = SimpleNamespace(**{name: runtime[name] for name in ("join", "exists")})
        inventory = functions(ROOT / "io/input.py", {"available_forecast_years"}, {"storage": storage})
        return fs, config, runtime["exists"], inventory["available_forecast_years"]

    def test_existing_country_file_is_found_despite_startup_timeout(self):
        fs, config, _, inventory = self.runtime()
        fs.exists.return_value = True
        self.assertTrue(config.exists(self.folder + "/" + self.pattern.format(2000)))
        fs.exists.reset_mock()
        config.exists.reset_mock()
        self.assertEqual(inventory((2000,), model_dir=self.folder, file_pattern=self.pattern), (2000,))
        fs.exists.assert_called_once_with(
            "aim4scale_training_2026/Forecasts_Country_0p25deg/AIFS_ENS_v2/tp/Ethiopia/"
            "AIFS_ENS_v2_tp_Ethiopia_2000_box025.nc")
        config.exists.assert_not_called()

    def test_genuinely_absent_years_are_still_skipped(self):
        fs, _, _, inventory = self.runtime()
        fs.exists.side_effect = lambda path: "_2000_" in path
        self.assertEqual(inventory((1999, 2000), model_dir=self.folder, file_pattern=self.pattern), (2000,))

    def test_cloud_errors_are_not_reported_as_missing_forecasts(self):
        for error in (PermissionError("GCS access denied"), TimeoutError("GCS timed out")):
            with self.subTest(error=error):
                fs, _, _, inventory = self.runtime()
                fs.exists.side_effect = error
                with self.assertRaises(type(error)):
                    inventory((2000,), model_dir=self.folder, file_pattern=self.pattern)

    def test_downloaded_country_file_does_not_need_gcs(self):
        fs, _, exists, _ = self.runtime(local="/local/country-data/forecast.nc")
        self.assertTrue(exists(self.folder + "/" + self.pattern.format(2000)))
        fs.exists.assert_not_called()

    def test_spawned_worker_recovers_from_its_own_startup_timeout(self):
        with ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn")) as executor:
            worker_pid, years = executor.submit(inventory_in_fresh_worker).result(timeout=15)
        self.assertNotEqual(worker_pid, os.getpid())
        self.assertEqual(years, (2000,))
