"""Observation selection must not replace a custom reference forecast's source."""

import ast
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]


class ReferenceSourceOverrideTests(unittest.TestCase):
    def setUp(self):
        source = ROOT / "run_momp.py"
        functions = [node for node in ast.parse(source.read_text()).body
                     if isinstance(node, ast.FunctionDef)
                     and node.name in {"apply_spec_to_cfg", "apply_env_vars_to_cfg",
                                       "validate_runtime_config"}]
        observation = {"dir": "/observations", "file_pattern": "CHIRPS_{}.nc", "var": "rain"}
        self.namespace = {
            "ast": ast, "os": os, "config": SimpleNamespace(_OBS_SETS={}),
            "_clean_str": lambda value: str(value or "").strip(),
            "_meta_get": lambda metadata, key, default: metadata.get(key, default),
            "_obs_meta": lambda key: observation if key.upper() == "CHIRPS" else None,
            "resolve_path": str,
        }
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"),  # noqa: S102
             self.namespace)

    def config(self, reference="custom"):
        return SimpleNamespace(
            ref_model=reference, ref_model_dir="/custom-reference",
            ref_model_file_pattern=("reference_{}.nc",), ref_model_var="reference_rain",
            obs_dir="/old-observations", obs_file_pattern=("{}.nc",), obs_var="old_rain")

    def apply_environment(self, cfg, **environment):
        with patch.dict(os.environ, environment, clear=True):
            self.namespace["apply_env_vars_to_cfg"](cfg)

    def assert_custom_reference(self, cfg, directory="/custom-reference"):
        self.assertEqual(cfg.ref_model_dir, directory)
        self.assertEqual(cfg.ref_model_file_pattern, ("reference_{}.nc",))
        self.assertEqual(cfg.ref_model_var, "reference_rain")

    def assert_observation_reference(self, cfg):
        self.assertEqual(cfg.ref_model_dir, "/observations")
        self.assertEqual(cfg.ref_model_file_pattern, ("CHIRPS_{}.nc",))
        self.assertEqual(cfg.ref_model_var, "rain")

    def test_environment_ground_truth_preserves_custom_reference(self):
        cfg = self.config()
        self.apply_environment(cfg, ROMP_GROUND_TRUTH="CHIRPS")
        self.assertEqual(cfg.obs_dir, "/observations")
        self.assertEqual(cfg.obs_var, "rain")
        self.assert_custom_reference(cfg)

    def test_environment_ground_truth_updates_climatology_reference(self):
        cfg = self.config("climatology")
        self.apply_environment(cfg, ROMP_GROUND_TRUTH="CHIRPS")
        self.assert_observation_reference(cfg)

    def test_environment_switch_to_climatology_uses_selected_observations(self):
        cfg = self.config()
        self.apply_environment(cfg, ROMP_BASELINE_FORECAST="climatology",
                               ROMP_GROUND_TRUTH="CHIRPS")
        self.assert_observation_reference(cfg)

    def test_spec_reference_path_survives_environment_observation_override(self):
        cfg = self.config()
        self.namespace["apply_spec_to_cfg"](cfg, {
            "ground_truth": "CHIRPS",
            "shared": {"baseline_forecast": "custom", "baseline_data_path": "/spec-reference"},
        }, "AIFS")
        self.apply_environment(cfg, ROMP_GROUND_TRUTH="CHIRPS")
        self.assert_custom_reference(cfg, directory="/spec-reference")

    def test_spec_ground_truth_preserves_existing_custom_reference(self):
        cfg = self.config()
        self.namespace["apply_spec_to_cfg"](cfg, {"ground_truth": "CHIRPS"}, "AIFS")
        self.assert_custom_reference(cfg)

    def test_spec_switch_to_climatology_uses_selected_observations(self):
        cfg = self.config()
        self.namespace["apply_spec_to_cfg"](cfg, {
            "ground_truth": "CHIRPS", "shared": {"baseline_forecast": "climatology"},
        }, "AIFS")
        self.assert_observation_reference(cfg)

    def test_runtime_rejects_cross_year_season_before_scientific_processing(self):
        cfg = SimpleNamespace(start_date=(2021, 10, 1), end_date=(2024, 3, 31))
        with self.assertRaisesRegex(ValueError, "Cross-year evaluation seasons are not supported"):
            self.namespace["validate_runtime_config"](cfg)

    def test_runtime_accepts_multiyear_evaluation_within_calendar_year(self):
        cfg = SimpleNamespace(
            start_date=(2021, 5, 1), end_date=(2024, 7, 31),
            init_type="daily", init_days=(), verification_window_list=((1, 15),),
            tolerance_days_list=(3,), max_forecast_day=15,
            day_bins=((1, 5), (6, 10), (11, 15)), probabilistic=False)
        self.namespace["validate_runtime_config"](cfg)
        self.assertEqual(cfg.day_bins, ((1, 5), (6, 10), (11, 15)))

    def test_runtime_rejects_insufficient_remaining_reference_members(self):
        cfg = SimpleNamespace(start_date=(2019, 5, 1), end_date=(2019, 7, 31),
                              years_clim=(2018, 2019), probabilistic=True, ref_model='climatology')
        with self.assertRaisesRegex(ValueError, 'at least 2 baseline years remaining'):
            self.namespace['validate_runtime_config'](cfg)
