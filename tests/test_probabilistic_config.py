"""Check configuration merging and precipitation units without loading weather files."""

import ast
import os
import runpy
import tempfile
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import xarray as xr


def functions(path, names, namespace):
    tree = ast.parse(Path(path).read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)  # noqa: S102


class ProbabilisticConfigTests(unittest.TestCase):
    def setUp(self):
        self.ns = {}
        functions("run_momp.py", {"_apply_forecast_window", "validate_runtime_config"}, self.ns)
        self.cfg = SimpleNamespace(
            verification_window_list=((1, 15), (16, 30)), tolerance_days_list=(3, 5),
            day_bins=((1, 5), (6, 10), (11, 15), (16, 20), (21, 25), (26, 30)),
            max_forecast_day=30, probabilistic=True, BS=True, RPS=True, AUC=True,
            skill_score=True, plot_heatmap_bss_auc=True, plot_panel_heatmap_skill=True,
            plot_bar_bss_rpss_auc=True, save_csv_score=True, parallel=True,
        )

    def test_advanced_override_does_not_lose_second_window_bins(self):
        self.ns["_apply_forecast_window"](self.cfg, "1-15")
        self.cfg.verification_window_list = ((1, 15), (16, 30))
        self.cfg.tolerance_days_list = (3, 5)
        self.cfg.max_forecast_day = 30
        self.ns["validate_runtime_config"](self.cfg)
        self.assertIn((26, 30), self.cfg.day_bins)

    def test_final_config_rejects_inconsistent_horizon_tolerances_and_bins(self):
        for attr, value in (("max_forecast_day", 15), ("tolerance_days_list", (3,)),
                            ("day_bins", ((1, 5), (6, 10), (11, 15)))):
            with self.subTest(attr=attr):
                cfg = SimpleNamespace(**{**vars(self.cfg), attr: value})
                with self.assertRaises(ValueError):
                    self.ns["validate_runtime_config"](cfg)

    def test_final_bins_pass_both_validators_for_single_and_both_windows(self):
        validator = runpy.run_path("apps/benchmarking/ROMP/momp/lib/assertion.py")["ROMPValidator"]
        for selection, expected_bins in (("1-15", ((1, 5), (6, 10), (11, 15))),
                                          ("16-30", ((16, 20), (21, 25), (26, 30))),
                                          (None, self.cfg.day_bins)):
            for model in ("gencast", "aifs_ens"):
                with self.subTest(selection=selection, model=model):
                    cfg = SimpleNamespace(**vars(self.cfg))
                    cfg.start_date, cfg.end_date = (2003, 5, 1), (2024, 7, 31)
                    cfg.start_year_clim, cfg.end_year_clim = 2007, 2024
                    cfg.model_list = (model,)
                    cfg.dry_extent, cfg.dry_spell, cfg.thresh_file = 21, 7, None
                    if selection:
                        self.ns["_apply_forecast_window"](cfg, selection)
                    self.ns["validate_runtime_config"](cfg)
                    self.assertEqual(cfg.day_bins, expected_bins)
                    self.assertTrue(validator(vars(cfg)).validate())

    def test_concurrency_override_reaches_setting_consumed_by_pipeline(self):
        self.cfg.run_years_concurrently = False
        self.ns["validate_runtime_config"](self.cfg)
        self.assertFalse(self.cfg.parallel)

    def test_missing_metrics_fail_before_expensive_plotting(self):
        self.cfg.AUC = False
        with self.assertRaisesRegex(ValueError, "AUC"):
            self.ns["validate_runtime_config"](self.cfg)

    def test_reliability_flag_survives_setting_dataclass(self):
        from dataclasses import asdict

        namespace = runpy.run_path("apps/benchmarking/ROMP/momp/lib/convention.py")
        setting = namespace["Setting"](Reliability=True, plot_reliability=True)
        self.assertTrue(asdict(setting)["Reliability"])

    def test_two_parameter_sources_have_matching_mode_defaults(self):
        flags = {"probabilistic", "members", "BS", "RPS", "AUC", "Reliability", "skill_score",
                 "plot_reliability", "plot_heatmap_bss_auc", "plot_panel_heatmap_skill",
                 "plot_bar_bss_rpss_auc", "show_panel", "model_list"}
        for probabilistic in (False, True):
            results = []
            for source in ("params.py", "params/config.in"):
                path = Path("apps/benchmarking/ROMP/momp") / source
                tree = ast.parse(path.read_text())
                assignments = [n for n in tree.body if isinstance(n, ast.Assign)
                               and any(isinstance(t, ast.Name) and t.id in flags for t in n.targets)]
                ns = {"config": SimpleNamespace(probabilistic=probabilistic)}
                exec(compile(ast.Module(body=assignments, type_ignores=[]), str(path), "exec"), ns)  # noqa: S102
                results.append({key: ns[key] for key in flags})
            self.assertEqual(*results)
            self.assertEqual(results[0]["plot_reliability"], probabilistic)

    def test_declared_mm_prevents_thousandfold_double_conversion(self):
        ns = {"warnings": warnings}
        functions("ai_weather/precipitation.py", {"precipitation_to_mm"}, ns)
        convert = ns["precipitation_to_mm"]
        mm = xr.DataArray([1.5, 25.0], attrs={"units": "mm"})
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            actual = convert(mm, 1000)
        self.assertEqual(captured, [])
        np.testing.assert_allclose(actual, mm)
        self.assertEqual(actual.attrs["units"], "mm")
        np.testing.assert_allclose(mm, [1.5, 25.0])
        np.testing.assert_allclose(convert(xr.DataArray([.025], attrs={"units": "m"}), 1000), [25.0])
        np.testing.assert_allclose(convert(xr.DataArray([.025]), 1000), [25.0])

    def test_processed_gencast_catalog_preserves_unlabeled_mm_values(self):
        tree = ast.parse(Path("ai_weather/config.py").read_text())
        catalog = next(n.value for n in tree.body if isinstance(n, ast.AnnAssign)
                       and isinstance(n.target, ast.Name) and n.target.id == "BENCHMARK_MODEL_CATALOG")
        entry = next(value for key, value in zip(catalog.keys, catalog.values, strict=True)
                     if isinstance(key, ast.Constant) and key.value == "GenCast")
        meta = {"unit_cvt": ast.literal_eval(next(
            value for key, value in zip(entry.keys, entry.values, strict=True)
            if isinstance(key, ast.Constant) and key.value == "unit_cvt"))}
        self.assertEqual(meta["unit_cvt"], 1.0)
        ns = {"warnings": warnings}
        functions("ai_weather/precipitation.py", {"precipitation_to_mm"}, ns)
        rainfall = xr.DataArray([-.5496635437011719, 267.7001953125, 2.716803789138794])
        with self.assertLogs("ai_weather.precipitation", level="INFO"):
            actual = ns["precipitation_to_mm"](rainfall, meta["unit_cvt"])
        np.testing.assert_allclose(actual, rainfall)
        self.assertEqual(actual.attrs["units"], "mm")

    def test_aifs_ensemble_mm_catalog_needs_no_override_warning(self):
        tree = ast.parse(Path("ai_weather/config.py").read_text())
        catalog = next(n.value for n in tree.body if isinstance(n, ast.AnnAssign)
                       and isinstance(n.target, ast.Name) and n.target.id == "BENCHMARK_MODEL_CATALOG")
        entry = next(value for key, value in zip(catalog.keys, catalog.values, strict=True)
                     if isinstance(key, ast.Constant) and key.value == "AIFS_ENS")
        factor = ast.literal_eval(next(value for key, value in zip(entry.keys, entry.values, strict=True)
                                      if isinstance(key, ast.Constant) and key.value == "unit_cvt"))
        self.assertEqual(factor, 1.0)
        ns = {"warnings": warnings}
        functions("ai_weather/precipitation.py", {"precipitation_to_mm"}, ns)
        rainfall = xr.DataArray([1.0, 26.2954], attrs={"units": "mm"})
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            actual = ns["precipitation_to_mm"](rainfall, factor)
        self.assertEqual(captured, [])
        np.testing.assert_allclose(actual, rainfall)

    def test_csv_disabled_keeps_summaries_for_panel_plots(self):
        ns = {"np": np, "pd": pd, "os": os, "get_target_bins": lambda *_: ["Days 1-5"],
              "tuple_to_str": lambda window: "-".join(map(str, window))}
        functions("apps/benchmarking/ROMP/momp/io/output.py", {"save_score_results"}, ns)
        bs = {"fair_brier_score": .2, "bin_fair_brier_scores": {"Days 1-5": .2}}
        auc = {"auc": .7, "bin_auc_scores": {"Days 1-5": .7}}
        score_results = {"BS": bs, "BS_ref": bs, "AUC": auc, "AUC_ref": auc,
                         "RPS": {"fair_rps": .3}, "skill_results": {
                             "fair_brier_skill_score": 0, "fair_rps_skill_score": 0,
                             "bin_fair_brier_skill_scores": {"Days 1-5": 0}}}
        with tempfile.TemporaryDirectory() as folder:
            binned, overall = ns["save_score_results"](
                score_results, model="test", verification_window=(1, 15),
                dir_out=folder, save_csv_score=False)
            self.assertEqual(list(Path(folder).iterdir()), [])
            self.assertEqual(binned["Bin"], ["Days 1-5"])
            self.assertEqual(overall["AUC"], .7)
