"""All Metrics includes every configured model and uses numeric lead ordering."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from ui.utils.metric_filters import (
    lead_window_sort_key,
    load_probabilistic_metric_table,
    metric_comparison_groups,
    metric_model_label,
    metric_model_options,
)


class MetricFilterTests(unittest.TestCase):
    def test_comparisons_separate_metric_families_and_keep_climatology_with_errors(self):
        models = ["AIFS Single", "FuXi", "AIFS Ensemble", "GenCast", "Traditional Climatology"]
        categories = dict(zip(models, ["Deterministic", "Deterministic", "Probabilistic",
                                      "Probabilistic", "Climatology"]))
        self.assertEqual(metric_comparison_groups(models, categories), [
            ("Onset error metrics", ["AIFS Single", "FuXi", "Traditional Climatology"]),
            ("Probabilistic skill metrics", ["AIFS Ensemble", "GenCast"]),
        ])
        self.assertEqual(metric_comparison_groups(["GenCast"], categories),
                         [("Probabilistic skill metrics", ["GenCast"])])
        self.assertEqual(metric_comparison_groups([], categories), [])

    def test_lead_windows_sort_numerically(self):
        windows = ["Days 16–30", "Days 2-7", "Days 31–45", "Days 1–15", "Unknown"]
        self.assertEqual(sorted(windows, key=lead_window_sort_key),
                         ["Days 1–15", "Days 2-7", "Days 16–30", "Days 31–45", "Unknown"])

    def test_all_configured_models_are_options_without_spatial_outputs(self):
        catalog = {"AIFS": {"model_name": "aifs_single", "probabilistic": False},
                   "AIFS_ENS": {"model_name": "aifs_ens", "probabilistic": True},
                   "GenCast": {"model_name": "gencast", "probabilistic": True}}
        models, categories = metric_model_options(catalog, ["AIFS Single", "Traditional Climatology"])
        self.assertEqual(models, ["AIFS Single", "AIFS Ensemble", "GenCast", "Traditional Climatology"])
        self.assertEqual(categories["AIFS Ensemble"], "Probabilistic")
        self.assertEqual(metric_model_label("AIFS Ensemble v2"), "AIFS Ensemble")

    def test_scalar_probabilistic_metrics_include_csv_only_models_and_preserve_missing(self):
        with tempfile.TemporaryDirectory() as folder:
            model_dir = Path(folder) / "aifs_ens"
            model_dir.mkdir()
            pd.DataFrame([{"Fair_Brier_Skill_Score": -.25, "Fair_RPS_Skill_Score": float("nan"),
                           "AUC": .577, "AUC_ref": .845}]).to_csv(model_dir / "overall_skill_scores_aifs_ens_16-30.csv", index=False)
            table, baselines, errors = load_probabilistic_metric_table(folder)
            self.assertEqual(errors, [])
            self.assertEqual(table[("Fair Brier Skill Score (%)", "Days 16–30", "AIFS Ensemble")], -25)
            self.assertEqual(table[("AUC", "Days 16–30", "AIFS Ensemble")], .577)
            self.assertIsNone(table[("Fair RPS Skill Score (%)", "Days 16–30", "AIFS Ensemble")])
            self.assertEqual(baselines[("AUC", "Days 16–30", "AIFS Ensemble")], .845)
            self.assertEqual(baselines[("Fair RPS Skill Score (%)", "Days 16–30", "AIFS Ensemble")], 0)

    def test_score_baselines_use_saved_references_or_skill_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            model_dir = Path(folder) / "gencast"
            model_dir.mkdir()
            source = model_dir / "overall_skill_scores_gencast_1-15.csv"
            pd.DataFrame([{"Fair_Brier_Score": .2, "Fair_Brier_Skill_Score": -.25,
                           "Fair_RPS": .3, "Fair_RPS_Skill_Score": .4,
                           "Reference_Fair_RPS": .5, "AUC": .8, "AUC_ref": .7}]).to_csv(source, index=False)
            _, baselines, errors = load_probabilistic_metric_table(folder)
            self.assertEqual(errors, [])
            self.assertAlmostEqual(baselines[("Fair Brier Score", "Days 1–15", "GenCast")], .16)
            self.assertEqual(baselines[("Fair RPS", "Days 1–15", "GenCast")], .5)
            data = pd.read_csv(source)
            data["Reference_Fair_Brier_Score"] = .18
            data["Fair_RPS_Skill_Score"] = 1.
            data = data.drop(columns="Reference_Fair_RPS")
            data.to_csv(source, index=False)
            _, baselines, _ = load_probabilistic_metric_table(folder)
            self.assertEqual(baselines[("Fair Brier Score", "Days 1–15", "GenCast")], .18)
            self.assertIsNone(baselines[("Fair RPS", "Days 1–15", "GenCast")])
