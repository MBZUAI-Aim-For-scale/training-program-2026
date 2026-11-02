"""Undefined skill scores must not crash or distort probabilistic plots."""

import ast
import os
import tempfile
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


class SkillBarPlotTests(unittest.TestCase):
    def setUp(self):
        self.ns = {"np": np, "plt": plt, "os": os,
                   "tuple_to_str": lambda window: "-".join(map(str, window))}
        for source, name in (("io/dict.py", "extract_overall_dict"),
                             ("graphics/panel_bar_skill.py", "panel_bar_bss_rpss_auc")):
            path = Path("apps/benchmarking/ROMP/momp") / source
            tree = ast.parse(path.read_text())
            function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
            exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), self.ns)  # noqa: S102
        self.addCleanup(plt.close, "all")

    def plot(self, scores, folder):
        return self.ns["panel_bar_bss_rpss_auc"](scores, (16, 30), dir_fig=folder, show_panel=False)

    def test_reported_nan_rpss_preserves_valid_bss_and_auc(self):
        scores = {"aifs_ens": {"Fair_Brier_Skill_Score": -.5587460877562356,
                               "Fair_RPS_Skill_Score": np.nan, "AUC": .5774466255265703,
                               "AUC_ref": .8447001210981766}}
        with tempfile.TemporaryDirectory() as folder:
            fig, (ax, ax2) = self.plot(scores, folder)
            self.assertTrue(np.isfinite(ax.get_xlim()).all())
            self.assertLess(ax.get_xlim()[0], -55.8746)
            self.assertGreater(ax.get_xlim()[1], 0)
            self.assertLessEqual(ax2.get_xlim()[0], .57744)
            self.assertTrue(any("RPSS: N/A" in text.get_text() for text in ax.texts))
            self.assertTrue((Path(folder) / "panel_bar_BSS_RPSS_AUC_16-30.png").exists())
            plt.close(fig)

    def test_all_missing_scores_have_finite_fallback_limits(self):
        scores = {"model": {"Fair_Brier_Skill_Score": np.nan,
                             "Fair_RPS_Skill_Score": np.inf, "AUC": np.nan, "AUC_ref": np.inf}}
        with tempfile.TemporaryDirectory() as folder:
            fig, axes = self.plot(scores, folder)
            for ax in axes:
                self.assertTrue(np.isfinite(ax.get_xlim()).all())
                self.assertLess(*ax.get_xlim())
                self.assertTrue(all(np.isfinite(patch.get_width()) for patch in ax.patches))
            plt.close(fig)

    def test_large_positive_skill_and_all_model_rows_are_visible(self):
        scores = {f"model_{i}": {"Fair_Brier_Skill_Score": 2.0,
                                 "Fair_RPS_Skill_Score": -.5, "AUC": .45, "AUC_ref": .7}
                  for i in range(4)}
        with tempfile.TemporaryDirectory() as folder:
            fig, (ax, _) = self.plot(scores, folder)
            self.assertGreater(ax.get_xlim()[1], 200)
            self.assertGreater(ax.get_ylim()[1], 3.2)
            self.assertLess(ax.get_ylim()[0], -.2)
            plt.close(fig)
