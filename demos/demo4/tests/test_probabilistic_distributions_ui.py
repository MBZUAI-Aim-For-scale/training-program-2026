"""Saved probability score grids provide spatial statistics and comparison colors."""

import ast
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import xarray as xr
from streamlit.testing.v1 import AppTest

from ui.utils.metric_filters import load_probabilistic_distribution_stats


class ProbabilisticDistributionTests(unittest.TestCase):
    def save_grid(self, root, model="aifs_ens", values=None):
        folder = Path(root) / model
        folder.mkdir(exist_ok=True)
        values = values or {
            "fair_brier_score": [[0., .25], [.5, np.nan]],
            "fair_brier_skill_score": [[-.5, 0.], [.5, np.nan]],
            "fair_rps": [[.1, .2], [.3, np.nan]],
            "fair_rps_skill_score": [[-.2, 0.], [.2, np.nan]],
            "auc": [[.6, .7], [.8, np.nan]],
            "reference_auc": [[.5, .65], [.9, .1]],
            "reference_fair_brier_score": [[.1, .3], [.6, .9]],
            "reference_fair_rps": [[.2, .3], [.4, .8]],
        }
        target = folder / f"probabilistic_metrics_{model}_1-15.nc"
        xr.Dataset({name: (("lat", "lon"), data) for name, data in values.items()},
                   coords={"lat": [8., 9.], "lon": [38., 39.]}).to_netcdf(target)
        return target

    def test_grid_statistics_use_finite_cells_and_scale_skill_to_percent(self):
        with tempfile.TemporaryDirectory() as root:
            self.save_grid(root)
            frame, errors = load_probabilistic_distribution_stats(root)
        self.assertEqual(errors, [])
        self.assertEqual(set(frame.Model), {"AIFS Ensemble"})
        by_metric = frame.set_index("Metric")
        brier = by_metric.loc["Fair Brier Score"]
        self.assertEqual(brier["Mean"], .25)
        self.assertEqual(brier["Median"], .25)
        self.assertAlmostEqual(brier["P90"], .45)
        self.assertAlmostEqual(brier["Reference Mean"], 1 / 3)
        skill = by_metric.loc["Fair Brier Skill Score (%)"]
        self.assertEqual(skill["Min"], -50.)
        self.assertEqual(skill["Max"], 50.)
        self.assertEqual(skill["Reference Mean"], 0.)
        self.assertEqual(skill["unit"], "%")
        self.assertAlmostEqual(by_metric.loc["AUC", "Reference Mean"], (.5 + .65 + .9) / 3)

    def test_csv_summaries_and_scalar_netcdf_are_not_distributions(self):
        with tempfile.TemporaryDirectory() as root:
            folder = Path(root) / "aifs_ens"
            folder.mkdir()
            pd.DataFrame([{"Fair_Brier_Score": .2}]).to_csv(
                folder / "overall_skill_scores_aifs_ens_1-15.csv", index=False)
            frame, errors = load_probabilistic_distribution_stats(root)
            self.assertTrue(frame.empty)
            self.assertFalse(errors)
            xr.Dataset({"fair_brier_score": .2}).to_netcdf(
                folder / "probabilistic_metrics_aifs_ens_1-15.nc")
            frame, errors = load_probabilistic_distribution_stats(root)
            self.assertTrue(frame.empty)
            self.assertIn("latitude/longitude score grids", errors[0])

    def test_selected_models_and_windows_do_not_read_other_grids(self):
        with tempfile.TemporaryDirectory() as root:
            self.save_grid(root)
            corrupt = Path(root) / "gencast"
            corrupt.mkdir()
            (corrupt / "probabilistic_metrics_gencast_1-15.nc").write_text("invalid")
            frame, errors = load_probabilistic_distribution_stats(
                root, models=["AIFS Ensemble"], windows=["Days 1–15"])
            self.assertFalse(errors)
            self.assertFalse(frame.empty)
            frame, errors = load_probabilistic_distribution_stats(root, windows=["Days 16–30"])
            self.assertTrue(frame.empty)
            self.assertFalse(errors)

    def test_probability_category_renders_gridded_cards_with_score_units(self):
        source = Path("ui/pages/short_medium_benchmarking.py").read_text()
        tree = ast.parse(source)
        definitions = "\n\n".join(ast.get_source_segment(source, node) for node in tree.body
                                  if isinstance(node, ast.FunctionDef) and node.name in {
                                      "build_metrics_table_html", "_distribution_cell_style"})
        page = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "page")
        metrics_tab = next(node for node in page.body if isinstance(node, ast.If)
                           and ast.unparse(node.test) == "res_tabs[1].open")
        setup = '''
from __future__ import annotations
import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path
from types import SimpleNamespace
class OpenTab:
    open = True
    def __enter__(self): return self
    def __exit__(self, *args): return False
res_tabs = [None, OpenTab()]
config = SimpleNamespace(ref_model='climatology')
CONFIG_BENCHMARK_MODEL_CATALOG = {
    'AIFS_ENS': {'model_name': 'aifs_ens', 'probabilistic': True}}
ROMP_DEMO_OUT_DIR = Path(st.session_state['test_output_root'])
map_outputs = []
def compute_metric_distribution_stats(): raise AssertionError('No onset grids should be read')
def is_metric_higher_better(metric): return 'Skill Score' in metric or 'AUC' in metric
def info_box(text, color): st.info(text)
def section_header(text, color): st.subheader(text)
'''
        with tempfile.TemporaryDirectory() as root:
            self.save_grid(root)
            pd.DataFrame([{"Fair_Brier_Score": .25}]).to_csv(
                Path(root) / "aifs_ens/overall_skill_scores_aifs_ens_1-15.csv", index=False)
            app = AppTest.from_string(setup + definitions + "\n" + ast.get_source_segment(source, metrics_tab))
            app.session_state["test_output_root"] = root
            app.session_state["metrics_model_category"] = "Probabilistic"
            app.run()
        self.assertFalse(app.exception)
        cards = next(item.value for item in app.markdown if "AIFS_ENSEMBLE - " in item.value)
        self.assertIn("Per-grid probabilistic scores", cards)
        self.assertIn("Fair Brier Skill Score (%)", cards)
        self.assertIn("34,197,94", cards)
        self.assertIn("239,68,68", cards)
        self.assertIn("<b>0.250</b>", cards)
        self.assertNotIn("day tolerance", cards)
        self.assertNotIn("0.2 d", cards)
        self.assertTrue(any("Rerun Benchmark" in item.value for item in app.info))
