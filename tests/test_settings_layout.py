"""Settings placement and metric comparison colors remain consistent."""

import ast
import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


class SettingsLayoutTests(unittest.TestCase):
    def test_advanced_controls_are_in_right_column_beside_masks(self):
        source = Path("ui/pages/short_medium_benchmarking.py").read_text()
        function = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)
                        and n.name == "render_shared_settings")
        script = '''import streamlit as st
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Any
config=SimpleNamespace(nc_mask='/configured/external/jjas_100mm_rainfall_mask_0p25.nc')
EXTERNAL_DATA_DIR=Path('/configured/external')
ONSET_REFERENCE_DIR=Path('/reference')
def section_header(title,*args): st.subheader(title)
''' + ast.get_source_segment(source, function) + '''
def advanced(): st.selectbox('Verification windows',['Days 1–15','Both windows'])
st.json(render_shared_settings(advanced_renderer=advanced))
'''
        app = AppTest.from_string(script).run(timeout=20)
        self.assertFalse(app.exception)
        locations = {}

        def walk(node, path=()):
            if getattr(node, "type", "") == "markdown":
                for label in ("MASKS AND BASELINE", "ADVANCED TEMPORAL & VERIFICATION SETTINGS"):
                    if label in node.value:
                        locations[label] = path
            for index, child in getattr(node, "children", {}).items():
                walk(child, path + (index,))

        walk(app.main)
        left, right = locations["MASKS AND BASELINE"], locations["ADVANCED TEMPORAL & VERIFICATION SETTINGS"]
        self.assertEqual(left[:-2], right[:-2])
        self.assertEqual((left[-2], right[-2]), (0, 1))
        self.assertTrue(any(box.label == "Verification windows" for box in app.selectbox))

    def test_summary_high_and_low_metric_directions_are_respected(self):
        path = Path("ui/pages/short_medium_benchmarking.py")
        source = path.read_text()
        function = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)
                        and n.name == "build_metrics_table_html")
        ns = {"is_metric_higher_better": lambda name: name == "AUC"}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), ns)  # noqa: S102
        for metric, baseline, better, worse in (("AUC", .5, .8, .3), ("Mean Absolute Error days", 5., 2., 8.)):
            with self.subTest(metric=metric):
                table = {(metric, "Days 1–15", "Traditional Climatology"): baseline,
                         (metric, "Days 1–15", "Better"): better, (metric, "Days 1–15", "Worse"): worse}
                html = ns["build_metrics_table_html"](table, ["Better", "Worse", "Traditional Climatology"], ["Days 1–15"])
                row = html.split(f"<tr><td>{metric}</td>", 1)[1]
                cells = row.split("</td>")
                self.assertIn("34,197,94", cells[0])
                self.assertIn("239,68,68", cells[1])
                self.assertIn("position:sticky", html)
