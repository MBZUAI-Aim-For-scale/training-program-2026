"""Metric tables display saved model/window columns without inventing empty ones."""

import ast
from pathlib import Path
import re
import unittest

import numpy as np


class MetricTableAvailabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / 'ui/pages/short_medium_benchmarking.py'
        function = next(node for node in ast.parse(path.read_text()).body
                        if isinstance(node, ast.FunctionDef) and node.name == 'build_metrics_table_html')
        namespace = {'np': np, 'is_metric_higher_better': lambda metric: 'AUC' in metric or 'Skill' in metric}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)  # noqa: S102
        cls.render = staticmethod(namespace['build_metrics_table_html'])

    def test_absent_window_and_model_are_removed(self):
        html = self.render({('Mean Absolute Error days', 'Days 1–15', 'AIFS Single'): 8.4},
                           ['AIFS Single', 'FuXi'], ['Days 1–15', 'Days 16–30'])
        self.assertIn("<th colspan='1'>Days 1–15</th>", html)
        self.assertNotIn('Days 16–30', html)
        self.assertNotIn('<th>FuXi</th>', html)
        self.assertNotIn('<td>-</td>', html)

    def test_each_window_has_only_its_available_models(self):
        html = self.render({('AUC', 'Days 1–15', 'AIFS Ensemble'): .72,
                            ('AUC', 'Days 16–30', 'GenCast'): .65},
                           ['AIFS Ensemble', 'GenCast'], ['Days 1–15', 'Days 16–30'])
        self.assertEqual(html.count("colspan='1'"), 2)
        self.assertEqual(html.count('<th>AIFS Ensemble</th>'), 1)
        self.assertEqual(html.count('<th>GenCast</th>'), 1)
        body = html.split('<tbody>')[1].split('</tbody>')[0]
        self.assertEqual(len(re.findall(r'<td(?:\s|>)', body)), 3)

    def test_recorded_undefined_score_is_kept_as_missing(self):
        html = self.render({('Fair Brier Score', 'Days 1–15', 'AIFS Ensemble'): .1,
                            ('Fair Brier Skill Score (%)', 'Days 1–15', 'AIFS Ensemble'): None},
                           ['AIFS Ensemble'], ['Days 1–15'])
        self.assertIn('Fair Brier Skill Score (%)', html)
        self.assertIn('<td>-</td>', html)

    def test_hidden_reference_values_do_not_create_rows_without_selected_model_scores(self):
        html = self.render({('AUC', 'Days 1–15', 'AIFS Ensemble'): .72,
                            ('Mean Absolute Error days', 'Days 1–15', 'Traditional Climatology'): 14.9},
                           ['AIFS Ensemble'], ['Days 1–15'])
        self.assertIn('AUC', html)
        self.assertNotIn('<td>Mean Absolute Error days</td>', html)

    def test_empty_selection_has_no_empty_table(self):
        self.assertEqual(self.render({}, ['AIFS Single'], ['Days 1–15', 'Days 16–30']), '')

    def test_all_nonfinite_model_window_column_is_removed(self):
        html = self.render({('AUC', 'Days 1–15', 'AIFS Ensemble'): .72,
                            ('AUC', 'Days 16–30', 'AIFS Ensemble'): np.nan},
                           ['AIFS Ensemble'], ['Days 1–15', 'Days 16–30'])
        self.assertIn('Days 1–15', html)
        self.assertNotIn('Days 16–30', html)
