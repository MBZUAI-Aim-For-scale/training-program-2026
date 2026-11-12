"""One stale model must not disable another model's shared-reference comparison."""

import ast
from pathlib import Path
import re
import sys
import unittest
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

SOURCE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE_ROOT))
from ui.utils.benchmark_report import reference_comparison_allowed


class ReferenceComparisonUITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = SOURCE_ROOT / 'ui/pages/short_medium_benchmarking.py'
        cls.tree = ast.parse(cls.source.read_text())
        cls.page = next(node for node in cls.tree.body
                        if isinstance(node, ast.FunctionDef) and node.name == 'page')

    def execute(self, nodes, namespace):
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(self.source), 'exec'), namespace)  # noqa: S102

    def test_tables_keep_matching_reference_colors_and_disable_only_stale_fallback(self):
        window = 'Days 1–15'
        mae = 'Mean Absolute Error days'
        table = {(mae, window, 'Fresh'): 5., (mae, window, 'Stale'): 1.,
                 (mae, window, 'Traditional Climatology'): 8.,
                 ('Fair RPS', window, 'Stale'): .2}
        nodes = [node for node in ast.walk(self.page) if (
            isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id in
                {'incompatible_spatial_comparisons', 'comparison_baselines'} for target in node.targets)
        ) or (
            isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and isinstance(node.value.func.value, ast.Name)
            and node.value.func.value.id == 'comparison_baselines'
            and node.value.func.attr == 'update'
        )]
        nodes.sort(key=lambda node: node.lineno)
        namespace = {
            'tbl': table, 'metric_model_label': lambda label: label,
            'map_outputs': [
                {'model_label': 'Fresh', 'windows': [window]},
                {'model_label': 'Stale', 'windows': [window],
                 'reference_incompatible_windows': ['Days 1-15']},
            ],
            'probability_baselines': {('Fair RPS', window, 'Stale'): .4},
            'np': np, 'is_metric_higher_better': lambda metric: False,
        }
        self.execute(nodes, namespace)
        baselines = namespace['comparison_baselines']
        self.assertIsNone(baselines[(mae, window, 'Stale')])
        self.assertNotIn((mae, window, 'Fresh'), baselines)
        self.assertEqual(baselines[('Fair RPS', window, 'Stale')], .4)
        function = next(node for node in self.tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == 'build_metrics_table_html')
        self.execute([function], namespace)
        rendered = namespace['build_metrics_table_html'](table, ['Fresh', 'Stale'], [window], baselines)
        fresh_style = re.search(r"<td style='([^']*)'>5\.0</td>", rendered).group(1)
        stale_style = re.search(r"<td style='([^']*)'>1\.0</td>", rendered).group(1)
        paired_style = re.search(r"<td style='([^']*)'>0\.200</td>", rendered).group(1)
        self.assertIn('34,197,94', fresh_style)
        self.assertNotIn('background-color', stale_style)
        self.assertIn('34,197,94', paired_style)

    def test_distribution_fallback_is_per_model_and_preserves_paired_references(self):
        assignment = next(node for node in ast.walk(self.page) if isinstance(node, ast.Assign)
                          and any(isinstance(target, ast.Name) and target.id == 'base_v'
                                  for target in node.targets))
        fallback = next(node for node in ast.walk(self.page) if isinstance(node, ast.If)
                        and isinstance(node.test, ast.BoolOp)
                        and any(isinstance(child, ast.Name) and child.id == 'base_v'
                                for child in ast.walk(node.test)))
        for model, paired, expected in [('Fresh', np.nan, 8.), ('Stale', np.nan, None),
                                         ('Stale', .4, .4)]:
            with self.subTest(model=model, paired=paired):
                namespace = {'row': {'Reference Mean': paired}, 'stat': 'Mean', 'np': np,
                             'baseline_vals': {('days 1-15', 'mae', 'mean'): 8.},
                             'win_k': 'days 1-15', 'met_k': 'mae', 'stat_k': 'mean',
                             'model_label': model, 'window_label': 'Days 1–15',
                             'incompatible_spatial_comparisons': {('Stale', 'Days 1-15')}}
                self.execute([assignment, fallback], namespace)
                self.assertEqual(namespace['base_v'], expected)

    def test_stale_model_does_not_set_matching_models_reference_color_intensity(self):
        nodes = [node for node in ast.walk(self.page) if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id in {'comparable', 'model_vals'}
                         for target in node.targets)]
        nodes.sort(key=lambda node: node.lineno)
        namespace = {
            'pd': pd, 'g': pd.DataFrame({'Model': ['Fresh', 'Stale', 'Traditional Climatology'],
                                         'Mean': [5., 100., 8.]}),
            'clim_mask': pd.Series([False, False, True]), 'stat': 'Mean', 'window': 'Days 1–15',
            'incompatible_spatial_comparisons': {('Stale', 'Days 1-15')},
        }
        self.execute(nodes, namespace)
        self.assertEqual(namespace['model_vals'].tolist(), [5.])

    def test_difference_and_swipe_guard_only_the_selected_reference_pair(self):
        reference = {'model_label': 'Traditional Climatology', 'windows': ['Days 1–15']}
        fresh = {'model_label': 'Fresh', 'windows': ['Days 1–15']}
        stale = {'model_label': 'Stale', 'windows': ['Days 1–15'],
                 'reference_incompatible_windows': ['Days 1–15']}
        for candidate_name, candidate_window in [('cout', 'cw'), ('rout', 'rw')]:
            guard = next(node for node in ast.walk(self.page) if isinstance(node, ast.If)
                         and isinstance(node.test, ast.BoolOp)
                         and isinstance(node.test.values[0], ast.Name)
                         and node.test.values[0].id == candidate_name
                         and any(isinstance(child, ast.Name) and child.id == 'reference_comparison_allowed'
                                 for child in ast.walk(node.test)))
            for left, right, allowed in [(fresh, reference, True), (stale, reference, False),
                                         (stale, fresh, True)]:
                with self.subTest(view=candidate_name, left=left['model_label'], right=right['model_label']):
                    loader = MagicMock(return_value=None)
                    warning = MagicMock()
                    namespace = {
                        'sel_out': left, 'wl': 'Days 1–15', candidate_window: 'Days 1–15',
                        candidate_name: {**right, 'path': 'reference.nc'},
                        'reference_comparison_allowed': reference_comparison_allowed,
                        'load_map_da_cached': loader, 'info_box': warning,
                        'ml': 'MAE', 'ual': '', 'uan': '', 'benchmark_coverage': 'Ethiopia',
                    }
                    self.execute([guard], namespace)
                    self.assertEqual(loader.called, allowed)
                    self.assertEqual(warning.called, not allowed)
