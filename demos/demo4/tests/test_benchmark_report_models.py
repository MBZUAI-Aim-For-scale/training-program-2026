"""Fresh report sessions count persisted evaluations rather than selections."""

import ast
from datetime import datetime
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from ui.utils.benchmark_report import (
    collect_benchmark_settings,
    collect_evaluated_models,
)


class BenchmarkReportModelTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)

    def artifact(self, name, contents=None):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if contents is None:
            contents = (b'CDF\x01fixture metadata' if path.suffix == '.nc'
                        else b'Fair_Brier_Score,Fair_RPS,AUC\n0.1,0.2,0.8\n')
        path.write_bytes(contents)
        return path

    def test_new_session_counts_existing_models_across_both_output_types(self):
        self.artifact('aifs_single/spatial_metrics_aifs_single_1-15.nc')
        self.artifact('aifs_ens/overall_skill_scores_aifs_ens_1-15.csv')
        rows = collect_evaluated_models(self.root)
        self.assertEqual([row['Model'] for row in rows], ['AIFS Ensemble', 'AIFS Single'])
        self.assertEqual({row['Category'] for row in rows}, {'Probabilistic', 'Deterministic'})
        self.assertTrue(all(row['Type'] == 'Stored benchmark results' for row in rows))
        self.assertTrue(all(row['Date Range'] == 'Not recorded' for row in rows))

    def test_windows_and_model_aliases_do_not_duplicate_model_count(self):
        self.artifact('aifs/spatial_metrics_aifs_1-15.nc')
        self.artifact('aifs_single/spatial_metrics_aifs_single_16-30.nc')
        self.artifact('aifs_single/overall_skill_scores_aifs_single_1-15.csv')
        rows = collect_evaluated_models(self.root)
        self.assertEqual(len(rows), 1)
        self.assertEqual(len(rows[0]['_output_paths']), 3)
        self.assertEqual(rows[0]['_output_windows'], ['Days 1–15', 'Days 16–30'])

    def test_catalog_and_current_inventory_do_not_count_models_without_outputs(self):
        self.artifact('aifs_single/spatial_metrics_aifs_single_1-15.nc')
        catalog = {'AIFS': {'model_name': 'aifs_single', 'probabilistic': False},
                   'GenCast': {'model_name': 'gencast', 'probabilistic': True}}
        inventory = [{'_exec_key': 'GenCast', '_model_name': 'gencast',
                      'Model': 'GenCast', 'Category': 'Probabilistic'}]
        rows = collect_evaluated_models(self.root, catalog, inventory)
        self.assertEqual([row['_exec_key'] for row in rows], ['AIFS'])

    def test_baselines_custom_reference_and_figures_are_excluded(self):
        for name in ('climatology', 'traditional_climatology', 'baseline', 'custom_reference'):
            self.artifact(f'{name}/spatial_metrics_{name}_1-15.nc')
        self.artifact('fuxi/figure.png', b'figure')
        self.artifact('gencast/onset_dates_gencast_1-15.nc')
        self.assertEqual(collect_evaluated_models(self.root, reference_models=('custom_reference',)), [])

    def test_empty_invalid_and_header_only_files_are_excluded(self):
        self.artifact('spatial_metrics_aifs_single_1-15.nc', b'')
        self.artifact('spatial_metrics_fuxi_1-15.nc', b'not netcdf')
        self.artifact('overall_skill_scores_gencast_1-15.csv', b'Fair_RPS,AUC\n')
        self.artifact('overall_skill_scores_graphcast_1-15.csv', b'temperature\n280\n')
        self.assertEqual(collect_evaluated_models(self.root), [])

    def test_inventory_labels_and_legacy_csv_scores_remain_supported(self):
        self.artifact('custom/overall_skill_scores_custom_1-15.csv', b'Metric,Score\nAUC,0.8\n')
        inventory = [{'_exec_key': 'Custom', '_model_name': 'custom', 'Model': 'Local Ensemble',
                      'Category': 'Probabilistic', 'Date Range': '2019 to 2024', 'Coverage': 'Ethiopia'}]
        rows = collect_evaluated_models(self.root, inventory_rows=inventory)
        self.assertEqual(rows[0]['Model'], 'Local Ensemble')
        self.assertEqual(rows[0]['Category'], 'Probabilistic')
        self.assertEqual(rows[0]['Coverage'], 'Not recorded')

    def test_missing_directory_has_no_evaluated_models(self):
        self.assertEqual(collect_evaluated_models(self.root / 'missing'), [])

    def test_actual_report_summary_and_pdf_context_use_persisted_models_in_a_new_session(self):
        self.artifact('aifs_single/spatial_metrics_aifs_single_1-15.nc')
        self.artifact('aifs_single/spatial_metrics_aifs_single_16-30.nc')
        self.artifact('aifs_ens/overall_skill_scores_aifs_ens_1-15.csv')
        self.artifact('climatology/spatial_metrics_climatology_1-15.nc')
        source = Path('ui/pages/short_medium_benchmarking.py')
        tree = ast.parse(source.read_text())
        page = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == 'page')

        def assigns(node, name):
            return isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == name for target in node.targets)

        summary_body = next(node.body for node in ast.walk(page)
                            if isinstance(node, (ast.If, ast.With))
                            and any(assigns(child, 'report_rows') for child in node.body))
        start = next(index for index, node in enumerate(summary_body) if assigns(node, 'report_rows'))
        end = next(index for index, node in enumerate(summary_body[start:], start)
                   if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                   and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == 'metric'
                   and node.value.args and isinstance(node.value.args[0], ast.Constant)
                   and node.value.args[0].value == 'Baseline')
        st = MagicMock()
        columns = [MagicMock() for _ in range(4)]
        st.columns.return_value = columns
        st.session_state = {}
        namespace = {
            'st': st, 'datetime': datetime, 'ROMP_DEMO_OUT_DIR': self.root,
            'collect_evaluated_models': collect_evaluated_models,
            'collect_benchmark_settings': collect_benchmark_settings,
            'CONFIG_BENCHMARK_MODEL_CATALOG': {}, 'model_registry': [], 'active_rows': [],
            'ground_truth': 'CHIRPS', 'benchmark_coverage': 'Ethiopia',
            'forecast_window': '1-15 days', 'shared_settings': {'baseline_forecast': 'climatology'},
            'map_outputs': [], 'scores_df': pd.DataFrame(), 'distribution_stats': pd.DataFrame(),
            'probability_scores': pd.DataFrame(), 'table_errors': [],
            'report_figures': {},
            'reference_warnings': [],
        }
        exec(compile(ast.Module(body=summary_body[start:end + 1], type_ignores=[]),
                     str(source), 'exec'), namespace)
        columns[0].metric.assert_called_once_with('Models Evaluated', 2)
        self.assertEqual(namespace['active_rows'], [])

        context_assignment = next(node for node in ast.walk(page) if assigns(node, 'report_context'))
        exec(compile(ast.Module(body=[context_assignment], type_ignores=[]),
                     str(source), 'exec'), namespace)
        context = namespace['report_context']
        self.assertIs(context['active_rows'], namespace['report_rows'])
        self.assertEqual(context['model_count_basis'], 'stored_outputs')
        self.assertEqual({row['Model'] for row in context['active_rows']},
                         {'AIFS Single', 'AIFS Ensemble'})

        # Exercise the real PDF implementation without importing the page's
        # unrelated scientific/data-discovery modules.
        from fpdf import FPDF
        names = {'_pdf_safe_text', '_normalize_dataframe_for_pdf',
                 'BenchmarkReportPDF', 'generate_benchmark_pdf_report'}
        nodes = [node for node in tree.body
                 if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
        nodes.insert(0, ast.ImportFrom(module='__future__',
                                      names=[ast.alias(name='annotations')], level=0))
        namespace.update(FPDF=FPDF, pd=pd, np=np, Path=Path)
        exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                     str(source), 'exec'), namespace)
        pdf_class = namespace['BenchmarkReportPDF']
        original = pdf_class.key_value_lines
        summary = []

        def capture(instance, items):
            summary.extend(items)
            return original(instance, items)

        with patch.object(pdf_class, 'key_value_lines', capture):
            report = namespace['generate_benchmark_pdf_report'](context)
        self.assertTrue(report.startswith(b'%PDF-'))
        self.assertTrue(report.rstrip().endswith(b'%%EOF'))
        self.assertIn(('Models Evaluated', 2), summary)
        self.assertIn(('Deterministic Models', 1), summary)
        self.assertIn(('Probabilistic Models', 1), summary)
