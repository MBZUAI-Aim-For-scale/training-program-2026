"""A scientific report summarizes saved experiments without dumping raw metadata."""

import ast
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from fpdf import FPDF
import numpy as np
import pandas as pd


class ReportContentSummaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = Path(__file__).resolve().parents[1] / 'ui/pages/short_medium_benchmarking.py'
        names = {'_pdf_safe_text', '_normalize_dataframe_for_pdf', 'BenchmarkReportPDF', 'generate_benchmark_pdf_report'}
        nodes = [node for node in ast.parse(source.read_text()).body
                 if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
        nodes.insert(0, ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0))
        namespace = {'FPDF': FPDF, 'pd': pd, 'np': np, 'Path': Path}
        exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),  # noqa: S102
                     str(source), 'exec'), namespace)
        cls.generate = staticmethod(namespace['generate_benchmark_pdf_report'])

    def records(self):
        baseline = range(2007, 2025)
        membership = {year: ','.join(str(member) for member in baseline if member != year)
                      for year in range(2000, 2025)}
        base = {'Window': 'Days 1-15', 'Evaluation years': '2003-2024',
                'Actual evaluation years': 'Not recorded', 'Baseline years': '2007-2024',
                'Reference calendar year': '2024', 'Initialization schedule': 'weekly',
                'Initialization days': '0,3', 'Evaluation start': '2003,5,1',
                'Evaluation end': '2024,7,29', 'Baseline forecast': 'climatology',
                'Reference policy': 'leave_one_year_out', 'Reference membership': 'Not recorded',
                'Reference years by verification year': 'Not recorded',
                'Country': 'Ethiopia', 'Ground truth': 'CHIRPS'}
        records = [{**base, 'Model': name, 'Output': f'spatial_metrics_{name}_1-15.nc'}
                   for name in ('aifs_ens', 'aifs_single', 'fuxi', 'gencast', 'graphcast')]
        records.append({**base, 'Model': 'aifs_ens', 'Output': 'overall_skill_scores_aifs_ens_1-15.csv',
                        'Evaluation years': '2000-2024', 'Actual evaluation years': '2000-2024',
                        'Evaluation start': '(2000, 5, 1)', 'Evaluation end': '(2024, 12, 31)',
                        'Initialization days': '(0, 3)',
                        'Reference membership': 'held_out_verification_year;leave_one_year_out',
                        'Reference years by verification year': str(membership)})
        records.append({**base, 'Model': 'gencast', 'Output': 'overall_skill_scores_gencast_1-15.csv',
                        'Actual evaluation years': '2003-2024',
                        'Initialization days': '(0, 3)',
                        'Reference membership': 'held_out_verification_year;leave_one_year_out',
                        'Reference years by verification year': str(membership)})
        return records

    def test_pdf_is_compact_readable_and_preserves_the_mixed_period_warning(self):
        if not shutil.which('pdftotext'):
            self.skipTest('pdftotext is needed for report content verification')
        records = self.records()
        original = [dict(record) for record in records]
        report = self.generate({
            'recorded_settings': records,
            'active_rows': [{'Model': model, 'Category': 'Probabilistic' if model in {'aifs_ens', 'gencast'}
                             else 'Deterministic'} for model in ('aifs_ens', 'aifs_single', 'fuxi', 'gencast', 'graphcast')],
            'ground_truth': 'IMERG', 'benchmark_coverage': 'Kenya',
            'shared_settings': {'baseline_start_year': 1981},
            'model_count_basis': 'stored_outputs',
        })
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'report.pdf'
            path.write_bytes(report)
            result = subprocess.run(['pdftotext', '-raw', str(path), '-'], check=True,
                                    capture_output=True, text=True)
        text = result.stdout
        self.assertLessEqual(text.count('\f'), 2)
        self.assertIn('AIFS Ensemble', text)
        self.assertIn('2000-2024', text)
        self.assertIn('2003-2024', text)
        self.assertIn('CHIRPS', text)
        self.assertIn('Ethiopia', text)
        self.assertNotIn('IMERG', text)
        self.assertNotIn('Kenya', text)
        self.assertNotIn('Reference years by verification year', text)
        self.assertNotIn("{2000:", text)
        self.assertNotIn('held_out_verification_year', text)
        self.assertNotIn('leave_one_year_out', text)
        self.assertNotIn('overall_skill_scores_aifs_ens_1-15.csv', text)
        self.assertNotIn('Current Dashboard Settings', text)
        self.assertNotIn('(2000, 5, 1)', text)
        self.assertEqual(records, original)
