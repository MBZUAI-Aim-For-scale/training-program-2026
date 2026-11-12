"""PDF tables preserve scientific labels, row identity and readable pagination."""

import ast
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

from fpdf import FPDF
import numpy as np
import pandas as pd


class BenchmarkPDFTableTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = Path(__file__).resolve().parents[1] / 'ui/pages/short_medium_benchmarking.py'
        names = {'_pdf_safe_text', '_normalize_dataframe_for_pdf', 'BenchmarkReportPDF'}
        nodes = [node for node in ast.parse(source.read_text()).body
                 if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
        nodes.insert(0, ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0))
        namespace = {'FPDF': FPDF, 'np': np, 'pd': pd, 'Path': Path}
        exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),  # noqa: S102
                     str(source), 'exec'), namespace)
        cls.pdf_type = namespace['BenchmarkReportPDF']
        cls.normalize = staticmethod(namespace['_normalize_dataframe_for_pdf'])

    def document(self):
        pdf = self.pdf_type(orientation='L', unit='mm', format='A4')
        pdf.set_auto_page_break(auto=True, margin=18)
        pdf.add_page()
        return pdf

    def text(self, pdf):
        if not shutil.which('pdftotext'):
            self.skipTest('pdftotext is needed to inspect actual PDF table content')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'table.pdf'
            pdf.output(path)
            result = subprocess.run(['pdftotext', '-raw', str(path), '-'],
                                    check=True, capture_output=True, text=True)
        return result.stdout

    def test_duplicate_columns_preserve_both_values(self):
        result = self.normalize(pd.DataFrame([[.1, .9]], columns=['Score', 'Score']))
        self.assertEqual(result.columns.tolist(), ['Score', 'Score_2'])
        self.assertEqual(result.iloc[0].tolist(), ['0.100', '0.900'])

    def test_small_nonzero_scores_are_not_rendered_as_zero(self):
        result = self.normalize(pd.DataFrame({'Score': [0., .00001234, -.00001234, .1]}))
        self.assertEqual(result['Score'].tolist(), ['0.000', '1.23e-05', '-1.23e-05', '0.100'])

    def test_mathematical_comparison_symbols_keep_their_meaning(self):
        pdf = self.document()
        pdf.dataframe_table(pd.DataFrame({'Criterion': ['Rainfall ≤ 20 mm', 'Onset ≥ day 5'], 'Score': [1., -2.]}))
        output = self.text(pdf)
        self.assertIn('Rainfall <= 20 mm', output)
        self.assertIn('Onset >= day 5', output)

    def test_meaningful_row_indexes_become_visible_identifiers(self):
        source = pd.DataFrame({'Score': [.1, .2]}, index=pd.Index(['AIFS Single', 'FuXi'], name='Model'))
        result = self.normalize(source)
        self.assertEqual(result['Model'].tolist(), ['AIFS Single', 'FuXi'])
        self.assertEqual(result['Score'].tolist(), ['0.100', '0.200'])
        self.assertEqual(self.normalize(pd.DataFrame({'Score': [.1]})).columns.tolist(), ['Score'])

    def test_hierarchical_rows_and_columns_keep_their_labels(self):
        index = pd.MultiIndex.from_tuples([('AIFS Single', 'Days 1-15')], names=['Model', 'Window'])
        columns = pd.MultiIndex.from_tuples([('Onset', 'MAE'), ('Onset', 'Miss rate')])
        result = self.normalize(pd.DataFrame([[8.4, .572]], index=index, columns=columns))
        self.assertEqual(result.columns.tolist(), ['Model', 'Window', 'Onset / MAE', 'Onset / Miss rate'])
        self.assertEqual(result.iloc[0].tolist(), ['AIFS Single', 'Days 1-15', '8.400', '0.572'])

    def test_long_metric_and_filename_are_wrapped_without_losing_text(self):
        metric = 'Fair Ranked Probability Skill Score against leave-one-year-out climatology'
        filename = 'spatial_metrics_aifs_ensemble_reference_verification_2007_2024_16_30_country_box025.nc'
        pdf = self.document()
        pdf.dataframe_table(pd.DataFrame({'Model': ['AIFS Ensemble'], 'Metric': [metric], 'File': [filename],
                                          'Score': [-12.345]}))
        compact = re.sub(r'\s+', '', self.text(pdf))
        self.assertIn(re.sub(r'\s+', '', metric), compact)
        self.assertIn(filename, compact)
        self.assertIn('-12.345', compact)

    def test_wide_table_continuations_repeat_row_identifiers(self):
        frame = pd.DataFrame({'Model': ['AIFS Ensemble'], 'Lead Window': ['Days 1-15'],
                              'Metric': ['Mean Absolute Error']})
        for index in range(18):
            frame[f'Scientific score {index:02d}'] = index + .123
        pdf = self.document()
        pdf.dataframe_table(frame)
        text = self.text(pdf)
        self.assertGreaterEqual(text.count('AIFS Ensemble'), 2)
        self.assertGreaterEqual(text.count('Days 1-15'), 2)
        self.assertGreaterEqual(text.count('Mean Absolute Error'), 2)
        self.assertIn('17.123', text)

    def test_vertical_pagination_repeats_complete_column_headers(self):
        metric = 'Fair Ranked Probability Skill Score'
        frame = pd.DataFrame({'Model': [f'Model {index:03d}' for index in range(80)],
                              'Metric': [metric] * 80, 'Score': np.arange(80) / 100})
        pdf = self.document()
        pdf.dataframe_table(frame)
        pages = self.text(pdf).split('\f')
        populated = [page for page in pages if 'Model 0' in page]
        self.assertGreaterEqual(len(populated), 2)
        for page in populated:
            self.assertIn('Model', page)
            self.assertIn('Metric', page)
            self.assertIn('Score', page)
        self.assertIn('Model 079', pages[-2])

    def test_single_tall_row_can_continue_without_truncation(self):
        text = '\n'.join(f'Observation detail {index:03d}' for index in range(100))
        pdf = self.document()
        pdf.dataframe_table(pd.DataFrame({'Model': ['AIFS Ensemble'], 'Details': [text]}))
        output = self.text(pdf)
        self.assertIn('Observation detail 000', output)
        self.assertIn('Observation detail 099', output)
        self.assertGreater(pdf.page_no(), 1)

    def test_section_heading_stays_with_its_first_paragraph(self):
        pdf = self.document()
        pdf.set_y(pdf.h - pdf.b_margin - 20)
        pdf.section_title('Notes')
        pdf.paragraph('A short closing note.')
        pages = self.text(pdf).split('\f')
        self.assertIn('Notes', pages[-2])
        self.assertIn('A short closing note.', pages[-2])
