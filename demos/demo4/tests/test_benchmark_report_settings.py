"""Downloaded reports describe the saved experiment, independent of controls."""

import ast
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import xarray as xr

from ui.utils import benchmark_report


class BenchmarkReportSettingsTests(unittest.TestCase):
    def test_mixed_baselines_omit_shared_reference_only_for_conflicting_window(self):
        outputs = [
            {'model_label': 'AIFS Single', 'windows': ['Days 1-15'],
             'metadata': {'years_clim': '2007,2008', 'init_type': 'weekly'}},
            {'model_label': 'GraphCast', 'windows': ['Days 1-15'],
             'metadata': {'years_clim': '2000,2001', 'init_type': 'weekly'}},
            {'model_label': 'Traditional Climatology', 'windows': ['Days 1-15', 'Days 16-30'],
             'metadata': {'years_clim': '(2000, 2001)', 'init_type': 'weekly'}},
        ]
        filtered = benchmark_report.compatible_reference_outputs(outputs)
        self.assertEqual(filtered[-1]['windows'], ['Days 16-30'])
        self.assertEqual(filtered[0]['windows'], ['Days 1-15'])
        self.assertIn('different benchmark settings', filtered[0]['reference_warning'])
        self.assertEqual(outputs[-1]['windows'], ['Days 1-15', 'Days 16-30'])

    def test_equivalent_metadata_encodings_keep_shared_reference(self):
        outputs = [
            {'model_label': 'AIFS Single', 'windows': ['Days 1-15'],
             'metadata': {'years_clim': '2007,2008', 'init_days': '0,3',
                          'init_type': 'daily', 'date_filter_year': 2020}},
            {'model_label': 'Traditional Climatology', 'windows': ['Days 1-15'],
             'metadata': {'years_clim': '(2007, 2008)', 'init_days': '(0, 3)',
                          'init_type': 'daily', 'date_filter_year': 2024}},
        ]
        self.assertIs(benchmark_report.compatible_reference_outputs(outputs), outputs)

    def test_weekly_calendar_mismatch_suppresses_shared_reference(self):
        outputs = [
            {'model_label': name, 'windows': ['Days 1-15'],
             'metadata': {'init_type': 'weekly', 'date_filter_year': year}}
            for name, year in [('AIFS Single', 2020), ('Traditional Climatology', 2024)]
        ]
        filtered = benchmark_report.compatible_reference_outputs(outputs)
        self.assertEqual([row['model_label'] for row in filtered], ['AIFS Single'])

    def test_disabled_mask_is_a_recorded_setting_and_conflicts_with_active_mask(self):
        outputs = [
            {'model_label': name, 'windows': ['Days 1-15'], 'metadata': {'nc_mask': mask}}
            for name, mask in [('AIFS Single', 'None'), ('Traditional Climatology', '/mask.nc')]
        ]
        filtered = benchmark_report.compatible_reference_outputs(outputs)
        self.assertEqual([row['model_label'] for row in filtered], ['AIFS Single'])

    def test_threshold_variable_and_onset_start_must_match(self):
        for field, left, right in [('thresh_var', 'wet', 'dry'), ('mok', '5,1', '6,1')]:
            with self.subTest(field=field):
                outputs = [
                    {'model_label': name, 'windows': ['Days 1-15'], 'metadata': {field: value}}
                    for name, value in [('AIFS Single', left), ('Traditional Climatology', right)]
                ]
                filtered = benchmark_report.compatible_reference_outputs(outputs)
                self.assertEqual([row['model_label'] for row in filtered], ['AIFS Single'])

    def test_country_reports_do_not_mix_nested_country_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            peru = root / 'countries/Peru'
            peru.mkdir(parents=True)
            for folder, model in [(root, 'aifs_single'), (peru, 'graphcast')]:
                xr.Dataset({'mean_mae': ('lat', [1.])}).to_netcdf(
                    folder / f'spatial_metrics_{model}_1-15.nc')
            self.assertEqual([row['Model'] for row in benchmark_report.collect_evaluated_models(root)],
                             ['AIFS Single'])
            self.assertEqual([row['Model'] for row in benchmark_report.collect_evaluated_models(peru)],
                             ['GraphCast'])

    def test_real_discovery_omits_a_shared_climatology_from_a_different_experiment(self):
        from ui.pages.short_medium_benchmarking import discover_map_outputs

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for model, years in [('aifs_single', '2007,2008'), ('climatology', '2001,2002')]:
                xr.Dataset({'mean_mae': (('lat', 'lon'), [[1.]])},
                           coords={'lat': [8.], 'lon': [39.]},
                           attrs={'years_clim': years}).to_netcdf(
                               root / f'spatial_metrics_{model}_1-15.nc')
            outputs = discover_map_outputs(output_root=root)
        self.assertEqual([output['model_label'] for output in outputs], ['AIFS Single'])
        self.assertIn('different benchmark settings', outputs[0]['reference_warning'])

    def test_real_pdf_renders_saved_years_separately_from_current_controls(self):
        from fpdf import FPDF

        source = Path(__file__).resolve().parents[1] / 'ui/pages/short_medium_benchmarking.py'
        names = {'_pdf_safe_text', '_normalize_dataframe_for_pdf',
                 'BenchmarkReportPDF', 'generate_benchmark_pdf_report'}
        nodes = [node for node in ast.parse(source.read_text()).body
                 if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
        nodes.insert(0, ast.ImportFrom(module='__future__',
                                      names=[ast.alias(name='annotations')], level=0))
        namespace = {'FPDF': FPDF, 'pd': pd, 'np': np, 'Path': Path}
        exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),  # noqa: S102
                     str(source), 'exec'), namespace)
        settings = [{'Model': 'AIFS Ensemble', 'Window': 'Days 1-15',
                     'Reference calendar year': '2020', 'Baseline years': '2007-2024'}]
        pdf_class = namespace['BenchmarkReportPDF']
        original = pdf_class.key_value_lines
        items = []

        def capture(instance, values):
            items.extend(values)
            return original(instance, values)

        with patch.object(pdf_class, 'key_value_lines', capture):
            report = namespace['generate_benchmark_pdf_report']({
                'recorded_settings': settings,
                'shared_settings': {'date_filter_year': 1990, 'baseline_start_year': 1981}})
        self.assertTrue(report.startswith(b'%PDF-'))
        self.assertIn(('Reference calendar year', '2020'), items)
        self.assertIn(('Baseline years', '2007-2024'), items)

    def test_spatial_output_preserves_per_model_calendar_and_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for model, calendar, baseline in (
                ("aifs_single", 2024, "2007,2008,2009"),
                ("aifs_ens", 2020, "2001,2003,2004"),
            ):
                xr.Dataset({"mae": ("lat", [1.])}, attrs={
                    "date_filter_year": calendar, "years_clim": baseline,
                    "years": "2021,2022", "init_type": "weekly", "init_days": "0,3",
                }).to_netcdf(root / f"spatial_metrics_{model}_1-15.nc")
            rows = benchmark_report.collect_evaluated_models(root)
            settings = benchmark_report.collect_benchmark_settings(rows)
        by_model = {row["Model"]: row for row in settings}
        self.assertEqual(by_model["AIFS Single"]["Reference calendar year"], "2024")
        self.assertEqual(by_model["AIFS Single"]["Baseline years"], "2007-2009")
        self.assertEqual(by_model["AIFS Ensemble"]["Reference calendar year"], "2020")
        self.assertEqual(by_model["AIFS Ensemble"]["Baseline years"], "2001, 2003-2004")
        self.assertEqual(by_model["AIFS Ensemble"]["Evaluation years"], "2021-2022")

    def test_csv_only_results_use_recorded_years_and_keep_window_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "overall_skill_scores_aifs_ens_16-30.csv").write_text(
                'Fair_RPS,Reference_Calendar_Year,Requested_Reference_Years,Evaluation_Years,Init_Type\n'
                '0.2,2020,"(2000, 2001, 2002)","(2022, 2023)",weekly\n')
            settings = benchmark_report.collect_benchmark_settings(
                benchmark_report.collect_evaluated_models(root))
        self.assertEqual(len(settings), 1)
        self.assertEqual(settings[0]["Window"], "Days 16-30")
        self.assertEqual(settings[0]["Reference calendar year"], "2020")
        self.assertEqual(settings[0]["Baseline years"], "2000-2002")

    def test_legacy_and_unreadable_outputs_never_invent_years(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "spatial_metrics_aifs_single_1-15.nc"
            path.write_bytes(b"CDF\x01invalid fixture")
            settings = benchmark_report.collect_benchmark_settings([
                {"Model": "AIFS Single", "_output_paths": [str(path)]}])
        self.assertEqual(settings[0]["Reference calendar year"], "Not recorded")
        self.assertEqual(settings[0]["Baseline years"], "Not recorded")

    def test_calendar_is_not_applicable_to_daily_and_monthly_schedules(self):
        with tempfile.TemporaryDirectory() as directory:
            for mode in ("daily", "monthly"):
                with self.subTest(mode=mode):
                    path = Path(directory) / "spatial_metrics_aifs_single_1-15.nc"
                    xr.Dataset(attrs={"init_type": mode, "date_filter_year": 2024,
                                      "years_clim": "2007,2008"}).to_netcdf(path)
                    settings = benchmark_report.collect_benchmark_settings([
                        {"Model": "AIFS Single", "_output_paths": [str(path)]}])
                    self.assertEqual(settings[0]["Reference calendar year"], "Not applicable")
