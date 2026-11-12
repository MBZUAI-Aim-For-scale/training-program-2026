"""PDF probability tables come directly from saved CSVs and score grids."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import xarray as xr

from ui.utils.benchmark_report import collect_pdf_probability_tables


class PDFProbabilityTableTests(unittest.TestCase):
    def write_scores(self, root, model='aifs_ens', values=None):
        folder = root / model
        folder.mkdir(parents=True, exist_ok=True)
        source = folder / f'overall_skill_scores_{model}_1-15.csv'
        pd.DataFrame([values if values is not None else {
            'Fair_Brier_Score': .113, 'Fair_Brier_Skill_Score': -.02526,
            'Fair_RPS': .570, 'Fair_RPS_Skill_Score': .12240,
            'AUC': .720, 'AUC_ref': .695,
        }]).to_csv(source, index=False)
        return source

    def test_csv_only_model_provides_saved_scores_and_percentage_units(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_scores(root)
            scores, distributions, errors = collect_pdf_probability_tables(root)
        self.assertEqual(errors, [])
        self.assertTrue(distributions.empty)
        self.assertEqual(list(scores.columns), ['Model', 'Lead Window', 'Metric', 'Score'])
        self.assertEqual(set(scores.Model), {'AIFS Ensemble'})
        self.assertEqual(set(scores['Lead Window']), {'Days 1–15'})
        by_metric = scores.set_index('Metric').Score
        self.assertAlmostEqual(by_metric['Fair Brier Skill Score (%)'], -2.526)
        self.assertAlmostEqual(by_metric['Fair RPS Skill Score (%)'], 12.240)
        self.assertEqual(by_metric['AUC'], .720)
        self.assertEqual(len(scores), 6)

    def test_saved_undefined_metric_remains_present(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_scores(root, values={'Fair_RPS': .2, 'Fair_RPS_Skill_Score': np.nan})
            scores, distributions, errors = collect_pdf_probability_tables(root)
        self.assertEqual(errors, [])
        self.assertTrue(distributions.empty)
        self.assertEqual(set(scores.Metric), {'Fair RPS', 'Fair RPS Skill Score (%)'})
        self.assertTrue(pd.isna(scores.set_index('Metric').loc['Fair RPS Skill Score (%)', 'Score']))

    def test_score_grid_provides_spatial_statistics_without_inventing_pooled_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / 'gencast'
            folder.mkdir()
            xr.Dataset({
                'fair_brier_score': (('lat', 'lon'), [[.1, .2], [.3, np.nan]]),
                'fair_brier_skill_score': (('lat', 'lon'), [[-.25, .5], [.75, np.nan]]),
            }, coords={'lat': [8., 9.], 'lon': [38., 39.]}).to_netcdf(
                folder / 'probabilistic_metrics_gencast_1-15.nc')
            scores, distributions, errors = collect_pdf_probability_tables(root)
        self.assertEqual(errors, [])
        self.assertTrue(scores.empty)
        self.assertEqual(set(distributions.Model), {'GenCast'})
        self.assertEqual(set(distributions['Lead Window']), {'Days 1–15'})
        by_metric = distributions.set_index('Metric')
        self.assertAlmostEqual(by_metric.loc['Fair Brier Score', 'Mean'], .2)
        self.assertEqual(by_metric.loc['Fair Brier Skill Score (%)', 'Min'], -25.)
        self.assertEqual(by_metric.loc['Fair Brier Skill Score (%)', 'Max'], 75.)
        self.assertEqual(by_metric.loc['Fair Brier Skill Score (%)', 'unit'], '%')

    def test_corrupt_files_are_reported_without_discarding_usable_model_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_scores(root)
            folder = root / 'gencast'
            folder.mkdir()
            (folder / 'overall_skill_scores_gencast_1-15.csv').write_text('')
            (folder / 'probabilistic_metrics_gencast_1-15.nc').write_text('not a NetCDF file')
            scores, distributions, errors = collect_pdf_probability_tables(root)
        self.assertEqual(set(scores.Model), {'AIFS Ensemble'})
        self.assertTrue(distributions.empty)
        self.assertEqual(len(errors), 2)
        self.assertTrue(any('overall_skill_scores_gencast' in message for message in errors))
        self.assertTrue(any('probabilistic_metrics_gencast' in message for message in errors))

    def test_empty_directory_and_nested_country_results_do_not_invent_tables(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_scores(root / 'countries/Peru')
            scores, distributions, errors = collect_pdf_probability_tables(root)
        self.assertTrue(scores.empty)
        self.assertTrue(distributions.empty)
        self.assertEqual(errors, [])
