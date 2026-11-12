"""Report setup summaries are compact, readable and honest about conflicting data."""

from itertools import permutations
import unittest

from ui.utils.benchmark_report import summarize_benchmark_settings


class ReportSetupSummaryTests(unittest.TestCase):
    def record(self, **updates):
        record = {
            'Model': 'aifs_ens', 'Window': 'Days 1–15',
            'Evaluation years': '2003-2024', 'Actual evaluation years': 'Not recorded',
            'Baseline years': '2007-2024',
            'Evaluation start': '(2003, 5, 1)', 'Evaluation end': '(2024, 7, 29)',
            'Initialization schedule': 'weekly', 'Initialization days': '(0, 3)',
            'Reference calendar year': '2024', 'Baseline forecast': 'climatology',
            'Reference policy': 'leave_one_year_out',
            'Reference membership': 'held_out_verification_year;leave_one_year_out',
            'Output': 'overall_skill_scores_aifs_ens_1-15.csv',
        }
        record.update(updates)
        return record

    def test_user_year_mapping_becomes_a_short_reference_description(self):
        baseline = range(2007, 2025)
        mapping = {target: ','.join(str(year) for year in baseline if year != target)
                   for target in range(2000, 2025)}
        rows, issues = summarize_benchmark_settings([
            self.record(**{'Reference years by verification year': str(mapping)})])
        self.assertEqual(issues, [])
        self.assertEqual(rows[0]['Model'], 'AIFS Ensemble')
        self.assertEqual(rows[0]['Reference'], 'Leave-one-year-out\n17-18 eligible years')
        self.assertLess(len(str(rows)), 500)
        self.assertNotIn('2000:', str(rows))
        self.assertNotIn('held_out_verification_year', str(rows))

    def test_equivalent_csv_and_netcdf_settings_share_one_row(self):
        csv = self.record(**{'Wet spell length': '3', 'Wet-day minimum': '1.0'})
        grid = self.record(**{'Output': 'spatial_metrics_aifs_ens_1-15.nc',
                              'Initialization days': '0,3', 'Evaluation start': '2003,5,1',
                              'Evaluation end': '2024,7,29', 'Wet spell length': '3.0',
                              'Wet-day minimum': '1'})
        rows, issues = summarize_benchmark_settings([csv, grid])
        self.assertEqual(issues, [])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['Results'], 'Onset errors\nProbability scores')
        self.assertEqual(rows[0]['Evaluation period'], '2003-2024\n01 May-29 Jul annually')
        self.assertEqual(rows[0]['Initialization'], 'Weekly: Mon, Thu\n2024 reference calendar')

    def test_different_evaluation_periods_remain_separate_and_are_explained(self):
        probability = self.record(**{'Evaluation years': '2000-2024',
                                      'Actual evaluation years': '2000-2024',
                                      'Evaluation start': '(2000, 5, 1)',
                                      'Evaluation end': '(2024, 12, 31)'})
        spatial = self.record(Output='spatial_metrics_aifs_ens_1-15.nc')
        rows, issues = summarize_benchmark_settings([probability, spatial])
        self.assertEqual(len(rows), 2)
        self.assertEqual(len(issues), 1)
        self.assertIn('Probability scores: 2000-2024', issues[0])
        self.assertIn('Onset errors: 2003-2024', issues[0])
        self.assertIn('reported separately', issues[0])
        self.assertIn('conflicting saved result settings', issues[0])

    def test_known_onset_criteria_conflicts_do_not_merge(self):
        rows, issues = summarize_benchmark_settings([
            self.record(**{'Wet spell length': '3'}),
            self.record(**{'Output': 'spatial_metrics_aifs_ens_1-15.nc', 'Wet spell length': '5'})])
        self.assertEqual(len(rows), 2)
        self.assertIn('Wet spell length', issues[0])
        self.assertIn('Probability scores: 3', issues[0])
        self.assertIn('Onset errors: 5', issues[0])

    def test_sparse_criteria_merge_only_with_an_unambiguous_matching_group(self):
        csv = self.record()
        grid = self.record(**{'Output': 'spatial_metrics_aifs_ens_1-15.nc',
                              'Country': 'Ethiopia', 'Wet spell length': '3'})
        rows, issues = summarize_benchmark_settings([csv, grid])
        self.assertEqual(len(rows), 1)
        self.assertEqual(issues, [])
        conflicting = self.record(**{'Output': 'spatial_metrics_aifs_ens_1-15.nc',
                                      'Country': 'Ethiopia', 'Wet spell length': '5'})
        rows, issues = summarize_benchmark_settings([csv, grid, conflicting])
        self.assertEqual(len(rows), 3)
        self.assertTrue(issues)

    def test_conflicting_partial_metadata_never_merges_by_input_order(self):
        records = [self.record(**{'Wet spell length': '3'}),
                   self.record(**{'Output': 'spatial_metrics_aifs_ens_1-15.nc', 'Wet-day minimum': '1'}),
                   self.record(**{'Output': 'spatial_metrics_aifs_ens_1-15.nc', 'Wet spell length': '5'})]
        for ordered in permutations(records):
            with self.subTest(order=[record.get('Wet spell length') for record in ordered]):
                rows, issues = summarize_benchmark_settings(ordered)
                self.assertEqual(len(rows), 3)
                self.assertTrue(issues)

    def test_actual_year_coverage_is_used_and_requested_coverage_remains_visible(self):
        rows, _ = summarize_benchmark_settings([
            self.record(**{'Actual evaluation years': '2021-2024'})])
        self.assertIn('2021-2024 (configured 2003-2024)', rows[0]['Evaluation period'])

    def test_missing_values_stay_unknown_without_repeated_raw_metadata(self):
        rows, issues = summarize_benchmark_settings([{'Model': 'gencast', 'Window': 'Days 1-15'}])
        self.assertEqual(issues, [])
        self.assertEqual(rows[0]['Model'], 'GenCast')
        for field in ('Evaluation period', 'Baseline years', 'Initialization', 'Reference'):
            self.assertEqual(rows[0][field], 'Unknown')
        rows, _ = summarize_benchmark_settings([{'Model': 'gencast', 'Reference calendar year': '2020'}])
        self.assertIn('Calendar: 2020', rows[0]['Initialization'])

    def test_missing_core_settings_are_not_described_as_proven_conflicts(self):
        for missing in ({'Baseline years': 'Not recorded'},
                        {'Reference policy': 'Not recorded', 'Reference membership': 'Not recorded'},
                        {'Initialization schedule': 'Not recorded'}):
            with self.subTest(missing=missing):
                rows, issues = summarize_benchmark_settings([
                    self.record(), self.record(**{'Output': 'spatial_metrics_aifs_ens_1-15.nc', **missing})])
                self.assertEqual(len(rows), 2)
                self.assertEqual(len(issues), 1)
                self.assertIn('incomplete saved settings', issues[0])
                self.assertIn('common experiment cannot be confirmed', issues[0])
                self.assertNotIn('conflicting', issues[0])

    def test_daily_and_monthly_schedules_do_not_claim_a_reference_calendar(self):
        daily, _ = summarize_benchmark_settings([
            self.record(**{'Initialization schedule': 'daily', 'Initialization days': 'None'})])
        monthly, _ = summarize_benchmark_settings([
            self.record(**{'Initialization schedule': 'monthly', 'Initialization days': '[1, 15]'})])
        self.assertEqual(daily[0]['Initialization'], 'Daily')
        self.assertEqual(monthly[0]['Initialization'], 'Monthly: days 1, 15')

    def test_unparseable_membership_does_not_invent_an_ensemble_size(self):
        rows, _ = summarize_benchmark_settings([
            self.record(**{'Reference years by verification year': '{2019: "unavailable"}'})])
        self.assertEqual(rows[0]['Reference'], 'Leave-one-year-out')

    def test_recorded_self_inclusion_is_not_labelled_as_independent(self):
        for mapping in ("{2019: '2019,2020'}", "{'2019': '2019,2020'}"):
            with self.subTest(mapping=mapping):
                rows, issues = summarize_benchmark_settings([
                    self.record(**{'Reference years by verification year': mapping})])
                self.assertEqual(rows[0]['Reference'], 'In-sample baseline\n2 eligible years')
                self.assertTrue(any('includes a verification year' in issue for issue in issues))
