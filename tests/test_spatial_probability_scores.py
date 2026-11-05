"""Per-grid probabilities preserve the paired ROMP score definitions."""

import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
import warnings

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "apps/benchmarking/ROMP"))
from momp.stats.score import calculate_auc, calculate_paired_brier, calculate_paired_rps
from momp.stats.spatial_probability import (
    calculate_spatial_probability_scores, save_spatial_probability_scores,
)


class SpatialProbabilityScoreTests(unittest.TestCase):
    def pairs(self):
        model_rows, reference_rows = [], []
        bounds = [(-np.inf, 0), (1, 5), (6, 10), (11, 15), (16, np.inf)]
        for cell in range(3):
            for case, truth_index in enumerate((0, 2, 4)):
                truth_index = 0 if cell == 2 else truth_index
                probabilities = np.array([.1, .2, .4, .2, .1])
                reference = np.full(5, .2)
                if cell == 1:
                    reference = np.eye(5)[truth_index]
                for index, (start, end) in enumerate(bounds):
                    common = dict(init_time=f"2024-06-{case + 1:02d}", lat=8. + cell,
                                  lon=39. + cell, bin_index=index, bin_label=f"bin {index}",
                                  bin_start=start, bin_end=end,
                                  observed_onset=int(index == truth_index), total_members=10)
                    model_rows.append({**common, "predicted_prob": probabilities[index]})
                    reference_rows.append({**common, "predicted_prob": reference[index],
                                           "total_members_with_onset": 1 if case == 1 and cell == 0 else 8})
        forecast, reference = pd.DataFrame(model_rows), pd.DataFrame(reference_rows)
        return {
            "forecast_obs_df": forecast[forecast.bin_index.between(1, 3)].copy(),
            "reference_obs_df": reference[reference.bin_index.between(1, 3)].copy(),
            "rps_forecast_obs_df": forecast,
            "rps_reference_obs_df": reference,
            "BS": {}, "RPS": {}, "AUC": {}, "skill_results": {},
        }

    def quiet(self):
        stack = contextlib.ExitStack()
        stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        stack.enter_context(warnings.catch_warnings())
        warnings.simplefilter("ignore", RuntimeWarning)
        return stack

    def test_each_cell_matches_paired_estimators_with_full_rps_partition(self):
        pairs = self.pairs()
        original = {key: value.copy(deep=True) for key, value in pairs.items()
                    if isinstance(value, pd.DataFrame)}
        with self.quiet():
            grid = calculate_spatial_probability_scores(pairs)
            for latitude in (8., 9., 10.):
                subset = {key: value[value.lat == latitude] for key, value in original.items()}
                bs, bs_ref = calculate_paired_brier(
                    subset["forecast_obs_df"], subset["reference_obs_df"])
                rps, rps_ref = calculate_paired_rps(
                    subset["rps_forecast_obs_df"], subset["rps_reference_obs_df"])
                expected = {"fair_brier_score": bs["fair_brier_score"],
                            "reference_fair_brier_score": bs_ref["fair_brier_score"],
                            "fair_rps": rps["fair_rps"], "reference_fair_rps": rps_ref["fair_rps"],
                            "auc": calculate_auc(subset["forecast_obs_df"])["auc"],
                            "reference_auc": calculate_auc(subset["reference_obs_df"])["auc"]}
                for name, value in expected.items():
                    np.testing.assert_allclose(grid[name].sel(lat=latitude, lon=latitude + 31), value,
                                               equal_nan=True)
            self.assertEqual(rps["n_forecasts"], 3)
        for key, value in original.items():
            pd.testing.assert_frame_equal(pairs[key], value)
        # Unsupported reference cases cannot contribute only on the forecast side.
        with self.quiet():
            supported = original["rps_forecast_obs_df"]
            reference = original["rps_reference_obs_df"]
            actual, _ = calculate_paired_rps(supported[supported.lat == 8], reference[reference.lat == 8])
        self.assertEqual(actual["n_forecasts"], 2)
        self.assertTrue(np.isnan(grid.auc.sel(lat=10, lon=41)))
        self.assertTrue(np.isnan(grid.fair_brier_skill_score.sel(lat=9, lon=40)))
        self.assertTrue(np.isnan(grid.fair_rps_skill_score.sel(lat=9, lon=40)))
        # Cartesian padding is missing data, not a scored zero.
        self.assertTrue(np.isnan(grid.fair_rps.sel(lat=8, lon=41)))

    def test_per_cell_skill_uses_its_own_reference(self):
        with self.quiet():
            grid = calculate_spatial_probability_scores(self.pairs())
        for metric in ("fair_brier", "fair_rps"):
            score_name = "fair_brier_score" if metric == "fair_brier" else "fair_rps"
            expected = 1 - grid[score_name].sel(lat=8, lon=39) / grid[f"reference_{score_name}"].sel(lat=8, lon=39)
            np.testing.assert_allclose(grid[f"{metric}_skill_score"].sel(lat=8, lon=39), expected)

    def test_netcdf_roundtrip_does_not_replace_pooled_csv_and_keeps_provenance(self):
        with tempfile.TemporaryDirectory() as directory, self.quiet():
            pooled = Path(directory) / "overall_skill_scores_aifs_ens_1-15.csv"
            pooled.write_text("AUC\n0.75\n")
            path = save_spatial_probability_scores(
                self.pairs(), model="aifs_ens", verification_window=(1, 15), dir_out=directory,
                years=(2024,), years_clim=(2010, 2020))
            self.assertEqual(path.name, "probabilistic_metrics_aifs_ens_1-15.nc")
            self.assertEqual(pooled.read_text(), "AUC\n0.75\n")
            with xr.open_dataset(path) as saved:
                self.assertEqual(set(saved.dims), {"lat", "lon"})
                self.assertEqual(saved.attrs["model"], "aifs_ens")
                self.assertEqual(saved.attrs["rps_category_order"], "chronological_complete_partition")
                self.assertEqual(saved.fair_brier_skill_score.attrs["units"], "1")
                self.assertFalse(np.isinf(saved.to_array()).any())
            self.assertFalse(list(Path(directory).glob("*.tmp")))

    def test_empty_input_does_not_create_fake_distributions(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNone(save_spatial_probability_scores(
                {}, model="gencast", verification_window=(16, 30), dir_out=directory))
            self.assertEqual(list(Path(directory).iterdir()), [])
