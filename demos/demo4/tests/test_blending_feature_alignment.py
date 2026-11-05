"""Interaction factor order must not change saved-model predictions."""

import unittest

import numpy as np
import pandas as pd
import patsy

from apps.onset_blending.predict import apply_blend_model as model


class FeatureAlignmentTests(unittest.TestCase):
    def test_reversed_interactions_preserve_values_and_training_order(self):
        frame = pd.DataFrame(
            {"clim": [2.0, 4.0], "ens": [3.0, 6.0], "single": [5.0, 7.0]}, index=[8, 3]
        )
        matrix = patsy.dmatrix("clim * ens * single", frame, return_type="dataframe")
        features = [
            "clim",
            "single",
            "ens",
            "clim:single",
            "clim:ens",
            "single:ens",
            "clim:single:ens",
        ]
        result = model.align_design_matrix(matrix, features)
        self.assertEqual(result.columns.tolist(), features)
        self.assertEqual(result.index.tolist(), [8, 3])
        np.testing.assert_allclose(
            result.values, [[2, 5, 3, 10, 6, 15, 30], [4, 7, 6, 28, 24, 42, 168]]
        )

    def test_genuinely_missing_feature_still_fails(self):
        matrix = patsy.dmatrix(
            "a * b", pd.DataFrame({"a": [2.0], "b": [3.0]}), return_type="dataframe"
        )
        with self.assertRaisesRegex(ValueError, "missing trained features.*a:c"):
            model.align_design_matrix(matrix, ["a:c"])

    def test_all_four_weekly_interaction_groups_align(self):
        frame = pd.DataFrame(
            {
                f"{name}_week{week}": [value]
                for week in range(1, 5)
                for name, value in [
                    ("prob_clim_mr", 2.0),
                    ("diff_aifs_ens", 3.0),
                    ("diff_aifs", 5.0),
                ]
            }
        )
        formula = " + ".join(
            f"prob_clim_mr_week{w} * diff_aifs_ens_week{w} * diff_aifs_week{w}"
            for w in range(1, 5)
        )
        matrix = patsy.dmatrix(formula, frame, return_type="dataframe")
        features = []
        for w in range(1, 5):
            c, a, e = f"prob_clim_mr_week{w}", f"diff_aifs_week{w}", f"diff_aifs_ens_week{w}"
            features.extend([c, a, e, f"{c}:{a}", f"{c}:{e}", f"{a}:{e}", f"{c}:{a}:{e}"])
        result = model.align_design_matrix(matrix, features)
        np.testing.assert_allclose(result.iloc[0], [2, 5, 3, 10, 6, 15, 30] * 4)


if __name__ == "__main__":
    unittest.main()
