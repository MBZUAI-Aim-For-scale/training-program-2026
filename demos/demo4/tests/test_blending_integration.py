"""Exercise the root blending runner with synthetic trained-model artifacts."""

import math
import os
import pickle
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class BlendingIntegrationTests(unittest.TestCase):
    def test_root_runner_applies_trained_scaler_and_exports_requested_issue(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            coefficients = pd.DataFrame(
                [
                    {
                        "class": label,
                        "feature": "predictor:companion",
                        "coefficient": math.log(4) if label == "week1" else 0.0,
                        "intercept": 0.0,
                    }
                    for label in ("week1", "week2", "week3", "week4", "later")
                ]
            )
            bundle = {
                "coefs": coefficients,
                "scaler": StandardScaler().fit(np.array([[0.0], [4.0]])),
                "features": ["predictor:companion"],
            }
            with (root / "coefs_blended_model_global_synthetic.pkl").open("wb") as stream:
                pickle.dump(bundle, stream)
            inputs = pd.DataFrame(
                {
                    "id": ["older-year", "target", "other-issue"],
                    "time": pd.to_datetime(["2025-06-15", "2026-06-15", "2026-06-16"]),
                    "year": [2025, 2026, 2026],
                    "predictor": [20.0, 4.0, 0.0],
                    "companion": [2.0, 1.0, 2.0],
                },
                index=[8, 3, 12],
            )
            input_path = root / "input.pkl"
            inputs.to_pickle(input_path)
            spec_path = root / "blend.yml"
            spec_path.write_text(
                yaml.safe_dump(
                    {
                        "run": {
                            "cutoff_mode": "clim_mok_date",
                            "training_years": [2024],
                            "cv_holdout_years": [2026],
                        },
                        "models": {
                            "formulas": {
                                "blended_model": {"text": "outcome ~ companion:predictor"}
                            }
                        },
                    }
                )
            )
            command = [
                sys.executable,
                str(PROJECT_ROOT / "run_blending.py"),
                "--year",
                "2026",
                "--issue_date",
                "2026-06-15",
                "--work_dir",
                str(root),
                "--coef_dir",
                str(root),
                "--coef_tag",
                "synthetic",
                "--blend_input",
                str(input_path),
                "--blend_spec",
                str(spec_path),
                "--skip_to",
                "6",
                "--stop_at",
                "7",
            ]
            env = os.environ.copy()
            env["ROMP_PYTHON"] = sys.executable
            completed = subprocess.run(
                command, cwd=root, env=env, capture_output=True, text=True, timeout=90, check=False
            )
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            predictions = pd.read_csv(root / "blended_model_global_year2026_preds.csv")
            self.assertEqual(set(predictions["id"]), {"target", "other-issue"})
            self.assertEqual(set(predictions["year"]), {2026})
            summary = pd.read_csv(root / "blend_output_summary_20260615.csv")
            self.assertEqual(summary["id"].tolist(), ["target"])
            self.assertEqual(summary["time"].tolist(), ["2026-06-15"])
            bins = ["week1", "week2", "week3", "week4", "later"]
            np.testing.assert_allclose(summary[bins].sum(axis=1), [1.0])
            np.testing.assert_allclose(summary[bins].iloc[0], [0.5, 0.125, 0.125, 0.125, 0.125])
            self.assertIn("Apply trained blend model", completed.stdout)
            self.assertFalse((root / "maps").exists())


if __name__ == "__main__":
    unittest.main()
