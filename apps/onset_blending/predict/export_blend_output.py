#!/usr/bin/env python3
"""
export_blend_output.py
Export blend predictions to CSV format for downstream consumption.

Reads the blended predictions pickle (produced by the blend evaluation step)
and writes a summary CSV with weekly probabilities for each cell, suitable
for map generation and downstream reporting.
"""
import os
import sys
import argparse
import pickle
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
# ==============================================================================
# Project Root Resolution
# ==============================================================================
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ==============================================================================
# Core export logic
# ==============================================================================
def _clean_probs5(P):
    """Sanitize 5-bin probability matrix: replace non-finite with 0, normalize rows."""
    P = np.array(P, dtype=float)
    P[~np.isfinite(P)] = 0.0
    P = np.maximum(P, 0.0)
    rs = P.sum(axis=1, keepdims=True)
    zero = (rs <= 0).flatten()
    if zero.any():
        P[zero] = 1.0 / P.shape[1]
    rs = P.sum(axis=1, keepdims=True)
    return P / rs


def export_blend_predictions(issue_date: str, spec: dict, preds_file: str, out_dir: str):
    """
    Export blend predictions to CSV format.

    Args:
        issue_date: Issue date in YYYY-MM-DD format
        spec: Blend specification dictionary
        preds_file: Path to predictions pickle file
        out_dir: Output directory for CSV
    """
    if not os.path.exists(preds_file):
        raise FileNotFoundError(f"Predictions file not found: {preds_file}")

    print(f"Loading predictions from: {preds_file}")
    with open(preds_file, "rb") as f:
        preds_df = pickle.load(f)

    print(f"Loaded predictions: {len(preds_df)} rows, {len(preds_df.columns)} columns")
    print(f"Columns: {preds_df.columns.tolist()}")

    # Expected probability columns (5-bin: week1..week4 + later)
    bins5 = ["week1", "week2", "week3", "week4", "later"]
    cv_cols = [f"cv_{b}" for b in bins5]

    required = ["id", "time", "year"] + cv_cols
    missing_cols = [column for column in required if column not in preds_df.columns]
    if missing_cols:
        raise ValueError(f"Missing prediction columns: {missing_cols}")
    target = pd.Timestamp(issue_date).date()
    dates = pd.to_datetime(preds_df["time"], errors="raise").dt.date
    preds_df = preds_df.loc[dates == target].copy()
    if preds_df.empty:
        raise ValueError(f"No predictions found for issue date {issue_date}")
    P5 = preds_df[cv_cols].to_numpy(dtype=float)
    if not np.isfinite(P5).all() or (P5 < 0).any() or (P5.sum(axis=1) <= 0).any():
        raise ValueError("Predictions contain invalid probabilities; refusing to export fabricated forecasts")
    P5_clean = _clean_probs5(P5)

    # Build summary DataFrame
    summary_df = preds_df[["id", "time", "year"]].copy() if "id" in preds_df.columns else pd.DataFrame(index=preds_df.index)

    # Add weekly probabilities
    for i, b in enumerate(bins5):
        summary_df[b] = P5_clean[:, i]

    # Add climatology columns if available
    clim_cols = [c for c in preds_df.columns if c.startswith("prob_clim_mr_")]
    for col in clim_cols:
        summary_df[col] = preds_df[col].values

    # Add outcome column if available
    if "outcome" in preds_df.columns:
        summary_df["outcome"] = preds_df["outcome"].values

    # Add onset threshold if available
    if "onset_threshold" in preds_df.columns:
        summary_df["onset_threshold"] = preds_df["onset_threshold"].values

    # Write output
    os.makedirs(out_dir, exist_ok=True)
    date_compact = issue_date.replace("-", "")
    output_csv = os.path.join(out_dir, f"blend_output_summary_{date_compact}.csv")

    summary_df.to_csv(output_csv, index=False)
    print(f"Wrote: {output_csv}")
    print(f"Rows: {len(summary_df)}, Cols: {len(summary_df.columns)}")
    print(f"Columns: {summary_df.columns.tolist()}")


# ==============================================================================
# Main
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Export blend predictions to CSV")
    parser.add_argument("--issue_date", required=True, help="Issue date YYYY-MM-DD")
    parser.add_argument("--spec_yaml", required=True, help="Path to blend spec YAML")
    parser.add_argument("--preds_file", required=True, help="Path to predictions pickle file")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    args = parser.parse_args()

    # Load spec
    with open(args.spec_yaml) as f:
        spec = yaml.safe_load(f)

    # Export
    export_blend_predictions(
        issue_date=args.issue_date,
        spec=spec,
        preds_file=args.preds_file,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()