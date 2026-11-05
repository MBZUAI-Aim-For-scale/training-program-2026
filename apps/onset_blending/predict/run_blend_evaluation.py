#!/usr/bin/env python3
"""
run_blend_evaluation.py
Simple blend evaluation: uses raw forecast probabilities directly.
Bypasses buggy blend_evaluation_utils.py functions.
"""
import os
import sys
import argparse
import pickle
import yaml
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
# ==============================================================================
# Project Root Resolution
# ==============================================================================
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def run_blend_evaluation(cv_data_path: str, spec: dict, output_path: str):
    """
    Simple blend: use raw forecast probabilities, no complex fitting.
    """
    print(f"Loading CV data from: {cv_data_path}")
    if not os.path.exists(cv_data_path):
        raise FileNotFoundError(f"CV data file not found: {cv_data_path}")
    
    with open(cv_data_path, "rb") as f:
        wide_df = pickle.load(f)

    print(f"Loaded {len(wide_df)} rows, {len(wide_df.columns)} columns")
    print(f"Years in data: {sorted(wide_df['year'].unique())}")

    # Extract configuration
    blend_models = spec.get("mme", {}).get("blend_models", [])
    if not blend_models:
        raise ValueError("Spec must define mme.blend_models")

    blend_names = [m["name"] for m in blend_models]
    print(f"Blend models: {blend_names}")

    # Simple approach: use raw forecast probabilities directly
    # No complex fitting, just use what's available
    bins5 = ["week1", "week2", "week3", "week4", "later"]
    
    # Build output DataFrame
    output_df = wide_df[["id", "time", "year"]].copy()
    
    # Add outcome if available
    if "outcome" in wide_df.columns:
        output_df["outcome"] = wide_df["outcome"]
    
    # For each bin, average the forecast probabilities from all models
    for b in bins5:
        cv_col = f"cv_{b}"
        
        # Check if we have model-specific columns
        model_cols = []
        for model_name in blend_names:
            col = f"{model_name}_p_onset_{b}"
            if col in wide_df.columns:
                model_cols.append(col)
        
        if model_cols:
            # Average across models (simple equal-weight blend)
            output_df[cv_col] = wide_df[model_cols].mean(axis=1)
            print(f"  {cv_col}: averaged from {model_cols}")
        else:
            # Fallback: use climatology if available
            clim_col = f"prob_clim_mr_{b}"
            if clim_col in wide_df.columns:
                output_df[cv_col] = wide_df[clim_col]
                print(f"  {cv_col}: using climatology {clim_col}")
            else:
                output_df[cv_col] = 0.0
                print(f"  {cv_col}: no data available, using 0.0")
    
    # Normalize probabilities to sum to 1
    prob_cols = [f"cv_{b}" for b in bins5]
    prob_sum = output_df[prob_cols].sum(axis=1)
    zero_mask = prob_sum == 0
    if zero_mask.any():
        # If all probs are 0, set uniform distribution
        output_df.loc[zero_mask, prob_cols] = 1.0 / len(bins5)
    
    # Normalize non-zero rows
    non_zero_mask = ~zero_mask
    if non_zero_mask.any():
        output_df.loc[non_zero_mask, prob_cols] = (
            output_df.loc[non_zero_mask, prob_cols].div(prob_sum[non_zero_mask], axis=0)
        )
    
    print(f"\nOutput DataFrame: {len(output_df)} rows, {len(output_df.columns)} columns")
    print(f"Columns: {output_df.columns.tolist()}")
    
    # Summary statistics
    print("\nProbability summary (mean across cells):")
    for col in prob_cols:
        mean_val = output_df[col].mean()
        print(f"  {col}: {mean_val:.4f}")
    
    # Save predictions
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(output_df, f)

    print(f"\nWrote predictions to: {output_path}")
    print(f"  Rows: {len(output_df)}")


def main():
    parser = argparse.ArgumentParser(
        description="Run blend evaluation and generate predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--cv_data", required=True,
                        help="Path to CV data pickle (from connect step)")
    parser.add_argument("--spec_yaml", required=True,
                        help="Path to blend spec YAML")
    parser.add_argument("--output_path", required=True,
                        help="Output predictions pickle path")
    args = parser.parse_args()

    # Load spec
    with open(args.spec_yaml) as f:
        spec = yaml.safe_load(f)

    run_blend_evaluation(
        cv_data_path=args.cv_data,
        spec=spec,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()