#!/usr/bin/env python3
# ==============================================================================
# Script: apps/onset_blending/python/pipelines/prepare_data/2_build_climatology.py
# ==============================================================================
# Purpose
#   Fit per-cell KDE climatology models from historical onset dates and
#   produce issue-date probability forecasts over within-season date windows.
#   Supports MULTIPLE climatology runs from a single spec.
#
# Usage (zero arguments — defaults come from etagrometaix.config)
#   python -m apps.onset_forecast.python.pipelines.prepare_data.2_build_climatology
#
#   # Or with a temp YAML written by the orchestrator:
#   python -m apps.onset_blending.python.pipelines.prepare_data.2_build_climatology \
#       --spec_yaml /tmp/spec_xxxxx.yml
# ==============================================================================

import os
import sys
import pickle
import yaml
import argparse
from pathlib import Path

# ==============================================================================
# Project Root Resolution
# ==============================================================================

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))



# Import everything from config — zero CLI arguments needed
from ai_weather.config import ONSET_FORECAST_IMD_CLIM_SPEC
from apps.onset_blending.python.prepare_data.climatology_utils import (
    get_paths_clim,
    get_climatology_options_from_run,
    read_gt_onset_from_tbl,
    filter_gt_training,
    build_issue_grid,
    compute_all_forecasts,
)


def load_spec(spec_yaml=None):
    """
    Load a spec dictionary either from a YAML file (--spec_yaml) or from
    the config.py default (ONSET_FORECAST_IMD_CLIM_SPEC).
    """
    if spec_yaml:
        path = Path(spec_yaml)
        if not path.exists():
            raise FileNotFoundError(f"Spec YAML not found: {path}")
        with open(path) as f:
            spec = yaml.safe_load(f)
        return spec

    # Default: use the config.py spec
    spec = ONSET_FORECAST_IMD_CLIM_SPEC.copy()
    spec["id"] = "imd_clim"
    return spec


def main():
    parser = argparse.ArgumentParser(
        description="Build KDE climatology forecasts. Defaults from etagrometaix.config.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--spec_yaml", default=None,
        help="Path to a temporary YAML spec file (written by the orchestrator). "
             "If not provided, uses ONSET_FORECAST_IMD_CLIM_SPEC from config.py."
    )
    args = parser.parse_args()

    # Load spec
    spec = load_spec(spec_yaml=args.spec_yaml)

    # Validate the spec structure
    if not spec.get("climatologies"):
        raise ValueError("Missing or empty 'climatologies' in spec.")
    
    required = ["train_year_min", "train_year_max", "test_year_min", "test_year_max",
                "season_start_md", "issue_end_md"]
    for nm, co in spec["climatologies"].items():
        miss = [f for f in required if f not in co]
        if miss:
            raise ValueError(f"Run '{nm}' missing fields: {', '.join(miss)}")
        has_fw = co.get("forecast_window") is not None
        has_hz = co.get("horizons") is not None
        if not has_fw and not has_hz:
            raise ValueError(f"Run '{nm}': provide forecast_window or horizons.")
        if has_fw and has_hz:
            raise ValueError(f"Run '{nm}': provide only one of forecast_window or horizons.")

    # Get paths from spec
    paths = get_paths_clim(spec)
    gt_wide_path = paths["gt_path"]
    if not os.path.exists(gt_wide_path):
        raise FileNotFoundError(f"Ground-truth file not found: {gt_wide_path}")

    with open(gt_wide_path, "rb") as f:
        gt_wide = pickle.load(f)

    # Run all climatology keys in the spec
    run_keys = list(spec["climatologies"].keys())
    print(f"Running {len(run_keys)} climatology configurations: {', '.join(run_keys)}")

    results = []
    for run_key in run_keys:
        co = spec["climatologies"][run_key]
        opt = get_climatology_options_from_run(co)

        gt = read_gt_onset_from_tbl(gt_wide, onset_col=opt["onset_col"])
        gt_train = filter_gt_training(gt, opt["train_year_min"], opt["train_year_max"])

        # Filter out cells with too few onset observations to fit a reliable KDE
        min_onset_years = co.get("min_onset_years", 10)
        onset_counts = gt_train.groupby("id")["onset_day"].count()
        valid_ids = onset_counts[onset_counts >= min_onset_years].index
        excluded = onset_counts[onset_counts < min_onset_years]
        if len(excluded) > 0:
            print(f"[{run_key}] Excluding {len(excluded)} cells with fewer than {min_onset_years} onset years.")
        gt_train = gt_train[gt_train["id"].isin(valid_ids)]
        if gt_train.empty:
            raise ValueError(f"[{run_key}] No cells remain after min_onset_years filter.")

        issue_grid = build_issue_grid(
            opt["test_year_min"], opt["test_year_max"],
            opt["season_start_md"], opt["issue_end_md"],
        )

        out = compute_all_forecasts(
            gt_train=gt_train,
            issue_grid=issue_grid,
            season_start_md=opt["season_start_md"],
            forecast_window=opt["forecast_window"],
            horizons=opt["horizons"],
            conditional=opt["conditional"],
            cv_by_year=opt["cv_by_year"],
        )

        forecast_tbl = out["forecasts"]
        out_stem = co.get("out_stem") or f"{paths['out_stem']}_{run_key}"
        out_rds = os.path.join(paths["out_dir"], f"{out_stem}.pkl")

        # Ensure output directory exists
        os.makedirs(paths["out_dir"], exist_ok=True)

        with open(out_rds, "wb") as f:
            pickle.dump(forecast_tbl, f)

        print(f"[{run_key}] Wrote: {out_rds}")
        results.append({"run": run_key, "out_pkl": out_rds})

    print(f"\nCompleted {len(results)} climatology runs.")
    return results


if __name__ == "__main__":
    main()