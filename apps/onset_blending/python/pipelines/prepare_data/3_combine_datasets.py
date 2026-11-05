#!/usr/bin/env python3
# ==============================================================================
# Script: apps/onset_forecast/python/pipelines/prepare_data/3_combine_datasets.py
# ==============================================================================
# Purpose
#   Combine climatology + forecast families + ground truth into one combined
#   dataset for modeling / downstream processing.
#
# Usage (zero arguments — defaults to combining AIFS + AIFS_ENS from config.py)
#   python -m etagrometaix.apps.onset_forecast.python.pipelines.prepare_data.3_combine_datasets
#
#   # Or select a different combine spec from config.py:
#   python -m etagrometaix.apps.onset_forecast.python.pipelines.prepare_data.3_combine_datasets \
#       --spec_id gencast_aifs
#
#   # Or with a temp YAML written by the orchestrator:
#   python -m etagrometaix.apps.onset_forecast.python.pipelines.prepare_data.3_combine_datasets \
#       --spec_yaml /tmp/spec_xxxxx.yml
# ==============================================================================
import argparse
import os
import sys
import pickle
import yaml
import numpy as np
from pathlib import Path
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# Import spec dictionaries from config
from ai_weather.config import (
    ONSET_FORECAST_COMBINE_CLIM_SPEC,
    ONSET_FORECAST_COMBINE_NGCM_AIFS_SPEC,
    ONSET_FORECAST_COMBINE_GENCAST_AIFS_SPEC,
)
from apps.onset_blending.python.pipelines._shared.misc import coalesce
from apps.onset_blending.python.prepare_data.combine_forecasts_utils import (
    read_ground_truth_wide,
    read_and_format_climatology_wide,
    format_forecast_family,
)


# Map spec_id names to the config dictionaries
SPEC_REGISTRY = {
    "clim": ONSET_FORECAST_COMBINE_CLIM_SPEC,
    "clim_mok_date": ONSET_FORECAST_COMBINE_CLIM_SPEC,
    "combine_template_clim_mok_date": ONSET_FORECAST_COMBINE_CLIM_SPEC,
    "ngcm_aifs": ONSET_FORECAST_COMBINE_NGCM_AIFS_SPEC,
    "combine_template_clim_mok_date_ngcm": ONSET_FORECAST_COMBINE_NGCM_AIFS_SPEC,
    "gencast_aifs": ONSET_FORECAST_COMBINE_GENCAST_AIFS_SPEC,
    "combine_template_clim_mok_date_gencast": ONSET_FORECAST_COMBINE_GENCAST_AIFS_SPEC,
}

# Default spec to use when no arguments are provided
DEFAULT_SPEC_ID = "clim"


def load_spec(spec_id=None, spec_yaml=None):
    """
    Load a spec dictionary either from a YAML file (--spec_yaml) or from
    the config.py registry (--spec_id).
    """
    if spec_yaml:
        path = Path(spec_yaml)
        if not path.exists():
            raise FileNotFoundError(f"Spec YAML not found: {path}")
        with open(path) as f:
            spec = yaml.safe_load(f)
        return spec

    # Use provided spec_id or default
    key = (spec_id or DEFAULT_SPEC_ID).lower()
    if key not in SPEC_REGISTRY:
        raise ValueError(
            f"Unknown spec_id '{spec_id}'. "
            f"Available: {', '.join(sorted(set(SPEC_REGISTRY.keys())))}"
        )
    spec = SPEC_REGISTRY[key].copy()
    spec["id"] = spec_id or DEFAULT_SPEC_ID
    return spec


def main():
    parser = argparse.ArgumentParser(
        description="Combine datasets into wide table. "
                    "Zero arguments — defaults to AIFS + AIFS_ENS from etagrometaix.config.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-i", "--spec_id", default=None,
        help=f"Spec ID name (one of: {', '.join(sorted(set(SPEC_REGISTRY.keys())))}). "
             f"Default: {DEFAULT_SPEC_ID}. Loads from etagrometaix.config."
    )
    parser.add_argument(
        "--spec_yaml", default=None,
        help="Path to a temporary YAML spec file (written by the orchestrator). "
             "Takes priority over --spec_id."
    )
    args = parser.parse_args()

    spec = load_spec(spec_id=args.spec_id, spec_yaml=args.spec_yaml)
    spec_id = spec.get("id", DEFAULT_SPEC_ID)

    # Read ground truth
    truth = read_ground_truth_wide(spec["input"]["ground_truth_wide_rds"])

    # Normalize ground truth column names for downstream compatibility.
    # The ground truth file uses 'mr_onset_date' / 'mr_onset_day', but the
    # combine and connect steps expect 'true_onset_date' / 'true_onset_day'.
    rename_map = {}
    if "mr_onset_date" in truth.columns and "true_onset_date" not in truth.columns:
        rename_map["mr_onset_date"] = "true_onset_date"
    if "mr_onset_day" in truth.columns and "true_onset_day" not in truth.columns:
        rename_map["mr_onset_day"] = "true_onset_day"
    if rename_map:
        truth = truth.rename(columns=rename_map)
        print(f"  Renamed ground truth columns: {rename_map}")

    # Read climatologies
    clim_list = []
    if spec["input"].get("climatologies"):
        for nm, x in spec["input"]["climatologies"].items():
            clim_list.append(
                read_and_format_climatology_wide(
                    path=x["rds"],
                    out_prefix=coalesce(x.get("out_prefix"), f"{nm}_p_onset"),
                )
            )
    elif spec["input"].get("climatology_rds"):
        clim_list.append(
            read_and_format_climatology_wide(
                path=spec["input"]["climatology_rds"],
                out_prefix="clim_p_onset",
            )
        )

    # Read forecast families
    forecast_parts = {
        nm: format_forecast_family(nm, conf)
        for nm, conf in spec.get("forecasts", {}).items()
    }
    forecast_daily = {nm: fp["daily"] for nm, fp in forecast_parts.items()}
    forecast_const = {nm: fp["constants"] for nm, fp in forecast_parts.items()}

    # Combine daily: clim + forecast daily tables
    daily_tables = clim_list + list(forecast_daily.values())

    join_type = "outer" if spec.get("options", {}).get("join") == "full" else "inner"

    # Normalize time, year, and id columns to ensure consistent types for merging.
    # This prevents 0-row results when the climatology and forecast tables have
    # different datetime representations.
    for i, tbl in enumerate(daily_tables):
        tbl = tbl.copy()
        tbl["time"] = pd.to_datetime(tbl["time"]).dt.date
        tbl["year"] = tbl["year"].astype(int)
        tbl["id"] = tbl["id"].astype(str)
        daily_tables[i] = tbl

    daily_wide = daily_tables[0]
    for tbl in daily_tables[1:]:
        daily_wide = daily_wide.merge(tbl, on=["id", "time", "year"], how=join_type)

    # Ensure year column is consistent after merge
    daily_wide["year"] = pd.to_datetime(daily_wide["time"]).dt.year

    # Merge ground truth
    daily_wide = daily_wide.merge(truth, on=["id", "year"], how="left")

    # Safety check: ensure true_onset_date exists after merge
    if "true_onset_date" not in daily_wide.columns:
        print("  Warning: 'true_onset_date' column not found after ground truth merge.")
        print(f"  Ground truth columns: {truth.columns.tolist()}")
        print("  Creating 'true_onset_date' column with NaT values.")
        daily_wide["true_onset_date"] = pd.NaT
        if "true_onset_day" not in daily_wide.columns:
            daily_wide["true_onset_day"] = np.nan

    # Optional trimming
    if spec.get("options", {}).get("trim_forecasts_after_true_onset", False):
        if "true_onset_date" in daily_wide.columns:
            mask = daily_wide["true_onset_date"].isna() | (
                pd.to_datetime(daily_wide["time"]) <= pd.to_datetime(daily_wide["true_onset_date"])
            )
            daily_wide = daily_wide[mask]
        else:
            print("  Warning: trim_forecasts_after_true_onset is True but "
                  "'true_onset_date' column not found. Skipping trim.")

    # Merge constants
    if forecast_const:
        const_tables = list(forecast_const.values())
        const_all = const_tables[0]
        for tbl in const_tables[1:]:
            const_all = const_all.merge(tbl, on=["id", "time", "year"], how="outer")
        daily_wide = daily_wide.merge(const_all, on=["id", "time", "year"], how="left")

    if daily_wide.empty:
        raise ValueError("No overlapping climatology and forecast rows; check issue dates and district identifiers")

    # Write output
    out_dir = spec["output"]["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    basename = spec.get("output", {}).get("basename", spec_id)
    out_pkl = os.path.join(out_dir, f"{basename}_combined_wide.pkl")

    with open(out_pkl, "wb") as f:
        pickle.dump(daily_wide, f)

    print(f"Wrote combined wide dataset: {out_pkl}")
    print(f"Rows: {len(daily_wide)}, Cols: {len(daily_wide.columns)}")


if __name__ == "__main__":
    main()