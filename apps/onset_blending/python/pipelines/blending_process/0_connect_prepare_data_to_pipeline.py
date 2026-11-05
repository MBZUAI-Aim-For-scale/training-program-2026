#!/usr/bin/env python3
# ==============================================================================
# Script: apps/onset_blending/python/pipelines/blending_process/0_connect_prepare_data_to_pipeline.py
# ==============================================================================
# Purpose
#   Convert day-level wide pickle from the prepare_data pipeline into a
#   weekly-bin pickle that 1_blend_evaluation.py expects.
#
# Usage (zero arguments — defaults come from etagrometaix.config)
#   python -m apps.onset_blending.python.pipelines.blending_process.0_connect_prepare_data_to_pipeline
#
#   # Or with a temp YAML written by the orchestrator:
#   python -m apps.onset_blendig.python.pipelines.blending_process.0_connect_prepare_data_to_pipeline \
#       --spec_yaml /tmp/spec_xxxxx.yml
# ==============================================================================

import argparse
import os
import sys
import yaml
from pathlib import Path

# ==============================================================================
# Project Root Resolution
# ==============================================================================
# __file__ is in: apps/onset_blending/python/pipelines/blending_process/
# Go up 5 levels to reach the root directory
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Import config and utilities
from ai_weather.config import (
    PROJECT_ROOT,
    YEAR,
    DEFAULT_MODEL_SINGLE,
    DEFAULT_MODEL_ENS,
    ONSET_PROCESSED_DIR,
)
from apps.onset_blending.python.blending_process.connect_utils import make_cv_rds_from_daylevel


# ==============================================================================
# Default Connect Spec
# ==============================================================================
# This is used when no --spec_id or --spec_yaml is provided
DEFAULT_CONNECT_SPEC = {
    "input_rds": str(ONSET_PROCESSED_DIR / str(YEAR) / "clim_combined_wide.pkl"),
    "output_rds": str(ONSET_PROCESSED_DIR / str(YEAR) / "cv_data_clim_mok_date_new_pipeline.pkl"),
    "forecast_models": [
        {"name": DEFAULT_MODEL_ENS},
        {"name": DEFAULT_MODEL_SINGLE},
    ],
}


def load_spec(spec_yaml=None):
    """
    Load a spec dictionary either from a YAML file (--spec_yaml) or from
    the default config.
    """
    if spec_yaml:
        path = Path(spec_yaml)
        if not path.exists():
            raise FileNotFoundError(f"Spec YAML not found: {path}")
        with open(path) as f:
            spec = yaml.safe_load(f)
        return spec

    # Default: use the config.py default
    return DEFAULT_CONNECT_SPEC.copy()


def main():
    parser = argparse.ArgumentParser(
        description="Convert daily combined data to weekly-bin RDS for blending. "
                    "Zero arguments — defaults from etagrometaix.config.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--spec_yaml", default=None,
        help="Path to a temporary YAML spec file (written by the orchestrator). "
             "If not provided, uses DEFAULT_CONNECT_SPEC from config.py."
    )
    args = parser.parse_args()

    spec = load_spec(spec_yaml=args.spec_yaml)

    wide_df = make_cv_rds_from_daylevel(spec=spec)

    print(f"Wrote: {spec['output_rds']}")
    print(f"Rows: {len(wide_df)}, Cols: {len(wide_df.columns)}")
    print(list(wide_df.columns))


if __name__ == "__main__":
    main()