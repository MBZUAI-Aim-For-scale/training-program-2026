#!/usr/bin/env python3
# ==============================================================================
# Script: apps/onset_blending/python/pipelines/prepare_data/1_process_raw_nc_files.py
# ==============================================================================
# Purpose
#   Process raw NetCDF rainfall files into monsoon-onset summary outputs.
#   Loads the spec from ai_weather.config (or from a --spec_yaml temp file
#   written by run_operational_pipeline.py / run_training.py) and processes
#   each NetCDF file year-by-year, writing per-year results to combined outputs.
#
# Usage (zero arguments — defaults to processing AIFS data from config.py)
#   python -m apps.onset_blending.python.pipelines.prepare_data.1_process_raw_nc_files
#
#   # Or with a temp YAML written by the orchestrator:
#   python -m ai_weather.apps.onset_blending.python.pipelines.prepare_data.1_process_raw_nc_files \
#       --spec_yaml /tmp/spec_xxxxx.yml
#
#   # Or select a different spec from config.py:
#   python -m apps.onset_blending.python.pipelines.prepare_data.1_process_raw_nc_files \
#       --spec_id gencast
# ==============================================================================
import sys
import argparse
import yaml
from pathlib import Path

# ==============================================================================
# Project Root Resolution
# ==============================================================================
# __file__ is in: apps/onset_blending/python/pipelines/prepare_data/
# Go up 5 levels to reach the root directory

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Import spec dictionaries from config
from ai_weather.config import (
    ONSET_FORECAST_AIFS_SPEC,
    ONSET_FORECAST_AIFS_ENS_SPEC,
    ONSET_FORECAST_GENCAST_SPEC,
    ONSET_FORECAST_NGCM_SPEC,
    ONSET_FORECAST_IMD_CLIM_SPEC,
)
from apps.onset_blending.python.prepare_data.nc_utils import run_single_pipeline


# Map spec_id names to the config dictionaries
SPEC_REGISTRY = {
    "aifs": ONSET_FORECAST_AIFS_SPEC,
    "aifs_ens": ONSET_FORECAST_AIFS_ENS_SPEC,
    "gencast": ONSET_FORECAST_GENCAST_SPEC,
    "ngcm": ONSET_FORECAST_NGCM_SPEC,
    "imd": ONSET_FORECAST_IMD_CLIM_SPEC,
    "imd_clim": ONSET_FORECAST_IMD_CLIM_SPEC,
    "imd_clim_mok_date": ONSET_FORECAST_IMD_CLIM_SPEC,
}

# Default spec to use when no arguments are provided
DEFAULT_SPEC_ID = "aifs"


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
            f"Available: {', '.join(sorted(SPEC_REGISTRY.keys()))}"
        )
    spec = SPEC_REGISTRY[key].copy()
    spec["id"] = spec_id or DEFAULT_SPEC_ID
    return spec


def main():
    parser = argparse.ArgumentParser(
        description="Process raw NetCDF files into onset tables. "
                    "Zero arguments — defaults to AIFS from ai_weather.config.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-i", "--spec_id", default=None,
        help=f"Spec ID name (one of: {', '.join(sorted(SPEC_REGISTRY.keys()))}). "
             f"Default: {DEFAULT_SPEC_ID}. Loads from ai_weather.config."
    )
    parser.add_argument(
        "--spec_yaml", default=None,
        help="Path to a temporary YAML spec file (written by the orchestrator). "
             "Takes priority over --spec_id."
    )
    args = parser.parse_args()

    spec = load_spec(spec_id=args.spec_id, spec_yaml=args.spec_yaml)
    run_single_pipeline(spec)


if __name__ == "__main__":
    main()
