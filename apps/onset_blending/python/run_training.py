#!/usr/bin/env python3
"""
# File: apps/onset_forecast/python/run_training.py
===============
Runs the full blending training pipeline in a single command. 
All defaults are loaded from ai_weather.config. Override only what you need.

Usage (run from repo root)
--------------------------
    # Zero arguments — uses all defaults from config.py
    python -m ai_weather.apps.onset_forecast.python.run_training

    # With overrides
    python -m ai_weather.apps.onset_forecast.python.run_training \\
        --model_single gencast \\
        --work_dir /custom/work/dir \\
        --results_dir /custom/results/dir \\
        --aifs_nc_folder /path/to/custom/nc/files
"""

import os
import sys
import argparse
import subprocess
import atexit
import tempfile
import yaml
import copy
from datetime import datetime
from pathlib import Path

# ==============================================================================
# Project Root Resolution
# ==============================================================================
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))




# ==============================================================================
# Project Root & Config Imports
# ==============================================================================
from ai_weather.config import (
    PROJECT_ROOT,
    ONSET_FORECAST_IMD_CLIM_SPEC,
    ONSET_FORECAST_AIFS_SPEC,
    ONSET_FORECAST_AIFS_ENS_SPEC,
    ONSET_FORECAST_GENCAST_SPEC,
    ONSET_FORECAST_NGCM_SPEC,
    ONSET_FORECAST_COMBINE_CLIM_SPEC,
    ONSET_FORECAST_COMBINE_GENCAST_AIFS_SPEC,
    ONSET_FORECAST_COMBINE_NGCM_AIFS_SPEC,
)

# ==============================================================================
# Temp spec management
# ==============================================================================
_temp_spec_files = []

def _cleanup_temp_specs():
    for path in _temp_spec_files:
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            pass

atexit.register(_cleanup_temp_specs)

def _set_nested(obj, keys, value):
    for k in keys[:-1]:
        if isinstance(obj, list):
            k = int(k)
        if isinstance(obj, dict):
            if k not in obj:
                obj[k] = {}
            obj = obj[k]
        elif isinstance(obj, list):
            obj = obj[k]
        else:
            raise TypeError(f"Cannot navigate into {type(obj)} with key {k}")
    last_key = keys[-1]
    if isinstance(obj, list):
        last_key = int(last_key)
    obj[last_key] = value

def patch_spec(base_spec, patches):
    spec = copy.deepcopy(base_spec)
    for dotted_key, value in patches:
        keys = dotted_key.split(".")
        _set_nested(spec, keys, value)
    return spec

def write_temp_yaml(spec_dict, prefix="spec"):
    fd, path = tempfile.mkstemp(suffix=".yml", prefix=f"{prefix}_")
    with os.fdopen(fd, 'w') as f:
        yaml.dump(spec_dict, f, default_flow_style=False, allow_unicode=True)
    _temp_spec_files.append(path)
    return path

# ==============================================================================
# Helpers
# ==============================================================================
def log(msg, level="INFO"):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] [{level}] {msg}", flush=True)

def abort(msg):
    log(msg, level="ERROR")
    sys.exit(1)

def check_output_exists(path, step_name):
    if not os.path.exists(path):
        abort(f"Step '{step_name}' completed but expected output not found:\n  {path}")
    log(f"Output verified: {path}")

def run_step(step_num, total, name, cmd, expected_output, dry_run):
    log(f"── Step {step_num}/{total}: {name} ──────────────────────────────")
    log(f"Command: {' '.join(str(c) for c in cmd)}")
    if dry_run:
        log("(dry run — skipping execution)")
        return
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        abort(f"Step '{name}' failed with exit code {result.returncode}.")
    if expected_output:
        check_output_exists(expected_output, name)
    log(f"Step {step_num}/{total} complete.\n")

# ==============================================================================
# Main
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Run the full blending training pipeline. Defaults from config.py.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--model_single", default="aifs", choices=["aifs", "gencast", "ngcm"],
                        help="Deterministic forecast model (default: aifs)")
    parser.add_argument("--model_ens", default="aifs_ens", choices=["aifs_ens", "aifs"],
                        help="Ensemble forecast model (default: aifs_ens)")
    
    parser.add_argument("--work_dir", default=str(PROJECT_ROOT / "data" / "onset_info" / "Monsoon_Data" / "Processed_Data" / "training"),
                        help="Working directory for intermediate files")
    parser.add_argument("--results_dir", default=str(PROJECT_ROOT / "data" / "onset_info" / "Monsoon_Data" / "results" / "dry_spell_aifs_aifs_ens"),
                        help="Output directory for blend evaluation results")
    
    parser.add_argument("--aifs_nc_folder", default=None, help="Override input.nc_folder for single model")
    parser.add_argument("--aifs_ens_nc_folder", default=None, help="Override input.nc_folder for ensemble model")
    parser.add_argument("--gt_path", default=None, help="Override ground truth wide pkl path")
    parser.add_argument("--blend_input", default=None, help="Override wide pipeline pkl for blend evaluation")
    
    parser.add_argument("--cores", type=int, default=None, help="Number of cores for blend evaluation")
    parser.add_argument("--skip_to", type=int, default=1, help="Skip to step N (1-indexed)")
    parser.add_argument("--stop_at", type=int, default=None, help="Stop after step N (1-indexed)")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without executing")

    args = parser.parse_args()

    work_dir = args.work_dir
    results_dir = args.results_dir
    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # --- Select base specs based on model choice ---
    spec_map = {
        "aifs": ONSET_FORECAST_AIFS_SPEC,
        "gencast": ONSET_FORECAST_GENCAST_SPEC,
        "ngcm": ONSET_FORECAST_NGCM_SPEC,
    }
    single_spec_base = spec_map[args.model_single]
    
    if args.model_single == "aifs":
        combine_spec_base = ONSET_FORECAST_COMBINE_CLIM_SPEC
    elif args.model_single == "gencast":
        combine_spec_base = ONSET_FORECAST_COMBINE_GENCAST_AIFS_SPEC
    elif args.model_single == "ngcm":
        combine_spec_base = ONSET_FORECAST_COMBINE_NGCM_AIFS_SPEC

    # --- Patch single model spec ---
    single_patches = [("output.out_dir", work_dir)]
    if args.aifs_nc_folder:
        single_patches.append(("input.nc_folder", args.aifs_nc_folder))
    single_spec = patch_spec(single_spec_base, single_patches)
    single_spec_yaml = write_temp_yaml(single_spec, f"{args.model_single}")

    # --- Patch ensemble model spec ---
    ens_patches = [("output.out_dir", work_dir)]
    if args.aifs_ens_nc_folder:
        ens_patches.append(("input.nc_folder", args.aifs_ens_nc_folder))
    ens_spec = patch_spec(ONSET_FORECAST_AIFS_ENS_SPEC, ens_patches)
    ens_spec_yaml = write_temp_yaml(ens_spec, f"{args.model_ens}")

    # --- Patch clim spec ---
    clim_gt_path = args.gt_path or ONSET_FORECAST_IMD_CLIM_SPEC["input"]["gt_path"]
    clim_patches = [
        ("output.out_dir", work_dir),
        ("paths.climatology_out_dir", work_dir),
        ("input.gt_path", clim_gt_path),
    ]
    clim_spec = patch_spec(ONSET_FORECAST_IMD_CLIM_SPEC, clim_patches)
    clim_spec_yaml = write_temp_yaml(clim_spec, "clim")
    
    clim_rds = os.path.join(work_dir, clim_spec["climatologies"]["clim"]["out_stem"] + ".pkl")
    clim_unc_rds = os.path.join(work_dir, clim_spec["climatologies"]["clim_unc"]["out_stem"] + ".pkl")

    # --- Patch combine spec ---
    combine_spec = copy.deepcopy(combine_spec_base)
    combine_spec["output"]["out_dir"] = work_dir
    if args.gt_path:
        combine_spec["input"]["ground_truth_wide_rds"] = args.gt_path
    combine_spec["input"]["climatologies"]["clim"]["rds"] = clim_rds
    combine_spec["input"]["climatologies"]["clim_unc"]["rds"] = clim_unc_rds
    
    single_pkl = os.path.join(work_dir, f"{args.model_single}_wide.pkl")
    ens_pkl = os.path.join(work_dir, f"{args.model_ens}_wide.pkl")
    
    if args.model_single in combine_spec["forecasts"]:
        combine_spec["forecasts"][args.model_single]["sources"][0]["file"] = single_pkl
    if args.model_ens in combine_spec["forecasts"]:
        combine_spec["forecasts"][args.model_ens]["sources"][0]["file"] = ens_pkl
        
    combine_spec_yaml = write_temp_yaml(combine_spec, "combine")

    # --- Build connect spec (dynamic) ---
    combine_basename = "combine_combined_wide.pkl"
    connect_input_rds = os.path.join(work_dir, combine_basename)
    connect_output_rds = args.blend_input or os.path.join(work_dir, "cv_data_clim_mok_date_new_pipeline.pkl")
    
    connect_spec = {
        "input_rds": connect_input_rds,
        "output_rds": connect_output_rds,
        "forecast_models": [{"name": args.model_ens}, {"name": args.model_single}],
    }
    connect_spec_yaml = write_temp_yaml(connect_spec, "connect")

    # --- Build blend spec (dynamic) ---
    blend_spec = {
        "mme": {"blend_models": [{"name": args.model_ens}, {"name": args.model_single}]},
        "extras": {"forecasts": [{"name": args.model_ens}, {"name": args.model_single}]},
        "models": {"formulas": {"default": {
            "enabled": True,
            "text": f"diff_{args.model_ens}_qx + diff_{args.model_single}_qx"
        }}},
        "run": {
            "pipeline_input_dir": work_dir,
            "pipeline_output_dir": results_dir
        }
    }
    blend_spec_yaml = write_temp_yaml(blend_spec, "blend")

    # --- Expected output paths ---
    imd_pkl = os.path.join(work_dir, "imd_clim_mok_date_wide.pkl")
    connect_pkl = connect_output_rds
    TOTAL = 7

    # --- Build steps list ---
    blend_eval_cmd = [
        sys.executable,
        "apps/onset_blending/python/pipelines/blending_process/1_blend_evaluation.py",
        "--spec_yaml", blend_spec_yaml,
        "--work_dir", work_dir,
        "--results_dir", results_dir,
    ]
    if args.cores is not None:
        blend_eval_cmd += ["--cores", str(args.cores)]

    steps = [
        (1, "Process IMD ground truth nc files", [
            sys.executable,
            "apps/onset_blending/python/pipelines/prepare_data/1_process_raw_nc_files.py",
            "--spec_yaml", clim_spec_yaml,
        ], imd_pkl),

        (2, f"Process {args.model_single} nc files", [
            sys.executable,
            "apps/onset_blending/python/pipelines/prepare_data/1_process_raw_nc_files.py",
            "--spec_yaml", single_spec_yaml,
        ], single_pkl),

        (3, f"Process {args.model_ens} nc files", [
            sys.executable,
            "apps/onset_blending/python/pipelines/prepare_data/1_process_raw_nc_files.py",
            "--spec_yaml", ens_spec_yaml,
        ], ens_pkl),

        (4, "Build climatology", [
            sys.executable,
            "apps/onset_blending/python/pipelines/prepare_data/2_build_climatology.py",
            "--spec_yaml", clim_spec_yaml,
        ], clim_rds),

        (5, "Combine datasets", [
            sys.executable,
            "apps/onset_blending/python/pipelines/prepare_data/3_combine_datasets.py",
            "--spec_yaml", combine_spec_yaml,
        ], None),

        (6, "Connect/prepare pipeline input", [
            sys.executable,
            "apps/onset_blending/python/pipelines/blending_process/0_connect_prepare_data_to_pipeline.py",
            "--spec_yaml", connect_spec_yaml,
        ], connect_pkl),

        (7, "Blend evaluation", blend_eval_cmd, None),
    ]

    # --- Run ---
    log("Starting training pipeline")
    log(f"Work dir    : {work_dir}")
    log(f"Results dir : {results_dir}")
    log(f"Models      : {args.model_single} + {args.model_ens}")
    
    if args.aifs_nc_folder:
        log(f"{args.model_single} nc_folder override: {args.aifs_nc_folder}")
    if args.aifs_ens_nc_folder:
        log(f"{args.model_ens} nc_folder override: {args.aifs_ens_nc_folder}")
    if args.gt_path:
        log(f"gt_path override: {args.gt_path}")
    if args.skip_to > 1:
        log(f"Skipping steps 1–{args.skip_to - 1} (--skip_to {args.skip_to})")
    if args.stop_at is not None:
        log(f"Stopping after step {args.stop_at} (--stop_at {args.stop_at})")
    if args.dry_run:
        log("DRY RUN — commands will be printed but not executed")
    print()

    for step_num, name, cmd, expected_output in steps:
        if step_num < args.skip_to:
            log(f"── Step {step_num}/{TOTAL}: {name} [SKIPPED] ──")
            continue

        if args.stop_at is not None and step_num > args.stop_at:
            log(f"── Step {step_num}/{TOTAL}: {name} [SKIPPED — stop_at={args.stop_at}] ──")
            continue

        run_step(step_num, TOTAL, name, cmd, expected_output, args.dry_run)

    if not args.dry_run:
        log("Pipeline complete.")
        log(f"Intermediate files : {work_dir}")
        log(f"Training outputs   : {results_dir}")
    else:
        log("Dry run complete — no files were created.")


if __name__ == "__main__":
    main()