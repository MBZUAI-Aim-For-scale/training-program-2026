"""
run_operational_pipeline.py
Runs the full operational blending pipeline for a given forecast year/issue date.
Usage (all defaults come from ai_weather.config):
# Zero arguments — uses YEAR, DEFAULT_ISSUE_DATE, DEFAULT_MODEL_SINGLE, etc.
python -m apps.onset_blending.predict.run_operational_pipeline
# Override only what you need
python -m apps.onset_blending.predict.run_operational_pipeline \
    --year 2026 --issue_date 2026-07-01 --model_single gencast
# Full override
python -m apps.onset_blending.predict.run_operational_pipeline \
    --year 2026 --issue_date 2026-06-15 --model_single aifs --model_ens aifs_ens \
    --work_dir /custom/path --coef_dir /custom/coef --coef_tag my_tag
"""
import os
import sys
import argparse
import subprocess
import atexit
import tempfile
import yaml
from pathlib import Path
from datetime import datetime
import copy

# ==============================================================================
# Project Root Resolution
# ==============================================================================
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Import EVERYTHING from config — no more hardcoded paths or tags
from ai_weather.blending_config import merge_spec, validate_runtime_overrides

from ai_weather.config import (
    YEAR,
    DEFAULT_WORK_DIR,
    DEFAULT_COEF_DIR,
    DEFAULT_COEF_TAG,
    DEFAULT_BLEND_MODEL,
    DEFAULT_REGION,
    DEFAULT_ISSUE_DATE,
    DEFAULT_MODEL_SINGLE,
    DEFAULT_MODEL_ENS,
    ONSET_FORECAST_AIFS_SPEC,
    ONSET_FORECAST_AIFS_ENS_SPEC,
    ONSET_FORECAST_GENCAST_SPEC,
    ONSET_FORECAST_NGCM_SPEC,
    ONSET_FORECAST_IMD_CLIM_SPEC,
    ONSET_FORECAST_COMBINE_CLIM_SPEC,
    ONSET_FORECAST_COMBINE_NGCM_AIFS_SPEC,
    ONSET_FORECAST_COMBINE_GENCAST_AIFS_SPEC,
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
    log(f"--- Step {step_num}/{total}: {name} ---")
    log(f"Command: {' '.join(str(c) for c in cmd)}")

    if dry_run:
        log("(dry run - skipping execution)")
        return

    # Resolve repository packages from every stage subprocess.
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")

    # A resumed run must not mistake a stale artifact for this step's output.
    previous = Path(expected_output).stat().st_mtime_ns if expected_output and Path(expected_output).exists() else None
    result = subprocess.run(cmd, cwd=REPO_ROOT, env=env)
    if result.returncode != 0:
        abort(f"Step '{name}' failed with exit code {result.returncode}.")

    if expected_output:
        check_output_exists(expected_output, name)
        if previous is not None and Path(expected_output).stat().st_mtime_ns == previous:
            abort(f"Step '{name}' did not refresh its expected output: {expected_output}")

    log(f"Step {step_num}/{total} complete.\n")


# ==============================================================================
# Main
# ==============================================================================
def _main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run the full operational blending pipeline. "
                    "All defaults come from ai_weather.config — override only what you need.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # EVERYTHING has a default from config.py now
    parser.add_argument("--year", type=int, default=YEAR,
                        help=f"Forecast year (default: {YEAR} from config)")
    parser.add_argument("--issue_date", default=DEFAULT_ISSUE_DATE,
                        help=f"Issue date YYYY-MM-DD (default: {DEFAULT_ISSUE_DATE} from config)")
    parser.add_argument("--model_single", default=DEFAULT_MODEL_SINGLE,
                        choices=["aifs", "gencast", "ngcm"],
                        help=f"Single deterministic model (default: {DEFAULT_MODEL_SINGLE})")
    parser.add_argument("--model_ens", default=None,
                        choices=["aifs_ens", "aifs"],
                        help=f"Ensemble model (default: {DEFAULT_MODEL_ENS})")
    parser.add_argument("--work_dir", default=None,
                        help=f"Working directory (default: {DEFAULT_WORK_DIR})")
    parser.add_argument("--coef_dir", default=str(DEFAULT_COEF_DIR),
                        help=f"Coefficients directory (default: {DEFAULT_COEF_DIR})")
    parser.add_argument("--coef_tag", default=DEFAULT_COEF_TAG,
                        help=f"Coefficient tag (default: {DEFAULT_COEF_TAG})")
    parser.add_argument("--blend_model", default=DEFAULT_BLEND_MODEL,
                        help=f"Blend model name (default: {DEFAULT_BLEND_MODEL})")
    parser.add_argument("--region", default=DEFAULT_REGION,
                        help=f"Region for maps (default: {DEFAULT_REGION})")

    # Optional overrides
    parser.add_argument("--aifs_nc_folder", default=None,
                        help="Override NetCDF folder for single model")
    parser.add_argument("--aifs_ens_nc_folder", default=None,
                        help="Override NetCDF folder for ensemble model")
    parser.add_argument("--gt_path", default=None,
                        help="Override ground truth path")
    parser.add_argument("--blend_input", default=None,
                        help="Override blend input path")

    # Control flags
    parser.add_argument("--skip_to", type=int, default=1,
                        help="Skip to this step number (1-based)")
    parser.add_argument("--stop_at", type=int, default=None,
                        help="Stop after this step number")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print commands without executing")

    parser.add_argument("--blend_spec", help="Override the trained model formula YAML")
    parser.add_argument("--check_inputs", action="store_true",
                        help="Inspect forecast calendars and shared dates without running the pipeline")
    parser.add_argument("--runtime_config", help="YAML/JSON per-stage overrides from the dashboard")
    args = parser.parse_args(argv)
    runtime = {}
    if args.runtime_config:
        with open(Path(args.runtime_config).expanduser()) as stream:
            runtime = validate_runtime_overrides(yaml.safe_load(stream))
    try:
        parsed_date = datetime.strptime(args.issue_date, "%Y-%m-%d").date()
    except ValueError:
        parser.error("issue_date must be YYYY-MM-DD")
    if parsed_date.year != args.year:
        parser.error("issue_date must be in the forecast year")
    if not 1 <= args.skip_to <= 8 or (args.stop_at is not None and not args.skip_to <= args.stop_at <= 8):
        parser.error("Step range must satisfy 1 <= skip_to <= stop_at <= 8")
    expected_ens = DEFAULT_MODEL_ENS if args.model_single == "aifs" else "aifs"
    args.model_ens = args.model_ens or expected_ens
    if args.model_ens != expected_ens:
        parser.error(f"{args.model_single} requires companion model {expected_ens}")
    args.work_dir = str(Path(args.work_dir or Path(DEFAULT_WORK_DIR).parent / str(args.year)).expanduser().resolve())
    args.coef_dir = str(Path(args.coef_dir).expanduser().resolve())
    for key in ("aifs_nc_folder", "aifs_ens_nc_folder", "gt_path", "blend_input", "blend_spec"):
        value = getattr(args, key)
        if value:
            setattr(args, key, str(Path(value).expanduser().resolve()))

    year = args.year
    issue_date = args.issue_date
    work_dir = args.work_dir
    date_compact = issue_date.replace("-", "")

    if args.check_inputs:
        from ai_weather.blending_inputs import inspect_blending_inputs

        report = inspect_blending_inputs({**vars(args), "runtime_overrides": runtime})
        for record in report["records"]:
            label = "OK" if record["valid"] else "MISMATCH"
            log(f"{record['source']}: {Path(record['file']).name} | "
                f"filename year {record['filename_year']} | decoded years {record['actual_years']} | {label}")
        dates = sorted(report["common_dates"], key=lambda value: abs((datetime.fromisoformat(value).date() - parsed_date).days))[:10]
        log("Nearby shared dates from correctly labelled files: " + (", ".join(sorted(dates)) or "none"))
        for error in report["errors"]:
            log(error, level="ERROR")
        if not report["ok"]:
            raise SystemExit(1)
        log("Input parameters and forecast calendars are consistent.")
        return

    if not args.dry_run:
        os.makedirs(work_dir, exist_ok=True)

    # --- Select base specs based on model choice ---
    spec_map = {
        "aifs": (ONSET_FORECAST_AIFS_SPEC, ONSET_FORECAST_COMBINE_CLIM_SPEC),
        "gencast": (ONSET_FORECAST_GENCAST_SPEC, ONSET_FORECAST_COMBINE_GENCAST_AIFS_SPEC),
        "ngcm": (ONSET_FORECAST_NGCM_SPEC, ONSET_FORECAST_COMBINE_NGCM_AIFS_SPEC),
    }
    if args.model_single not in spec_map:
        abort(f"Unknown model_single: {args.model_single}")

    single_spec_base, combine_spec_base = spec_map[args.model_single]
    ens_spec_base = ONSET_FORECAST_AIFS_ENS_SPEC if args.model_ens == "aifs_ens" else ONSET_FORECAST_AIFS_SPEC
    single_spec_base = merge_spec(single_spec_base, runtime.get("single", {}))
    ens_spec_base = merge_spec(ens_spec_base, runtime.get("ensemble", {}))
    clim_spec_base = merge_spec(ONSET_FORECAST_IMD_CLIM_SPEC, runtime.get("climatology", {}))
    combine_spec_base = merge_spec(combine_spec_base, runtime.get("combine", {}))
    validate_runtime_overrides({
        "single": single_spec_base, "ensemble": ens_spec_base,
        "climatology": clim_spec_base, "combine": combine_spec_base,
    })

    # --- Patch single model spec (with output.basename for correct filename) ---
    single_patches = [
        ("output.out_dir", work_dir),
        ("output.basename", f"{args.model_single}_{year}"),
        ("input.file_regex", single_spec_base["input"]["file_regex"].replace(str(YEAR), str(year))),
    ]
    if args.aifs_nc_folder:
        single_patches.append(("input.nc_folder", args.aifs_nc_folder))
    single_spec = patch_spec(single_spec_base, single_patches)
    single_spec_yaml = write_temp_yaml(single_spec, f"{args.model_single}_{year}")

    # --- Patch ensemble model spec (with output.basename for correct filename) ---
    ens_patches = [
        ("output.out_dir", work_dir),
        ("output.basename", f"{args.model_ens}_{year}"),
        ("input.file_regex", ens_spec_base["input"]["file_regex"].replace(str(YEAR), str(year))),
    ]
    if args.aifs_ens_nc_folder:
        ens_patches.append(("input.nc_folder", args.aifs_ens_nc_folder))
    ens_spec = patch_spec(ens_spec_base, ens_patches)
    ens_spec_yaml = write_temp_yaml(ens_spec, f"{args.model_ens}_{year}")

    # --- Patch clim spec ---
    clim_patches = [
        ("output.out_dir", work_dir),
        ("paths.climatology_out_dir", work_dir),
        ("input.gt_path", args.gt_path or clim_spec_base["input"]["gt_path"]),
    ]
    clim_spec = patch_spec(clim_spec_base, clim_patches)
    for options in clim_spec["climatologies"].values():
        options["test_year_min"] = year
        options["test_year_max"] = year
    clim_spec_yaml = write_temp_yaml(clim_spec, f"imd_clim_{year}")

    # --- Expected output paths ---
    single_pkl = os.path.join(work_dir, f"{args.model_single}_{year}_wide.pkl")
    ens_pkl = os.path.join(work_dir, f"{args.model_ens}_{year}_wide.pkl")
    clim_rds = os.path.join(work_dir, clim_spec["climatologies"]["clim"]["out_stem"] + ".pkl")
    clim_unc_rds = os.path.join(work_dir, clim_spec["climatologies"]["clim_unc"]["out_stem"] + ".pkl")

    # --- Patch combine spec ---
    combine_spec = copy.deepcopy(combine_spec_base)
    combine_spec["output"]["out_dir"] = work_dir
    # Note: 3_combine_datasets.py appends "_combined_wide.pkl" to basename,
    # so we do NOT include "_combined" here
    combine_spec["output"]["basename"] = f"{args.model_single}_{args.model_ens}_{year}"

    combine_spec["input"]["ground_truth_wide_rds"] = clim_spec["input"]["gt_path"]

    combine_spec["input"]["climatologies"]["clim"]["rds"] = clim_rds
    combine_spec["input"]["climatologies"]["clim_unc"]["rds"] = clim_unc_rds

    if args.model_single in combine_spec["forecasts"]:
        combine_spec["forecasts"][args.model_single]["sources"][0]["file"] = single_pkl
        combine_spec["forecasts"][args.model_single]["sources"][0]["years"] = f"{year}:{year}"

    if args.model_ens in combine_spec["forecasts"]:
        combine_spec["forecasts"][args.model_ens]["sources"][0]["file"] = ens_pkl
        combine_spec["forecasts"][args.model_ens]["sources"][0]["years"] = f"{year}:{year}"

    combine_spec_yaml = write_temp_yaml(combine_spec, f"combine_{year}")

    # --- Build connect spec ---
    combine_basename = f"{args.model_single}_{args.model_ens}_{year}_combined_wide.pkl"
    connect_input_rds = os.path.join(work_dir, combine_basename)
    connect_output_rds = args.blend_input or os.path.join(
        work_dir, f"cv_data_clim_mok_date_new_pipeline_{year}.pkl")

    # Keep rain predictor windows aligned with the configured onset definition.
    connect_spec = {
        "input_rds": connect_input_rds,
        "output_rds": connect_output_rds,
        "forecast_models": [
            {
                "name": args.model_ens,
                "variants": ["clim_mok_date"],
                "rain_predictors": [{"agg": "diff", "window": ens_spec["options"]["window"]}],
            },
            {
                "name": args.model_single,
                "variants": ["clim_mok_date"],
                "rain_predictors": [{"agg": "diff", "window": single_spec["options"]["window"]}],
            },
        ],
    }
    connect_spec_yaml = write_temp_yaml(connect_spec, f"connect_{year}")

    # Use the formula used in training; operational prediction must apply saved coefficients.
    suffix = "" if args.model_single == "aifs" else f"_{args.model_single}"
    blend_path = args.blend_spec or REPO_ROOT / "apps/onset_blending/specs/2025_blend" / f"cv_models_clim_mok_date_2026{suffix}.yml"
    with open(blend_path) as stream:
        blend_spec = yaml.safe_load(stream)
    blend_spec["run"]["cv_holdout_years"] = [year]
    blend_spec_yaml = write_temp_yaml(blend_spec, f"blend_{year}")

    if not args.dry_run and args.skip_to <= 6 and (args.stop_at is None or args.stop_at >= 6):
        coefficient = Path(args.coef_dir) / f"coefs_{args.blend_model}_global_{args.coef_tag}.pkl"
        if not coefficient.is_file():
            abort(f"Trained coefficient file not found: {coefficient}")

    if not args.dry_run:
        from apps.onset_blending.python.prepare_data.nc_utils import validate_forecast_inputs

        stop = args.stop_at or 8
        for stage, forecast_spec in ((2, single_spec), (3, ens_spec)):
            if args.skip_to <= stage <= stop:
                validate_forecast_inputs(forecast_spec, issue_date if stop >= 7 else None)

    # --- Final output paths ---
    preds_pkl = os.path.join(work_dir, f"{args.blend_model}_global_year{year}_preds.pkl")
    export_csv = os.path.join(work_dir, f"blend_output_summary_{date_compact}.csv")

    TOTAL = 8
    steps = [
        (1, "Build climatology", [
            sys.executable, "-m",
            "apps.onset_blending.python.pipelines.prepare_data.2_build_climatology",
            "--spec_yaml", clim_spec_yaml,
        ], clim_rds),

        (2, f"Process {args.model_single} nc files", [
            sys.executable, "-m",
            "apps.onset_blending.python.pipelines.prepare_data.1_process_raw_nc_files",
            "--spec_yaml", single_spec_yaml,
        ], single_pkl),

        (3, f"Process {args.model_ens} nc files", [
            sys.executable, "-m",
            "apps.onset_blending.python.pipelines.prepare_data.1_process_raw_nc_files",
            "--spec_yaml", ens_spec_yaml,
        ], ens_pkl),

        (4, "Combine datasets", [
            sys.executable, "-m",
            "apps.onset_blending.python.pipelines.prepare_data.3_combine_datasets",
            "--spec_yaml", combine_spec_yaml,
        ], connect_input_rds),

        (5, "Connect/prepare pipeline input", [
            sys.executable, "-m",
            "apps.onset_blending.python.pipelines.blending_process.0_connect_prepare_data_to_pipeline",
            "--spec_yaml", connect_spec_yaml,
        ], connect_output_rds),

        (6, "Apply trained blend model", [
            sys.executable, "-m",
            "apps.onset_blending.predict.apply_blend_model",
            "--spec_yaml", blend_spec_yaml,
            "--input_path", connect_output_rds,
            "--model", args.blend_model, "--year", str(year),
            "--coef_dir", args.coef_dir, "--coef_tag", args.coef_tag,
            "--out_dir", work_dir,
        ], preds_pkl),

        (7, "Export blend output", [
            sys.executable, "-m",
            "apps.onset_blending.predict.export_blend_output",
            "--issue_date", issue_date,
            "--spec_yaml", blend_spec_yaml,
            "--preds_file", preds_pkl,
            "--out_dir", work_dir,
        ], export_csv),

        (8, "Generate maps", [
            sys.executable, "-m",
            "apps.onset_blending.predict.run_maps",
            "--input_file", export_csv,
            "--output_path", work_dir,
            "--region", args.region,
        ], os.path.join(work_dir, "maps", f"prob_weeks1-4_{date_compact}.png")),
    ]

    map_options = runtime.get("maps", {})
    map_command = steps[-1][2]
    for key in ("country_boundary", "district_boundary", "onset_file"):
        if map_options.get(key):
            map_command.extend([f"--{key}", str(map_options[key])])
    if map_options.get("zoom_to_data"):
        map_command.append("--zoom_to_data")

    log(f"Starting operational pipeline")
    log(f"  year        : {year}")
    log(f"  issue_date  : {issue_date}")
    log(f"  models      : {args.model_single} + {args.model_ens}")
    log(f"  work_dir    : {work_dir}")
    log(f"  coef_dir    : {args.coef_dir}")
    log(f"  coef_tag    : {args.coef_tag}")

    if args.dry_run:
        log("DRY RUN - commands will be printed but not executed")
    print()

    for step_num, name, cmd, expected_output in steps:
        if step_num < args.skip_to:
            log(f"--- Step {step_num}/{TOTAL}: {name} [SKIPPED] ---")
            continue

        if args.stop_at is not None and step_num > args.stop_at:
            log(f"--- Step {step_num}/{TOTAL}: {name} [SKIPPED - stop_at={args.stop_at}] ---")
            continue

        run_step(step_num, TOTAL, name, cmd, expected_output, args.dry_run)

    if not args.dry_run:
        log(f"Pipeline complete. Outputs in: {work_dir}")
    else:
        log("Dry run complete - no pipeline outputs were created.")


def main(argv=None):
    try:
        return _main(argv)
    finally:
        _cleanup_temp_specs()
        _temp_spec_files.clear()


if __name__ == "__main__":
    main()