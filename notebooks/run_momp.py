#!/usr/bin/env python3
"""
MOMP / ROMP benchmarking execution script.

Phase 0 + Phase 1 updates:
- Normalized catalog metadata handling
- Deterministic / Probabilistic categorization
- Benchmark spec support through --spec or ROMP_SPEC_PATH
- One model per run
- Fixed sys.argv sanitization to prevent driver.py argument conflicts
- Fixed path resolution to prevent duplicate folder names
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.metadata
import json
import os
import runpy
import subprocess
import sys
from datetime import date
from importlib.metadata import PackageNotFoundError
from pathlib import Path

# ============================================================================
# Path Bootstrap - Makes script runnable from anywhere
# ============================================================================
def _bootstrap_paths() -> Path:
    """
    Makes run_momp.py runnable from anywhere without installing the project.
    Returns the project root directory.
    
    Fixed: Walks up the directory tree to find the actual project root
    (containing config.py, data/, and apps/) instead of hardcoding
    folder names that can cause path duplication.
    """
    env_root = os.getenv("AI_WEATHER_REPO")
    
    if env_root:
        root = Path(env_root).expanduser().resolve()
    else:
        here = Path(__file__).resolve()
        
        # Walk up the directory tree to find the project root
        # The project root is the directory that contains config.py, data/, and apps/
        root = None
        for p in [here.parent] + list(here.parent.parents):
            if (p / "config.py").exists() and (p / "data").exists() and (p / "apps").exists():
                root = p
                break
        
        if root is None:
            # Fallback to here.parent.parent
            root = here.parent.parent
    
    # Add paths to sys.path
    romp_root = root / "apps" / "benchmarking" / "ROMP"
    
    for p in (root, romp_root):
        if p.exists():
            p_str = str(p)
            if p_str not in sys.path:
                sys.path.insert(0, p_str)
    
    # Set environment variables
    os.environ.setdefault("AI_WEATHER_REPO", str(root))
    os.environ.setdefault("PROJ_ROOT", str(root))
    os.environ.setdefault("ROMP_DATA_DIR", str(root / "data"))
    
    return root


PROJECT_ROOT = _bootstrap_paths()

# Now safe to import config
import config

# ============================================================================
# Paths - Use getattr with fallbacks for robustness
# ============================================================================
ROMP_ROOT = config.ROMP_ROOT.resolve()
MOMP_PKG_DIR = config.MOMP_PKG_DIR.resolve()

# Support both naming conventions
ROMP_OUT_DIR = getattr(config, "ROMP_OUT_DIR", getattr(config, "ROMP_DEMO_OUT_DIR", None))
ROMP_FIG_DIR = getattr(config, "ROMP_FIG_DIR", getattr(config, "ROMP_DEMO_FIG_DIR", None))

if ROMP_OUT_DIR is None:
    # Fallback to default paths
    ROMP_OUT_DIR = PROJECT_ROOT / "data" / "ROMP_OUT" / "et" / "output"
    ROMP_FIG_DIR = PROJECT_ROOT / "data" / "ROMP_OUT" / "et" / "figure"

ROMP_OUT_DIR = Path(ROMP_OUT_DIR)
ROMP_FIG_DIR = Path(ROMP_FIG_DIR)

if str(ROMP_ROOT) not in sys.path:
    sys.path.insert(0, str(ROMP_ROOT))

import momp  # noqa: E402,F401
import momp.params as params  # noqa: E402
from momp.lib.loader import get_cfg  # noqa: E402


# ============================================================================
# Lead windows / tolerances
# ============================================================================
LEAD_SETTINGS = {
    "verification_window_list": ((1, 15), (16, 30)),
    "tolerance_days_list": (3, 5),
    "max_forecast_day": 30,
    "day_bins": ((1, 5), (6, 10), (11, 15), (16, 20), (21, 25), (26, 30)),
}

DETERMINISTIC_FLAGS = dict(
    probabilistic=False,
    FAR=True,
    MAE=True,
    MR=True,
    BS=False,
    RPS=False,
    AUC=False,
    Reliability=False,
    skill_score=False,
    plot_spatial_far_mr_mae=True,
    plot_panel_heatmap_error=True,
    plot_heatmap_bss_auc=False,
    plot_reliability=False,
    plot_panel_heatmap_skill=False,
    plot_bar_bss_rpss_auc=False,
)

PROBABILISTIC_FLAGS = dict(
    probabilistic=True,
    members="All",
    onset_percentage_threshold=0.5,
    BS=True,
    RPS=True,
    AUC=True,
    Reliability=True,
    skill_score=True,
    FAR=False,
    MAE=False,
    MR=False,
    plot_heatmap_bss_auc=True,
    plot_reliability=True,
    plot_panel_heatmap_skill=True,
    plot_bar_bss_rpss_auc=True,
    plot_spatial_far_mr_mae=False,
    plot_panel_heatmap_error=False,
)

VALID_MODES = ["det", "deterministic", "prob", "probabilistic"]


# ============================================================================
# Normalization helpers
# ============================================================================
def _clean_str(value):
    return str(value).strip() if value is not None else ""


def _meta_get(meta, key, default=None):
    if not isinstance(meta, dict):
        return default

    if key in meta:
        return meta[key]

    target = key.strip().lower()

    for k, v in meta.items():
        if str(k).strip().lower() == target:
            return v

    return default


def _normalize_category(value):
    if isinstance(value, dict):
        if bool(_meta_get(value, "probabilistic", False)):
            return "Probabilistic"

        category = _clean_str(_meta_get(value, "category", ""))
    else:
        category = _clean_str(value)

    c = category.lower()

    if (
        c.startswith("prob")
        or "ensemble" in c
        or "gencast" in c
        or "aifs_ens" in c
        or "_ens" in c
        or " ens" in c
    ):
        return "Probabilistic"

    if c.startswith("det"):
        return "Deterministic"

    return "Deterministic"


def _model_name(meta, fallback: str = "") -> str:
    name = _clean_str(_meta_get(meta, "model_name", ""))

    if name:
        return name

    label = _clean_str(_meta_get(meta, "label", ""))

    if label and "·" in label:
        return label.split("·")[-1].strip()

    return label or fallback


def _obs_meta(obs_key: str):
    catalog = getattr(config, "BENCHMARK_OBS_CATALOG", {}) or {}

    target = _clean_str(obs_key).upper()

    for key, meta in catalog.items():
        if _clean_str(key).upper() == target:
            return meta

    return None


def resolve_path(p):
    if p is None:
        return None

    p = Path(str(p))

    return str(p if p.is_absolute() else PROJECT_ROOT / p)


# ============================================================================
# Mode helpers
# ============================================================================
def is_prob_mode(mode):
    mode = mode.lower()

    if mode not in VALID_MODES:
        raise ValueError(
            f"Invalid mode '{mode}'. Choose 'det'/'deterministic' or "
            "'prob'/'probabilistic'."
        )

    return mode in ("prob", "probabilistic")


def select_models(mode, model_key=None):
    """
    Return a list of (catalog_key, meta) for the requested mode.
    """
    target = "Probabilistic" if is_prob_mode(mode) else "Deterministic"

    selected = []

    for raw_key, meta in config.BENCHMARK_MODEL_CATALOG.items():
        catalog_key = _clean_str(raw_key)

        if not catalog_key:
            continue

        category = _normalize_category(meta)

        if category != target:
            continue

        selected.append((catalog_key, meta))

    if model_key:
        model_key_clean = _clean_str(model_key).lower()

        selected = [
            (k, m)
            for k, m in selected
            if k.lower() == model_key_clean
            or _model_name(m, k).lower() == model_key_clean
        ]

        if not selected:
            raise ValueError(
                f"Model '{model_key}' not found in category '{target}' of "
                "BENCHMARK_MODEL_CATALOG."
            )

    if not selected:
        raise ValueError(
            f"No models found in BENCHMARK_MODEL_CATALOG for category '{target}'."
        )

    return selected


# ============================================================================
# importlib.metadata patch
# ============================================================================
@contextlib.contextmanager
def patched_package_metadata():
    original_version = importlib.metadata.version
    original_distribution = importlib.metadata.distribution

    class DummyDistribution:
        version = "0.0.1"

    def patched_version(pkg_name):
        if pkg_name.lower() == "momp":
            return "0.0.1"

        try:
            return original_version(pkg_name)
        except PackageNotFoundError:
            return "0.0.1"

    def patched_distribution(pkg_name):
        if pkg_name.lower() == "momp":
            return DummyDistribution()

        return original_distribution(pkg_name)

    importlib.metadata.version = patched_version
    importlib.metadata.distribution = patched_distribution

    try:
        yield
    finally:
        importlib.metadata.version = original_version
        importlib.metadata.distribution = original_distribution


# ============================================================================
# Base paths
# ============================================================================
def apply_base_paths(cfg):
    cfg.work_dir = str(ROMP_ROOT)
    cfg.pkg_dir = str(MOMP_PKG_DIR)

    cfg.obs_dir = resolve_path(
        getattr(cfg, "obs_dir", "data/external/ENACTS_regridded_025")
    )

    cfg.ref_model_dir = resolve_path(
        getattr(cfg, "ref_model_dir", "data/external/ENACTS")
    )

    cfg.nc_mask = resolve_path(
        getattr(cfg, "nc_mask", "data/external/jjas_100mm_rainfall_mask_0p25.nc")
    )

    if getattr(cfg, "shpfile_dir", None):
        cfg.shpfile_dir = resolve_path(cfg.shpfile_dir)

    cfg.ref_model = "climatology"
    cfg.layout = ("model", "verification_window")


# ============================================================================
# Benchmark spec support
# ============================================================================
def _iso_to_tuple(value):
    try:
        d = date.fromisoformat(str(value))
        return (d.year, d.month, d.day)
    except Exception:
        return None


def _apply_forecast_window(cfg, window: str):
    window = str(window).lower()

    if "1-15" in window:
        cfg.verification_window_list = ((1, 15),)
        cfg.tolerance_days_list = (3,)
        cfg.max_forecast_day = 15
        cfg.day_bins = ((1, 5), (6, 10), (11, 15))

    elif "16-30" in window:
        cfg.verification_window_list = ((16, 30),)
        cfg.tolerance_days_list = (5,)
        cfg.max_forecast_day = 30
        cfg.day_bins = ((16, 20), (21, 25), (26, 30))


def apply_spec_to_cfg(cfg, spec: dict, model_key: str):
    """
    Applies BenchmarkSpec settings to MOMP config.
    """
    if not isinstance(spec, dict):
        return

    # Ground truth / observation source
    ground_truth = _clean_str(spec.get("ground_truth", ""))

    if ground_truth:
        obs_meta = _obs_meta(ground_truth)

        if obs_meta:
            cfg.obs = ground_truth

            obs_dir = _clean_str(_meta_get(obs_meta, "dir", ""))

            if obs_dir:
                cfg.obs_dir = str(Path(obs_dir).resolve())

            file_pattern = _clean_str(_meta_get(obs_meta, "file_pattern", "{}.nc"))
            cfg.obs_file_pattern = (file_pattern,)

            cfg.obs_var = _clean_str(_meta_get(obs_meta, "var", "precip"))

    # Benchmark coverage / region
    coverage = _clean_str(spec.get("benchmark_coverage", ""))

    if coverage:
        if coverage.lower().startswith("ethiopia"):
            cfg.region = "Ethiopia"
        else:
            cfg.region = coverage

    # Forecast window
    forecast_window = _clean_str(spec.get("forecast_window", ""))

    if forecast_window:
        _apply_forecast_window(cfg, forecast_window)

    # Shared settings
    shared = spec.get("shared", {})

    if shared.get("wet_day_threshold") is not None:
        cfg.wet_threshold = float(shared["wet_day_threshold"])

    if shared.get("minimum_wet_day_rainfall") is not None:
        cfg.wet_init = float(shared["minimum_wet_day_rainfall"])

    if shared.get("wet_spell_length") is not None:
        cfg.wet_spell = int(shared["wet_spell_length"])

    if shared.get("dry_spell_limit") is not None:
        cfg.dry_spell = int(shared["dry_spell_limit"])

    if shared.get("dry_spell_search_extension") is not None:
        cfg.dry_extent = int(shared["dry_spell_search_extension"])

    area_mask_file = shared.get("area_mask_file")

    if area_mask_file is not None:
        area_mask_file = _clean_str(area_mask_file)

        if area_mask_file.lower() in {"", "none", "no mask"}:
            cfg.nc_mask = None
        else:
            cfg.nc_mask = resolve_path(area_mask_file)

    onset_threshold_file = shared.get("onset_threshold_file")

    if onset_threshold_file is not None:
        onset_threshold_file = _clean_str(onset_threshold_file)

        if onset_threshold_file.lower() in {"", "dataset default", "default"}:
            cfg.thresh_file = None
        else:
            cfg.thresh_file = resolve_path(onset_threshold_file)

    baseline_forecast = _clean_str(shared.get("baseline_forecast", ""))

    if baseline_forecast:
        cfg.ref_model = baseline_forecast

    baseline_data_path = _clean_str(shared.get("baseline_data_path", ""))

    if baseline_data_path:
        cfg.ref_model_dir = resolve_path(baseline_data_path)

    # Model-specific settings
    models = spec.get("models", [])

    model_spec = None

    for m in models:
        exec_key = _clean_str(m.get("exec_key", "")).lower()
        model_name = _clean_str(m.get("model_name", "")).lower()
        target = _clean_str(model_key).lower()

        if exec_key == target or model_name == target:
            model_spec = m
            break

    if not model_spec:
        return

    eval_start = model_spec.get("eval_start")

    if eval_start:
        start_tuple = _iso_to_tuple(eval_start)

        if start_tuple:
            cfg.start_date = start_tuple

    eval_end = model_spec.get("eval_end")

    if eval_end:
        end_tuple = _iso_to_tuple(eval_end)

        if end_tuple:
            cfg.end_date = end_tuple

    if model_spec.get("baseline_start_year") is not None:
        cfg.start_year_clim = int(model_spec["baseline_start_year"])

    if model_spec.get("baseline_end_year") is not None:
        cfg.end_year_clim = int(model_spec["baseline_end_year"])

    init_days = model_spec.get("init_days")

    if init_days is not None:
        try:
            cfg.init_days = tuple(sorted(set(int(x) for x in init_days)))
        except Exception:
            cfg.init_days = (0, 3)

    cfg.run_years_concurrently = bool(model_spec.get("run_years_concurrently", True))

    if model_spec.get("ensemble_forecast") is not None:
        cfg.ensemble_forecast = bool(model_spec["ensemble_forecast"])


# ============================================================================
# Model configuration
# ============================================================================
def configure_model(cfg, catalog_key: str, meta, mode):
    """
    Configure cfg for exactly ONE model.
    """
    name = _model_name(meta, catalog_key)

    cfg.model_list = (name,)

    cfg.model_dir_list = (resolve_path(_meta_get(meta, "dir", "")),)
    cfg.model_var_list = (_clean_str(_meta_get(meta, "var", "tp")),)
    cfg.unit_cvt_list = (_meta_get(meta, "unit_cvt"),)
    cfg.file_pattern_list = (
        _clean_str(_meta_get(meta, "file_pattern", "{}.nc")),
    )

    for key, value in LEAD_SETTINGS.items():
        setattr(cfg, key, value)

    if len(cfg.verification_window_list) != len(cfg.tolerance_days_list):
        raise ValueError(
            "verification_window_list and tolerance_days_list must match in length."
        )

    flags = PROBABILISTIC_FLAGS if is_prob_mode(mode) else DETERMINISTIC_FLAGS

    for key, value in flags.items():
        setattr(cfg, key, value)

    cfg.show_plot = False
    cfg.show_panel = False

    # Use the correct attribute names with fallbacks
    out_dir = ROMP_OUT_DIR / name
    fig_dir = ROMP_FIG_DIR / name

    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    cfg.dir_out = str(out_dir)
    cfg.dir_fig = str(fig_dir)


def sync_params(cfg):
    for attr in dir(cfg):
        if not attr.startswith("_"):
            setattr(params, attr, getattr(cfg, attr))


def print_summary(cfg):
    print(f"work_dir      : {cfg.work_dir}")
    print(f"pkg_dir       : {cfg.pkg_dir}")
    print(f"obs_dir       : {cfg.obs_dir}")
    print(f"ref_model     : {cfg.ref_model}")
    print(f"Probabilistic : {cfg.probabilistic}")
    print(f"Model         : {cfg.model_list}")
    print(f"Model path    : {cfg.model_dir_list}")
    print(f"Windows       : {cfg.verification_window_list}")
    print(f"Tolerances    : {cfg.tolerance_days_list}")
    print(f"dir_out       : {cfg.dir_out}")
    print(f"dir_fig       : {cfg.dir_fig}")
    print("-" * 60)


# ============================================================================
# sys.argv sanitization for driver.py
# ============================================================================
@contextlib.contextmanager
def sanitized_driver_argv(driver_path: Path, mode: str):
    """
    Temporarily replace sys.argv so driver.py / momp-run argparse sees only:

        driver.py --mode det

    instead of:

        run_momp.py --mode det --model AIFS --spec ...

    This fixes:

        momp-run: error: unrecognized arguments: --model AIFS
    """
    original_argv = sys.argv.copy()

    driver_mode = mode.lower()

    if driver_mode == "deterministic":
        driver_mode = "det"

    if driver_mode == "probabilistic":
        driver_mode = "prob"

    sys.argv = [
        str(driver_path),
        "--mode",
        driver_mode,
    ]

    try:
        yield
    finally:
        sys.argv = original_argv


# ============================================================================
# Execution
# ============================================================================
def run_single_model(mode, catalog_key, meta):
    """
    Run the MOMP driver for one model in the current process.
    """
    name = _model_name(meta, catalog_key)

    print("=" * 60)
    print(f"MOMP [{mode.upper()}]  model={name}  (catalog key: {catalog_key})")
    print("=" * 60)

    driver_path = MOMP_PKG_DIR / "driver.py"

    if not driver_path.exists():
        raise FileNotFoundError(f"Driver not found at {driver_path}")

    if hasattr(config, "setup_earthdata_auth"):
        config.setup_earthdata_auth()

    with patched_package_metadata():
        cfg = get_cfg()

        apply_base_paths(cfg)
        configure_model(cfg, catalog_key, meta, mode)

        spec_path = os.getenv("ROMP_SPEC_PATH")

        if spec_path and Path(spec_path).exists():
            try:
                spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
                apply_spec_to_cfg(cfg, spec, catalog_key)
            except Exception as exc:
                print(f"Warning: could not apply benchmark spec: {exc}")

        sync_params(cfg)
        print_summary(cfg)

        print(f"Running MOMP driver pipeline at {driver_path}...")

        # ----------------------------------------------------------------------
        # Important fix:
        # driver.py should not receive --model, --spec, or --in-process.
        # ----------------------------------------------------------------------
        with sanitized_driver_argv(driver_path, mode):
            try:
                runpy.run_path(str(driver_path), run_name="__main__")
            except SystemExit as exc:
                code = exc.code

                if code is None or code == 0:
                    pass
                else:
                    raise RuntimeError(
                        f"MOMP driver exited with code {code} for model '{name}'."
                    ) from exc

    print(f"\nMOMP [{mode.upper()}] {name}: completed.")


def execute_momp_benchmark(mode="det", model_key=None, in_process=False):
    models = select_models(mode, model_key)

    print(
        f"Models to run ({mode}): "
        f"{[_model_name(m, k) for k, m in models]}"
    )

    if len(models) == 1 or in_process:
        for key, meta in models:
            run_single_model(mode, key, meta)

        return 0

    failures = []

    for key, meta in models:
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--mode",
            mode,
            "--model",
            key,
        ]

        spec_path = os.getenv("ROMP_SPEC_PATH")

        if spec_path:
            cmd.extend(["--spec", spec_path])

        print(f"\n>>> {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            env=os.environ.copy(),
        )

        if result.returncode != 0:
            failures.append((_model_name(meta, key), result.returncode))

    print("\n" + "=" * 60)

    if failures:
        for name, code in failures:
            print(f"FAILED: {name} (exit code {code})")

        return 1

    print("All models completed successfully.")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Run MOMP benchmarking pipeline (one model per run).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--mode",
        type=str.lower,
        default=os.getenv("ROMP_MODE", "det"),
        choices=VALID_MODES,
        help="'det' (deterministic) or 'prob' (probabilistic)",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Run only this model (catalog key or model_name), e.g. GraphCast",
    )

    parser.add_argument(
        "--spec",
        type=str,
        default=None,
        help="Path to benchmark_spec.json",
    )

    parser.add_argument(
        "--in-process",
        action="store_true",
        help="Run all models in this process instead of one subprocess per model",
    )

    args = parser.parse_args()

    if args.spec:
        os.environ["ROMP_SPEC_PATH"] = str(Path(args.spec).resolve())

    sys.exit(
        execute_momp_benchmark(
            args.mode,
            args.model,
            args.in_process,
        )
    )


if __name__ == "__main__":
    main()