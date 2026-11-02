"""
run_momp.py
MOMP / ROMP benchmarking execution script.
Executes evaluation pipelines for selected weather models.
Parses Streamlit UI runtime parameters, spec files, and GCS catalog configs.
"""
from __future__ import annotations
import argparse
import ast
import contextlib
import fnmatch
import importlib.metadata
import json
import os
import subprocess
import sys
from datetime import date
from importlib.metadata import PackageNotFoundError
from pathlib import Path

# ------------------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
if (PROJECT_ROOT / "ai_weather" / "config.py").is_file():
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from ai_weather import config

    sys.modules["config"] = config
else:
    import config

LOCAL_ROMP_ROOT = PROJECT_ROOT / "apps" / "benchmarking" / "ROMP"
if (LOCAL_ROMP_ROOT / "momp" / "driver.py").is_file():
    ROMP_ROOT = LOCAL_ROMP_ROOT
else:
    ROMP_ROOT = Path(config.ROMP_ROOT).expanduser().resolve()
MOMP_PKG_DIR = ROMP_ROOT / "momp"

config.ROMP_ROOT = ROMP_ROOT
config.MOMP_PKG_DIR = MOMP_PKG_DIR
config.work_dir = str(ROMP_ROOT)
config.pkg_dir = str(MOMP_PKG_DIR)

# Make `config` importable no matter where the script is launched from
for _p in (PROJECT_ROOT, ROMP_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# Data locations come from config.py
DATA_DIR = Path(config.DATA_DIR).expanduser()
EXTERNAL_DATA_DIR = Path(config.EXTERNAL_DATA_DIR).expanduser()

ROMP_OUT_DIR = Path(getattr(config, "ROMP_OUT_DIR", getattr(config, "ROMP_DEMO_OUT_DIR", PROJECT_ROOT / "romp_output"))).expanduser()
ROMP_FIG_DIR = Path(getattr(config, "ROMP_FIG_DIR", getattr(config, "ROMP_DEMO_FIG_DIR", PROJECT_ROOT / "romp_figure"))).expanduser()

import momp  # noqa: E402,F401
import momp.params as params  # noqa: E402
from ai_weather.runtime import check_runtime_python, select_runtime_python
from momp.lib.loader import get_cfg, get_setting  # noqa: E402
from momp.lib.assertion import ROMPValidator  # noqa: E402

# ------------------------------------------------------------------------------
# Lead windows / tolerances (Length matched: 2 items each)
# ------------------------------------------------------------------------------
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

# ------------------------------------------------------------------------------
# Normalization helpers
# ------------------------------------------------------------------------------
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
    if p is None or str(p).strip() == "":
        return None
    s = str(p).strip()
    if config.is_uri(s):
        clean_path = config._strip(s)
        return f"gs://{clean_path}"

    p_obj = Path(s).expanduser()
    if p_obj.is_absolute():
        return str(p_obj)

    candidates = [PROJECT_ROOT / p_obj, ROMP_ROOT / p_obj]
    if p_obj.parts and p_obj.parts[0].lower() == "data":
        candidates.insert(0, DATA_DIR.joinpath(*p_obj.parts[1:]))
    for cand in candidates:
        if cand.exists():
            return str(cand.resolve())
    return str(candidates[0])

# ------------------------------------------------------------------------------
# Mode helpers
# ------------------------------------------------------------------------------
def is_prob_mode(mode):
    mode = mode.lower()
    if mode not in VALID_MODES:
        raise ValueError(
            f"Invalid mode '{mode}'. Choose 'det'/'deterministic' or "
            "'prob'/'probabilistic'."
        )
    return mode in ("prob", "probabilistic")

def select_models(mode, model_key=None):
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

# ------------------------------------------------------------------------------
# importlib.metadata patch
# ------------------------------------------------------------------------------
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

# ------------------------------------------------------------------------------
# Base paths
# ------------------------------------------------------------------------------
def apply_base_paths(cfg):
    cfg.work_dir = str(ROMP_ROOT)
    cfg.pkg_dir = str(MOMP_PKG_DIR)

    # Resolve paths dynamically from config.py
    obs_dir = getattr(config, "obs_dir", None) or getattr(config, "OBS_DIR", None)
    ref_dir = getattr(config, "ref_model_dir", None) or getattr(config, "obs_dir", None)
    mask_file = getattr(config, "nc_mask", None) or getattr(config, "JJAS_MASK_FILE", None)

    cfg.obs_dir = resolve_path(obs_dir) if obs_dir else resolve_path(EXTERNAL_DATA_DIR / "CHIRPS")
    cfg.ref_model_dir = resolve_path(ref_dir) if ref_dir else cfg.obs_dir
    cfg.nc_mask = resolve_path(mask_file) if mask_file else resolve_path(EXTERNAL_DATA_DIR / "jjas_100mm_rainfall_mask_0p25.nc")

    cfg.verification_window_list = ((1, 15), (16, 30))
    cfg.tolerance_days_list = (3, 5)

    if hasattr(config, "obs_file_pattern"):
        cfg.obs_file_pattern = config.obs_file_pattern
    if hasattr(config, "obs_var"):
        cfg.obs_var = config.obs_var
    if hasattr(config, "ref_model_file_pattern"):
        cfg.ref_model_file_pattern = config.ref_model_file_pattern
    if hasattr(config, "ref_model_var"):
        cfg.ref_model_var = config.ref_model_var
    if hasattr(config, "region"):
        cfg.region = config.region

    if getattr(cfg, "shpfile_dir", None):
        cfg.shpfile_dir = resolve_path(cfg.shpfile_dir)
    else:
        cfg.shpfile_dir = resolve_path(EXTERNAL_DATA_DIR / "shapefile")

    cfg.ref_model = getattr(config, "ref_model", "climatology")
    cfg.layout = ("model", "verification_window")

# ------------------------------------------------------------------------------
# Benchmark spec support
# ------------------------------------------------------------------------------
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
    elif "16-30" in window:
        cfg.verification_window_list = ((16, 30),)
        cfg.tolerance_days_list = (5,)
        cfg.max_forecast_day = 30

def apply_spec_to_cfg(cfg, spec: dict, model_key: str):
    if not isinstance(spec, dict):
        return
    ground_truth = _clean_str(spec.get("ground_truth", ""))
    if ground_truth:
        obs_meta = _obs_meta(ground_truth)
        if obs_meta:
            cfg.obs = ground_truth
            obs_dir = _clean_str(_meta_get(obs_meta, "dir", ""))
            if obs_dir:
                cfg.obs_dir = resolve_path(obs_dir)
            file_pattern = _clean_str(_meta_get(obs_meta, "file_pattern", "{}.nc"))
            cfg.obs_file_pattern = (file_pattern,)
            cfg.obs_var = _clean_str(_meta_get(obs_meta, "var", "precip"))
    coverage = _clean_str(spec.get("benchmark_coverage", ""))
    if coverage:
        cfg.region = coverage
    forecast_window = _clean_str(spec.get("forecast_window", ""))
    if forecast_window:
        _apply_forecast_window(cfg, forecast_window)
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
            cfg.init_days = tuple(sorted(set(int(x)) for x in init_days))
        except Exception:
            cfg.init_days = (0, 3)
    cfg.run_years_concurrently = bool(model_spec.get("run_years_concurrently", True))
    if model_spec.get("ensemble_forecast") is not None:
        cfg.ensemble_forecast = bool(model_spec["ensemble_forecast"])

# ------------------------------------------------------------------------------
# Apply Advanced Temporal & Verification Parameters from Environment
# ------------------------------------------------------------------------------
def apply_env_vars_to_cfg(cfg):
    """
    Apply temporal, verification, and Streamlit UI parameters from environment variables.
    """
    def _parse_tuple(val):
        if not val:
            return None
        try:
            return ast.literal_eval(val)
        except Exception:
            return None

    # Streamlit Date Evaluation Windows
    if val := os.getenv("ROMP_EVAL_START"):
        if t := _iso_to_tuple(val):
            cfg.start_date = t
    if val := os.getenv("ROMP_EVAL_END"):
        if t := _iso_to_tuple(val):
            cfg.end_date = t

    # Streamlit Baseline Climatology Years
    if val := os.getenv("ROMP_BASELINE_START_YEAR"):
        try:
            cfg.start_year_clim = int(val)
        except ValueError:
            pass
    if val := os.getenv("ROMP_BASELINE_END_YEAR"):
        try:
            cfg.end_year_clim = int(val)
        except ValueError:
            pass

    # Streamlit Forecast Window
    if val := os.getenv("ROMP_FORECAST_WINDOW"):
        _apply_forecast_window(cfg, val)

    # Streamlit Regional Coverage
    if val := os.getenv("ROMP_BENCHMARK_COVERAGE"):
        cfg.region = str(val)

    for key, env_name, converter in [
        ("wet_threshold", "ROMP_WET_THRESHOLD", float),
        ("wet_init", "ROMP_WET_INIT", float),
        ("wet_spell", "ROMP_WET_SPELL", int),
        ("dry_spell", "ROMP_DRY_SPELL", int),
        ("dry_extent", "ROMP_DRY_EXTENT", int),
    ]:
        if val := os.getenv(env_name):
            try:
                setattr(cfg, key, converter(val))
            except ValueError:
                pass

    # Preserve explicit empty values: the UI uses them to disable a mask or
    # select scalar onset criteria instead of a threshold file.
    for key, env_name in (("nc_mask", "ROMP_AREA_MASK_FILE"),
                          ("thresh_file", "ROMP_ONSET_THRESHOLD_FILE")):
        if env_name in os.environ:
            value = os.environ[env_name].strip()
            setattr(cfg, key, resolve_path(value) if value else None)
    if "ROMP_BASELINE_FORECAST" in os.environ:
        cfg.ref_model = os.environ["ROMP_BASELINE_FORECAST"].strip() or "climatology"

    # Verification and Advanced Temporal Settings
    if val := os.getenv("ROMP_INIT_TYPE"):
        cfg.init_type = val
    if val := os.getenv("ROMP_DATE_FILTER_YEAR"):
        try:
            cfg.date_filter_year = int(val)
        except ValueError:
            pass
    if val := os.getenv("ROMP_VERIFICATION_WINDOW_LIST"):
        parsed = _parse_tuple(val)
        if parsed:
            cfg.verification_window_list = parsed
    if val := os.getenv("ROMP_TOLERANCE_DAYS_LIST"):
        parsed = _parse_tuple(val)
        if parsed:
            cfg.tolerance_days_list = parsed
    if val := os.getenv("ROMP_MAX_FORECAST_DAY"):
        try:
            cfg.max_forecast_day = int(val)
        except ValueError:
            pass

    if "ROMP_INIT_DAYS" in os.environ:
        val = os.environ["ROMP_INIT_DAYS"]
        try:
            cfg.init_days = tuple(sorted(set(int(day.strip()) for day in val.split(",") if day.strip())))
        except ValueError:
            pass

    for key, env_name in [
        ("run_years_concurrently", "ROMP_RUN_YEARS_CONCURRENTLY"),
        ("ensemble_forecast", "ROMP_ENSEMBLE_FORECAST"),
    ]:
        if val := os.getenv(env_name):
            normalized = val.strip().lower()
            if normalized in {"1", "true", "yes", "on"}:
                setattr(cfg, key, True)
            elif normalized in {"0", "false", "no", "off"}:
                setattr(cfg, key, False)

    if val := os.getenv("ROMP_GROUND_TRUTH"):
        obs_key = _clean_str(val).upper()
        obs_meta = _obs_meta(obs_key)
        cfg.obs = obs_key
        obs_settings = getattr(config, "_OBS_SETS", {}).get(obs_key, {})
        if obs_meta:
            obs_settings = {
                "dir": _meta_get(obs_meta, "dir", ""),
                "pattern": _meta_get(obs_meta, "file_pattern", "{}.nc"),
                "var": _meta_get(obs_meta, "var", "precip"),
            }
        if obs_settings:
            obs_dir = _clean_str(obs_settings.get("dir", ""))
            if obs_dir:
                cfg.obs_dir = resolve_path(obs_dir)
            patterns = obs_settings.get("pattern", "{}.nc")
            cfg.obs_file_pattern = tuple(patterns) if isinstance(patterns, (list, tuple)) else (str(patterns),)
            cfg.obs_var = _clean_str(obs_settings.get("var", "precip"))
            cfg.ref_model_dir = cfg.obs_dir
            cfg.ref_model_file_pattern = cfg.obs_file_pattern
            cfg.ref_model_var = cfg.obs_var

    for key, env_name in [
        ("mok", "ROMP_MOK"),
        ("years", "ROMP_YEARS"),
        ("years_clim", "ROMP_YEARS_CLIM"),
        ("fallback_date", "ROMP_FALLBACK_DATE"),
    ]:
        if val := os.getenv(env_name):
            if val.lower() == "none":
                setattr(cfg, key, None)
            else:
                parsed = _parse_tuple(val)
                setattr(cfg, key, parsed if parsed is not None else val)

# ------------------------------------------------------------------------------
# Model configuration
# ------------------------------------------------------------------------------
def validate_runtime_config(cfg):
    """Check the final merged settings, after spec and dashboard overrides."""
    mode = getattr(cfg, "init_type", "weekly")
    days = getattr(cfg, "init_days", (0, 3))
    if mode not in {"weekly", "daily", "monthly"}:
        raise ValueError("Initialization schedule must be weekly, daily or monthly.")
    if mode != "daily":
        low, high = (0, 6) if mode == "weekly" else (1, 31)
        if not days or any(type(day) is not int or not low <= day <= high for day in days):
            raise ValueError(f"{mode.title()} initialization days must be nonempty integers in {low}–{high}.")
    windows = tuple(tuple(window) for window in cfg.verification_window_list)
    if not windows or len(windows) != len(cfg.tolerance_days_list):
        raise ValueError("Verification windows and tolerances must be nonempty and match in length.")
    for start, end in windows:
        if start < 1 or end < start or end > cfg.max_forecast_day:
            raise ValueError(f"Verification window {(start, end)} exceeds forecast horizon {cfg.max_forecast_day} or is invalid.")
        bins = [tuple(b) for b in cfg.day_bins if b[0] >= start and b[1] <= end]
        covered = [day for low, high in bins for day in range(low, high + 1)]
        if sorted(covered) != list(range(start, end + 1)):
            raise ValueError(f"Day bins {cfg.day_bins} do not partition verification window {(start, end)}.")
    # Keep the full bin catalog until all overrides are merged, then pass only
    # active bins to the downstream validator and calculation stages.
    cfg.day_bins = tuple(tuple(b) for b in cfg.day_bins
                         if any(b[0] >= start and b[1] <= end for start, end in windows))
    if getattr(cfg, "run_years_concurrently", None) is not None:
        cfg.parallel = cfg.run_years_concurrently
    if cfg.probabilistic:
        required = set()
        if cfg.skill_score:
            required.update(("BS", "RPS"))
        if cfg.plot_heatmap_bss_auc or cfg.plot_panel_heatmap_skill or cfg.plot_bar_bss_rpss_auc or cfg.save_csv_score:
            required.update(("BS", "RPS", "AUC", "skill_score"))
        missing = sorted(flag for flag in required if not getattr(cfg, flag, False))
        if missing:
            raise ValueError(f"Enabled probabilistic outputs require these metrics: {', '.join(missing)}.")


def configure_model(cfg, catalog_key: str, meta, mode):
    name = _model_name(meta, catalog_key)
    cfg.model_list = (name,)
    model_dir = resolve_path(_meta_get(meta, "dir", ""))
    if not model_dir or not _path_exists(model_dir):
        raise FileNotFoundError(
            f"Model data directory for '{name}' not found: {model_dir!r}. "
            "Check BENCHMARK_MODEL_CATALOG['dir'] in config.py."
        )
    cfg.model_dir_list = (model_dir,)
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
    out_dir = ROMP_OUT_DIR / name
    fig_dir = ROMP_FIG_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    cfg.dir_out = str(out_dir)
    cfg.dir_fig = str(fig_dir)

def _path_exists(path) -> bool:
    """True if a local path or a gs:// object/folder exists."""
    return config.exists(str(path)) if config.is_uri(path) else Path(str(path)).expanduser().exists()

def _list_nc(folder):
    """Names of the .nc files directly in a local folder or gs:// folder."""
    return sorted(Path(f).name for f in config.glob(config.join(folder, "*.nc")))

def validate_paths(cfg):
    checks = {
        "obs_dir": getattr(cfg, "obs_dir", None),
        "ref_model_dir": getattr(cfg, "ref_model_dir", None),
        "nc_mask": getattr(cfg, "nc_mask", None),
        "thresh_file": getattr(cfg, "thresh_file", None),
    }
    if getattr(cfg, "polygon", False):
        checks["shpfile_dir"] = getattr(cfg, "shpfile_dir", None)
    for i, d in enumerate(getattr(cfg, "model_dir_list", ()) or ()):
        checks[f"model_dir_list[{i}]"] = d
    missing = [
        f"  {name}: {path}"
        for name, path in checks.items()
        if path and not _path_exists(path)
    ]
    if missing:
        raise FileNotFoundError(
            "These configured paths do not exist:\n" + "\n".join(missing)
            + f"\n(data root is {DATA_DIR})"
        )
    problems = []
    
    region_clean = getattr(cfg, "region", "Ethiopia").replace(" ", "_")
    sd, ed = getattr(cfg, "start_date", None), getattr(cfg, "end_date", None)
    
    # 1. Validate Observations
    obs_dir = getattr(cfg, "obs_dir", None)
    obs_pat_raw = getattr(cfg, "obs_file_pattern", None)
    if obs_dir and sd and ed:
        names = _list_nc(obs_dir)
        absent = []
        for y in range(sd[0], ed[0] + 1):
            patterns_to_try = []
            if isinstance(obs_pat_raw, (list, tuple)):
                patterns_to_try.extend(p.format(y) for p in obs_pat_raw if "{}" in p)
            elif isinstance(obs_pat_raw, str) and "{}" in obs_pat_raw:
                patterns_to_try.append(obs_pat_raw.format(y))
            
            patterns_to_try.extend([f"CHIRPS_{region_clean}_{y}.nc", f"{y}.nc"])
            if not any(pat in names for pat in patterns_to_try):
                absent.append(y)

        if absent:
            problems.append(
                f"obs: missing rainfall files in {obs_dir} for years {absent}; "
                f"first files found: {names[:5] or 'none'}"
            )

    # 2. Validate Reference Model
    ref_dir = getattr(cfg, "ref_model_dir", None)
    ref_pat_raw = getattr(cfg, "ref_model_file_pattern", None) or "{}.nc"
    if ref_dir:
        names = _list_nc(ref_dir)
        patterns_to_check = []
        if isinstance(ref_pat_raw, (list, tuple)):
            patterns_to_check.extend(p.replace("{}", "*") for p in ref_pat_raw if "{}" in p)
        elif isinstance(ref_pat_raw, str):
            patterns_to_check.append(ref_pat_raw.replace("{}", "*"))
        patterns_to_check.extend(["*.nc", f"CHIRPS_{region_clean}_*.nc"])

        matched = any(
            fnmatch.fnmatch(n, pat) 
            for pat in patterns_to_check 
            for n in names
        )
        if not matched:
            problems.append(
                f"ref model: nothing matches patterns {patterns_to_check} in {ref_dir}; "
                f"first files found: {names[:5] or 'none'}"
            )

    if problems:
        raise FileNotFoundError("Data file check failed:\n  " + "\n  ".join(problems))

def sync_params(cfg):
    for attr in dir(cfg):
        if not attr.startswith("_"):
            setattr(params, attr, getattr(cfg, attr))
    from momp.lib.loader import set_cfg

    set_cfg(cfg)

def print_summary(cfg):
    print(f"work_dir      : {cfg.work_dir} ")
    print(f"pkg_dir       : {cfg.pkg_dir} ")
    print(f"obs_dir       : {cfg.obs_dir} ")
    print(f"obs pattern   : {getattr(cfg, 'obs_file_pattern', None)}  var={getattr(cfg, 'obs_var', None)} ")
    print(f"ref_model_dir : {getattr(cfg, 'ref_model_dir', None)}  pattern={getattr(cfg, 'ref_model_file_pattern', None)} ")
    print(f"Years         : {getattr(cfg, 'start_date', None)} -> {getattr(cfg, 'end_date', None)} ")
    print(f"ref_model     : {cfg.ref_model} ")
    print(f"Probabilistic : {cfg.probabilistic} ")
    print(f"Model         : {cfg.model_list} ")
    print(f"Model path    : {cfg.model_dir_list} ")
    print(f"Windows       : {cfg.verification_window_list} ")
    print(f"Tolerances    : {cfg.tolerance_days_list} ")
    print(f"dir_out       : {cfg.dir_out} ")
    print(f"dir_fig       : {cfg.dir_fig} ")
    print("-" * 60)

# ------------------------------------------------------------------------------
# Execution
# ------------------------------------------------------------------------------
def run_single_model(mode, catalog_key, meta):
    name = _model_name(meta, catalog_key)
    print("=" * 60)
    print(f"MOMP [{mode.upper()}]  model={name}  (catalog key: {catalog_key}) ")
    print("=" * 60)
    driver_path = MOMP_PKG_DIR / "driver.py"
    if not driver_path.exists():
        raise FileNotFoundError(f"Driver not found at {driver_path} ")
    config.setup_earthdata_auth()
    with patched_package_metadata():
        cfg = get_cfg(validate=False)
        apply_base_paths(cfg)
        configure_model(cfg, catalog_key, meta, mode)

        spec_path = os.getenv("ROMP_SPEC_PATH")
        spec_file = Path(spec_path).expanduser() if spec_path else None
        if spec_file and spec_file.exists():
            try:
                spec = json.loads(spec_file.read_text(encoding="utf-8"))
                apply_spec_to_cfg(cfg, spec, catalog_key)
            except Exception as exc:
                print(f"Warning: could not apply benchmark spec: {exc}")
                
        # Apply environment variables passed from Streamlit UI
        apply_env_vars_to_cfg(cfg)
        validate_runtime_config(cfg)

        ROMPValidator(vars(cfg)).validate()
        
        validate_paths(cfg)
        sync_params(cfg)
        print_summary(cfg)
        print(f"Running MOMP driver pipeline at {driver_path}...")
        from momp.driver import run_momp

        run_momp(cfg=cfg, setting=get_setting())
    print(f"\nMOMP [{mode.upper()}] {name}: completed.")

def execute_momp_benchmark(mode="det", model_key=None, in_process=False):
    python = select_runtime_python(PROJECT_ROOT)
    if python != os.path.abspath(sys.executable):
        cmd = [python, str(Path(__file__).resolve()), "--mode", mode]
        if model_key:
            cmd.extend(["--model", model_key])
        if in_process:
            cmd.append("--in-process")
        print(f"Using scientific Python: {python}", flush=True)
        return subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=os.environ.copy(), check=False).returncode
    print(check_runtime_python(python, os.environ.copy(), PROJECT_ROOT), flush=True)
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
            python,
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
        os.environ["ROMP_SPEC_PATH"] = str(Path(args.spec).expanduser().resolve())
    sys.exit(
        execute_momp_benchmark(
            args.mode,
            args.model,
            args.in_process,
        )
    )

if __name__ == "__main__":
    main()
