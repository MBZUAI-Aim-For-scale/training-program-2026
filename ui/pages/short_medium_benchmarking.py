"""
ui/pages/short_medium_benchmarking.py
High-performance Streamlit page for short and medium-range forecast benchmarking.
Includes memory-safe Matplotlib rendering, cached NetCDF metadata discovery with mtime keying,
metric-direction aware performance color coding matrix, grouped spatial metric distribution cards,
padded shapefile clipping & boundary overlay rendering for spatial maps, persistent Woreda selection,
robust argument filtering for onset analysis, and error-proof tab routing.
"""
from __future__ import annotations
from ai_weather.benchmark_job import BenchmarkJob
from ui.utils.parameter_help import PARAMETER_HELP
from ui.utils.benchmark_messages import cancellation_warning, execution_status
from ai_weather.spatial import climatology_boundary_path
from ai_weather.spatial import apply_jjas_mask

import os
import sys

import base64
import importlib
import inspect
import io
import json
import re
import subprocess
import warnings
from calendar import monthrange
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
import streamlit as st
import xarray as xr
from PIL import Image, ImageDraw
import config
from ai_weather.map_extent import geographic_extent
from ai_weather.precipitation import precipitation_to_mm
from ai_weather.runtime import check_runtime_python, select_runtime_python
from ai_weather.spatial import clip_map_artist, clip_to_boundary, load_boundary_frame
from ui.utils.verification_settings import initialization_matches_schedule, parse_verification_settings, temporal_settings_errors

# ======================================================================
# Environment snapshot at startup.
# Subprocesses are launched with this copy so nothing that runs inside
# the Streamlit process leaks into them unexpectedly.
# ======================================================================
_STARTUP_ENV: Dict[str, str] = dict(os.environ)

# ======================================================================
# Safe config access
# ======================================================================
PROJECT_ROOT = Path(getattr(config, "PROJECT_ROOT", Path.cwd())).resolve()
ROMP_OUTPUT_ROOT = Path(__file__).resolve().parents[2] / "data" / "ROMP_OUTPUT"

ROMP_DEMO_FIG_DIR = ROMP_OUTPUT_ROOT / "figures"
ROMP_DEMO_OUT_DIR = ROMP_OUTPUT_ROOT
EXTERNAL_DATA_DIR = Path(
    getattr(
        config,
        "EXTERNAL_DATA_DIR",
        PROJECT_ROOT / "data" / "external",
    )
)
ONSET_REFERENCE_DIR = Path(
    getattr(
        config,
        "ONSET_REFERENCE_DIR",
        EXTERNAL_DATA_DIR / "onset_reference",
    )
)

BENCHMARK_OBS_CATALOG = getattr(config, "BENCHMARK_OBS_CATALOG", {})
CONFIG_BENCHMARK_MODEL_CATALOG = getattr(config, "BENCHMARK_MODEL_CATALOG", {})

try:
    from config import get_region_presets
except Exception:
    def get_region_presets() -> List[Dict[str, Any]]:
        return []

try:
    from apps.benchmarking.benchmarking import (
        BENCHMARK_METHOD,
        get_benchmark_presets,
    )
except Exception:
    BENCHMARK_METHOD = {}
    def get_benchmark_presets() -> Dict[str, Any]:
        return {}

# ======================================================================
# Optional & Safe Imports
# ======================================================================
try:
    from apps.benchmarking.inventory import (
        build_model_inventory,
        build_obs_inventory,
        readiness_summary,
        inventory_to_dataframe,
    )
    PHASE_2_AVAILABLE = True
except Exception:
    PHASE_2_AVAILABLE = False

try:
    from apps.benchmarking.fairness import (
        build_fair_comparison_plan,
        plan_to_dataframe,
    )
    PHASE_3_AVAILABLE = True
except Exception:
    PHASE_3_AVAILABLE = False

try:
    from apps.benchmarking.verification import (
        collect_scores_from_momp_outputs,
        compute_skill_vs_baseline,
    )
    PHASE_4_AVAILABLE = True
except Exception:
    PHASE_4_AVAILABLE = False

try:
    from apps.benchmarking.uncertainty import (
        compute_metric_uncertainty,
        compute_significance_vs_baseline,
    )
    PHASE_5_AVAILABLE = True
except Exception:
    PHASE_5_AVAILABLE = False

try:
    from apps.benchmarking.dashboard import build_provenance
    PHASE_6_AVAILABLE = True
except Exception:
    PHASE_6_AVAILABLE = False

try:
    import momp.app.onset_time_series as ots
    OTS_AVAILABLE = True
except Exception:
    OTS_AVAILABLE = False

try:
    from fpdf import FPDF
    PDF_AVAILABLE = True
except Exception:
    PDF_AVAILABLE = False
    FPDF = object

# ======================================================================
# 1. Global & Dynamic Configuration
# ======================================================================

LIVE_FORECAST_RECENT_YEARS = getattr(config, "LIVE_FORECAST_RECENT_YEARS", 2)

REFERENCE_CATEGORY_ORDER = getattr(
    config,
    "REFERENCE_CATEGORY_ORDER",
    [
        "Deterministic AI",
        "Probabilistic AI",
        "Climatology / Baseline",
        "Other",
    ],
)

WINDOW_TOLERANCE_DAYS = {
    "Days 1–15": 3,
    "Days 16–30": 5,
}

# Metric Direction Keywords
HIGHER_IS_BETTER_KEYWORDS = {
    "auc", "bss", "rpss", "skill_score", "skill score", "reliability",
    "accuracy", "pod", "csi", "tss", "ets", "pcc", "r2", "correlation"
}


def is_metric_higher_better(metric_name: str) -> bool:
    """True if higher values indicate better forecast skill (e.g. AUC, BSS, RPSS)."""
    m = str(metric_name).strip().lower()
    return any(k in m for k in HIGHER_IS_BETTER_KEYWORDS)

# ----------------------------------------------------------------------
# 2. Dynamic Model Resolution & Backward-Compatible Dict
# ----------------------------------------------------------------------
MODEL_DISPLAY_OVERRIDES: Dict[str, str] = getattr(
    config,
    "MODEL_DISPLAY_OVERRIDES",
    {
        "AIFS": "AIFS Single",
        "AIFS_ENS": "AIFS Ensemble",
        "FuXi": "FuXi",
        "GraphCast": "GraphCast",
        "GenCast": "GenCast",
    },
)


def get_model_display_label(model_key: str) -> str:
    """Dynamically resolves a human-readable display label for ANY model."""
    if not model_key:
        return "Unknown Model"

    key_str = str(model_key).strip()
    catalog = getattr(config, "BENCHMARK_MODEL_CATALOG", {}) or {}

    if key_str in catalog and isinstance(catalog[key_str], dict):
        meta = catalog[key_str]
        label = meta.get("model_name") or meta.get("label") or meta.get("display_name")
        if label:
            return str(label)

    if key_str in MODEL_DISPLAY_OVERRIDES:
        return MODEL_DISPLAY_OVERRIDES[key_str]

    tokens = re.split(r"[_\-\s]+", key_str)
    formatted = []
    for token in tokens:
        t_lower = token.lower()
        if t_lower in {"ai", "ecmwf", "gfs", "nwp", "ens", "fuxi", "aifs", "ml"}:
            formatted.append(t_lower.upper() if t_lower != "fuxi" else "FuXi")
        elif re.match(r"^v\d+$", t_lower):
            formatted.append(f"v{t_lower[1:]}")
        else:
            formatted.append(token.capitalize())

    return " ".join(formatted)


class DynamicModelLabelsDict(dict):
    """Dict wrapper that resolves unknown key lookups dynamically.

    Prevents NameError while formatting future models on the fly.
    """

    def __getitem__(self, key):
        return get_model_display_label(key)

    def __contains__(self, key):
        return True

    def get(self, key, default=None):
        res = get_model_display_label(key)
        return res if res != "Unknown Model" else (default if default is not None else str(key))


# Explicitly defined for legacy imports and lookups across the app
MODEL_DISPLAY_LABELS = DynamicModelLabelsDict(MODEL_DISPLAY_OVERRIDES)

# ----------------------------------------------------------------------
# 3. Extensible Spatial Metrics & Admin Columns
# ----------------------------------------------------------------------
DEFAULT_MAP_METRICS = {
    "False Alarm Rate": {
        "keywords": ["false_alarm_rate", "false_alarm", "far"],
        "fraction": True,
        "unit": "%",
        "threshold": 35.0,
        "cmap": "Reds",
        "description": "Fraction of forecast onset grid cells that were not observed onset cells.",
    },
    "Miss Rate": {
        "keywords": ["miss_rate", "miss", "mr"],
        "fraction": True,
        "unit": "%",
        "threshold": 60.0,
        "cmap": "Reds",
        "description": "Fraction of observed onset grid cells that the forecast missed.",
    },
    "Mean Absolute Error": {
        "keywords": ["mean_absolute_error", "mean_mae", "mae"],
        "fraction": False,
        "unit": "days",
        "threshold": 7.0,
        "cmap": "YlOrRd",
        "description": "Average absolute error in onset timing, measured in days.",
    },
}

MAP_METRICS: Dict[str, Dict[str, Any]] = getattr(
    config, "BENCHMARK_MAP_METRICS", DEFAULT_MAP_METRICS
)

DEFAULT_ADMIN_COLUMN_CANDIDATES = {
    "Admin1": ["ADM1_EN", "ADM1_NAME", "ADM1_NM", "NAME_1", "ADM1", "admin1", "Region", "State", "Province"],
    "Admin2": ["ADM2_EN", "ADM2_NAME", "ADM2_NM", "NAME_2", "ADM2", "admin2", "Zone", "District", "County"],
    "Admin3": ["ADM3_EN", "ADM3_NAME", "ADM3_NM", "NAME_3", "ADM3", "adm3_name", "admin3", "Woreda", "Subcounty"],
}

MAP_ADMIN_COLUMN_CANDIDATES: Dict[str, List[str]] = getattr(
    config, "MAP_ADMIN_COLUMN_CANDIDATES", DEFAULT_ADMIN_COLUMN_CANDIDATES
)

# ----------------------------------------------------------------------
# 4. Figure Constants & Helpers
# ----------------------------------------------------------------------
DEFAULT_FIGURE_HUMAN_LABELS = {
    "panel_portrait_mae_far_mr_AIFS_30day.png": "Panel Portrait: MAE / FAR / MR - AIFS (30-day)",
    "panel_portrait_mae_far_mr_prob_30day.png": "Panel Portrait: MAE / FAR / MR - Probabilistic (30-day)",
    "spatial_metrics_climatology_1-15.png": "Spatial Metrics: Climatology (Days 1-15)",
    "spatial_metrics_climatology_16-30.png": "Spatial Metrics: Climatology (Days 16-30)",
    "spatial_metrics_prob_1-15.png": "Spatial Metrics: Probabilistic (Days 1-15)",
    "spatial_metrics_prob_16-30.png": "Spatial Metrics: Probabilistic (Days 16-30)",
}


FIGURE_HUMAN_LABELS: Dict[str, str] = getattr(config, "FIGURE_HUMAN_LABELS", DEFAULT_FIGURE_HUMAN_LABELS)
STATIC_FIGURE_HUMAN_LABELS = FIGURE_HUMAN_LABELS

FIGURE_HUMAN_LABELS: Dict[str, str] = getattr(
    config, "FIGURE_HUMAN_LABELS", DEFAULT_FIGURE_HUMAN_LABELS
)
STATIC_FIGURE_HUMAN_LABELS = FIGURE_HUMAN_LABELS  # Compatibility alias


DEFAULT_FIGURE_TYPE_TOKENS = {
    "Spatial Metrics": ["spatial_metrics", "spatial"],
    "Portrait Panel": ["panel_portrait", "portrait"],
    "Reliability Diagram": ["reliability", "reliability_diagram"],
    "Skill Scores Heatmap": ["skill_scores", "heatmap", "skill_score"],
    "Climatology Onset": ["climatology_onset", "clim_onset"],
}

FIGURE_TYPE_TOKENS: Dict[str, List[str]] = getattr(config, "FIGURE_TYPE_TOKENS", DEFAULT_FIGURE_TYPE_TOKENS)


def classify_figure_type(filename: str) -> str:
    fname = str(filename).lower()
    for label, tokens in FIGURE_TYPE_TOKENS.items():
        if any(token in fname for token in tokens):
            return label
    return "Other"


def format_figure_label(path_str: str) -> str:
    filename = Path(path_str).name


    # 1. Explicit lookup
    if filename in FIGURE_HUMAN_LABELS:
        return FIGURE_HUMAN_LABELS[filename]

    stem = Path(filename).stem.lower()
    fig_type = classify_figure_type(filename)
    if fig_type == "Other":
        fig_type = Path(filename).stem.replace("_", " ").title()

    # 2. Extract metrics
    metrics_found = []
    if "mae_far_mr" in stem:
        metrics_found.append("MAE / FAR / MR")
    else:
        if "mae" in stem:
            metrics_found.append("MAE")
        if "far" in stem:
            metrics_found.append("FAR")
        if "mr" in stem or "miss" in stem:
            metrics_found.append("Miss Rate")
        if "crps" in stem:
            metrics_found.append("CRPS")
        if "brier" in stem:
            metrics_found.append("Brier")

    metric_str = ", ".join(metrics_found)

    # 3. Extract model/subject (sorted by string length descending to match longest key first)
    subject = None
    if "climatology" in stem or "clim" in stem:
        subject = "Climatology"
    elif "prob" in stem or "ensemble" in stem or "ens" in stem:
        subject = "Probabilistic"
    else:
        catalog = getattr(config, "BENCHMARK_MODEL_CATALOG", {}) or {}
        sorted_catalog_keys = sorted(catalog.keys(), key=len, reverse=True)
        for k in sorted_catalog_keys:
            if k.lower() in stem:
                subject = get_model_display_label(k)
                break

        if not subject:
            sorted_override_keys = sorted(MODEL_DISPLAY_OVERRIDES.keys(), key=len, reverse=True)
            for k in sorted_override_keys:
                if k.lower() in stem:
                    subject = MODEL_DISPLAY_OVERRIDES[k]
                    break

    # 4. Extract lead window
    window = None
    if "1-15" in stem or "1_15" in stem:
        window = "Days 1-15"
    elif "16-30" in stem or "16_30" in stem:
        window = "Days 16-30"
    elif "30day" in stem or "30_day" in stem:
        window = "30-day"
    else:
        match = re.search(r"(\d+)[_\-]?day", stem)
        if match:
            window = f"{match.group(1)}-day"

    # 5. Assemble title cleanly
    title_head = fig_type
    if metric_str and metric_str not in fig_type:
        title_head = f"{fig_type}: {metric_str}"

    details = []
    if subject:
        details.append(subject)
    if window:
        details.append(f"({window})")

    return f"{title_head} - {' '.join(details)}" if details else title_head


def render_benchmark_figure(path_str: str) -> None:
    path = Path(path_str)
    if not path.exists() or path.stat().st_size == 0:
        st.warning(f"Figure unavailable or empty: {path.name}")
        return
    try:
        st.image(str(path), width='stretch')
        st.caption(format_figure_label(str(path)))
    except Exception as e:
        st.warning(f"Could not display figure {path.name}: {type(e).__name__} - {e}")


# ======================================================================
# Generic helpers
# ======================================================================
def _clean(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _meta_get(meta: Dict[str, Any], key: str, default: Any = None) -> Any:
    if not isinstance(meta, dict):
        return default
    if key in meta:
        return meta[key]
    target = key.strip().lower()
    for k, v in meta.items():
        if str(k).strip().lower() == target:
            return v
    return default


def _find_var(ds: xr.Dataset, keywords: List[str]) -> Optional[str]:
    for kw in keywords:
        for v in ds.data_vars:
            if kw in str(v).lower():
                return str(v)
    return None

def normalize_aifs_precipitation_dataset(ds: xr.Dataset, unit_cvt=None) -> xr.Dataset:
    var_aliases = ["tp", "tp06", "tp24", "pr", "precip", "total_precipitation"]
    target_var = "RAINFALL"

    # 1. Rename variable to RAINFALL if needed
    for alias in var_aliases:
        if alias in ds.data_vars and target_var not in ds.data_vars:
            ds = ds.rename({alias: target_var})
            break

    # 2. Convert declared units; small rainfall values alone do not imply metres.
    if target_var in ds.data_vars:
        ds = ds.copy()
        ds[target_var] = precipitation_to_mm(ds[target_var], unit_cvt)

    return ds


def preprocess_aifs_benchmark_file(file_path: str) -> str:
    p = Path(file_path)
    if not p.exists():
        return file_path

    try:
        ds = _safe_open_dataset(str(p))
        if any(name in ds.data_vars for name in ("tp", "tp06", "tp24", "pr", "precip", "total_precipitation", "RAINFALL")):
            ds_corrected = normalize_aifs_precipitation_dataset(
                ds, config.BENCHMARK_MODEL_CATALOG["AIFS"].get("unit_cvt"))
            # Save back scaled NetCDF file
            ds_corrected.to_netcdf(str(p))
            ds_corrected.close()
        ds.close()
    except Exception as e:
        st.warning(f"Note on preprocessing {p.name}: {e}")

    return str(p)

def _safe_open_dataset(path_str: str) -> xr.Dataset:
    path = Path(path_str)
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"File {path_str} does not exist or is empty.")

    engines = ["netcdf4", "h5netcdf", "scipy"]
    last_err = None

    for engine in engines:
        try:
            return xr.open_dataset(str(path), engine=engine)
        except Exception as e:
            last_err = e
            try:

                # Fallback disabling time decoding if non-standard calendar/units cause errors
                return xr.open_dataset(str(path), engine=engine, decode_times=False)
            except Exception:
                continue


    # Final fallback default open
    try:
        return xr.open_dataset(str(path))
    except Exception:
        raise last_err or RuntimeError(f"Could not open dataset: {path_str}")


def _render_variable_plot(ds: xr.Dataset, var_name: str, title: str):
    fig = None
    try:
        da = ds[var_name].squeeze()
        if "time" in da.dims:
            da = da.isel(time=0)
        
        is_map = {"lat", "lon"}.issubset(da.dims)
        if is_map:
            zones = load_boundary_frame(climatology_boundary_path(config))
            da = apply_jjas_mask(da, getattr(config, "nc_mask", None))
            da = clip_to_boundary(da, climatology_boundary_path(config))
        fig, ax = plt.subplots(figsize=(8, 4))
        image = da.plot(ax=ax, cmap="YlGnBu")
        if is_map:
            west, south, east, north = zones.total_bounds
            extent = geographic_extent(da.lon, da.lat, region_bounds=(west, east, south, north))
            clip_map_artist(image, ax, climatology_boundary_path(config))
            zones.boundary.plot(ax=ax, color="black", linewidth=1.0, zorder=10)
            ax.set_xlim(extent[:2])
            ax.set_ylim(extent[2:])
            ax.set_aspect("equal", adjustable="box")
            ax.set_anchor("C")
        ax.set_title(f"{title} - {var_name}")
        fig.tight_layout()
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"Could not render plot for {var_name}: {e}")
    finally:
        if fig is not None:
            plt.close(fig)


def format_nc_label(path_str: str) -> str:
    return Path(path_str).stem.replace("_", " ").title()



def classify_figure_type(filename: str) -> str:
    fname = str(filename).lower()
    for label, tokens in FIGURE_TYPE_TOKENS.items():
        for token in tokens:
            if token in fname:
                return label
    return "Other"


def format_figure_label(path_str: str) -> str:
    name = Path(path_str).name
    if name in FIGURE_HUMAN_LABELS:
        return FIGURE_HUMAN_LABELS[name]
    stem = Path(path_str).stem.lower()
    if stem.startswith("panel_portrait"):
        label = "Panel Portrait"
    elif stem.startswith("spatial_metrics"):
        label = "Spatial Metrics"
    elif "reliability" in stem:
        label = "Reliability Diagram"
    elif "skill_scores_heatmap" in stem or "heatmap" in stem:
        label = "Skill Scores Heatmap"
    elif "climatology_onset" in stem:
        label = "Climatology Onset"
    else:
        label = Path(path_str).stem.replace("_", " ").title()

    metric = None
    if "mae_far_mr" in stem:
        metric = "MAE / FAR / MR"
    elif "mae" in stem:
        metric = "MAE"
    elif "far" in stem:
        metric = "FAR"
    elif "mr" in stem:
        metric = "Miss Rate"

    subject = None
    if "climatology" in stem:
        subject = "Climatology"
    elif "aifs_ens" in stem:
        subject = "AIFS Ensemble"
    elif "prob" in stem:
        subject = "Probabilistic"
    elif "aifs" in stem:
        subject = "AIFS"
    elif "fuxi" in stem:
        subject = "FuXi"
    elif "graphcast" in stem:
        subject = "GraphCast"
    elif "gencast" in stem:
        subject = "GenCast"

    window = None
    if "1-15" in stem:
        window = "Days 1-15"
    elif "16-30" in stem:
        window = "Days 16-30"
    elif "30day" in stem:
        window = "30-day"

    parts = [label]
    if metric:
        parts.append(metric)
    details = []
    if subject:
        details.append(subject)
    if window:
        details.append(window)
    if details:
        parts.append("- " + " - ".join(details))
    return " ".join(parts)


def render_benchmark_figure(path_str: str) -> None:
    """Safely render a benchmark figure, skipping corrupt or empty files."""
    path = Path(path_str)
    if not path.exists():
        st.warning(f"Figure not found: {path.name}")
        return
    if path.stat().st_size == 0:
        st.warning(f"Figure is empty (0 bytes): {path.name}")
        return
    try:
        st.image(path_str, width='stretch')
        st.caption(format_figure_label(path_str))
    except Exception as e:
        st.warning(f"Could not display figure {path.name}: {type(e).__name__}")


# ======================================================================
# Fast Auto-detection & Metadata Caching
# ======================================================================
def _extract_time_range_from_nc(nc_path: str):
    try:
        with xr.open_dataset(nc_path, decode_cf=False, decode_times=False) as ds:
            time_name = next(
                (str(name) for name in ds.coords if "time" in str(name).lower() or str(name).lower() in {"t", "date"}),
                None
            )
            if not time_name or time_name not in ds:
                return None, None
            
            vals = np.asarray(ds[time_name].values).ravel()
            if vals.size == 0:
                return None, None
            
            try:
                with xr.open_dataset(nc_path, decode_cf=True, decode_times=True) as ds_decoded:
                    da = ds_decoded[time_name]
                    min_t = pd.Timestamp(da.values.min()).date()
                    max_t = pd.Timestamp(da.values.max()).date()
                    return min_t, max_t
            except Exception:
                return None, None
    except Exception:
        return None, None


def _scan_filename_date_range(nc_files: List[Path]):
    pairs: List[tuple] = []
    years: set = set()
    for path in nc_files:
        for match in re.finditer(r"((?:19|20)\d{2})(?:[-_]?(\d{2}))?", path.stem):
            try:
                year = int(match.group(1))
            except Exception:
                continue
            years.add(year)
            month_raw = match.group(2)
            if month_raw:
                try:
                    month = int(month_raw)
                    if 1 <= month <= 12:
                        pairs.append((year, month))
                except Exception:
                    continue

    if pairs:
        min_y, min_m = min(pairs)
        max_y, max_m = max(pairs)
        return (
            date(min_y, min_m, 1),
            date(max_y, max_m, monthrange(max_y, max_m)[1]),
        )
    if years:
        return date(min(years), 1, 1), date(max(years), 12, 31)
    return None, None


def _detect_model_data_date_range(model_dir: str):
    if not model_dir or str(model_dir).strip() in {"", "."}:
        return None, None

    model_dir = str(model_dir)
    remote = config.is_uri(model_dir)
    try:
        if not config.exists(model_dir):
            return None, None
        if remote:
            nc_files = config.glob(model_dir.rstrip("/") + "/**/*.nc")
        else:
            nc_files = [str(p) for p in Path(model_dir).rglob("*.nc")]
    except Exception:
        return None, None

    if not nc_files:
        return None, None

    def file_year(p) -> int:
        m = re.search(r"(?:19|20)\d{2}", Path(str(p)).stem)
        return int(m.group(0)) if m else 0

    files_sorted = sorted(nc_files, key=lambda p: (file_year(p), Path(str(p)).name))
    n_each = 2 if remote else 3
    sample = (
        files_sorted
        if len(files_sorted) <= 2 * n_each
        else files_sorted[:n_each] + files_sorted[-n_each:]
    )

    starts, ends = [], []
    for nc_path in sample:
        s, e = _extract_time_range_from_nc(nc_path)
        if s and e:
            starts.append(s)
            ends.append(e)

    if starts and ends:
        return min(starts), max(ends)

    return _scan_filename_date_range([Path(str(p)) for p in files_sorted])


@st.cache_data(show_spinner=False, ttl=600)
def detect_benchmark_models(force_refresh: bool = False) -> List[Dict[str, Any]]:
    detected: List[Dict[str, Any]] = []
    current_year = datetime.now().year
    catalog = CONFIG_BENCHMARK_MODEL_CATALOG or {}

    for raw_key, meta in catalog.items():
        key = _clean(raw_key)
        if not key:
            continue

        model_name = _clean(_meta_get(meta, "model_name", key))
        label = MODEL_DISPLAY_LABELS.get(key, model_name or key)
        dir_raw = _clean(_meta_get(meta, "dir", ""))
        if not dir_raw:
            continue

        start_date, end_date = _detect_model_data_date_range(dir_raw)
        if start_date is None or end_date is None:
            continue

        probabilistic = bool(_meta_get(meta, "probabilistic", False))

        is_live = end_date.year >= current_year - LIVE_FORECAST_RECENT_YEARS
        type_label = "LIVE FORECAST" if is_live else "HISTORICAL"
        if probabilistic:
            type_label += " - Ensemble"

        if (
            start_date.month == 1
            and start_date.day == 1
            and end_date.month == 12
            and end_date.day == 31
        ):
            date_range_label = f"{start_date.year} to {end_date.year}"
        else:
            date_range_label = f"{start_date:%Y-%m} to {end_date:%Y-%m}"

        detected.append(
            {
                "Model": label,
                "Category": "Probabilistic" if probabilistic else "Deterministic",
                "Type": type_label,
                "Coverage": "AIWP",
                "Date Range": date_range_label,
                "_row_key": key,
                "_exec_key": key,
                "_model_name": model_name,
                "_start_date": start_date,
                "_end_date": end_date,
                "_category": "Probabilistic" if probabilistic else "Deterministic",
                "_probabilistic": probabilistic,
            }
        )

    preferred = ["AIFS", "AIFS_ENS", "FuXi", "GraphCast", "GenCast"]
    detected.sort(
        key=lambda r: preferred.index(r["_row_key"])
        if r["_row_key"] in preferred
        else 999
    )
    return detected


def available_observation_years(ground_truth: str) -> List[int]:
    meta = BENCHMARK_OBS_CATALOG.get(ground_truth) or getattr(config, "_OBS_SETS", {}).get(ground_truth, {})
    folder = meta.get("dir")
    if not folder:
        return []
    patterns = meta.get("pattern", "{}.nc")
    if isinstance(patterns, str):
        patterns = (patterns,)
    names = {Path(p).name for p in config.glob(config.join(folder, "*.nc"))}
    return [year for year in range(1900, date.today().year + 1)
            if any(pattern.format(year) in names for pattern in patterns)]


def compute_common_model_window(active_rows: List[Dict[str, Any]], observation_years=None):
    starts = [r["_start_date"] for r in active_rows if r.get("_start_date")]
    ends = [r["_end_date"] for r in active_rows if r.get("_end_date")]
    if not starts or not ends:
        return None, None
    cs, ce = max(starts), min(ends)
    if observation_years is not None:
        if not observation_years:
            return None, None
        cs = max(cs, date(min(observation_years), 1, 1))
        ce = min(ce, date(max(observation_years), 12, 31))
    return (cs, ce) if cs <= ce else (None, None)


# ======================================================================
# Map helpers
# ======================================================================
def _map_find_window_dim(da: xr.DataArray) -> Optional[str]:
    for dim in da.dims:
        if "window" in str(dim).lower():
            return str(dim)
    return None


def _map_infer_window_label(nc_path: Path) -> str:
    s = str(nc_path).lower()
    if any(t in s for t in ["16-30", "16_30", "days_16_30", "d16_30", "lead_16_30"]):
        return "Days 16–30"
    if any(t in s for t in ["1-15", "1_15", "days_1_15", "d1_15", "lead_1_15"]):
        return "Days 1–15"
    return "Days 1–15"


def _map_infer_model_label(nc_path: Path) -> Optional[str]:
    """
    Fixed labeling logic: Check explicit file name indicators FIRST (e.g. climatology),
    before falling back to parent folder names.
    """
    fname = nc_path.name.lower()
    parent_name = nc_path.parent.name.lower()

    # 1. Filename checks (highest priority)
    if "climatology" in fname or "traditional" in fname:
        return "Traditional Climatology"
    if "aifs_ens" in fname or "aifs_ensemble" in fname:
        return "AIFS Ensemble"
    if "gencast" in fname:
        return "GenCast"
    if "graphcast" in fname:
        return "GraphCast"
    if "fuxi" in fname:
        return "FuXi"
    if "aifs" in fname:
        return "AIFS Single"

    # 1. Check parent folder name first if stored in model subdirectories
    if nc_path.parent != ROMP_DEMO_OUT_DIR:
        if any(k in parent_name for k in ["aifs_ens", "aifs_ensemble"]):
            return "AIFS Ensemble"
        if "aifs" in parent_name:
            return "AIFS Single"
        if "fuxi" in parent_name:
            return "FuXi"
        if "graphcast" in parent_name:
            return "GraphCast"
        if "gencast" in parent_name:
            return "GenCast"
        if any(k in parent_name for k in ["climatology", "traditional"]):
            return "Traditional Climatology"

    # 2. Check explicit filename prefixes
    if "climatology" in fname or "traditional" in fname:
        return "Traditional Climatology"
    if "aifs_ens" in fname or "aifs_ensemble" in fname:
        return "AIFS Ensemble"
    if "gencast" in fname:
        return "GenCast"
    if "graphcast" in fname:
        return "GraphCast"
    if "fuxi" in fname:
        return "FuXi"
    if "aifs" in fname:
        return "AIFS Single"

    return None


@st.cache_data(show_spinner=False, ttl=300)
def discover_map_outputs(force_refresh: bool = False) -> List[Dict[str, Any]]:
    outputs: List[Dict[str, Any]] = []
    if not ROMP_DEMO_OUT_DIR.exists():
        return outputs

    for nc_path in sorted(ROMP_DEMO_OUT_DIR.rglob("*.nc")):
        first_var = None
        try:
            mtime = nc_path.stat().st_mtime
            with xr.open_dataset(nc_path, decode_cf=False, decode_times=False) as ds:
                contingency_definition = ds.attrs.get("contingency_definition")
                for mm in MAP_METRICS.values():
                    var = _find_var(ds, mm["keywords"])
                    if var:
                        first_var = var
                        break
                if not first_var:
                    continue

                da = ds[first_var]
                wdim = _map_find_window_dim(da)
                if wdim:
                    nw = int(da.sizes[wdim])
                    if nw == 1:
                        windows = [_map_infer_window_label(nc_path)]
                    elif nw == 2:
                        windows = ["Days 1–15", "Days 16–30"]
                    else:
                        windows = [f"Window {i + 1}" for i in range(nw)]
                else:
                    windows = [_map_infer_window_label(nc_path)]
        except Exception:
            continue

        model_label = _map_infer_model_label(nc_path)
        if model_label is None:
            continue

        outputs.append(
            {
                "path": str(nc_path.resolve()),
                "model_label": model_label,
                "windows": windows,
                "mtime": mtime,
                "contingency_definition": contingency_definition,
            }
        )
    latest_by_model_window = {}
    for output in outputs:
        for window in output["windows"]:
            key = (output["model_label"], window)
            previous = latest_by_model_window.get(key)
            if previous is None or output["mtime"] >= previous["mtime"]:
                latest_by_model_window[key] = output

    deduplicated = {}
    for (model_label, window), output in latest_by_model_window.items():
        key = (output["path"], model_label, output["mtime"])
        if key not in deduplicated:
            deduplicated[key] = {**output, "windows": []}
        deduplicated[key]["windows"].append(window)

    return sorted(deduplicated.values(), key=lambda item: item["model_label"])


def _map_select_window(da: xr.DataArray, wl: str) -> xr.DataArray:
    wdim = _map_find_window_dim(da)
    if wdim:
        return da.isel(
            {
                wdim: 1
                if wl == "Days 16–30" and da.sizes[wdim] > 1
                else 0
            }
        )
    return da


def _map_lon_lat_names(da: xr.DataArray):
    lon = next((c for c in ["lon", "longitude"] if c in da.coords), None)
    lat = next((c for c in ["lat", "latitude"] if c in da.coords), None)
    return lon, lat


@st.cache_resource(show_spinner=False)
def _map_load_admin_shapefile():
    try:
        import geopandas as gpd
    except Exception:
        return None

    candidates: List[Path] = []
    for attr in [
        "SHAPEFILE_PATH",
        "ZONE_WOREDA_SHP",
        "ETH_GEOJSON_PATH",
        "STUDYAREA_SHAPEFILE",
        "WOREDA_SHAPEFILE_PATH",
        "SHAPEFILE_DIR",
    ]:
        val = getattr(config, attr, None)
        if val:
            p = Path(str(val))
            if p.is_file():
                candidates.append(p)
            elif p.is_dir():
                candidates.extend(
                    sorted(p.glob("*.geojson")) + sorted(p.glob("*.shp"))
                )

    for path in list(dict.fromkeys(candidates)):
        try:
            gdf = gpd.read_file(path)
            if gdf.empty or "geometry" not in gdf.columns:
                continue
            inv = ~gdf.geometry.is_valid
            if inv.any():
                gdf.loc[inv, "geometry"] = gdf.loc[inv, "geometry"].buffer(0)
            gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
            if not gdf.empty:
                return gdf
        except Exception:
            continue
    return None


def _map_admin_column(gdf, al: str) -> Optional[str]:
    if gdf is None:
        return None
    cmap = {str(c).strip().lower(): c for c in gdf.columns}
    for c in MAP_ADMIN_COLUMN_CANDIDATES.get(al, []):
        if c.lower() in cmap:
            return cmap[c.lower()]
    tk = al.lower().replace("admin", "adm")
    for c in gdf.columns:
        if tk in str(c).lower():
            return c
    return None


@st.cache_data(show_spinner=False, ttl=300)
def map_admin_options(al: str) -> List[str]:
    gdf = _map_load_admin_shapefile()
    if gdf is None:
        return []
    col = _map_admin_column(gdf, al)
    if not col:
        return []
    vals = gdf[col].dropna().astype(str).str.strip()
    return sorted(vals[vals != ""].unique().tolist())


@st.cache_data(show_spinner=False, ttl=600)
def map_admin_bounds(al: str, an: str):
    if not an or an == "All":
        return None
    gdf = _map_load_admin_shapefile()
    if gdf is None:
        return None
    col = _map_admin_column(gdf, al)
    if not col:
        return None
    mask = gdf[col].astype(str).str.strip().str.lower().eq(an.strip().lower())
    subset = gdf[mask]
    if subset.empty:
        return None
    try:
        return tuple(subset.total_bounds)
    except Exception:
        return None


def _get_admin_subset(al: str, an: str):
    """Returns the GeoDataFrame subset for a specific admin level and name."""
    if not al or not an or an == "All":
        return None
    gdf = _map_load_admin_shapefile()
    if gdf is None:
        return None
    col = _map_admin_column(gdf, al)
    if not col:
        return None
    mask = gdf[col].astype(str).str.strip().str.lower().eq(an.strip().lower())
    subset = gdf[mask]
    return subset if not subset.empty else None


def _map_clip_bbox(da: xr.DataArray, bbox, pad_ratio: float = 0.08) -> xr.DataArray:
    """Clips DataArray to bbox with spatial padding to prevent shapefile boundary edge truncation."""
    lon, lat = _map_lon_lat_names(da)
    if not lon or not lat:
        return da

    minx, miny, maxx, maxy = bbox

    # Add spatial padding buffer (at least 0.25 degrees or pad_ratio of span)
    dx = max((maxx - minx) * pad_ratio, 0.25)
    dy = max((maxy - miny) * pad_ratio, 0.25)

    minx_padded, maxx_padded = minx - dx, maxx + dx
    miny_padded, maxy_padded = miny - dy, maxy + dy

    lats = da[lat].values
    lat_slice = (
        slice(maxy_padded, miny_padded)
        if len(lats) > 1 and lats[0] > lats[-1]
        else slice(miny_padded, maxy_padded)
    )
    try:
        return da.sel({lon: slice(minx_padded, maxx_padded), lat: lat_slice})
    except Exception:
        return da


@st.cache_data(show_spinner=False, ttl=300)
def load_map_da_cached(
    nc_path_str: str,
    ml: str,
    wl: str,
    al: str = "",
    an: str = "",
    file_mtime: float = 0.0,
) -> Optional[xr.DataArray]:
    try:
        path = Path(nc_path_str)
        if not path.exists():
            return None

        with xr.open_dataset(path) as ds:
            mm = MAP_METRICS.get(ml)
            if not mm:
                return None

            var = _find_var(ds, mm["keywords"])
            if not var:
                return None

            da = _map_select_window(ds[var], wl)
            da = apply_jjas_mask(da, getattr(config, "nc_mask", None))
            da = clip_to_boundary(da, climatology_boundary_path(config))
            while len(da.dims) > 2:
                da = da.isel({da.dims[0]: 0})

            if {"lat", "lon"}.issubset(da.dims):
                try:
                    from momp.utils.land_mask import shp_mask

                    da = shp_mask(da, region=getattr(config, "region", "Ethiopia"))
                except Exception:
                    pass

            if mm.get("fraction", False):
                arr = np.asarray(da.values, dtype=float)
                if np.isfinite(arr).any() and float(np.nanmax(arr)) <= 1.5:
                    da = da * 100.0

            if al and an and an != "All":
                bbox = map_admin_bounds(al, an)
                if bbox:
                    da = _map_clip_bbox(da, bbox)

            return da.load()
    except Exception:
        return None


def plot_map(
    da,
    title,
    cmap="viridis",
    unit="",
    vmin=None,
    vmax=None,
    diverging=False,
    cbar_kwargs=None,
    admin_gdf=None,
    boundary_path=None,
):
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=140)
    try:
        lon, lat = _map_lon_lat_names(da)
        zones = None
        if lon and lat:
            boundary_path = boundary_path or climatology_boundary_path(config)
            zones = load_boundary_frame(boundary_path)
            da = clip_to_boundary(apply_jjas_mask(da, getattr(config, "nc_mask", None)), boundary_path)
        arr = np.asarray(da.values, dtype=float)
        has_data = np.isfinite(arr).any()
        if diverging:
            lim = (float(np.nanmax(np.abs(arr))) or 1.0) if has_data else 1.0
            vmin, vmax, cmap = -lim, lim, "RdBu_r"
        elif not has_data:
            vmin = 0 if vmin is None else vmin
            vmax = 1 if vmax is None else vmax

        kwargs = {}
        if cbar_kwargs is not None:
            kwargs["cbar_kwargs"] = cbar_kwargs
        elif unit or title:
            kwargs["cbar_kwargs"] = {"label": unit or title}
        if not has_data:
            kwargs = {"add_colorbar": False}

        if lon and lat:
            image = da.plot.imshow(
                ax=ax,
                x=lon,
                y=lat,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                **kwargs,
            )
            clip_map_artist(image, ax, boundary_path)
            zones.boundary.plot(ax=ax, color="black", linewidth=1.0, zorder=10)

            # Overlay Admin Boundary Polygon Line
            if admin_gdf is not None and not admin_gdf.empty:
                admin_gdf.boundary.plot(ax=ax, color="black", linewidth=1.4, zorder=11)

            # Explicit padding limits
            lons_v = da[lon].values
            lats_v = da[lat].values
            frame = admin_gdf if admin_gdf is not None and not admin_gdf.empty else zones
            west, south, east, north = frame.total_bounds
            region_bounds = (west, east, south, north)
            west, east, south, north = geographic_extent(lons_v, lats_v, da.transpose(lat, lon).values, region_bounds=region_bounds)
            ax.set_xlim(west, east)
            ax.set_ylim(south, north)
            ax.set_aspect("equal", adjustable="box")
            ax.set_anchor("C")
        else:
            im = ax.imshow(
                arr,
                origin="lower",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )
            cb_label = unit or title
            fig.colorbar(im, ax=ax, label=cb_label)

        ax.set_title(title, fontsize=10)
        if not has_data:
            ax.text(.5, .5, "No valid metric values", transform=ax.transAxes, ha="center", va="center",
                    bbox={"facecolor": "white", "alpha": .9, "edgecolor": "none"}, zorder=12)
        plt.tight_layout()
        return fig
    except Exception as exc:
        plt.close(fig)
        st.warning(f"Could not render map {title}: {exc}")
        return None


def fig_to_bytes(fig, tight: bool = True) -> bytes:
    try:
        buf = io.BytesIO()
        fig.savefig(
            buf,
            format="png",
            dpi=140,
            bbox_inches="tight" if tight else None,
            pad_inches=0 if not tight else 0.1,
        )
        buf.seek(0)
        return buf.read()
    finally:
        plt.close(fig)


def _shared_vmin_vmax(*arrays):
    fins = [
        a[np.isfinite(a)]
        for a in (np.asarray(a, dtype=float) for a in arrays)
    ]
    fins = [f for f in fins if f.size]
    if not fins:
        return None, None
    all_vals = np.concatenate(fins)
    vmin, vmax = float(all_vals.min()), float(all_vals.max())
    if vmin == vmax:
        vmax = vmin + 1.0
    return vmin, vmax


def create_swipe_comparison_image(
    left_bytes: bytes,
    right_bytes: bytes,
    split_pct: float,
):
    try:
        img_l = Image.open(io.BytesIO(left_bytes)).convert("RGB")
        img_r = Image.open(io.BytesIO(right_bytes)).convert("RGB")
    except Exception:
        return None

    if img_l.size != img_r.size:
        img_r = img_r.resize(img_l.size, Image.Resampling.LANCZOS)

    w, h = img_l.size
    sx = max(0, min(w, int(w * split_pct / 100.0)))
    comp = Image.new("RGB", (w, h), color="white")
    if sx > 0:
        comp.paste(img_l.crop((0, 0, sx, h)), (0, 0))
    if sx < w:
        comp.paste(img_r.crop((sx, 0, w, h)), (sx, 0))

    draw = ImageDraw.Draw(comp)
    if 0 < sx < w:
        draw.line([(sx, 0), (sx, h)], fill=(37, 99, 235), width=4)
    return comp


def subtract_maps(da_a, da_b):
    try:
        da_a, da_b = xr.align(da_a, da_b, join="inner")
        return da_a - da_b
    except Exception:
        return None


@st.cache_data(show_spinner=False, ttl=300)
def compute_model_comparison_cached(ml, wl, al, an):
    outs = discover_map_outputs()
    ual = al if al and al != "All" else ""
    uan = an if an and an != "All" else ""
    recs: Dict[str, List[float]] = {}

    for item in outs:
        if wl not in item.get("windows", []):
            continue
        da = load_map_da_cached(item["path"], ml, wl, ual, uan, file_mtime=item.get("mtime", 0.0))
        if da is None:
            continue
        arr = np.asarray(da.values, dtype=float)
        if not np.isfinite(arr).any():
            continue
        recs.setdefault(item["model_label"], []).append(float(np.nanmean(arr)))

    rows = [
        {"Model": k, ml: float(np.nanmean(v))}
        for k, v in recs.items()
        if v
    ]
    df = pd.DataFrame(rows)
    return df.sort_values(ml).reset_index(drop=True) if not df.empty else df


# ======================================================================
# Metrics Table Helpers
# ======================================================================
def _extract_spatial_mean(da, convert_fraction=False):
    while len(da.dims) > 2:
        da = da.isel({da.dims[0]: 0})
    arr = np.asarray(da.values, dtype=float)
    if not np.isfinite(arr).any():
        return None
    val = float(np.nanmean(arr))
    return val / 100.0 if convert_fraction and val > 1.5 else val


@st.cache_data(show_spinner=False, ttl=300)
def load_metric_values_by_window_cached(nc_path_str: str, file_mtime: float = 0.0):
    nc_path = Path(nc_path_str)
    dw = _map_infer_window_label(nc_path)
    res: Dict[str, Dict[str, float]] = {}

    try:
        with xr.open_dataset(nc_path) as ds:
            for mn, kws, is_frac in [
                ("Mean Absolute Error days", ["mean_absolute_error", "mean_mae", "mae"], False),
                ("Miss Rate fraction", ["miss_rate", "miss", "mr"], True),
                ("False Alarm Rate fraction", ["false_alarm_rate", "false_alarm", "far"], True),
            ]:
                var = _find_var(ds, kws)
                if not var:
                    continue
                wdim = _map_find_window_dim(ds[var])
                if wdim:
                    nw = int(ds[var].sizes[wdim])
                    wls = [dw] if nw == 1 else (["Days 1–15", "Days 16–30"] if nw == 2 else [f"Window {i + 1}" for i in range(nw)])

                    for i, wl in enumerate(wls[:nw]):
                        val = _extract_spatial_mean(ds[var].isel({wdim: i}), is_frac)
                        if val is not None:
                            res.setdefault(wl, {})[mn] = val
                else:
                    val = _extract_spatial_mean(ds[var], is_frac)
                    if val is not None:
                        res.setdefault(dw, {})[mn] = val
    except Exception:
        pass
    return res

WINDOW_TOLERANCE_DAYS = {
    "Days 1–15": 3,
    "Days 16–30": 5,
}

def _extract_spatial_mean(da, convert_fraction=False):
    while len(da.dims) > 2:
        da = da.isel({da.dims[0]: 0})
    arr = np.asarray(da.values, dtype=float)
    if not np.isfinite(arr).any():
        return None
    val = float(np.nanmean(arr))
    return val / 100.0 if convert_fraction and val > 1.5 else val


@st.cache_data(show_spinner=False, ttl=120)
def load_metric_values_by_window_cached(nc_path_str: str):
    nc_path = Path(nc_path_str)
    dw = _map_infer_window_label(nc_path)
    res: Dict[str, Dict[str, float]] = {}

    try:
        with xr.open_dataset(nc_path) as ds:
            for mn, kws, is_frac in [
                (
                    "Mean Absolute Error days",
                    ["mean_absolute_error", "mean_mae", "mae"],
                    False,
                ),
                (
                    "Miss Rate fraction",
                    ["miss_rate", "miss", "mr"],
                    True,
                ),
                (
                    "False Alarm Rate fraction",
                    ["false_alarm_rate", "false_alarm", "far"],
                    True,
                ),
            ]:
                var = _find_var(ds, kws)
                if not var:
                    continue
                wdim = _map_find_window_dim(ds[var])
                if wdim:
                    nw = int(ds[var].sizes[wdim])
                    if nw == 1:
                        wls = [dw]
                    elif nw == 2:
                        wls = ["Days 1–15", "Days 16–30"]
                    else:
                        wls = [f"Window {i + 1}" for i in range(nw)]
                    for i, wl in enumerate(wls[:nw]):
                        val = _extract_spatial_mean(
                            ds[var].isel({wdim: i}),
                            is_frac,
                        )
                        if val is not None:
                            res.setdefault(wl, {})[mn] = val
                else:
                    val = _extract_spatial_mean(ds[var], is_frac)
                    if val is not None:
                        res.setdefault(dw, {})[mn] = val
    except Exception:
        pass
    return res


def build_metrics_table_html(table, models, windows):
    metric_order = [
        "Mean Absolute Error days",
        "Miss Rate fraction",
        "False Alarm Rate fraction",
    ]
    present = {k[0] for k in table.keys()}
    ordered = [m for m in metric_order if m in present] + sorted(
        present - set(metric_order)
    )
    clim = "Traditional Climatology"
    css = (
        "<style>"
        ".all-metrics-table{width:100%;border-collapse:collapse;"
        f"font-size:0.9rem;min-width:{180 + 100 * len(models) * len(windows)}px}}"
        ".all-metrics-table th,.all-metrics-table td{border:1px solid #e5e7eb;"
        "padding:6px 8px;text-align:right;min-width:100px}"
        ".all-metrics-table th{background:#f8fafc;color:#0f1b2a;font-weight:600}"
        ".all-metrics-table td:first-child,.all-metrics-table th:first-child{"
        "text-align:left}"
        ".all-metrics-table tbody td:first-child,"
        ".all-metrics-table thead tr:first-child th:first-child{"
        "min-width:180px;position:sticky;left:0;z-index:2;background:#f8fafc}"
        "</style>"
    )
    parts = [
        css,
        '<div style="overflow-x:auto;">'
        '<table class="all-metrics-table">'
        "<thead><tr><th rowspan='2'>METRIC</th>",
    ]
    for w in windows:
        parts.append(f"<th colspan='{len(models)}'>{w}</th>")
    parts.append("</tr><tr>")
    for _ in windows:
        for m in models:
            parts.append(f"<th>{m}</th>")
    parts.append("</tr></thead><tbody>")

    for metric in ordered:
        parts.append(f"<tr><td>{metric}</td>")
        higher_better = is_metric_higher_better(metric)
        for w in windows:
            cv = table.get((metric, w, clim))
            diffs = [
                abs(table.get((metric, w, m)) - cv)
                for m in models
                if m != clim
                and table.get((metric, w, m)) is not None
                and cv is not None
            ]
            scale = max(diffs) if diffs else None
            for m in models:
                val = table.get((metric, w, m))
                if val is None:
                    parts.append("<td>-</td>")
                    continue
                style = ""
                if m == clim:
                    style = (
                        "background-color: rgba(148,163,184,0.12);"
                        "font-weight:600;"
                    )
                elif cv is not None and scale:
                    diff = val - cv
                    if abs(diff) > 1e-6:
                        better = (diff > 0) if higher_better else (diff < 0)
                        intensity = min(abs(diff) / scale, 1.0)
                        alpha = 0.08 + (0.35 * intensity)
                        color = "34,197,94" if better else "239,68,68"
                        style = (
                            f"background-color: rgba({color},{alpha:.2f});"
                        )
                fmt = (
                    f"{val:.1f}"
                    if metric.startswith("Mean Absolute Error")
                    else f"{val:.3f}"
                )
                parts.append(f"<td style='{style}'>{fmt}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table></div>")
    return "".join(parts)


def _format_distribution_value(value: float, is_fraction: bool) -> str:
    if value is None:
        return "-"
    try:
        value = float(value)
    except Exception:
        return "-"
    if not np.isfinite(value):
        return "-"
    return f"{value:.1f}%" if is_fraction else f"{value:.1f} d"


def _distribution_cell_style(value, baseline_value, scale, higher_better, is_baseline=False):
    neutral = "background-color:rgba(148,163,184,0.12);color:#334155;font-weight:normal;"
    if is_baseline or baseline_value is None or not scale:
        return neutral

    difference = float(value) - float(baseline_value)
    if abs(difference) <= 1e-5:
        return neutral

    better = difference > 0 if higher_better else difference < 0
    intensity = min(abs(difference) / scale, 1.0)
    alpha = 0.08 + 0.35 * intensity
    color = "34,197,94" if better else "239,68,68"
    text_color = "#15803d" if better else "#b91c1c"
    font_weight = "600" if better else "normal"
    return f"background-color:rgba({color},{alpha:.2f});color:{text_color};font-weight:{font_weight};"



@st.cache_data(show_spinner=False, ttl=120)
def compute_metric_distribution_stats() -> pd.DataFrame:
    outs = discover_map_outputs()
    unique_outputs = {}
    for item in outs:
        model_label = item.get("model_label")
        if not model_label:
            continue
        path = item.get("path")
        if not path:
            continue
        for wl in item.get("windows", []):
            unique_outputs.setdefault((model_label, wl), path)

    rows = []
    for (model_label, wl), path in unique_outputs.items():
        for metric_key, metric_meta in MAP_METRICS.items():
            da = load_map_da_cached(path, metric_key, wl, "", "")
            if da is None:
                continue
            arr = np.asarray(da.values, dtype=float).ravel()
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                continue
            rows.append(
                {
                    "Model": model_label,
                    "Lead Window": wl,
                    "Metric": metric_key,
                    "Mean": float(np.mean(arr)),
                    "Min": float(np.min(arr)),
                    "Median": float(np.median(arr)),
                    "P75": float(np.percentile(arr, 75)),
                    "P90": float(np.percentile(arr, 90)),
                    "Max": float(np.max(arr)),
                    "fraction": bool(metric_meta.get("fraction", False)),
                }
            )
    return pd.DataFrame(rows)



@st.cache_data(show_spinner=False, ttl=60)
def compute_metric_distribution_stats() -> pd.DataFrame:
    outs = discover_map_outputs()
    unique_outputs = {}
    for item in outs:
        model_label = item.get("model_label")
        path = item.get("path")
        mtime = item.get("mtime", 0.0)
        if model_label and path:
            for wl in item.get("windows", []):
                unique_outputs[(model_label, wl)] = (path, mtime)

    rows = []
    for (model_label, wl), (path, mtime) in unique_outputs.items():
        for metric_key, metric_meta in MAP_METRICS.items():
            da = load_map_da_cached(path, metric_key, wl, "", "", file_mtime=mtime)
            if da is None:
                continue
            arr = np.asarray(da.values, dtype=float).ravel()
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                continue
            rows.append(
                {
                    "Model": model_label,
                    "Lead Window": wl,
                    "Metric": metric_key,
                    "Mean": float(np.mean(arr)),
                    "Min": float(np.min(arr)),
                    "Median": float(np.median(arr)),
                    "P75": float(np.percentile(arr, 75)),
                    "P90": float(np.percentile(arr, 90)),
                    "Max": float(np.max(arr)),
                    "fraction": bool(metric_meta.get("fraction", False)),
                }
            )
    return pd.DataFrame(rows)

def build_grouped_distribution_cards_html(dist_df: pd.DataFrame) -> str:
    """
    Renders spatial metric distribution cards grouped by Model and Lead Window.
    Color coding: Green = better (lower than climatology), Red = worse.
    Bolding: Bold = best (minimum) value across all evaluated models.
    """
    if dist_df is None or dist_df.empty:
        return "<p>No spatial metric distribution data available.</p>"

    # Sort ordering
    metric_order = {"False Alarm Rate": 0, "Miss Rate": 1, "Mean Absolute Error": 2}
    model_order = {
        "AIFS Single": 0,
        "FuXi": 1,
        "GraphCast": 2,
        "AIFS Ensemble": 3,
        "GenCast": 4,
        "Traditional Climatology": 5,
    }
    window_order = {"Days 1–15": 0, "Days 16–30": 1}

    df = dist_df.copy()
    df["metric_sort"] = df["Metric"].map(lambda m: metric_order.get(m, 999))
    df["model_sort"] = df["Model"].map(lambda m: model_order.get(m, 999))
    df["window_sort"] = df["Lead Window"].map(lambda w: window_order.get(w, 999))
    df = df.sort_values(["model_sort", "window_sort", "metric_sort"])

    stat_cols = ["Mean", "Min", "Median", "P75", "P90", "Max"]

    # Pre-calculate Best (Minimum) and Climatology baseline values
    best_vals = {}
    clim_vals = {}

    for (window, metric), g in df.groupby(["Lead Window", "Metric"]):
        clim_row = g[g["Model"].str.lower().str.contains("clim")]
        for stat in stat_cols:
            vals = pd.to_numeric(g[stat], errors="coerce").dropna()
            if not vals.empty:
                best_vals[(window, metric, stat)] = float(vals.min())
            if not clim_row.empty and pd.notna(clim_row[stat].values[0]):
                clim_vals[(window, metric, stat)] = float(clim_row[stat].values[0])

    # Updated CSS stylesheet with theme-adaptive (light & dark mode) contrast
            css = """
            <style>
            .all-metrics-table {
                width: 100%;
                border-collapse: collapse;
                font-size: 0.88rem;
                margin-bottom: 1rem;
            }
            .all-metrics-table th, .all-metrics-table td {
                border: 1px solid #cbd5e1;
                padding: 7px 12px;
                text-align: right;
            }
            .all-metrics-table th {
                background: #f1f5f9;
                color: #0f172a;
                font-weight: 700;
                text-transform: uppercase;
                font-size: 0.8rem;
            }
            .all-metrics-table td:first-child, .all-metrics-table th:first-child {
                text-align: left;
                font-weight: 600;
            }
            .all-metrics-table td:last-child, .all-metrics-table th:last-child {
                text-align: center;
                color: #64748b;
            }
            
            /* Light Theme Styles */
            .val-better {
                background-color: rgba(34, 197, 94, 0.22) !important;
                color: #15803d !important;
                font-weight: 600;
            }
            .val-worse {
                background-color: rgba(239, 68, 68, 0.22) !important;
                color: #b91c1c !important;
            }
            .val-neutral {
                background-color: rgba(241, 245, 249, 0.6) !important;
                color: #334155 !important;
            }

            /* Dark Theme Overrides */
            @media (prefers-color-scheme: dark) {
                .all-metrics-table th {
                    background: #1e293b !important;
                    color: #f8fafc !important;
                }
                .all-metrics-table td, .all-metrics-table th {
                    border-color: #334155 !important;
                }
                .val-better {
                    background-color: rgba(34, 197, 94, 0.28) !important;
                    color: #4ade80 !important;
                    font-weight: 600;
                }
                .val-worse {
                    background-color: rgba(239, 68, 68, 0.28) !important;
                    color: #f87171 !important;
                }
                .val-neutral {
                    background-color: rgba(51, 65, 85, 0.4) !important;
                    color: #cbd5e1 !important;
                }
            }
            </style>
            """

    html_parts = [css]

    for (model_label, window_label), group_df in df.groupby(["Model", "Lead Window"], sort=False):
        display_model = "Climatology" if "clim" in model_label.lower() else model_label
        title_model = display_model.replace(" ", "_").upper()
        title_window = window_label.upper()

        tolerance_days = WINDOW_TOLERANCE_DAYS.get(window_label)
        if tolerance_days is None:
            if any(t in window_label.lower() for t in ["16-30", "16–30", "16_30"]):
                tolerance_days = 5
            else:
                tolerance_days = 3

        is_clim = "clim" in model_label.lower()

        html_parts.append(f"""
        <div class="dist-card">
            <div class="dist-header">{title_model} - {title_window}</div>
            <div class="dist-subheader">+/{tolerance_days} day tolerance</div>
            <table class="dist-table">
                <thead>
                    <tr>
                        <th>METRIC</th>
                        <th>MEAN</th>
                        <th>MIN</th>
                        <th>MEDIAN</th>
                        <th>P75</th>
                        <th>P90</th>
                        <th>MAX</th>
                        <th>UNIT</th>
                    </tr>
                </thead>
                <tbody>
        """)

        for _, row in group_df.iterrows():
            metric = row["Metric"]
            is_fraction = bool(row.get("fraction", "rate" in metric.lower()))
            unit_str = "fraction" if is_fraction else "days"

            html_parts.append(f"<tr><td>{metric}</td>")

            for stat in stat_cols:
                val = row.get(stat, np.nan)
                if pd.isna(val) or not np.isfinite(val):
                    html_parts.append("<td>-</td>")
                    continue

                val_float = float(val)
                val_str = f"{val_float:.1f}%" if is_fraction else f"{val_float:.1f} d"

                # 1. Bold best (minimum) value
                best_v = best_vals.get((window_label, metric, stat))
                if best_v is not None and abs(val_float - best_v) < 1e-5:
                    val_str = f"<b>{val_str}</b>"

                # 2. Color code vs Climatology baseline
                clim_v = clim_vals.get((window_label, metric, stat))
                if is_clim or clim_v is None:
                    cls_name = "val-neutral"
                else:
                    diff = val_float - clim_v
                    if abs(diff) < 1e-5:
                        cls_name = "val-neutral"
                    elif diff < 0:
                        cls_name = "val-better"  # Lower than climatology (Green)
                    else:
                        cls_name = "val-worse"   # Higher than climatology (Red)

                html_parts.append(f'<td class="{cls_name}">{val_str}</td>')

            html_parts.append(f"<td>{unit_str}</td></tr>")

        html_parts.append("""
                </tbody>
            </table>
        </div>
        """)

    return "".join(html_parts)

# ======================================================================
# Reference / Benchmark Outputs Helpers
# ======================================================================
def classify_reference_asset(path: Path) -> str:
    s = str(path).lower()
    prob_keys = ["aifs_ens", "ens", "ensemble", "probabilistic", "gencast", "brier", "rps", "reliability", "spread", "rank"]
    clim_keys = ["climatology", "baseline", "traditional"]
    det_keys = ["aifs", "fuxi", "graphcast", "deterministic", "mae", "far", "mr"]
    if any(k in s for k in prob_keys):
        return "Probabilistic AI"
    if any(k in s for k in clim_keys):
        return "Climatology / Baseline"
    if any(k in s for k in det_keys):
        return "Deterministic AI"
    return "Other"


@st.cache_data(show_spinner=False, ttl=300)
def collect_reference_figures() -> Dict[str, List[str]]:
    grouped = {cat: [] for cat in REFERENCE_CATEGORY_ORDER}
    if not ROMP_DEMO_FIG_DIR.exists():
        return grouped
    for path in sorted(ROMP_DEMO_FIG_DIR.rglob("*.png")):
        cat = classify_reference_asset(path)
        grouped.setdefault(cat, []).append(str(path.resolve()))
    return grouped


# ======================================================================
# Onset Time Series Helpers
# ======================================================================
def _onset_config_kwargs() -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in vars(config).items():
        if key.startswith("_") or not key.islower():
            continue
        if inspect.ismodule(value) or inspect.isclass(value) or callable(value):
            continue
        out[key] = str(value) if isinstance(value, Path) else value
    return out



@st.cache_data(show_spinner=False, ttl=600)
def _discover_dataset_structure(obs_dir_path: str):
    """
    Cached dataset inspector: Scans files once every 10 minutes (ttl=600)
    to prevent blocking the Streamlit main execution thread.
    """
    if config.is_uri(obs_dir_path):
        try:
            file_paths = config.glob(obs_dir_path.rstrip("/") + "/*.nc")
        except Exception:
            file_paths = []
    else:
        p = Path(obs_dir_path)
        file_paths = [str(f) for f in p.glob("*.nc")] if p.exists() else []

    discovered_patterns = []
    discovered_years = set()

    for fp in file_paths:
        fname = Path(fp).name
        # Find 4-digit years in filename (e.g., 1980–2030)
        matches = list(re.finditer(r"(19\d{2}|20\d{2})", fname))
        for m in matches:
            year = int(m.group(1))
            discovered_years.add(year)
            pattern = fname[:m.start()] + "{}" + fname[m.end():]
            if pattern not in discovered_patterns:
                discovered_patterns.append(pattern)

    if not discovered_patterns:
        discovered_patterns = ["{}.nc"]

    return tuple(discovered_patterns), sorted(discovered_years)


@st.cache_data(show_spinner=False, ttl=600)
def _prepare_local_obs_dir(remote_or_local_path: str) -> str:
    """
    Cached local dir preparer: Ensures local copies exist without 
    re-downloading on every UI rerun.
    """
    path_str = str(remote_or_local_path)

    # 1. Check local data candidates first
    local_candidates = [
        Path(path_str) if not config.is_uri(path_str) else None,
        Path(getattr(config, "PROJECT_ROOT", ".")) / "data" / "external" / "CHIRPS_regridded_025",
        Path(getattr(config, "PROJECT_ROOT", ".")) / "data" / "external" / "CHIRPS_IMERG",
        Path(getattr(config, "PROJECT_ROOT", ".")) / "data" / "external" / "CHIRPS",
        Path(getattr(config, "PROJECT_ROOT", ".")) / "data" / "external" / "CHIRPS_cache",
    ]

    for cand in local_candidates:
        if cand and cand.exists() and cand.is_dir():
            if list(cand.glob("*.nc")):
                return str(cand.resolve())

    # 2. Cache remote files locally if given a GCS URI
    if config.is_uri(path_str):
        cache_dir = Path(getattr(config, "PROJECT_ROOT", ".")) / "data" / "external" / "CHIRPS_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        try:
            remote_nc_files = config.glob(path_str.rstrip("/") + "/*.nc")
            if remote_nc_files:
                import fsspec
                fs, _ = fsspec.core.url_to_fs(path_str)
                for r_file in remote_nc_files:
                    fname = Path(r_file).name
                    local_file = cache_dir / fname
                    if not local_file.exists() or local_file.stat().st_size == 0:
                        with fs.open(r_file, "rb") as src, open(local_file, "wb") as dst:
                            dst.write(src.read())
                if list(cache_dir.glob("*.nc")):
                    return str(cache_dir.resolve())
        except Exception:
            pass

    return path_str


def _get_onset_base_kwargs():
    raw_obs_dir = (
        getattr(config, "CHIRPS_025_DIR", None)
        or getattr(config, "CHIRPS_DIR", None)
        or getattr(config, "OBS_DIR", None)
        or getattr(config, "ref_model_dir", None)
        or Path(getattr(config, "PROJECT_ROOT", ".")) / "data" / "external" / "CHIRPS_regridded_025"
    )
    obs_dir = _prepare_local_obs_dir(str(raw_obs_dir))
    auto_patterns, _ = _discover_dataset_structure(obs_dir)

    default_params = {
        "start_year_clim": getattr(config, "start_year_clim", getattr(config, "START_YEAR_CLIM", 1981)),
        "end_year_clim": getattr(config, "end_year_clim", getattr(config, "END_YEAR_CLIM", 2010)),
        "start_date": getattr(config, "start_date", "04-01"),
        "end_date": getattr(config, "end_date", "10-31"),
        "fallback_date": getattr(config, "fallback_date", "10-31"),
        "wet_threshold": getattr(config, "wet_threshold", 20.0),
        "wet_init": getattr(config, "wet_init", 1.0),
        "wet_spell": getattr(config, "wet_spell", 3),
        "dry_spell": getattr(config, "dry_spell", 7),
        "dry_threshold": getattr(config, "dry_threshold", 1.0),
        "dry_extent": getattr(config, "dry_extent", 0),
        "mok": getattr(config, "mok", (5, 15)),
        "save_nc_climatology": False,
    }

    base_kwargs = {
        "obs": getattr(config, "OBS_NAME", "CHIRPS"),
        "obs_dir": obs_dir,
        "obs_file_pattern": auto_patterns,
        "obs_var": getattr(config, "OBS_VAR", getattr(config, "obs_var", "RAINFALL")),
        "ref_model_dir": obs_dir,
        "ref_model_file_pattern": auto_patterns,
        "ref_model_var": getattr(config, "REF_MODEL_VAR", getattr(config, "ref_model_var", "RAINFALL")),
        "dir_fig": str(ROMP_DEMO_FIG_DIR),
        **default_params,
    }
    base_kwargs = {**_onset_config_kwargs(), **base_kwargs}

    FORBIDDEN_KWARGS = {"year", "years", "lat", "lon", "lat_select", "lon_select"}
    base_kwargs = {
        k: v for k, v in base_kwargs.items()
        if str(k).lower() not in FORBIDDEN_KWARGS
    }
    return base_kwargs, obs_dir


@st.cache_data(show_spinner=False, ttl=600)
def _available_years():
    base_kwargs, obs_dir = _get_onset_base_kwargs()
    _, years = _discover_dataset_structure(obs_dir)
    return years or list(range(2015, 2024))

@st.cache_data(show_spinner=False, ttl=300)
def _snap_to_grid(lat, lon, data_dir_str):
    files = config.glob(str(data_dir_str).rstrip("/") + "/*.nc")
    sample = files[0] if files else None
    if not sample:
        return lat, lon
    try:
        with config.open_dataset(sample) as ds:
            lat_key = next((k for k in ds.variables if str(k).lower() in ("lat", "latitude", "y")), None)
            lon_key = next((k for k in ds.variables if str(k).lower() in ("lon", "longitude", "x")), None)
            if not lat_key or not lon_key:
                return lat, lon
            nlat = ds[lat_key].sel({lat_key: lat}, method="nearest").values.item()
            nlon = ds[lon_key].sel({lon_key: lon}, method="nearest").values.item()
        return float(nlat), float(nlon)
    except Exception:
        return lat, lon


@st.cache_data(show_spinner=False, ttl=600)
def _find_shapefile_path():
    hint = (
        getattr(config, "WOREDA_SHP", None)
        or getattr(config, "SHAPEFILE_PATH", None)
        or getattr(config, "ZONE_WOREDA_SHP", None)
    )
    if hint and Path(hint).is_file():
        return str(Path(hint).resolve())
    root = Path(getattr(config, "DATA_DIR", Path(config.PROJECT_ROOT) / "data"))
    found = list(root.rglob("*.shp")) if root.is_dir() else []
    preferred = [p for p in found if any(k in p.name.lower() for k in ("woreda", "wereda", "adm3"))]
    best = (preferred or found or [None])[0]
    return str(best.resolve()) if best else None


@st.cache_resource(show_spinner=False)
def _load_woredas_cached(shp_path_str):
    try:
        import geopandas as gpd
    except Exception:
        return {}
    if not shp_path_str:
        return {}
    try:
        gdf = gpd.read_file(shp_path_str)
        if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(4326)
        NAME_COLS = ("woreda", "w_name", "wname", "woreda_name", "wor_name", "adm3_en", "adm3_name", "admin3name", "name")
        ZONE_COLS = ("zone", "z_name", "zone_name", "adm2_en", "admin2name")
        lower = {c.lower(): c for c in gdf.columns}
        name_col = next((lower[c] for c in NAME_COLS if c in lower), None)
        if name_col is None:
            name_col = next((c for c in gdf.columns if c != gdf.geometry.name and gdf[c].dtype == object), None)
        if name_col is None:
            return {}
        zone_col = next((lower[c] for c in ZONE_COLS if c in lower), None)
        pts = gdf.geometry.representative_point()
        out = {}
        for i, name in enumerate(gdf[name_col].astype(str)):
            label = f"{name} ({gdf[zone_col].iloc[i]})" if zone_col else name
            if label in out:
                label = f"{label} #{i}"
            out[label] = (float(pts.iloc[i].y), float(pts.iloc[i].x))
        return dict(sorted(out.items()))
    except Exception:
        return {}


_ONSET_WORKER_SOURCE = r"""
import os, pickle, sys, warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.getcwd())

payload = pickle.loads(Path(sys.argv[1]).read_bytes())
out_dir = Path(sys.argv[2])
out_dir.mkdir(parents=True, exist_ok=True)
sys.argv = [sys.argv[0]]

import config
import momp.app.onset_time_series as ots

real_close = plt.close
plt.show = lambda *a, **k: None
plt.close = lambda *a, **k: None

kwargs = payload.get("kwargs", {})
for k in ["year", "years", "lat", "lon", "lat_select", "lon_select"]:
    kwargs.pop(k, None)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    ots.obs_onset_analysis(
        year=payload["year"],
        lat_select=payload["lat_select"],
        lon_select=payload["lon_select"],
        **kwargs,
    )

saved = 0
for num in plt.get_fignums():
    fig = plt.figure(num)
    if not fig.get_axes():
        continue
    fig.savefig(out_dir / f"fig_{saved:02d}.png", format="png", dpi=110, bbox_inches="tight")
    saved += 1

real_close("all")
print(f"[onset] saved {saved} figure(s)")
"""


def _analyze_onset_streamlit(year, lat, lon):
    import pickle
    import tempfile

    base_kwargs, CHIRPS_DIR = _get_onset_base_kwargs()
    grid_lat, grid_lon = _snap_to_grid(lat, lon, str(CHIRPS_DIR))
    pngs: List[bytes] = []

    # Clean kwargs to guarantee no duplicated year argument
    for k in ["year", "years", "lat", "lon", "lat_select", "lon_select"]:
        base_kwargs.pop(k, None)

    env = {}
    _SAFE_ENV_VARS = [
        "PATH", "HOME", "USER", "SHELL", "TERM", "LANG", "LC_ALL",
        "VIRTUAL_ENV", "CONDA_PREFIX", "CONDA_DEFAULT_ENV",
    ]
    for key in _SAFE_ENV_VARS:
        if key in _STARTUP_ENV:
            env[key] = _STARTUP_ENV[key]

    env["PYTHONPATH"] = str(PROJECT_ROOT.resolve())
    env["PYTHONUNBUFFERED"] = "1"
    env["MPLBACKEND"] = "Agg"
    env["MPLCONFIGDIR"] = str((PROJECT_ROOT / ".mplcache").resolve())
    Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        payload_file = Path(tmp) / "payload.pkl"
        out_dir = Path(tmp) / "figs"
        try:
            payload_file.write_bytes(
                pickle.dumps(
                    {
                        "year": int(year),
                        "lat_select": grid_lat,
                        "lon_select": grid_lon,
                        "kwargs": base_kwargs,
                    }
                )
            )
            from types import SimpleNamespace

            _worker_out: List[str] = []
            worker_python = select_runtime_python(PROJECT_ROOT)
            check_runtime_python(worker_python, env, PROJECT_ROOT, _spawn_no_fork)
            _rc, _timed_out = _spawn_no_fork(
                [worker_python, "-c", _ONSET_WORKER_SOURCE, str(payload_file), str(out_dir)],
                env,
                str(PROJECT_ROOT.resolve()),
                900,
                _worker_out.append,
            )
            if _timed_out:
                raise subprocess.TimeoutExpired("onset analysis", 900)
            proc = SimpleNamespace(returncode=_rc, stdout="\n".join(_worker_out), stderr="")
        except subprocess.TimeoutExpired:
            st.error("Onset analysis timed out after 15 minutes.")
            return pngs, grid_lat, grid_lon
        except Exception as e:
            st.error(f"Could not start onset analysis: {type(e).__name__}: {e}")
            return pngs, grid_lat, grid_lon

        if out_dir.exists():
            pngs = [f.read_bytes() for f in sorted(out_dir.glob("*.png"))]

        if proc.returncode != 0:
            err_lines = (proc.stderr or proc.stdout or "").strip().splitlines()
            st.error("Error during onset analysis: " + (err_lines[-1] if err_lines else f"exit code {proc.returncode}"))
        elif not pngs:
            st.warning("Onset analysis finished but produced no figures.")

    return pngs, grid_lat, grid_lon


# ======================================================================
# Execution Bridge
# ======================================================================
def _locate_run_momp() -> Optional[Path]:
    for c in [PROJECT_ROOT / "run_momp.py", Path.cwd() / "run_momp.py"]:
        if c.exists():
            return c.resolve()
    return None


def _spawn_no_fork(argv, env, cwd, timeout, on_line=None, cancel_event=None):
    """
    Run `argv` in `cwd` using posix_spawn on Unix to prevent memory fragmentation and fork-safety issues.
    """
    import select
    import signal
    import time
    from ai_weather.benchmark_job import track_worker, untrack_worker

    started = time.time()

    if not hasattr(os, "posix_spawn"):
        proc = subprocess.Popen(
            argv, cwd=cwd, env=env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        timed_out = False
        try:
            for line in proc.stdout:
                if on_line:
                    on_line(line.rstrip("\n"))
                if cancel_event is not None and cancel_event.is_set():
                    proc.kill()
                    break
                if time.time() - started > timeout:
                    proc.kill()
                    timed_out = True
                    break
        finally:
            proc.stdout.close()
            proc.wait()
        return (130 if cancel_event is not None and cancel_event.is_set() else 124 if timed_out else proc.returncode), timed_out

    read_fd, write_fd = os.pipe()
    actions = [
        (os.POSIX_SPAWN_OPEN, 0, os.devnull, os.O_RDONLY, 0),
        (os.POSIX_SPAWN_DUP2, write_fd, 1),
        (os.POSIX_SPAWN_DUP2, write_fd, 2),
        (os.POSIX_SPAWN_CLOSE, read_fd),
        (os.POSIX_SPAWN_CLOSE, write_fd),
    ]
    shell_argv = ["/bin/sh", "-c", 'cd "$1" && shift && exec "$@"', "sh", str(cwd), *map(str, argv)]

    try:
        pid = os.posix_spawn("/bin/sh", shell_argv, env, file_actions=actions, setpgroup=0)
    except BaseException:
        os.close(read_fd)
        os.close(write_fd)
        raise
    os.close(write_fd)
    track_worker(pid)
    status = None
    termination_attempted = False

    def poll_exit():
        nonlocal status
        if status is None:
            exited, child_status = os.waitpid(pid, os.WNOHANG)
            if exited:
                status = child_status

    def terminate_group():
        nonlocal termination_attempted
        termination_attempted = True
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            poll_exit()
            return
        except PermissionError:
            poll_exit()
            if status is None:
                raise
        deadline = time.time() + 0.2
        while status is None and time.time() < deadline:
            poll_exit()
            if status is None:
                time.sleep(0.02)
        try:
            # The group may already be gone after graceful termination.
            os.killpg(pid, 0)
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except PermissionError:
            poll_exit()
            if status is None:
                raise
            if on_line:
                on_line("[stop-warning] The main benchmark process exited, but the operating system blocked verification of remaining worker cleanup.")

    timed_out = False
    buf = b""
    try:
        while True:
            if cancel_event is not None and cancel_event.is_set():
                terminate_group()
                break
            left = timeout - (time.time() - started)
            if left <= 0:
                terminate_group()
                timed_out = True
                break

            ready, _, _ = select.select([read_fd], [], [], min(left, 0.5))
            if not ready:
                continue

            chunk = os.read(read_fd, 65536)
            if not chunk:
                break

            buf += chunk
            *done, buf = buf.split(b"\n")
            for raw in done:
                if on_line:
                    on_line(raw.decode("utf-8", "replace").rstrip("\r"))

        if buf and on_line:
            on_line(buf.decode("utf-8", "replace").rstrip("\r"))
    except BaseException as exc:
        if isinstance(exc, PermissionError) and on_line:
            on_line(f"[stop-warning] The operating system denied process management for PID {pid}. Worker shutdown is not confirmed.")
        if not termination_attempted:
            terminate_group()
        raise
    finally:
        os.close(read_fd)
        if status is None:
            _, status = os.waitpid(pid, 0)
        untrack_worker(pid)

    rc = os.waitstatus_to_exitcode(status)
    return (130 if cancel_event is not None and cancel_event.is_set() else 124 if timed_out else rc), timed_out


def run_momp_benchmark(
    mode: str,
    model_key: Optional[str] = None,
    run_config: Optional[Dict[str, Any]] = None,
    progress=None,
    cancel_event=None,
) -> Dict[str, Any]:
    mode = mode.lower()
    if mode not in ("det", "deterministic", "prob", "probabilistic"):
        return {
            "mode": mode, "returncode": 1, "ok": False,
            "stdout": "", "stderr": f"Invalid mode: {mode}",
            "runtime_dir": str(ROMP_DEMO_OUT_DIR),
        }

    rmp = _locate_run_momp()
    if rmp is None:
        return {
            "mode": mode, "returncode": 1, "ok": False,
            "stdout": "", "stderr": "run_momp.py not found.",
            "runtime_dir": str(ROMP_DEMO_OUT_DIR),
        }

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(PROJECT_ROOT.resolve()), env.get("PYTHONPATH", "")])
    env["MPLCONFIGDIR"] = str((PROJECT_ROOT / ".mplcache").resolve())
    Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    env["PYTHONUNBUFFERED"] = "1"

    # CPU Thread Allocation caps to prevent CPU core exhaustion and Streamlit freezes
    env["OMP_NUM_THREADS"] = "4"
    env["MKL_NUM_THREADS"] = "4"
    env["OPENBLAS_NUM_THREADS"] = "4"
    env["VECLIB_MAXIMUM_THREADS"] = "4"
    env["NUMEXPR_NUM_THREADS"] = "4"

    if run_config:
        for key, env_name in (("area_mask_file", "ROMP_AREA_MASK_FILE"),
                              ("onset_threshold_file", "ROMP_ONSET_THRESHOLD_FILE"),
                              ("baseline_forecast", "ROMP_BASELINE_FORECAST")):
            if key in run_config:
                env[env_name] = str(run_config[key] or "")
        if run_config.get("eval_start"):
            env["ROMP_EVAL_START"] = str(run_config["eval_start"])
        if run_config.get("eval_end"):
            env["ROMP_EVAL_END"] = str(run_config["eval_end"])
        if run_config.get("baseline_start_year"):
            env["ROMP_BASELINE_START_YEAR"] = str(run_config["baseline_start_year"])
        if run_config.get("baseline_end_year"):
            env["ROMP_BASELINE_END_YEAR"] = str(run_config["baseline_end_year"])
        if "initialization_days" in run_config:
            env["ROMP_INIT_DAYS"] = ",".join(str(int(x)) for x in run_config.get("initialization_days", []))
        env["ROMP_RUN_YEARS_CONCURRENTLY"] = "1" if run_config.get("run_years_concurrently", True) else "0"
        env["ROMP_ENSEMBLE_FORECAST"] = "1" if run_config.get("ensemble_forecast", False) else "0"
        if run_config.get("forecast_window"):
            env["ROMP_FORECAST_WINDOW"] = str(run_config["forecast_window"]).replace(" days", "")
        if run_config.get("ground_truth"):
            env["ROMP_GROUND_TRUTH"] = str(run_config["ground_truth"])
        if run_config.get("benchmark_coverage"):
            env["ROMP_BENCHMARK_COVERAGE"] = str(run_config["benchmark_coverage"])

        for key, env_name in [
            ("wet_day_threshold", "ROMP_WET_THRESHOLD"),
            ("minimum_wet_day_rainfall", "ROMP_WET_INIT"),
            ("wet_spell_length", "ROMP_WET_SPELL"),
            ("dry_spell_limit", "ROMP_DRY_SPELL"),
            ("dry_spell_search_extension", "ROMP_DRY_EXTENT"),
        ]:
            if run_config.get(key) is not None:
                env[env_name] = str(run_config[key])

        if run_config.get("init_type"):
            env["ROMP_INIT_TYPE"] = str(run_config["init_type"])
        if run_config.get("date_filter_year"):
            env["ROMP_DATE_FILTER_YEAR"] = str(run_config["date_filter_year"])
        if run_config.get("verification_window_list"):
            env["ROMP_VERIFICATION_WINDOW_LIST"] = str(run_config["verification_window_list"])
        if run_config.get("tolerance_days_list"):
            env["ROMP_TOLERANCE_DAYS_LIST"] = str(run_config["tolerance_days_list"])
        if run_config.get("max_forecast_day"):
            env["ROMP_MAX_FORECAST_DAY"] = str(run_config["max_forecast_day"])

    cmd = [sys.executable, str(rmp), "--mode", mode]
    if model_key:
        cmd.extend(["--model", model_key])

    try:
        import time
        cmd[0] = select_runtime_python(PROJECT_ROOT)
        from collections import deque
        def spawn(argv, child_env, cwd, timeout, on_line=None):
            return _spawn_no_fork(argv, child_env, cwd, timeout, on_line, cancel_event=cancel_event)
        runtime_info = check_runtime_python(cmd[0], env, PROJECT_ROOT, spawn)
        live = st.empty() if progress is None else None
        lines = deque([runtime_info], maxlen=2000)
        stop_warnings = []
        started = time.time()
        last_draw = 0.0

        def _on_line(line: str) -> None:
            nonlocal last_draw
            lines.append(line)
            if line.startswith("[stop-warning]"):
                stop_warnings.append(line.removeprefix("[stop-warning]").strip())
            if progress is not None:
                progress(line)
                return
            now = time.time()
            if now - last_draw > 1.0:
                live.code(f"[{int(now - started)}s elapsed] {' '.join(cmd[-4:])}\n" + "\n".join(list(lines)[-15:]))
                last_draw = now

        returncode, timed_out = spawn(cmd, env, str(PROJECT_ROOT.resolve()), 1800, _on_line)

        text = "\n".join(lines)
        if timed_out:
            text += "\nTimed out after 30 minutes."

        return {
            "mode": mode,
            "returncode": returncode,
            "runtime_dir": str(ROMP_DEMO_OUT_DIR),
            "stdout": text,
            "stderr": "",
            "ok": (not timed_out) and returncode == 0 and not (cancel_event is not None and cancel_event.is_set()),
            "cancelled": cancel_event is not None and cancel_event.is_set(),
            "stop_warning": " ".join(stop_warnings),
            "command": " ".join(cmd),
        }
    except Exception as e:
        import traceback
        cancelled = cancel_event is not None and cancel_event.is_set()
        return {
            "mode": mode, "returncode": 130 if cancelled else 1, "ok": False, "cancelled": cancelled,
            "stop_warning": "The operating system denied worker shutdown. Some processes may still be running." if cancelled and isinstance(e, PermissionError) else "",
            "runtime_dir": str(ROMP_DEMO_OUT_DIR),
            "stdout": "",
            "stderr": f"Could not run benchmark process: {type(e).__name__}: {e!r}\n" + traceback.format_exc(),
            "command": " ".join(cmd),
        }


# ======================================================================
# UI Components
# ======================================================================
def _inject_theme_css() -> None:
    p = Path("ui/components/styles/page_styles/benchmarking.css")
    css = p.read_text() if p.exists() else ""
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def info_box(text: str, kind: str = "") -> None:
    st.markdown(f'<div class="ibox {kind}">{text}</div>', unsafe_allow_html=True)


def section_header(text: str, accent: str = "sky") -> None:
    st.markdown(
        f'<div class="sec-hd {accent}"><span class="sec-hd-accent"></span>{text}</div>',
        unsafe_allow_html=True,
    )


BENCHMARK_COVERAGE_COUNTRIES = (
    ("🇸🇳", "Senegal"), ("🇷🇼", "Rwanda"), ("🇨🇴", "Colombia"), ("🇵🇪", "Peru"),
    ("🇧🇴", "Bolivia"), ("🇵🇭", "Philippines"), ("🇮🇩", "Indonesia"), ("🇵🇰", "Pakistan"),
    ("🇳🇬", "Nigeria"), ("🇪🇹", "Ethiopia"), ("🇰🇪", "Kenya"), ("🇸🇱", "Sierra Leone"),
    ("🇨🇱", "Chile"), ("🇧🇩", "Bangladesh"),
)
DEFAULT_BENCHMARK_COVERAGE = "Ethiopia"
CONFIGURED_COVERAGE = {"Ethiopia"}


def _coverage_label(name: str) -> str:
    flags = {n: f for f, n in BENCHMARK_COVERAGE_COUNTRIES}
    label = f"{flags[name]} {name}" if name in flags else name
    return f"{label} (Default)" if name == DEFAULT_BENCHMARK_COVERAGE else label


def render_kpi_row(models: int, presets: int, status: str, runtime: str) -> None:
    st.markdown(
        f"""
        <div class="kpi-row">
            <div class="kpi sky">
                <div class="kpi-label">Models</div>
                <div class="kpi-value">{models}</div>
            </div>
            <div class="kpi teal">
                <div class="kpi-label">Presets</div>
                <div class="kpi-value">{presets}</div>
            </div>
            <div class="kpi green">
                <div class="kpi-label">Status</div>
                <div class="kpi-value" style="font-size:1.1rem;">{status}</div>
            </div>
            <div class="kpi amber">
                <div class="kpi-label">Output Dir</div>
                <div class="kpi-value" style="font-size:0.85rem;word-break:break-all;">{runtime}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_shared_settings(ground_truth: str = "CHIRPS", advanced_renderer=None, run_windows_renderer=None) -> Dict[str, Any]:
    section_header("Shared settings", "sky")
    st.markdown("Optional settings applied to every selected model in this benchmark.")

    st.markdown("**EVENT DETECTION**")
    c1, c2, c3 = st.columns(3)
    wet_day_threshold = c1.number_input("Wet-day threshold", min_value=0.0, value=float(getattr(config, "wet_threshold", 20)), step=0.5, format="%.1f", key="shared_wet_day_threshold")
    minimum_wet_day_rainfall = c2.number_input("Minimum wet-day rainfall", min_value=0.0, value=float(getattr(config, "wet_init", 1)), step=0.1, format="%.1f", key="shared_min_wet_day")
    wet_spell_length = c3.number_input("Wet spell length", min_value=1, value=3, format="%d", key="shared_wet_spell")

    c4, c5, _ = st.columns(3)
    dry_spell_limit = c4.number_input("Dry spell limit", min_value=0, value=7, format="%d", key="shared_dry_spell")
    dry_spell_search_extension = c5.number_input("Dry spell search extension", min_value=0, value=int(getattr(config, "dry_extent", 21)), format="%d", key="shared_dry_ext")

    masks_column, advanced_column = st.columns(2, gap="large")
    with masks_column:
        st.markdown("**MASKS AND BASELINE**")
        default_mask = EXTERNAL_DATA_DIR / "jjas_100mm_rainfall_mask_0p25.nc"
        seasonal_mask = EXTERNAL_DATA_DIR / "jjas_seasonal_mask_0p25.nc"
        mask_options = {
            "No mask": "",
            "JJAS 100mm rainfall mask": str(default_mask),
            "JJAS seasonal 0.25 mask": str(seasonal_mask),
            "Custom path": "__custom__",
        }
        configured_mask = str(getattr(config, "nc_mask", None) or "")
        default_mask_choice = next((label for label, value in mask_options.items() if value == configured_mask), "Custom path")
        area_mask_choice = st.selectbox("Area mask file", list(mask_options.keys()),
                                       index=list(mask_options).index(default_mask_choice), key="shared_area_mask_from_config")
        area_mask_file = mask_options[area_mask_choice]
        if area_mask_choice == "Custom path":
            area_mask_file = st.text_input("Area mask file path", value=configured_mask, key="shared_area_mask_custom")

        default_threshold = ONSET_REFERENCE_DIR / "thresholds_df.csv"
        threshold_options = {
            "Dataset default": "",
            "Reference thresholds CSV": str(default_threshold),
            "Custom path": "__custom__",
        }
        onset_threshold_choice = st.selectbox("Onset threshold file", list(threshold_options.keys()), index=0, key="shared_onset_thresh")
        onset_threshold_file = threshold_options[onset_threshold_choice]
        if onset_threshold_choice == "Custom path":
            onset_threshold_file = st.text_input("Onset threshold file path", value="", key="shared_onset_custom")

        baseline_forecast_choice = st.selectbox("Baseline forecast", ["climatology", "custom"], index=0, key="shared_baseline_fc")
        baseline_forecast = st.text_input("Baseline forecast name", value="climatology", key="shared_baseline_custom") if baseline_forecast_choice == "custom" else "climatology"
        if run_windows_renderer is not None:
            run_windows_renderer()

    with advanced_column:
        st.markdown("**ADVANCED TEMPORAL & VERIFICATION SETTINGS**")
        if advanced_renderer is None:
            st.info("Select models above to configure initialization schedules and verification windows.")
        else:
            advanced_renderer()

    return {
        "wet_day_threshold": float(wet_day_threshold),
        "minimum_wet_day_rainfall": float(minimum_wet_day_rainfall),
        "wet_spell_length": int(wet_spell_length),
        "dry_spell_limit": int(dry_spell_limit),
        "dry_spell_search_extension": int(dry_spell_search_extension),
        "area_mask_file": str(area_mask_file or "").strip(),
        "onset_threshold_file": str(onset_threshold_file or "").strip(),
        "baseline_forecast": str(baseline_forecast or "climatology").strip(),
    }


# <<<<<<< HEAD
# @st.dialog("Figure Inspector", width="large")
# def _open_figure_modal(path_str: str) -> None:
# =======
# ======================================================================
# PDF Report Helpers
# ======================================================================
def _pdf_safe_text(value: Any) -> str:
    text = str(value)
    replacements = {
        "\u2013": "-", "\u2014": "-", "\u00b7": "-", "\u2713": "Yes",
        "\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text.encode("latin-1", "ignore").decode("latin-1")


def _normalize_dataframe_for_pdf(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = pd.DataFrame(index=df.index)
    seen = set()
    for col in df.columns:
        base = str(col).strip() or "column"
        name = base
        suffix = 1
        while name in seen:
            suffix += 1
            name = f"{base}_{suffix}"
        seen.add(name)

        def _fmt(x):
            try:
                if pd.isna(x):
                    return ""
            except Exception:
                pass
            if isinstance(x, (int, np.integer)):
                return str(int(x))
            if isinstance(x, (float, np.floating)):
                return f"{x:.3f}"
            return str(x)

        out[name] = df[col].map(_fmt)
    return out


class BenchmarkReportPDF(FPDF):
    MIN_COL_WIDTH = 18

    def _usable_width(self) -> float:
        return getattr(self, "epw", self.w - self.l_margin - self.r_margin)

    def header(self):
        if self.page_no() > 1:
            self.set_font("Helvetica", "B", 9)
            self.set_text_color(15, 23, 42)
            self.set_x(self.l_margin)
            self.cell(0, 8, _pdf_safe_text("Forecast Benchmarking Report"))
            self.ln(4)
            self.set_draw_color(226, 232, 240)
            self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
            self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(100, 116, 139)
        self.set_x(self.l_margin)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

    def ensure_space(self, height: float = 12):
        if self.get_y() + height > self.h - self.b_margin:
            self.add_page()
            self.set_x(self.l_margin)

    def section_title(self, title: str):
        self.ensure_space(18)
        self.set_x(self.l_margin)
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(15, 23, 42)
        self.multi_cell(0, 7, _pdf_safe_text(title))
        self.set_draw_color(59, 130, 246)
        line_y = self.get_y() + 1
        self.line(self.l_margin, line_y, self.l_margin + 45, line_y)
        self.ln(4)

    def sub_title(self, title: str):
        self.ensure_space(12)
        self.set_x(self.l_margin)
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(30, 41, 59)
        self.multi_cell(0, 6, _pdf_safe_text(title))
        self.ln(1)

    def paragraph(self, text: str):
        self.ensure_space(10)
        self.set_x(self.l_margin)
        self.set_font("Helvetica", "", 9)
        self.set_text_color(51, 65, 85)
        self.multi_cell(0, 5, _pdf_safe_text(text))
        self.ln(2)

    def key_value_lines(self, items: List[tuple]):
        for key, value in items:
            self.ensure_space(8)
            self.set_x(self.l_margin)
            self.set_font("Helvetica", "", 9)
            self.set_text_color(51, 65, 85)
            self.multi_cell(0, 6, _pdf_safe_text(f"{key}: {value}"))

    def _fit_text(self, text: str, width: float) -> str:
        try:
            text = _pdf_safe_text(text)
        except Exception:
            text = ""
        if width <= 6 or not text:
            return ""
        max_width = max(1.0, width - 2)
        try:
            if self.get_string_width(text) <= max_width:
                return text
            while text and self.get_string_width(text + "..") > max_width:
                text = text[:-1]
            return text + ".." if text else ""
        except Exception:
            return ""

    def dataframe_table(self, df: pd.DataFrame):
        if df is None or df.empty:
            return
        table_df = _normalize_dataframe_for_pdf(df)
        all_cols = [str(c) for c in table_df.columns]
        if not all_cols:
            return
        usable_width = self._usable_width()
        max_cols_per_table = max(1, int(usable_width // self.MIN_COL_WIDTH))
        col_chunks = [all_cols[i:i + max_cols_per_table] for i in range(0, len(all_cols), max_cols_per_table)]
        
        for chunk_idx, cols in enumerate(col_chunks):
            if not cols:
                continue
            if chunk_idx > 0:
                self.ln(2)
                self.set_x(self.l_margin)
                self.set_font("Helvetica", "I", 7)
                self.set_text_color(100, 116, 139)
                start_col = chunk_idx * max_cols_per_table + 1
                end_col = chunk_idx * max_cols_per_table + len(cols)
                self.cell(0, 5, _pdf_safe_text(f"Table continued - columns {start_col} to {end_col}"))
                self.ln(6)
            n_cols = len(cols)
            widths = [usable_width / n_cols] * n_cols
            row_height = 6
            header_font = 8 if n_cols <= 3 else (7 if n_cols <= 5 else 6)
            body_font = 7 if n_cols <= 3 else (6 if n_cols <= 5 else 5)

            def draw_header():
                self.set_x(self.l_margin)
                self.set_font("Helvetica", "B", header_font)
                self.set_fill_color(241, 245, 249)
                self.set_text_color(15, 23, 42)
                for i, col_name in enumerate(cols):
                    cell_width = widths[i]
                    label = self._fit_text(col_name, cell_width) if cell_width > 6 else ""
                    self.cell(cell_width, row_height + 1, label, border=1, align="C", fill=True)
                self.ln(row_height + 1)

            self.ensure_space(25)
            draw_header()
            self.set_font("Helvetica", "", body_font)
            self.set_text_color(30, 41, 59)
            for _, row in table_df.iterrows():
                if self.get_y() + row_height > self.h - self.b_margin - 2:
                    self.add_page()
                    draw_header()
                    self.set_font("Helvetica", "", body_font)
                    self.set_text_color(30, 41, 59)
                self.set_x(self.l_margin)
                for i, col_name in enumerate(cols):
                    cell_width = widths[i]
                    txt = self._fit_text(row.get(col_name, ""), cell_width) if cell_width > 6 else ""
                    self.cell(cell_width, row_height, txt, border=1)
                self.ln(row_height)
            self.ln(3)

    def add_image_path(self, image_path: str, caption: str = ""):
        image_path = Path(image_path)
        if not image_path.exists():
            return
        try:
            with Image.open(image_path) as img:
                img_w, img_h = img.size
            if img_w <= 0 or img_h <= 0:
                return
            usable_width = self._usable_width()
            max_w = max(20, usable_width - 10)
            target_h = max_w * img_h / img_w
            if self.get_y() + target_h + 18 > self.h - self.b_margin:
                self.add_page()
            available_h = self.h - self.get_y() - self.b_margin - 18
            if target_h > available_h and available_h > 10:
                scale = available_h / target_h
                target_w = max_w * scale
                target_h = available_h
            else:
                target_w = max_w
            if target_w < 15 or target_h < 15:
                return
            x = self.l_margin + (usable_width - target_w) / 2
            start_y = self.get_y()
            self.image(str(image_path), x=x, y=start_y, w=target_w, h=target_h)
            self.set_y(start_y + target_h + 2)
            self.set_x(self.l_margin)
            if caption:
                self.set_font("Helvetica", "I", 8)
                self.set_text_color(71, 85, 105)
                self.multi_cell(0, 5, _pdf_safe_text(caption), align="C")
                self.ln(4)
        except Exception:
            self.paragraph(f"Could not embed image: {image_path.name}")


def generate_benchmark_pdf_report(context: Dict[str, Any]) -> bytes:
    pdf = BenchmarkReportPDF(orientation="L", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(15, 23, 42)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(0, 12, "Forecast Benchmarking Report", align="C")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(71, 85, 105)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(0, 7, _pdf_safe_text(context.get("generated_at", "")), align="C")
    pdf.ln(6)

    pdf.section_title("Executive Summary")
    active_rows = context.get("active_rows", [])
    shared_settings = context.get("shared_settings", {})
    det_count = sum(1 for r in active_rows if r.get("Category") == "Deterministic")
    prob_count = sum(1 for r in active_rows if r.get("Category") == "Probabilistic")
    pdf.key_value_lines([
        ("Ground Truth", context.get("ground_truth", "")),
        ("Benchmark Coverage", context.get("benchmark_coverage", "")),
        ("Forecast Window", context.get("forecast_window", "")),
        ("Baseline Forecast", shared_settings.get("baseline_forecast", "climatology")),
        ("Models Evaluated", len(active_rows)),
        ("Deterministic Models", det_count),
        ("Probabilistic Models", prob_count),
    ])

    if active_rows:
        models_df = pd.DataFrame([
            {
                "Model": r.get("Model", ""),
                "Category": r.get("Category", ""),
                "Type": r.get("Type", ""),
                "Coverage": r.get("Coverage", ""),
                "Date Range": r.get("Date Range", ""),
            }
            for r in active_rows
        ])
        if not models_df.empty:
            pdf.section_title("Models Evaluated")
            pdf.dataframe_table(models_df)

    if shared_settings:
        pdf.section_title("Shared Settings")
        pdf.key_value_lines([(str(k).replace("_", " ").title(), v) for k, v in shared_settings.items()])

    map_outputs = context.get("map_outputs", [])
    if map_outputs:
        outputs_df = pd.DataFrame([
            {
                "Model": o.get("model_label", ""),
                "Windows": ", ".join(o.get("windows", [])),
                "File": Path(o.get("path", "")).name,
            }
            for o in map_outputs
        ])
        if not outputs_df.empty:
            pdf.section_title("Available Model Outputs")
            pdf.dataframe_table(outputs_df)

    scores_df = context.get("scores_df")
    if scores_df is not None and not scores_df.empty:
        pdf.section_title("Verification Scores")
        pdf.dataframe_table(scores_df)

    distribution_stats = context.get("distribution_stats")
    if distribution_stats is not None and not distribution_stats.empty:
        pdf.section_title("Detailed Metric Distributions")
        pdf.dataframe_table(distribution_stats)

    figures = context.get("figures", {})
    has_figures = any(paths for paths in figures.values())
    if has_figures:
        pdf.section_title("Benchmark Output Figures")
        for category, paths in figures.items():
            if not paths:
                continue
            pdf.sub_title(category)
            type_groups: Dict[str, List[str]] = {}
            for path_str in paths:
                path = Path(path_str)
                if not path.exists():
                    continue
                ftype = classify_figure_type(path.name)
                type_groups.setdefault(ftype, []).append(str(path))
            for ftype, fpaths in type_groups.items():
                pdf.paragraph(f"{ftype} ({len(fpaths)})")
                for path_str in fpaths:
                    pdf.add_image_path(path_str, format_figure_label(path_str))

    pdf.section_title("Notes")
    pdf.paragraph(
        "This report was generated automatically from the benchmark dashboard. "
        "Metrics are computed from model output files available at generation time."
    )

    return bytes(pdf.output())

@st.dialog("Figure Inspector", width="large")
def _open_figure_modal(path_str: str) -> None:
    """
    Modal overlay dialog for high-resolution figure viewing and direct downloading.
    """
    path = Path(path_str)
    if path.exists() and path.stat().st_size > 0:
        st.image(str(path), width='stretch')
        st.caption(format_figure_label(str(path)))
        with open(path, "rb") as f:
            img_bytes = f.read()
        st.download_button(
            label="📥 Download High-Res PNG",
            data=img_bytes,
            file_name=path.name,
            mime="image/png",
            key=f"dl_modal_btn_{path.stem}",
            type="primary",
            width='stretch',
        )


@st.cache_data(show_spinner=False, max_entries=24)
def _benchmark_figure_preview(path_str: str, mtime_ns: int, size: int) -> bytes:
    """Cache display-sized PNGs; file metadata invalidates regenerated figures."""
    with Image.open(path_str) as source:
        source.thumbnail((1600, 1200), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        source.save(buffer, format="PNG")
    return buffer.getvalue()


def render_benchmark_figure_with_download(
    path_str: str, 
    key_suffix: str = "", 
    enable_zoom: bool = True
) -> None:
    """
    Safely renders a benchmark figure thumbnail with embedded Download 
    and Zoom/Expand modal trigger buttons.
    """
    path = Path(path_str)

    if not path.exists() or path.stat().st_size == 0:
        st.warning(f"Figure unavailable: {path.name}")
        return

    try:
        stat = path.stat()
        st.image(_benchmark_figure_preview(str(path), stat.st_mtime_ns, stat.st_size), width='stretch')
        st.caption(format_figure_label(str(path)))

        # Action bar: Download and Zoom controls side-by-side
        btn_col1, btn_col2 = st.columns([1, 1])

        with btn_col1:
            st.download_button(
                label="📥 Download",
                data=path.read_bytes,
                file_name=path.name,
                mime="image/png",
                key=f"dl_btn_{path.stem}_{key_suffix}",
                width='stretch',
                on_click="ignore",
            )

        with btn_col2:
            if enable_zoom and hasattr(st, "dialog"):
                if st.button(
                    "Zoom Modal", 
                    key=f"zoom_btn_{path.stem}_{key_suffix}", 
                    width='stretch'
                ):
                    _open_figure_modal(str(path))
    except Exception as e:
        st.warning(f"Could not display figure {path.name}: {type(e).__name__} - {e}")


def render_reference_figure_gallery(paths, view_mode, filter_key):
    """Render one bounded page instead of decoding every benchmark figure."""
    if not paths:
        info_box("No figures match the selected dropdown filters.", "amber")
        return
    page_size = 6
    page_count = (len(paths) + page_size - 1) // page_size
    page = st.selectbox("Figure page", range(1, page_count + 1),
                        format_func=lambda value: f"{value} of {page_count}",
                        key=f"figure_page_{filter_key}_{view_mode}")
    start = (page - 1) * page_size
    page_paths = paths[start:start + page_size]
    st.caption(f"Showing figures {start + 1}–{start + len(page_paths)} of {len(paths)}")
    if view_mode == "Full Width":
        for index, path in enumerate(page_paths, start):
            render_benchmark_figure_with_download(path, key_suffix=f"full_{index}")
            st.divider()
    else:
        for offset in range(0, len(page_paths), 2):
            columns = st.columns(2)
            for column_index, (column, path) in enumerate(zip(columns, page_paths[offset:offset + 2])):
                with column:
                    render_benchmark_figure_with_download(path, key_suffix=f"grid_{start + offset + column_index}")
# ======================================================================
# Main Page
# ======================================================================
@st.cache_data(show_spinner=False, ttl=300, max_entries=32)
def inspect_model_temporal_coverage(model_key, first_year, last_year):
    """Read only initialization/lead coordinates, never load forecast fields."""
    meta = CONFIG_BENCHMARK_MODEL_CATALOG.get(model_key, {})
    folder, pattern = meta.get("dir"), meta.get("file_pattern", "{}.nc")
    dates, horizons, failures = [], {}, []
    for year in range(first_year, last_year + 1):
        path = config.join(folder, pattern.format(year))
        if not config.exists(path):
            continue
        try:
            with config.open_dataset(path) as ds:
                time_name = next((name for name in ds.coords if str(name).casefold()
                                  in {"time", "init_time", "initialization_time", "forecast_reference_time"}), None)
                if time_name:
                    times = pd.to_datetime(ds[time_name].values, errors="coerce")
                    dates.extend((int(t.year), int(t.month), int(t.day)) for t in times if not pd.isna(t))
                lead_name = next((name for name in ds.coords if str(name).casefold() in {"day", "step", "lead", "lead_time"}), None)
                if lead_name:
                    values = np.asarray(ds[lead_name].values)
                    if np.issubdtype(values.dtype, np.timedelta64):
                        values = values / np.timedelta64(1, "D")
                    else:
                        values = values.astype(float)
                        units = str(ds[lead_name].attrs.get("units", "")).casefold()
                        for unit, divisor in (("hour", 24), ("minute", 1440), ("second", 86400)):
                            if unit in units:
                                values = values / divisor
                                break
                    finite = values[np.isfinite(values)]
                    if finite.size:
                        horizons[year] = int(np.floor(finite.max()))
        except Exception as exc:
            failures.append(f"{year}: {type(exc).__name__}: {exc}")
    return {"dates": dates, "horizons": horizons, "failures": failures}


def render_verification_window_settings(row_key, forecast_window):
    options = ["Days 1–15", "Days 16–30", "Both windows"]
    selected = 0 if forecast_window == "1-15 days" else 1
    choice = st.selectbox("Verification windows", options, index=selected,
                          key=f"verification_windows_{row_key}_{selected}")
    windows = (((1, 15),), ((16, 30),), ((1, 15), (16, 30)))[options.index(choice)]
    defaults = dict(zip(getattr(config, "verification_window_list", ((1, 15), (16, 30))),
                        getattr(config, "tolerance_days_list", (3, 5))))
    columns = st.columns(len(windows))
    tolerances = []
    errors = []
    for column, window in zip(columns, windows):
        default = int(defaults.get(window, 3 if window[1] <= 15 else 5))
        tolerance = column.number_input(f"Tolerance: days {window[0]}–{window[1]}",
            min_value=0, max_value=window[1] - window[0] + 1,
            value=max(0, min(default, window[1] - window[0] + 1)), step=1,
            key=f"verification_tolerance_{row_key}_{window[0]}_{window[1]}")
        if tolerance is None:
            errors.append(f"Enter a tolerance for days {window[0]}–{window[1]}.")
            tolerance = default
        tolerances.append(int(tolerance))
    required_horizon = max(end for _, end in windows)
    st.number_input("Max Forecast Day (automatic)", min_value=required_horizon,
                    max_value=required_horizon, value=required_horizon, disabled=True,
                    key=f"verification_horizon_{row_key}_{required_horizon}")
    try:
        parse_verification_settings(repr(windows), repr(tuple(tolerances)), day_bins=config.day_bins)
    except ValueError as exc:
        errors.append(str(exc))
    return {"verification_window_list": repr(windows), "tolerance_days_list": repr(tuple(tolerances)),
            "max_forecast_day": required_horizon}, errors


def render_initialization_settings(row_key):
    modes = ["weekly", "daily", "monthly"]
    default_mode = getattr(config, "init_type", "weekly")
    default_mode = default_mode if default_mode in modes else "weekly"
    mode = st.selectbox("Initialization schedule", modes, index=modes.index(default_mode),
                        format_func=str.title, key=f"initialization_schedule_{row_key}")
    if mode == "weekly":
        defaults = [int(day) for day in getattr(config, "init_days", (0, 3)) if 0 <= int(day) <= 6] if default_mode == mode else [0, 3]
        weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        days = st.multiselect("Initialization weekdays", list(range(7)), default=defaults or [0, 3],
                             format_func=lambda day: weekdays[day], key=f"initialization_weekdays_{row_key}")
    elif mode == "monthly":
        defaults = [int(day) for day in getattr(config, "init_days", (1, 15)) if 1 <= int(day) <= 31] if default_mode == mode else [1, 15]
        days = st.multiselect("Initialization days of month", list(range(1, 32)), default=defaults or [1, 15],
                             key=f"initialization_monthdays_{row_key}")
        st.caption("Days that do not exist in a month are skipped.")
    else:
        days = list(range(7))
        st.caption("Daily scheduling uses every available date within the evaluation season.")
    reference_year = int(getattr(config, "date_filter_year", 2024))
    reference_year = st.number_input("Reference calendar year", min_value=1900, max_value=2100,
        value=max(1900, min(reference_year, 2100)), disabled=mode != "weekly",
        help="Weekly scheduling uses this year to align weekdays across historical forecast years. It does not change evaluation years.",
        key=f"initialization_reference_year_{row_key}")
    return {"init_type": mode, "initialization_days": sorted(days),
            "date_filter_year": int(reference_year) if reference_year is not None else None}


def _record_finished_benchmark(job):
    snapshot = job.snapshot()
    runs = snapshot["runs"]
    completed = snapshot["status"] == "completed"
    cancelled = snapshot["status"] == "cancelled"
    if completed:
        st.session_state["benchmark_settings_signature"] = job.signature
    st.session_state["benchmark_last_run"] = {
        "ok": completed, "cancelled": cancelled,
        "returncode": 130 if cancelled else max((int(r.get("returncode", 1)) for r in runs), default=1),
        "mode": runs[-1].get("mode", "") if runs else "",
        "runtime_dir": runs[-1].get("runtime_dir", str(ROMP_DEMO_OUT_DIR)) if runs else str(ROMP_DEMO_OUT_DIR),
        "command": "\n".join(r.get("command", "") for r in runs), "runs": runs,
    }
    st.session_state.pop("benchmark_active_job", None)
    st.cache_data.clear()


@st.fragment(run_every=1)
def render_benchmark_job():
    job = st.session_state.get("benchmark_active_job")
    if job is None:
        return
    snapshot = job.snapshot()
    if snapshot["done"]:
        _record_finished_benchmark(job)
        st.rerun()
        return
    st.info("Stopping benchmark…" if snapshot["stopping"] else "Benchmark running. You can stop it below.")
    if st.button("Stop Benchmark", key="btn_stop_benchmark", disabled=snapshot["stopping"]):
        job.cancel()
        st.info("Stop requested; terminating benchmark workers.")
    if snapshot["logs"]:
        for line in snapshot["logs"]:
            if line.startswith("[stop-warning]"):
                st.warning(line.removeprefix("[stop-warning]").strip())
        st.code("\n".join(snapshot["logs"][-20:]), language="text")


def page() -> None:
    _inject_theme_css()

    presets = get_benchmark_presets()
    title_column, guide_column = st.columns([3, 1])
    with title_column:
        st.markdown('<div class="page-title">Forecast Benchmarking</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="page-subtitle">'
            "Evaluate AI weather model skill against observational references "
            "across the Ethiopian highlands."
            "</div>",
            unsafe_allow_html=True,
        )

    with guide_column:
        with st.popover("Parameter guide", width="stretch"):
            st.caption("Expand a parameter for its meaning and usage.")
            for parameter, instruction in PARAMETER_HELP.items():
                with st.expander(parameter):
                    st.markdown(instruction)

    last_run = st.session_state.get("benchmark_last_run")
    status_text = ("Stop requested" if cancellation_warning(last_run) else "Cancelled") if last_run and last_run.get("cancelled") else (
        "Completed" if last_run and last_run.get("ok") else "Failed" if last_run else "Idle")
    render_kpi_row(
        len(CONFIG_BENCHMARK_MODEL_CATALOG or {}),
        len(presets),
        status_text,
        str(ROMP_DEMO_OUT_DIR.name),
    )

    # Benchmark Inputs
    section_header("Benchmark Inputs", "violet")
    bi1, bi2, bi3 = st.columns(3)
    ground_truth = bi1.selectbox("Ground truth", ["CHIRPS", "ENACTS"])
    rp = get_region_presets()
    preset_names = [p["name"] for p in rp] if rp else []
    coverage_options = [name for _, name in BENCHMARK_COVERAGE_COUNTRIES]
    coverage_options += [n for n in preset_names if n not in coverage_options]

    benchmark_coverage = bi2.selectbox(
        "Benchmark coverage",
        coverage_options,
        index=coverage_options.index(DEFAULT_BENCHMARK_COVERAGE),
        format_func=_coverage_label,
    )

    coverage_ready = benchmark_coverage in CONFIGURED_COVERAGE or benchmark_coverage in preset_names
    if not coverage_ready:
        bi2.caption(f"⚠️ No data configured for {benchmark_coverage} yet. Run Benchmark is disabled.")
    forecast_window = bi3.selectbox("Forecast window", ["1-15 days", "16-30 days"])

    # Models
    section_header("Models", "teal")
    model_registry = detect_benchmark_models()
    active_rows: List[Dict[str, Any]] = []

    if not model_registry:
        info_box("No forecast model data was detected.", "amber")
    else:
        mrr = {r["_row_key"]: r for r in model_registry}
        category_options = ["All", "Deterministic", "Probabilistic"]
        selected_category_filter = st.radio("Filter by category", category_options, horizontal=True, key="model_category_filter")
        visible_registry = model_registry if selected_category_filter == "All" else [r for r in model_registry if r.get("Category") == selected_category_filter]

        if not visible_registry:
            info_box(f"No {selected_category_filter} models detected.", "amber")
        else:
            table = st.dataframe(
                pd.DataFrame(visible_registry)[["Model", "Category", "Type", "Coverage", "Date Range"]],
                hide_index=True,
                selection_mode="multi-row",
                on_select="rerun",
                key="model_table_selection",
                width='stretch',
            )
            try:
                sel_idx = table.selection.rows
            except Exception:
                sel_idx = st.session_state.get("model_table_selection", {}).get("selection", {}).get("rows", [])
            sel_keys = [visible_registry[int(i)]["_row_key"] for i in sel_idx if 0 <= int(i) < len(visible_registry)]
            active_rows = [mrr[k] for k in sel_keys if k in mrr]
            st.session_state.selected_model_rows = sel_keys
            
            det_count = sum(1 for r in active_rows if r.get("Category") == "Deterministic")
            prob_count = sum(1 for r in active_rows if r.get("Category") == "Probabilistic")
            if active_rows:
                st.markdown(
                    f"""
                    <div style="display:flex; flex-wrap:wrap; gap:0.5rem; margin:0.5rem 0;">
                        <span class="metric-pill">Deterministic: <strong>{det_count}</strong></span>
                        <span class="metric-pill">Probabilistic: <strong>{prob_count}</strong></span>
                        <span class="metric-pill">Total: <strong>{len(active_rows)}</strong></span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    model_run_windows: Dict[str, Dict[str, Any]] = {}
    verification_problems = []
    observation_years = available_observation_years(ground_truth)

    def render_model_run_windows():
        section_header("Model run windows", "amber")
        if not active_rows:
            st.caption("Select models above to set evaluation dates and baseline years.")
        if active_rows:
            cs, ce = compute_common_model_window(active_rows, observation_years)
            if cs and ce:
                st.info(f"Shared model/observation window: {cs:%Y-%m-%d} to {ce:%Y-%m-%d}.")
            for row in active_rows:
                rk = row["_row_key"]
                avs, ave = row["_start_date"], row["_end_date"]
                if observation_years:
                    shared_start = max(avs, date(min(observation_years), 1, 1))
                    shared_end = min(ave, date(max(observation_years), 12, 31))
                    if shared_start < shared_end:
                        avs, ave = shared_start, shared_end
                    else:
                        verification_problems.append(f"{row['Model']}: no shared model/observation evaluation period.")
                des, dee = (max(cs, avs), min(ce, ave)) if cs and ce else (avs, ave)
                if des > dee:
                    des, dee = avs, ave
                with st.expander(f"{row['Model']} - available {row['Date Range']}"):
                    st.caption("The month/day range defines the season repeated in each evaluation year. Dates use shared model/observation coverage; baseline years use observation coverage.")
                    c1, c2 = st.columns(2)
                    if st.session_state.get(f"es_{rk}") is not None and not avs <= st.session_state[f"es_{rk}"] <= ave:
                        st.session_state[f"es_{rk}"] = min(max(st.session_state[f"es_{rk}"], avs), ave)
                    eval_start = c1.date_input("Evaluation Start", value=des, min_value=avs, max_value=ave, key=f"es_{rk}")
                    if eval_start is None:
                        verification_problems.append(f"{row['Model']}: select Evaluation Start.")
                        eval_start = des
                    if st.session_state.get(f"ee_{rk}") is not None and not eval_start <= st.session_state[f"ee_{rk}"] <= ave:
                        st.session_state[f"ee_{rk}"] = min(max(st.session_state[f"ee_{rk}"], eval_start), ave)
                    eval_end = c2.date_input("Evaluation End", value=max(dee, eval_start), min_value=eval_start, max_value=ave, key=f"ee_{rk}")
                    if eval_end is None:
                        verification_problems.append(f"{row['Model']}: select Evaluation End.")
                        eval_end = max(dee, eval_start)
                    first_year = min(observation_years) if observation_years else avs.year
                    last_year = max(observation_years) if observation_years else ave.year
                    default_start = max(first_year, min(int(getattr(config, "start_year_clim", des.year)), last_year))
                    if st.session_state.get(f"bsy_{rk}") is not None and not first_year <= st.session_state[f"bsy_{rk}"] <= last_year:
                        st.session_state[f"bsy_{rk}"] = min(max(int(st.session_state[f"bsy_{rk}"]), first_year), last_year)
                    c3, c4 = st.columns(2)
                    bsy = c3.number_input("Baseline Start Year", min_value=first_year, max_value=last_year, value=default_start, key=f"bsy_{rk}")
                    if bsy is None:
                        verification_problems.append(f"{row['Model']}: enter Baseline Start Year.")
                        bsy = default_start
                    if st.session_state.get(f"bey_{rk}") is not None and not int(bsy) <= st.session_state[f"bey_{rk}"] <= last_year:
                        st.session_state[f"bey_{rk}"] = min(max(int(st.session_state[f"bey_{rk}"]), int(bsy)), last_year)
                    default_end = max(int(bsy), min(int(getattr(config, "end_year_clim", dee.year)), last_year))
                    bey = c4.number_input("Baseline End Year", min_value=int(bsy), max_value=last_year, value=default_end, key=f"bey_{rk}")
                    if bey is None:
                        verification_problems.append(f"{row['Model']}: enter Baseline End Year.")
                        bey = default_end
                
                    model_run_windows[rk] = {
                        "eval_start": eval_start,
                        "eval_end": eval_end,
                        "baseline_start_year": int(bsy),
                        "baseline_end_year": int(bey),
                    }

    def render_advanced_settings():
        if not active_rows:
            st.info("Select models above to configure their schedules and verification windows.")
            return
        st.caption("Per-model controls apply to the next benchmark run. The summary updates as you edit.")
        for index, row in enumerate(active_rows):
            rk = row["_row_key"]
            with st.expander(row["Model"], expanded=index == 0):
                initialization = render_initialization_settings(rk)
                verification, errors = render_verification_window_settings(rk, forecast_window)
                model_run_windows[rk].update({**initialization, **verification})
                verification_problems.extend(f"{row['Model']}: {error}" for error in errors)
                windows = parse_verification_settings(verification["verification_window_list"])[0]
                window_label = ", ".join(f"{start}–{end}" for start, end in windows)
                st.caption(f"Next run: {initialization['init_type'].title()} · Days {window_label} · Horizon {verification['max_forecast_day']} days")

    shared_settings = render_shared_settings(ground_truth, advanced_renderer=render_advanced_settings, run_windows_renderer=render_model_run_windows)
    dry_extension = shared_settings["dry_spell_search_extension"]
    if 0 < dry_extension < shared_settings["dry_spell_limit"]:
        verification_problems.append("Dry spell search extension must be 0 (disabled) or at least the dry spell limit.")

    if PHASE_2_AVAILABLE or PHASE_3_AVAILABLE:
        section_header("Data Readiness & Fair Comparison", "green")
        if PHASE_2_AVAILABLE:
            with st.expander("Data Readiness Report", expanded=False):
                try:
                    m_inv = build_model_inventory()
                    o_inv = build_obs_inventory()
                    summary = readiness_summary(m_inv + o_inv)
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Ready", summary.get("Ready", 0))
                    c2.metric("Warning", summary.get("Warning", 0))
                    c3.metric("No data", summary.get("No data", 0))
                    c4.metric("Error", summary.get("Error", 0))
                    st.dataframe(inventory_to_dataframe(m_inv + o_inv), hide_index=True, width='stretch')
                except Exception as e:
                    st.warning(f"Could not load inventory: {e}")

    # Run Benchmark Action
    observation_problems = []
    for row in active_rows:
        window = model_run_windows[row["_row_key"]]
        errors = temporal_settings_errors({**window, "probabilistic": row.get("_probabilistic", False)}, observation_years)
        observation_problems.extend(f"{row['Model']}: {error}" for error in errors)
        if not errors:
            coverage = inspect_model_temporal_coverage(row["_exec_key"], window["eval_start"].year, window["eval_end"].year)
            if coverage["failures"]:
                observation_problems.append(f"{row['Model']}: cannot read forecast metadata: {'; '.join(coverage['failures'])}")
            elif not coverage["dates"]:
                observation_problems.append(f"{row['Model']}: no forecast initialization dates are available for these years. Adjust the evaluation period or model data configuration.")
            elif not any(initialization_matches_schedule(parts, window) for parts in coverage["dates"]):
                observation_problems.append(f"{row['Model']}: no forecast initialization dates match this schedule and evaluation season. Adjust the selected days or dates.")
            required_days = window["max_forecast_day"] + max(0, int(shared_settings["wet_spell_length"]) - 1)
            too_short = [year for year, end in coverage["horizons"].items() if end < required_days]
            if too_short:
                observation_problems.append(f"{row['Model']}: forecast files for {too_short} need at least {required_days} days to verify the selected window and wet spell.")
    for problem in observation_problems + verification_problems:
        st.error(problem)
    st.markdown('<div class="run-section">', unsafe_allow_html=True)
    c1, _ = st.columns([1, 2])
    run_clicked = c1.button("Run Benchmark", key="btn_run", type="primary", width='stretch', disabled=not coverage_ready or bool(observation_problems or verification_problems) or st.session_state.get("benchmark_active_job") is not None)
    st.markdown("</div>", unsafe_allow_html=True)

    import hashlib
    settings_signature = hashlib.sha256(json.dumps({
        "shared": shared_settings, "windows": model_run_windows,
        "models": [r["_exec_key"] for r in active_rows],
        "ground_truth": ground_truth, "coverage": benchmark_coverage,
        "forecast_window": forecast_window,
    }, sort_keys=True, default=str).encode()).hexdigest()
    previous_signature = st.session_state.get("benchmark_settings_signature")
    if previous_signature != settings_signature:
        # Settings edits do not change files on disk. Keep cached results until
        # a benchmark runs; clearing here reloads every results tab on each edit.
        if not run_clicked and st.session_state.get("benchmark_last_run", {}).get("ok"):
            st.warning("Benchmark settings changed. Run Benchmark to compute results for these settings; existing outputs belong to an earlier run.")

    if run_clicked and not active_rows:
        st.warning("Select at least one model in the Models table, then click Run Benchmark.")

    if run_clicked and active_rows and st.session_state.get("benchmark_active_job") is None:
        plans = []
        for row in active_rows:
            rc = dict(model_run_windows.get(row["_row_key"], {}))
            rc.update(shared_settings)
            rc.update({"ground_truth": ground_truth, "benchmark_coverage": benchmark_coverage,
                       "forecast_window": forecast_window})
            plans.append({"mode": "prob" if row.get("_probabilistic") else "det",
                          "model_key": row["_exec_key"], "label": row.get("Model", row["_exec_key"]), "config": rc})
        st.session_state["benchmark_active_job"] = BenchmarkJob(run_momp_benchmark, plans, settings_signature).start()
        st.rerun()

    if st.session_state.get("benchmark_active_job") is not None:
        render_benchmark_job()

    last_run = st.session_state.get("benchmark_last_run")
    if last_run:
        if last_run.get("cancelled"):
            cleanup_warning = cancellation_warning(last_run)
            if cleanup_warning:
                st.warning("Stop requested. Worker shutdown could not be fully verified.")
                st.caption(cleanup_warning)
            else:
                st.info("Benchmark cancelled by you. Remaining queued models were skipped.")
            st.caption("Results from the interrupted run may be incomplete. Use Run Benchmark to generate a complete set of results.")
        elif last_run.get("ok"):
            st.success("Latest benchmark run completed successfully.")
        else:
            st.error("Latest benchmark run failed. See the execution log below for the cause.")
        with st.expander("Latest benchmark execution log", expanded=not last_run.get("ok") and not last_run.get("cancelled")):
            results = last_run.get("runs", [])
            for index, result in enumerate(results):
                cancelled = last_run.get("cancelled") and index == len(results) - 1
                st.write(f"{result.get('model', 'Benchmark')} — {execution_status(result, cancelled=cancelled)}")
                if result.get("command"):
                    st.code(result["command"], language="bash")
                log = "\n".join(str(result.get(key) or "") for key in ("stdout", "stderr")).strip()
                st.code(log or "The process produced no output.", language="text")

    # Results Section
    section_header("Results", "sky")
    if st.button("Refresh saved results", key="benchmark_refresh_results",
                 help="Reload saved metrics, maps and figures after a benchmark completes."):
        st.cache_data.clear()
        st.rerun()
    tab_labels = ["Metric Map", "All Metrics", "Benchmark Outputs"]
    if OTS_AVAILABLE:
        tab_labels.append("Onset Analysis")
    if PHASE_4_AVAILABLE:
        tab_labels.append("Skill vs Baseline")
    if PHASE_5_AVAILABLE:
        tab_labels.append("Uncertainty")
    if PDF_AVAILABLE or PHASE_6_AVAILABLE:
        tab_labels.append("Benchmark Report & Export")

    res_tabs = st.tabs(tab_labels, key="benchmark_result_tabs", on_change="rerun")
    map_outputs = discover_map_outputs() if any(
        tab.open and label in {"Metric Map", "All Metrics", "Benchmark Report & Export"}
        for tab, label in zip(res_tabs, tab_labels)) else []

    # Tab 1: Metric Map
    if res_tabs[0].open:
        with res_tabs[0]:
            if not map_outputs:
                info_box("No map outputs found.", "amber")
            else:
                c1, c2, c3, c4 = st.columns(4)
                ml = c1.selectbox("Metric", list(MAP_METRICS.keys()), key="m_ml")
                all_w: set = set()
                for o in map_outputs:
                    all_w.update(o["windows"])
                wl = c2.selectbox("Lead Time", sorted(list(all_w)), key="m_wl")
                m_opts = sorted({o["model_label"] for o in map_outputs if wl in o["windows"]})
                model_label = c3.selectbox("Model", m_opts, key="m_mod")
                sf_avail = _map_load_admin_shapefile() is not None
                al = c4.selectbox("Subregion level", ["All", "Admin1", "Admin2", "Admin3"] if sf_avail else ["All"], disabled=not sf_avail, key="m_al")
                an = "All"
                if al != "All":
                    ao = map_admin_options(al)
                    if ao:
                        an = st.selectbox("Subregion", ["All"] + ao, key="m_an")

                sel_out = next((o for o in map_outputs if o["model_label"] == model_label and wl in o["windows"]), None)
                if sel_out:
                    ual = "" if al == "All" else al
                    uan = "" if an == "All" else an
                    base_da = load_map_da_cached(sel_out["path"], ml, wl, ual, uan, file_mtime=sel_out.get("mtime", 0.0))
                    admin_sub = _get_admin_subset(ual, uan)

                    if base_da is not None:
                        mm = MAP_METRICS[ml]
                        sub_tabs = st.tabs(["Values", "Skill", "Difference", "Swipe Compare"], key="benchmark_map_tabs", on_change="rerun")


                        # Sub-tab 1: Values
                        if sub_tabs[0].open:
                            with sub_tabs[0]:
                                fig = plot_map(
                                    base_da,
                                    f"{ml} - {model_label} - {wl}",
                                    cmap=mm["cmap"],
                                    unit=mm["unit"],
                                    admin_gdf=admin_sub
                                )
                                if fig:
                                    st.pyplot(fig)
                                    plt.close(fig)
                                df = compute_model_comparison_cached(ml, wl, ual, uan)
                                if not df.empty:
                                    df["Selected"] = np.where(df["Model"] == model_label, "Yes", "")
                                    st.dataframe(df, width='stretch', hide_index=True)

                        if sub_tabs[1].open:
                            with sub_tabs[1]:
                                thr = mm["threshold"]
                                skill_da = (base_da <= thr).astype(float).where(base_da.notnull())
                                cbar_kw = {
                                    "label": f"Skill Status ({ml})",
                                    "ticks": [0, 1],
                                    "format": matplotlib.ticker.FixedFormatter(
                                        [f"Fail (>{thr}{mm['unit']})", f"Pass (\u2264{thr}{mm['unit']})"]
                                    ),
                                }
                                fig = plot_map(
                                    skill_da,
                                    f"Skill - {ml} (Threshold \u2264 {thr}{mm['unit']})",
                                    cmap="RdYlGn",
                                    vmin=0,
                                    vmax=1,
                                    cbar_kwargs=cbar_kw,
                                    admin_gdf=admin_sub,
                                    boundary_path=climatology_boundary_path(config) if "climatology" in model_label.casefold() else None,
                                )
                                if fig:
                                    st.pyplot(fig)
                                    plt.close(fig)
                                st.caption(
                                    f"**Skill Classification**: Green = Skillful ({ml} \u2264 {thr}{mm['unit']}), "
                                    f"Red = Unskillful ({ml} > {thr}{mm['unit']})."
                                )

                        if sub_tabs[2].open:
                            with sub_tabs[2]:
                                dc1, dc2 = st.columns(2)
                                default_cmp_idx = 0
                                for i, m in enumerate(m_opts):
                                    if m != model_label:
                                        default_cmp_idx = i
                                        break
                                cm = dc1.selectbox("Compare against model", m_opts, index=default_cmp_idx, key="d_cm")
                                lead_opts_diff = [w for w in ["Days 1–15", "Days 16–30"] if w in all_w]
                                default_lead_idx = 0
                                for i, w in enumerate(lead_opts_diff):
                                    if w != wl:
                                        default_lead_idx = i
                                        break
                                cw = dc2.selectbox("Compare against lead time", lead_opts_diff, index=default_lead_idx, key="d_cw")

                                if cm == model_label and cw == wl:
                                    info_box("Cannot compare a map with itself.", "amber")
                                else:
                                    cout = next((o for o in map_outputs if o["model_label"] == cm and cw in o["windows"]), None)
                                    if cout:
                                        cda = load_map_da_cached(cout["path"], ml, cw, ual, uan, file_mtime=cout.get("mtime", 0.0))
                                        if cda is not None:
                                            dda = subtract_maps(base_da, cda)
                                            if dda is not None:
                                                fig = plot_map(
                                                    dda,
                                                    f"Difference - {model_label} minus {cm}",
                                                    diverging=True,
                                                    admin_gdf=admin_sub
                                                )
                                                if fig:
                                                    st.pyplot(fig)
                                                    plt.close(fig)

                        if sub_tabs[3].open:
                            with sub_tabs[3]:
                                sc1, sc2, _ = st.columns(3)
                                cb = sc1.radio("Compare by", ["Model", "Lead Time"], horizontal=True, key="s_cb")
                                rm, rw = model_label, wl
                                if cb == "Model":
                                    default_swipe_idx = 0
                                    for i, m in enumerate(m_opts):
                                        if m != model_label:
                                            default_swipe_idx = i
                                            break
                                    rm = sc2.selectbox("Right model", m_opts, index=default_swipe_idx, key="s_rm")
                                else:
                                    swipe_lead_opts = [w for w in ["Days 1–15", "Days 16–30"] if w in all_w]
                                    default_swipe_lead_idx = 0
                                    for i, w in enumerate(swipe_lead_opts):
                                        if w != wl:
                                            default_swipe_lead_idx = i
                                            break
                                    rw = sc2.selectbox("Right lead time", swipe_lead_opts, index=default_swipe_lead_idx, key="s_rw")

                                is_same = (cb == "Model" and rm == model_label) or (cb == "Lead Time" and rw == wl)
                                if is_same:
                                    info_box("Cannot compare a map with itself.", "amber")
                                else:
                                    rout = next((o for o in map_outputs if o["model_label"] == rm and rw in o["windows"]), None)
                                    if rout:
                                        rda = load_map_da_cached(rout["path"], ml, rw, ual, uan, file_mtime=rout.get("mtime", 0.0))
                                        if rda is not None:
                                            sp = st.slider("Drag to swipe", 0, 100, 50, key="s_sp")
                                            vmin, vmax = _shared_vmin_vmax(base_da.values, rda.values)

                                            lf = plot_map(
                                                base_da,
                                                f"Left: {model_label} ({wl})",
                                                cmap=mm["cmap"],
                                                unit=mm["unit"],
                                                vmin=vmin,
                                                vmax=vmax,
                                                admin_gdf=admin_sub
                                            )
                                            rf = plot_map(
                                                rda,
                                                f"Right: {rm} ({rw})",
                                                cmap=mm["cmap"],
                                                unit=mm["unit"],
                                                vmin=vmin,
                                                vmax=vmax,
                                                admin_gdf=admin_sub
                                            )

                                            if lf and rf:
                                                lb = fig_to_bytes(lf, False)
                                                rb = fig_to_bytes(rf, False)
                                                comp = create_swipe_comparison_image(lb, rb, sp)
                                                if comp:
                                                    lc1, lc2 = st.columns(2)
                                                    lc1.markdown(f"**Left Map (0–{sp}%):** {model_label} - {wl}")
                                                    lc2.markdown(f"**Right Map ({sp}–100%):** {rm} - {rw}")
                                                    st.image(comp, width='stretch')
                                    else:
                                        info_box(f"Output for {rm} ({rw}) not available.", "amber")

    # Tab 2: All Metrics
    if res_tabs[1].open:
        with res_tabs[1]:
            from ui.utils.metric_filters import (
                lead_window_sort_key, load_probabilistic_metric_table,
                metric_comparison_groups, metric_model_label, metric_model_options,
            )

            st.markdown("### ALL METRICS")
            tbl: Dict[tuple, float] = {}
            for output in map_outputs:
                values = load_metric_values_by_window_cached(output["path"])
                for window, metrics in values.items():
                    for metric, value in metrics.items():
                        tbl[(metric, window, metric_model_label(output["model_label"]))] = value
            probability_table, summary_errors = load_probabilistic_metric_table(ROMP_DEMO_OUT_DIR)
            tbl.update(probability_table)
            models, categories = metric_model_options(
                CONFIG_BENCHMARK_MODEL_CATALOG, {key[2] for key in tbl} |
                {output["model_label"] for output in map_outputs} |
                {getattr(config, "ref_model", "climatology")})
            lead_options = sorted({key[1] for key in tbl} | {"Days 1–15", "Days 16–30"}, key=lead_window_sort_key)
            category_options = ["All"] + [category for category in ("Deterministic", "Probabilistic", "Climatology", "Other")
                                          if category in categories.values()]
            category = st.radio("Metric model category", category_options, horizontal=True, key="metrics_model_category")
            eligible = [model for model in models if category == "All" or categories[model] == category]
            all_models = st.checkbox("All models", value=True, key=f"metrics_all_models_{category}")
            chosen_models = st.multiselect("Models to compare", eligible, default=eligible, disabled=all_models,
                                          key=f"metrics_models_{category}")
            om = eligible if all_models else chosen_models
            ow = st.multiselect("Lead times", lead_options, default=lead_options,
                                key=f"metrics_leads_{tuple(lead_options)}")
            ow = sorted(ow, key=lead_window_sort_key)
            st.caption("N/A means a metric is not computed for that model, no result was saved for that lead time, or the score is mathematically undefined. RPSS requires supported forecast/reference cases and a finite, nonzero reference Fair RPS. BSS/RPSS percentages and AUC are included from probabilistic summaries when available.")
            st.caption("Onset error metrics and probabilistic scores are shown separately because they evaluate different forecast types. MAE includes timing errors for false alarms and can exceed the length of the lead-time window.")
            st.caption("Scroll the table horizontally to view all models and lead times on smaller screens.")
            if summary_errors:
                st.warning("Some probabilistic summaries could not be read: " + "; ".join(summary_errors))
            if om and ow:
                for heading, group_models in metric_comparison_groups(om, categories):
                    st.markdown(f"#### {heading}")
                    selected = {key: value for key, value in tbl.items() if key[2] in group_models and key[1] in ow}
                    present_metrics = {key[0] for key in selected}
                    if not present_metrics:
                        present_metrics = ({"Fair Brier Score", "Fair Brier Skill Score (%)", "Fair RPS", "Fair RPS Skill Score (%)", "AUC", "Reference AUC"}
                                           if categories[group_models[0]] == "Probabilistic"
                                           else {"Mean Absolute Error days", "Miss Rate fraction", "False Alarm Rate fraction"})
                    # Keep configured models visible before they have generated outputs.
                    for metric in present_metrics:
                        for window in ow:
                            for model in group_models:
                                selected.setdefault((metric, window, model), None)
                    st.markdown(build_metrics_table_html(selected, group_models, ow).replace("<td>-</td>", "<td>N/A</td>"), unsafe_allow_html=True)
                legacy_outputs = [output for output in map_outputs if
                                  metric_model_label(output["model_label"]) in om and
                                  set(output["windows"]) & set(ow) and
                                  output.get("contingency_definition") != "timing_matched_v2"]
                if legacy_outputs:
                    st.warning("Saved onset error results predate the timing-miss correction. Rerun these models for the selected lead times to regenerate miss rates; historical grids do not retain the event counts needed to repair them.")
                missing = [model for model in om if not any(
                    key[2] == model and key[1] in ow and value is not None and np.isfinite(value)
                    for key, value in tbl.items())]
                if missing:
                    st.info("No saved results for the selected lead times: " + ", ".join(missing))
            else:
                st.info("Select at least one model and one lead time to compare results.")

            st.divider()
            section_header("Detailed Metric Distributions", "violet")
            st.caption(
                "Distribution statistics computed directly from gridded model outputs. "
                "Green = better (lower than baseline), Red = worse (higher than baseline). Bold = best value."
            )
            spatial_models = {metric_model_label(output["model_label"]) for output in map_outputs}
            stats_df = compute_metric_distribution_stats() if om and ow and set(om) & spatial_models else pd.DataFrame()
            if not stats_df.empty:
                stats_df = stats_df.copy()
                stats_df["Model"] = stats_df["Model"].map(metric_model_label)
                stats_df = stats_df[stats_df["Model"].isin(om) & stats_df["Lead Window"].isin(ow)]
            if stats_df.empty:
                info_box("No gridded distributions are available for these selections. Scalar probabilistic results, when available, are shown above.", "amber")
            else:
                # Normalize string helper to prevent Unicode en-dash / hyphen key lookup misses
                def _norm(s):
                    return str(s).replace("–", "-").strip().lower()

                stat_cols = ["Mean", "Min", "Median", "P75", "P90", "Max"]

                # Pre-calculate Climatology baseline & Best values
                best_vals = {}
                baseline_vals = {}
                comparison_scales = {}

                for (window, metric), g in stats_df.groupby(["Lead Window", "Metric"]):
                    win_k = _norm(window)
                    met_k = _norm(metric)
                    higher_better = is_metric_higher_better(metric)

                    # Check for Climatology / Baseline model
                    clim_mask = g["Model"].apply(lambda m: any(k in _norm(m) for k in ["clim", "trad", "base", "ref"]))
                    clim_row = g[clim_mask]

                    for stat in stat_cols:
                        stat_k = _norm(stat)
                        vals = pd.to_numeric(g[stat], errors="coerce").dropna()
                        if not vals.empty:
                            # Best value: minimum for error metrics, maximum for skill metrics
                            best_vals[(win_k, met_k, stat_k)] = float(vals.max()) if higher_better else float(vals.min())

                        if not clim_row.empty:
                            clim_v = pd.to_numeric(clim_row[stat], errors="coerce").dropna()
                            if not clim_v.empty:
                                baseline_value = float(clim_v.iloc[0])
                                baseline_vals[(win_k, met_k, stat_k)] = baseline_value

                                # Calculate scale for intensity
                                model_vals = pd.to_numeric(g.loc[~clim_mask, stat], errors="coerce").dropna()
                                differences = (model_vals - baseline_value).abs()
                                if not differences.empty:
                                    comparison_scales[(win_k, met_k, stat_k)] = float(differences.max())

                # Sort ordering
                metric_order = {"false alarm rate": 0, "miss rate": 1, "mean absolute error": 2}
                model_order = {
                    "aifs single": 0, "fuxi": 1, "graphcast": 2,
                    "aifs ensemble": 3, "gencast": 4,
                    "traditional climatology": 5, "climatology": 5
                }

                df_sorted = stats_df.copy()
                df_sorted["metric_sort"] = df_sorted["Metric"].apply(lambda m: metric_order.get(_norm(m), 999))
                df_sorted["model_sort"] = df_sorted["Model"].apply(lambda m: model_order.get(_norm(m), 999))
                df_sorted["window_sort"] = df_sorted["Lead Window"].apply(lead_window_sort_key)
                df_sorted = df_sorted.sort_values(["window_sort", "model_sort", "metric_sort"])

                html_blocks = []
                for (model_label, window_label), group_df in df_sorted.groupby(["Model", "Lead Window"], sort=False):
                    win_k = _norm(window_label)
                    is_clim = any(k in _norm(model_label) for k in ["clim", "trad", "base", "ref"])
                    display_model = "CLIMATOLOGY" if is_clim else str(model_label).replace(" ", "_").upper()
                    display_window = str(window_label).upper()
                    tolerance_days = 5 if "16" in win_k else 3

                    rows_html = []
                    for _, row in group_df.iterrows():
                        metric = str(row["Metric"])
                        met_k = _norm(metric)
                        higher_better = is_metric_higher_better(metric)
                        is_fraction = bool(row.get("fraction", False)) or any(k in met_k for k in ["rate", "far", "mr"])

                        cells_html = [
                            f'<td style="border:1px solid #cbd5e1; padding:8px 12px; text-align:left; '
                            f'font-weight:600; background-color:#ffffff; color:#0f172a;">{metric}</td>'
                        ]

                        for stat in stat_cols:
                            stat_k = _norm(stat)
                            val = row.get(stat, np.nan)

                            if pd.isna(val) or not np.isfinite(val):
                                cells_html.append(
                                    '<td style="border:1px solid #cbd5e1; padding:8px 12px; text-align:right; '
                                    'background-color:#ffffff; color:#94a3b8;">-</td>'
                                )
                                continue

                            val_float = float(val)
                            formatted = f"{val_float:.1f}%" if is_fraction else f"{val_float:.1f} d"

                            # Apply bolding if best value across models
                            best_v = best_vals.get((win_k, met_k, stat_k))
                            if best_v is not None and abs(val_float - best_v) <= 1e-5:
                                formatted = f"<b>{formatted}</b>"

                            # Color coding vs baseline using _distribution_cell_style
                            base_v = baseline_vals.get((win_k, met_k, stat_k))
                            scale = comparison_scales.get((win_k, met_k, stat_k))
                            comparison_style = _distribution_cell_style(
                                val_float,
                                base_v,
                                scale,
                                higher_better,
                                is_baseline=is_clim,
                            )

                            cell_style = (
                                f"border:1px solid #cbd5e1; padding:8px 12px; text-align:right; {comparison_style}"
                            )
                            cells_html.append(f'<td style="{cell_style}">{formatted}</td>')

                        unit_str = "%" if is_fraction else "days"
                        cells_html.append(
                            f'<td style="border:1px solid #cbd5e1; padding:8px 12px; text-align:center; '
                            f'color:#64748b; background-color:#ffffff;">{unit_str}</td>'
                        )
                        rows_html.append(f"<tr>{''.join(cells_html)}</tr>")

                    card_html = (
                        f'<div style="margin-bottom:1.75rem; background:#ffffff; border:1px solid #e2e8f0; '
                        f'border-radius:8px; padding:16px; box-shadow:0 1px 3px rgba(0,0,0,0.05);">'
                        f'<div style="font-size:1rem; font-weight:700; color:#0f172a; text-transform:uppercase;">'
                        f'{display_model} - {display_window}</div>'
                        f'<div style="font-size:0.8rem; color:#64748b; margin-bottom:12px;">'
                        f'+/-{tolerance_days} day tolerance</div>'
                        f'<table style="width:100%; border-collapse:collapse; font-size:0.88rem;">'
                        f'<thead><tr style="background-color:#f1f5f9; color:#0f172a; text-transform:uppercase; font-size:0.8rem;">'
                        f'<th style="border:1px solid #cbd5e1; padding:8px 12px; text-align:left;">METRIC</th>'
                        f'<th style="border:1px solid #cbd5e1; padding:8px 12px; text-align:right;">MEAN</th>'
                        f'<th style="border:1px solid #cbd5e1; padding:8px 12px; text-align:right;">MIN</th>'
                        f'<th style="border:1px solid #cbd5e1; padding:8px 12px; text-align:right;">MEDIAN</th>'
                        f'<th style="border:1px solid #cbd5e1; padding:8px 12px; text-align:right;">P75</th>'
                        f'<th style="border:1px solid #cbd5e1; padding:8px 12px; text-align:right;">P90</th>'
                        f'<th style="border:1px solid #cbd5e1; padding:8px 12px; text-align:right;">MAX</th>'
                        f'<th style="border:1px solid #cbd5e1; padding:8px 12px; text-align:center;">UNIT</th>'
                        f'</tr></thead><tbody>'
                        f'{"".join(rows_html)}'
                        f'</tbody></table></div>'
                    )
                    html_blocks.append(card_html)

                st.markdown("".join(html_blocks), unsafe_allow_html=True)
    # Tab 3: Benchmark Outputs
    if res_tabs[2].open:
        with res_tabs[2]:
            st.markdown("MOMP benchmark output figures and datasets.")
            momp_figures = collect_reference_figures()

            if not momp_figures:
                info_box("No MOMP output figures found. Run a benchmark first.", "amber")
            else:
                # Gather non-empty categories
                avail_cats = [cat for cat in REFERENCE_CATEGORY_ORDER if momp_figures.get(cat)]
                if not avail_cats:
                    avail_cats = [cat for cat, paths in momp_figures.items() if paths]

                # Dropdown & Layout Controls Row
                fc1, fc2, fc3, fc4 = st.columns([1.2, 1.2, 1.5, 1.0])

                selected_cat = fc1.selectbox(
                    "Category",
                    ["All Categories"] + avail_cats,
                    key="fig_sel_cat"
                )

                # Filter paths by category
                if selected_cat == "All Categories":
                    cat_paths = [p for paths in momp_figures.values() for p in paths]
                else:
                    cat_paths = momp_figures.get(selected_cat, [])

                # Group paths by figure type for second dropdown
                type_groups: Dict[str, List[str]] = {}
                for path_str in cat_paths:
                    ftype = classify_figure_type(Path(path_str).name)
                    type_groups.setdefault(ftype, []).append(path_str)

                avail_types = sorted(type_groups.keys())
                selected_type = fc2.selectbox(
                    "Figure Type",
                    ["All Figure Types"] + avail_types,
                    key="fig_sel_type"
                )

                # Filter paths by figure type
                if selected_type == "All Figure Types":
                    filtered_paths = cat_paths
                else:
                    filtered_paths = type_groups.get(selected_type, [])

                # Map formatted human labels back to file paths
                label_to_path: Dict[str, str] = {}
                fig_options = []
                for p in filtered_paths:
                    lbl = format_figure_label(p)
                    if lbl in label_to_path:
                        lbl = f"{lbl} ({Path(p).name})"
                    label_to_path[lbl] = p
                    fig_options.append(lbl)
                fig_options.append("Show All Figures in Selection")

                selected_fig_label = fc3.selectbox(
                    "Select Figure",
                    fig_options,
                    key="fig_sel_item"
                )

                view_mode = fc4.radio(
                    "Layout View",
                    ["Grid", "Full Width"],
                    horizontal=True,
                    key="fig_view_mode",
                )

                st.divider()

                # Render single selected figure or batch selection according to view mode
                if selected_fig_label != "Show All Figures in Selection":
                    target_path = label_to_path.get(selected_fig_label)
                    if target_path:
                        render_benchmark_figure_with_download(target_path, key_suffix="single")
                else:
                    render_reference_figure_gallery(filtered_paths, view_mode, f"{selected_cat}_{selected_type}")

            st.divider()
            section_header("NetCDF Data Explorer", "sky")
            if st.checkbox("Load NetCDF Data Explorer", value=False, key="load_nc_explorer"):
                nc_files = sorted(str(p.resolve()) for p in ROMP_DEMO_OUT_DIR.rglob("*.nc"))

                if nc_files:
                    sel = st.selectbox("Dataset", nc_files, format_func=format_nc_label, key="nc_explorer_sel")
                    try:
                        with _safe_open_dataset(sel) as ds:
                            # Filter out coordinate variables or time-only bounds
                            vars_list = [v for v in ds.data_vars if "time" not in v.lower() and "bounds" not in v.lower()]
                            if vars_list:
                                vn = st.selectbox("Variable", vars_list, key="nc_explorer_var")
                                _render_variable_plot(ds, vn, format_nc_label(sel))
                            else:
                                info_box("No plottable data variables found in dataset.", "amber")
                    except Exception as e:
                        info_box(f"Could not open dataset: {e}", "amber")
                else:
                    info_box("No reference datasets available.", "amber")
            else:
                st.caption("Enable the data explorer to load and map a NetCDF dataset.")
    # Tab 4: Onset Analysis
    if OTS_AVAILABLE and "Onset Analysis" in tab_labels:
        if res_tabs[tab_labels.index("Onset Analysis")].open:
            with res_tabs[tab_labels.index("Onset Analysis")]:
                section_header("Onset Time Series Analysis", "teal")
                mode = st.radio("Location Selection Mode", ["Coordinates", "Woreda"], horizontal=True, key="ots_mode")
                years = _available_years()
                default_yr_idx = years.index(2021) if 2021 in years else len(years) - 1
                year = st.selectbox("Year", years, index=default_yr_idx, key="ots_year")

                lat, lon, label = None, None, None
                if mode == "Coordinates":
                    c1, c2 = st.columns(2)
                    lat = c1.number_input("Latitude", value=8.5, step=0.25, format="%.4f", key="ots_lat")
                    lon = c2.number_input("Longitude", value=39.2, step=0.25, format="%.4f", key="ots_lon")
                    label = f"Lat {lat:.4f}, Lon {lon:.4f}"
                else:
                    shp_path = _find_shapefile_path()
                    if shp_path:
                        woredas = _load_woredas_cached(shp_path)
                        if woredas:
                            search_q = st.text_input("Filter Woreda", value=st.session_state.get("ots_search", ""), key="ots_search")
                            woreda_list = list(woredas.keys())
                            filtered_woredas = [w for w in woreda_list if search_q.lower() in w.lower()] if search_q.strip() else woreda_list

                            if not filtered_woredas:
                                st.warning("No woredas match the filter query.")
                            else:
                                prev_woreda = st.session_state.get("ots_selected_woreda_name")
                                default_idx = (
                                    filtered_woredas.index(prev_woreda)
                                    if prev_woreda and prev_woreda in filtered_woredas
                                    else 0
                                )
                                selected_woreda = st.selectbox(
                                    "Select Woreda",
                                    filtered_woredas,
                                    index=default_idx,
                                    key="ots_woreda_selectbox"
                                )
                                if selected_woreda:
                                    st.session_state["ots_selected_woreda_name"] = selected_woreda
                                    lat, lon = woredas[selected_woreda]
                                    label = selected_woreda
                        else:
                            st.warning("No woredas found in shapefile.")
                    else:
                        st.warning("Shapefile not found.")

                if st.button("Run Onset Analysis", type="primary", key="btn_run_ots") and lat is not None and lon is not None:
                    with st.spinner("Analyzing onset timeseries..."):
                        pngs, grid_lat, grid_lon = _analyze_onset_streamlit(year, lat, lon)
                        st.session_state["onset_pngs"] = pngs
                        st.session_state["onset_meta"] = {
                            "year": year,
                            "lat": lat,
                            "lon": lon,
                            "label": label,
                            "grid_lat": grid_lat,
                            "grid_lon": grid_lon,
                        }

                if "onset_pngs" in st.session_state and st.session_state["onset_pngs"]:
                    meta = st.session_state["onset_meta"]
                    st.info(
                        f"**Results for {meta.get('label') or 'Selected Location'}** | "
                        f"Year: {meta['year']} | "
                        f"Snapped to Grid: Lat {meta['grid_lat']:.4f}, Lon {meta['grid_lon']:.4f}"
                    )
                    for img_bytes in st.session_state["onset_pngs"]:
                        st.image(img_bytes, width='stretch')

    # Tab 5: Skill vs Baseline
    if PHASE_4_AVAILABLE and "Skill vs Baseline" in tab_labels:
        if res_tabs[tab_labels.index("Skill vs Baseline")].open:
            with res_tabs[tab_labels.index("Skill vs Baseline")]:
                section_header("Skill versus Climatology", "green")
                try:
                    scores = collect_scores_from_momp_outputs(ROMP_DEMO_OUT_DIR)
                    if not scores.empty:
                        skill_df = compute_skill_vs_baseline(scores, baseline_model="Climatology")
                        if not skill_df.empty:
                            st.dataframe(skill_df, hide_index=True, width='stretch')
                        else:
                            info_box("No climatology baseline found for skill comparison.", "amber")
                    else:
                        info_box("Run a benchmark to generate verification scores.", "amber")
                except Exception as e:
                    info_box(f"Verification engine error: {e}", "rose")

    # Tab 6: Uncertainty
    if PHASE_5_AVAILABLE and "Uncertainty" in tab_labels:
        if res_tabs[tab_labels.index("Uncertainty")].open:
            with res_tabs[tab_labels.index("Uncertainty")]:
                section_header("Uncertainty & Statistical Significance", "amber")
                st.caption("Bootstrap confidence intervals and significance testing against Climatology.")
                if st.button("Compute Uncertainty", key="btn_compute_uncertainty"):
                    with st.spinner("Computing bootstrap uncertainty..."):
                        try:
                            unc_df = compute_metric_uncertainty(ROMP_DEMO_OUT_DIR)
                            sig_df = compute_significance_vs_baseline(ROMP_DEMO_OUT_DIR, baseline_model="Climatology")
                            st.session_state["unc_df"] = unc_df
                            st.session_state["sig_df"] = sig_df
                        except Exception as e:
                            st.error(f"Error: {e}")
                if "unc_df" in st.session_state:
                    st.markdown("**Metric Uncertainty**")
                    st.dataframe(st.session_state["unc_df"], hide_index=True, width='stretch')
                    st.markdown("**Significance vs Baseline**")
                    st.dataframe(st.session_state["sig_df"], hide_index=True, width='stretch')

    # Tab 7: Benchmark Report & Export
    if (PDF_AVAILABLE or PHASE_6_AVAILABLE) and "Benchmark Report & Export" in tab_labels:
        if res_tabs[tab_labels.index("Benchmark Report & Export")].open:
            with res_tabs[tab_labels.index("Benchmark Report & Export")]:
                section_header("Benchmark Report & Export", "teal")
                st.caption("Generate executive summary reports and inspect workflow provenance metadata.")

                # ======================================================================
                # Provenance Metadata (if available)
                # ======================================================================
                if PHASE_6_AVAILABLE:
                    st.markdown("### Provenance Metadata")
                    try:
                        prov = build_provenance()
                        st.json(prov)
                    except Exception as e:
                        info_box(f"Could not build provenance metadata: {e}", "amber")

                # ======================================================================
                # PDF Generation
                # ======================================================================
                if PDF_AVAILABLE:
                    st.markdown("### Executive Summary Export")
                    st.markdown(
                        "Generate a complete scientific PDF report including benchmark inputs, "
                        "model configuration, verification scores, metric distributions, and benchmark output figures."
                    )

                    # Executive Summary Metrics
                    st.markdown("#### Report Summary")
                    c1, c2, c3, c4 = st.columns(4)
                    det_count = sum(1 for r in active_rows if r.get("Category") == "Deterministic")
                    prob_count = sum(1 for r in active_rows if r.get("Category") == "Probabilistic")
                    c1.metric("Models Evaluated", len(active_rows))
                    c2.metric("Ground Truth", ground_truth)
                    c3.metric("Forecast Window", forecast_window)
                    c4.metric("Baseline", shared_settings.get("baseline_forecast", "climatology"))

                    # Fair comparison window info
                    fair_plan_dict = st.session_state.get("fair_comparison_plan")
                    if fair_plan_dict:
                        eval_years = fair_plan_dict.get("evaluation_years", [])
                        if eval_years:
                            st.info(
                                f"**Fair Comparison Window:** {min(eval_years)} - {max(eval_years)} "
                                f"({len(eval_years)} common years)"
                            )
                        else:
                            st.warning("Fair comparison plan did not find common evaluation years.")

                    st.divider()

                    # Generate PDF Button
                    st.markdown("#### Download Complete Report")
                    generate_clicked = st.button(
                        "Generate Complete PDF Report",
                        type="primary",
                        width="stretch",
                        key="btn_generate_pdf_report",
                    )

                    if generate_clicked:
                        with st.spinner("Generating complete PDF report..."):
                            try:
                                # Collect verification scores
                                scores_df = pd.DataFrame()
                                if PHASE_4_AVAILABLE:
                                    try:
                                        scores_df = collect_scores_from_momp_outputs(ROMP_DEMO_OUT_DIR)
                                    except Exception:
                                        scores_df = pd.DataFrame()

                                # Collect distribution statistics
                                try:
                                    distribution_stats = compute_metric_distribution_stats()
                                except Exception:
                                    distribution_stats = pd.DataFrame()

                                # Collect benchmark figures
                                try:
                                    report_figures = collect_reference_figures()
                                except Exception:
                                    report_figures = {}

                                # Build report context
                                report_context = {
                                    "generated_at": datetime.utcnow().isoformat() + "Z",
                                    "ground_truth": ground_truth,
                                    "benchmark_coverage": benchmark_coverage,
                                    "forecast_window": forecast_window,
                                    "shared_settings": shared_settings,
                                    "active_rows": active_rows,
                                    "fair_plan": st.session_state.get("fair_comparison_plan"),
                                    "map_outputs": map_outputs,
                                    "scores_df": scores_df,
                                    "distribution_stats": distribution_stats,
                                    "uncertainty_df": st.session_state.get("unc_df"),
                                    "significance_df": st.session_state.get("sig_df"),
                                    "figures": report_figures,
                                }

                                # Generate PDF
                                pdf_bytes = generate_benchmark_pdf_report(report_context)

                                # Store in session state
                                st.session_state["benchmark_pdf_bytes"] = pdf_bytes
                                st.session_state["benchmark_pdf_file"] = (
                                    f"benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                                )
                                st.success("PDF report generated successfully.")
                            except Exception as e:
                                st.error(f"PDF generation failed: {e}")
                                import traceback
                                with st.expander("Technical details"):
                                    st.code(traceback.format_exc())

                    # Download Button (appears after generation)
                    pdf_bytes = st.session_state.get("benchmark_pdf_bytes")
                    pdf_file = st.session_state.get("benchmark_pdf_file")
                    if pdf_bytes and pdf_file:
                        st.download_button(
                            "Download Complete PDF Report",
                            data=pdf_bytes,
                            file_name=pdf_file,
                            mime="application/pdf",
                            key="btn_download_pdf_report",
                            type="primary",
                            width="stretch",
                        )
                else:
                    st.error(
                        "PDF generation requires `fpdf2`. Install it with: `pip install fpdf2`"
                    )
if __name__ == "__main__":
    page()
