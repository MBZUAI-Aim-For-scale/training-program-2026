"""
ui/pages/short_medium_benchmarking.py
High-performance Streamlit page for short and medium-range forecast benchmarking.
Includes memory-safe Matplotlib rendering, cached NetCDF metadata discovery with mtime keying,
metric-direction aware performance color coding matrix, grouped spatial metric distribution cards,
padded shapefile clipping & boundary overlay rendering for spatial maps, persistent Woreda selection,
robust argument filtering for onset analysis, and error-proof tab routing.
"""
from __future__ import annotations

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
        "AIFS": "AIFS Single v2",
        "AIFS_ENS": "AIFS Ensemble v2",
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
        st.image(str(path), use_container_width=True)
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

def normalize_aifs_precipitation_dataset(ds: xr.Dataset) -> xr.Dataset:
    var_aliases = ["tp", "tp06", "tp24", "pr", "precip", "total_precipitation"]
    target_var = "RAINFALL"

    # 1. Rename variable to RAINFALL if needed
    for alias in var_aliases:
        if alias in ds.data_vars and target_var not in ds.data_vars:
            ds = ds.rename({alias: target_var})
            break

    # 2. Convert meters (m) to millimeters (mm)
    if target_var in ds.data_vars:
        da = ds[target_var]
        units = str(da.attrs.get("units", "")).lower()
        max_val = float(da.max().values) if da.size > 0 else 0.0


        # If max precipitation value is < 1.0 (indicating meters) or units attribute is 'm'
        if units in ["m", "meters", "meter"] or max_val < 1.0:
            ds[target_var] = ds[target_var] * 1000.0
            ds[target_var].attrs["units"] = "mm"

    return ds


def preprocess_aifs_benchmark_file(file_path: str) -> str:
    p = Path(file_path)
    if not p.exists():
        return file_path

    try:
        ds = _safe_open_dataset(str(p))
        if "tp" in ds.data_vars or "tp06" in ds.data_vars or (
            "RAINFALL" in ds.data_vars and float(ds["RAINFALL"].max().values) < 1.0
        ):
            ds_corrected = normalize_aifs_precipitation_dataset(ds)
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
        
        fig, ax = plt.subplots(figsize=(8, 4))
        da.plot(ax=ax, cmap="YlGnBu")
        ax.set_title(f"{title} - {var_name}")
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
        st.image(path_str, use_container_width=True)
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
                ds_decoded = xr.open_dataset(nc_path, decode_cf=True, decode_times=True)
                da = ds_decoded[time_name]
                min_t = pd.Timestamp(da.values.min()).date()
                max_t = pd.Timestamp(da.values.max()).date()
                ds_decoded.close()
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


def compute_common_model_window(active_rows: List[Dict[str, Any]]):
    starts = [r["_start_date"] for r in active_rows if r.get("_start_date")]
    ends = [r["_end_date"] for r in active_rows if r.get("_end_date")]
    if not starts or not ends:
        return None, None
    cs, ce = max(starts), min(ends)
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
        return "AIFS Ensemble v2"
    if "gencast" in fname:
        return "GenCast"
    if "graphcast" in fname:
        return "GraphCast"
    if "fuxi" in fname:
        return "FuXi"
    if "aifs" in fname:
        return "AIFS Single v2"

    # 1. Check parent folder name first if stored in model subdirectories
    if nc_path.parent != ROMP_DEMO_OUT_DIR:
        if any(k in parent_name for k in ["aifs_ens", "aifs_ensemble"]):
            return "AIFS Ensemble v2"
        if "aifs" in parent_name:
            return "AIFS Single v2"
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
        return "AIFS Ensemble v2"
    if "gencast" in fname:
        return "GenCast"
    if "graphcast" in fname:
        return "GraphCast"
    if "fuxi" in fname:
        return "FuXi"
    if "aifs" in fname:
        return "AIFS Single v2"

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
):
    arr = np.asarray(da.values, dtype=float)
    if not np.isfinite(arr).any():
        return None

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=140)
    try:
        if diverging:
            lim = float(np.nanmax(np.abs(arr))) or 1.0
            vmin, vmax, cmap = -lim, lim, "RdBu_r"

        lon, lat = _map_lon_lat_names(da)
        kwargs = {}
        if cbar_kwargs is not None:
            kwargs["cbar_kwargs"] = cbar_kwargs
        elif unit or title:
            kwargs["cbar_kwargs"] = {"label": unit or title}

        if lon and lat:
            da.plot.imshow(
                ax=ax,
                x=lon,
                y=lat,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                **kwargs,
            )

            # Overlay Admin Boundary Polygon Line
            if admin_gdf is not None and not admin_gdf.empty:
                try:
                    admin_gdf.boundary.plot(ax=ax, color="black", linewidth=1.2, zorder=5)
                except Exception:
                    pass

            # Explicit padding limits
            lons_v = da[lon].values
            lats_v = da[lat].values
            if lons_v.size > 1 and lats_v.size > 1:
                ax.set_xlim(float(np.min(lons_v)), float(np.max(lons_v)))
                ax.set_ylim(float(np.min(lats_v)), float(np.max(lats_v)))
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
        plt.tight_layout()
        return fig
    except Exception:
        plt.close(fig)
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
        "font-size:0.9rem;min-width:700px}"
        ".all-metrics-table th,.all-metrics-table td{border:1px solid #e5e7eb;"
        "padding:6px 8px;text-align:right}"
        ".all-metrics-table th{background:#f8fafc;color:#0f1b2a;font-weight:600}"
        ".all-metrics-table td:first-child,.all-metrics-table th:first-child{"
        "text-align:left}"
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
                        better = diff < 0
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
        "AIFS Single v2": 0,
        "FuXi": 1,
        "GraphCast": 2,
        "AIFS Ensemble v2": 3,
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
            _rc, _timed_out = _spawn_no_fork(
                [sys.executable, "-c", _ONSET_WORKER_SOURCE, str(payload_file), str(out_dir)],
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


def _spawn_no_fork(argv, env, cwd, timeout, on_line=None):
    """
    Run `argv` in `cwd` using posix_spawn on Unix to prevent memory fragmentation and fork-safety issues.
    """
    import select
    import signal
    import time

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
                if time.time() - started > timeout:
                    proc.kill()
                    timed_out = True
                    break
        finally:
            proc.stdout.close()
            proc.wait()
        return (124 if timed_out else proc.returncode), timed_out

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
        pid = os.posix_spawn("/bin/sh", shell_argv, env, file_actions=actions)
    except BaseException:
        os.close(read_fd)
        os.close(write_fd)
        raise
    os.close(write_fd)

    timed_out = False
    buf = b""
    try:
        while True:
            left = timeout - (time.time() - started)
            if left <= 0:
                os.kill(pid, signal.SIGKILL)
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
    except BaseException:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        raise
    finally:
        os.close(read_fd)
        _, status = os.waitpid(pid, 0)

    rc = os.waitstatus_to_exitcode(status)
    return (124 if timed_out else rc), timed_out


def run_momp_benchmark(
    mode: str,
    model_key: Optional[str] = None,
    run_config: Optional[Dict[str, Any]] = None,
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
        live = st.empty()
        lines: List[str] = []
        started = time.time()
        last_draw = 0.0

        def _on_line(line: str) -> None:
            nonlocal last_draw
            lines.append(line)
            now = time.time()
            if now - last_draw > 1.0:
                live.code(f"[{int(now - started)}s elapsed] {' '.join(cmd[-4:])}\n" + "\n".join(lines[-15:]))
                last_draw = now

        returncode, timed_out = _spawn_no_fork(cmd, env, str(PROJECT_ROOT.resolve()), 1800, _on_line)

        text = "\n".join(lines)
        if timed_out:
            text += "\nTimed out after 30 minutes."

        return {
            "mode": mode,
            "returncode": returncode,
            "runtime_dir": str(ROMP_DEMO_OUT_DIR),
            "stdout": text,
            "stderr": "",
            "ok": (not timed_out) and returncode == 0,
            "command": " ".join(cmd),
        }
    except Exception as e:
        import traceback
        return {
            "mode": mode, "returncode": 1, "ok": False,
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
    ("🇳🇬", "Nigeria"), ("🇪ET", "Ethiopia"), ("🇰🇪", "Kenya"), ("🇸🇱", "Sierra Leone"),
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


def render_shared_settings(ground_truth: str = "CHIRPS") -> Dict[str, Any]:
    section_header("Shared settings", "sky")
    st.markdown("Optional settings applied to every selected model in this benchmark.")

    st.markdown("**EVENT DETECTION**")
    c1, c2, c3 = st.columns(3)
    wet_day_threshold = c1.number_input("Wet-day threshold", min_value=0.0, value=20.0, step=0.5, format="%.1f", key="shared_wet_day_threshold")
    minimum_wet_day_rainfall = c2.number_input("Minimum wet-day rainfall", min_value=0.0, value=1.0, step=0.1, format="%.1f", key="shared_min_wet_day")
    wet_spell_length = c3.number_input("Wet spell length", min_value=1, value=3, format="%d", key="shared_wet_spell")

    c4, c5, _ = st.columns(3)
    dry_spell_limit = c4.number_input("Dry spell limit", min_value=0, value=7, format="%d", key="shared_dry_spell")
    dry_spell_search_extension = c5.number_input("Dry spell search extension", min_value=0, value=0, format="%d", key="shared_dry_ext")

    st.markdown("**MASKS AND BASELINE**")
    m1, m2 = st.columns(2)
    default_mask = EXTERNAL_DATA_DIR / "jjas_100mm_rainfall_mask_0p25.nc"
    seasonal_mask = EXTERNAL_DATA_DIR / "jjas_seasonal_mask_0p25.nc"
    mask_options = {
        "No mask": "",
        "JJAS 100mm rainfall mask": str(default_mask),
        "JJAS seasonal 0.25 mask": str(seasonal_mask),
        "Custom path": "__custom__",
    }
    area_mask_choice = m1.selectbox("Area mask file", list(mask_options.keys()), index=0, key="shared_area_mask")
    area_mask_file = mask_options[area_mask_choice]
    if area_mask_choice == "Custom path":
        area_mask_file = m2.text_input("Area mask file path", value="", key="shared_area_mask_custom")

    t1, t2 = st.columns(2)
    default_threshold = ONSET_REFERENCE_DIR / "thresholds_df.csv"
    threshold_options = {
        "Dataset default": "",
        "Reference thresholds CSV": str(default_threshold),
        "Custom path": "__custom__",
    }
    onset_threshold_choice = t1.selectbox("Onset threshold file", list(threshold_options.keys()), index=0, key="shared_onset_thresh")
    onset_threshold_file = threshold_options[onset_threshold_choice]
    if onset_threshold_choice == "Custom path":
        onset_threshold_file = t2.text_input("Onset threshold file path", value="", key="shared_onset_custom")

    b1, b2 = st.columns(2)
    baseline_forecast_choice = b1.selectbox("Baseline forecast", ["climatology", "custom"], index=0, key="shared_baseline_fc")
    baseline_forecast = b2.text_input("Baseline forecast name", value="climatology", key="shared_baseline_custom") if baseline_forecast_choice == "custom" else "climatology"

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
        st.image(str(path), use_container_width=True)
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
            use_container_width=True,
        )


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
        st.image(str(path), use_container_width=True)
        st.caption(format_figure_label(str(path)))

        with open(path, "rb") as f:
            img_bytes = f.read()

        # Action bar: Download and Zoom controls side-by-side
        btn_col1, btn_col2 = st.columns([1, 1])

        with btn_col1:
            st.download_button(
                label="📥 Download",
                data=img_bytes,
                file_name=path.name,
                mime="image/png",
                key=f"dl_btn_{path.stem}_{key_suffix}",
                use_container_width=True,
            )

        with btn_col2:
            if enable_zoom and hasattr(st, "dialog"):
                if st.button(
                    "Zoom Modal", 
                    key=f"zoom_btn_{path.stem}_{key_suffix}", 
                    use_container_width=True
                ):
                    _open_figure_modal(str(path))
    except Exception as e:
        st.warning(f"Could not display figure {path.name}: {type(e).__name__} - {e}")
# ======================================================================
# Main Page
# ======================================================================
def page() -> None:
    _inject_theme_css()

    presets = get_benchmark_presets()
    st.markdown('<div class="page-title">Forecast Benchmarking</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">'
        "Evaluate AI weather model skill against observational references "
        "across the Ethiopian highlands."
        "</div>",
        unsafe_allow_html=True,
    )

    last_run = st.session_state.get("benchmark_last_run")
    status_text = "Completed" if last_run and last_run.get("ok") else ("Failed" if last_run else "Idle")
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
                use_container_width=True,
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

    # Model run windows
    section_header("Model run windows", "amber")
    model_run_windows: Dict[str, Dict[str, Any]] = {}
    if active_rows:
        cs, ce = compute_common_model_window(active_rows)
        if cs and ce:
            st.info(f"Fair-comparison window automatically set to {cs:%Y-%m-%d} to {ce:%Y-%m-%d}.")
        for row in active_rows:
            rk = row["_row_key"]
            avs, ave = row["_start_date"], row["_end_date"]
            des, dee = (max(cs, avs), min(ce, ave)) if cs and ce else (avs, ave)
            if des > dee:
                des, dee = avs, ave
            with st.expander(f"{row['Model']} - available {row['Date Range']}"):
                c1, c2 = st.columns(2)
                eval_start = c1.date_input("Evaluation Start", value=des, min_value=avs, max_value=ave, key=f"es_{rk}")
                eval_end = c2.date_input("Evaluation End", value=dee, min_value=avs, max_value=ave, key=f"ee_{rk}")
                c3, c4 = st.columns(2)
                bsy = c3.number_input("Baseline Start Year", min_value=avs.year, max_value=ave.year, value=des.year, key=f"bsy_{rk}")
                bey = c4.number_input("Baseline End Year", min_value=avs.year, max_value=ave.year, value=dee.year, key=f"bey_{rk}")
                init_days = st.multiselect("Initialization Days", list(range(16)), default=[0, 3], key=f"id_{rk}")
                
                with st.expander("Advanced Temporal & Verification Settings"):
                    c5, c6 = st.columns(2)
                    init_type = c5.selectbox("Init Type", ["weekly", "daily", "monthly"], index=0, key=f"init_type_{rk}")
                    date_filter_year = c6.number_input("Date Filter Year", min_value=2000, max_value=2030, value=2024, key=f"date_filter_year_{rk}")
                    
                    c7, c8 = st.columns(2)
                    verification_window_list = c7.text_input("Verification Window List", value="((1, 15), (16, 30))", key=f"vwl_{rk}")
                    tolerance_days_list = c8.text_input("Tolerance Days List", value="(3, 5)", key=f"tdl_{rk}")
                    
                    c9, _ = st.columns(2)
                    max_forecast_day = c9.number_input("Max Forecast Day", min_value=1, max_value=60, value=30, key=f"max_forecast_day_{rk}")

                model_run_windows[rk] = {
                    "eval_start": eval_start,
                    "eval_end": eval_end,
                    "baseline_start_year": int(bsy),
                    "baseline_end_year": int(bey),
                    "initialization_days": sorted(init_days),
                    "init_type": init_type,
                    "date_filter_year": int(date_filter_year),
                    "verification_window_list": verification_window_list,
                    "tolerance_days_list": tolerance_days_list,
                    "max_forecast_day": int(max_forecast_day),
                }

    shared_settings = render_shared_settings(ground_truth)

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
                    st.dataframe(inventory_to_dataframe(m_inv + o_inv), hide_index=True, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not load inventory: {e}")

    # Run Benchmark Action
    st.markdown('<div class="run-section">', unsafe_allow_html=True)
    c1, _ = st.columns([1, 2])
    run_clicked = c1.button("Run Benchmark", key="btn_run", type="primary", use_container_width=True, disabled=not coverage_ready)
    st.markdown("</div>", unsafe_allow_html=True)

    if run_clicked and active_rows:
        runs: List[Dict[str, Any]] = []
        with st.spinner("Running benchmark..."):
            for row in active_rows:
                rk, ek = row["_row_key"], row["_exec_key"]
                rc = dict(model_run_windows.get(rk, {}))
                rc.update(shared_settings)
                rc.update({
                    "ground_truth": ground_truth,
                    "benchmark_coverage": benchmark_coverage,
                    "forecast_window": forecast_window,
                })
                mode = "prob" if row.get("_probabilistic") else "det"
                result = run_momp_benchmark(mode, ek, rc)
                result["model"] = row.get("Model", ek)
                runs.append(result)

        st.cache_data.clear()
        st.session_state["benchmark_last_run"] = {
            "ok": all(r.get("ok") for r in runs),
            "returncode": max(int(r.get("returncode", 1)) for r in runs),
            "mode": runs[-1].get("mode", ""),
            "runtime_dir": runs[-1].get("runtime_dir", str(ROMP_DEMO_OUT_DIR)),
            "command": "\n".join(r.get("command", "") for r in runs),
            "runs": runs,
        }
        st.rerun()

    # Results Section
    section_header("Results", "sky")
    map_outputs = discover_map_outputs()
    tab_labels = ["Metric Map", "All Metrics", "Benchmark Outputs"]
    if OTS_AVAILABLE:
        tab_labels.append("Onset Analysis")
    if PHASE_4_AVAILABLE:
        tab_labels.append("Skill vs Baseline")
    if PHASE_5_AVAILABLE:
        tab_labels.append("Uncertainty")
    if PDF_AVAILABLE or PHASE_6_AVAILABLE:
        tab_labels.append("Benchmark Report & Export")

    res_tabs = st.tabs(tab_labels)

    # Tab 1: Metric Map
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
                    sub_tabs = st.tabs(["Values", "Skill", "Difference", "Swipe Compare"])
                    

                    # Sub-tab 1: Values
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
                            st.dataframe(df, use_container_width=True, hide_index=True)

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
                            admin_gdf=admin_sub
                        )
                        if fig:
                            st.pyplot(fig)
                            plt.close(fig)
                        st.caption(
                            f"**Skill Classification**: Green = Skillful ({ml} \u2264 {thr}{mm['unit']}), "
                            f"Red = Unskillful ({ml} > {thr}{mm['unit']})."
                        )

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
                                    lf = plot_map(base_da, f"Left: {model_label} ({wl})", cmap=mm["cmap"], unit=mm["unit"], vmin=vmin, vmax=vmax)
                                    rf = plot_map(rda, f"Right: {rm} ({rw})", cmap=mm["cmap"], unit=mm["unit"], vmin=vmin, vmax=vmax)
                                    
                                    if lf and rf:
                                        lb = fig_to_bytes(lf, False)
                                        rb = fig_to_bytes(rf, False)
                                        comp = create_swipe_comparison_image(lb, rb, sp)
                                        if comp:
                                            lc1, lc2 = st.columns(2)
                                            lc1.markdown(f"**Left Map (0–{sp}%):** {model_label} - {wl}")
                                            lc2.markdown(f"**Right Map ({sp}–100%):** {rm} - {rw}")
                                            st.image(comp, use_container_width=True)
                            else:
                                info_box(f"Output for {rm} ({rw}) not available.", "amber")

    # Tab 2: All Metrics
    with res_tabs[1]:
        st.markdown("### ALL METRICS")
        tbl: Dict[tuple, float] = {}
        mods: set = set()
        wins: set = set()
        for o in map_outputs:
            vals = load_metric_values_by_window_cached(o["path"])
            for w, metrics in vals.items():
                for mn, val in metrics.items():
                    tbl[(mn, w, o["model_label"])] = val
                    mods.add(o["model_label"])
                    wins.add(w)
        if tbl:
            pref_m = ["GraphCast", "FuXi", "AIFS Single v2", "AIFS Ensemble v2", "GenCast", "Traditional Climatology"]
            om = [m for m in pref_m if m in mods] + sorted(mods - set(pref_m))
            ow = [w for w in ["Days 1–15", "Days 16–30"] if w in wins] + sorted(wins - {"Days 1–15", "Days 16–30"})
            st.markdown(build_metrics_table_html(tbl, om, ow), unsafe_allow_html=True)

        st.divider()
        section_header("Detailed Metric Distributions", "violet")
        st.caption(
            "Distribution statistics computed directly from gridded model outputs. "
            "Green = better than baseline, Red = worse than baseline. Bold = best value."
            "Green = better (lower than baseline), Red = worse (higher than baseline). Bold = best value."

        )
        stats_df = compute_metric_distribution_stats()
        if stats_df.empty:
            info_box("No distribution statistics could be computed from the available outputs.", "amber")
        else:

            def _norm(s):
                return str(s).replace("–", "-").strip().lower()

            stat_cols = ["Mean", "Min", "Median", "P75", "P90", "Max"]
            best_vals = {}
            baseline_vals = {}
            comparison_scales = {}
            
            for (window, metric), g in stats_df.groupby(["Lead Window", "Metric"]):
                win_k = _norm(window)
                met_k = _norm(metric)
                higher_better = is_metric_higher_better(metric)
                
                clim_mask = g["Model"].apply(lambda m: any(k in _norm(m) for k in ["clim", "trad", "base", "ref"]))
                clim_row = g[clim_mask]
                
                for stat in stat_cols:
                    stat_k = _norm(stat)
                    vals = pd.to_numeric(g[stat], errors="coerce").dropna()
                    
                    if not vals.empty:
                        best_vals[(win_k, met_k, stat_k)] = float(vals.max()) if higher_better else float(vals.min())
                        
                    if not clim_row.empty:
                        clim_v = pd.to_numeric(clim_row[stat], errors="coerce").dropna()
                        if not clim_v.empty:
                            comparison_key = (win_k, met_k, stat_k)
                            baseline_value = float(clim_v.iloc[0])
                            baseline_vals[comparison_key] = baseline_value
                            model_vals = pd.to_numeric(g.loc[~clim_mask, stat], errors="coerce").dropna()
                            differences = (model_vals - baseline_value).abs()
                            if not differences.empty:
                                comparison_scales[comparison_key] = float(differences.max())

            metric_order = {"false alarm rate": 0, "miss rate": 1, "mean absolute error": 2}
            model_order = {"aifs single v2": 0, "fuxi": 1, "graphcast": 2, "aifs ensemble v2": 3, "gencast": 4, "traditional climatology": 5, "climatology": 5}
            window_order = {"days 1-15": 0, "days 16-30": 1}
            
            df_sorted = stats_df.copy()
            df_sorted["metric_sort"] = df_sorted["Metric"].apply(lambda m: metric_order.get(_norm(m), 999))
            df_sorted["model_sort"] = df_sorted["Model"].apply(lambda m: model_order.get(_norm(m), 999))
            df_sorted["window_sort"] = df_sorted["Lead Window"].apply(lambda w: window_order.get(_norm(w), 999))
            df_sorted = df_sorted.sort_values(["model_sort", "window_sort", "metric_sort"])

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

                    cells_html = [f'<td style="border:1px solid #cbd5e1; padding:8px 12px; text-align:left; font-weight:600; background-color:#ffffff; color:#0f172a;">{metric}</td>']
                    
                    for stat in stat_cols:
                        stat_k = _norm(stat)
                        val = row.get(stat, np.nan)
                        
                        if pd.isna(val) or not np.isfinite(val):
                            cells_html.append('<td style="border:1px solid #cbd5e1; padding:8px 12px; text-align:right; background-color:#ffffff; color:#94a3b8;">-</td>')
                            continue

                        val_float = float(val)
                        formatted = f"{val_float:.1f}%" if is_fraction else f"{val_float:.1f} d"

                        best_v = best_vals.get((win_k, met_k, stat_k))
                        if best_v is not None and abs(val_float - best_v) <= 1e-5:
                            formatted = f"<b>{formatted}</b>"

                        base_v = baseline_vals.get((win_k, met_k, stat_k))
                        scale = comparison_scales.get((win_k, met_k, stat_k))
                        
                        comparison_style = _distribution_cell_style(
                            val_float,
                            base_v,
                            scale,
                            higher_better,
                            is_baseline=is_clim,
                        )
                        cell_style = f"border:1px solid #cbd5e1; padding:8px 12px; text-align:right; {comparison_style}"
                        cells_html.append(f'<td style="{cell_style}">{formatted}</td>')
                        
                    unit_str = "fraction" if is_fraction else "days"
                    cells_html.append(f'<td style="border:1px solid #cbd5e1; padding:8px 12px; text-align:center; color:#64748b; background-color:#ffffff;">{unit_str}</td>')
                    rows_html.append(f"<tr>{''.join(cells_html)}</tr>")
                    
                card_html = (
                    f'<div style="margin-bottom:1.75rem; background:#ffffff; border:1px solid #e2e8f0; border-radius:8px; padding:16px; box-shadow:0 1px 3px rgba(0,0,0,0.05);">'
                    f'<div style="font-size:1rem; font-weight:700; color:#0f172a; text-transform:uppercase;">{display_model} - {display_window}</div>'
                    f'<div style="font-size:0.8rem; color:#64748b; margin-bottom:12px;">+/-{tolerance_days} day tolerance</div>'
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
            # Normalize string helper to prevent Unicode en-dash / hyphen key lookup misses
            def _norm(s):
                return str(s).replace("–", "-").strip().lower()

            stat_cols = ["Mean", "Min", "Median", "P75", "P90", "Max"]
            
            # Pre-calculate Climatology baseline & Minimum (Best) values
            best_vals = {}
            baseline_vals = {}
            
            for (window, metric), g in stats_df.groupby(["Lead Window", "Metric"]):
                win_k = _norm(window)
                met_k = _norm(metric)
                
                # Check for Climatology / Baseline model
                clim_mask = g["Model"].apply(lambda m: any(k in _norm(m) for k in ["clim", "trad", "base", "ref"]))
                clim_row = g[clim_mask]
                
                for stat in stat_cols:
                    stat_k = _norm(stat)
                    vals = pd.to_numeric(g[stat], errors="coerce").dropna()
                    
                    if not vals.empty:
                        best_vals[(win_k, met_k, stat_k)] = float(vals.min())
                        
                    if not clim_row.empty:
                        clim_v = pd.to_numeric(clim_row[stat], errors="coerce").dropna()
                        if not clim_v.empty:
                            baseline_vals[(win_k, met_k, stat_k)] = float(clim_v.iloc[0])
                    elif not vals.empty:
                        # Fallback baseline if no explicit Climatology row exists
                        baseline_vals[(win_k, met_k, stat_k)] = float(vals.mean())

            # Sort ordering
            metric_order = {"false alarm rate": 0, "miss rate": 1, "mean absolute error": 2}
            model_order = {"aifs single v2": 0, "fuxi": 1, "graphcast": 2, "aifs ensemble v2": 3, "gencast": 4, "traditional climatology": 5, "climatology": 5}
            window_order = {"days 1-15": 0, "days 16-30": 1}
            
            df_sorted = stats_df.copy()
            df_sorted["metric_sort"] = df_sorted["Metric"].apply(lambda m: metric_order.get(_norm(m), 999))
            df_sorted["model_sort"] = df_sorted["Model"].apply(lambda m: model_order.get(_norm(m), 999))
            df_sorted["window_sort"] = df_sorted["Lead Window"].apply(lambda w: window_order.get(_norm(w), 999))
            df_sorted = df_sorted.sort_values(["model_sort", "window_sort", "metric_sort"])

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
                    is_fraction = bool(row.get("fraction", False)) or any(k in met_k for k in ["rate", "far", "mr"])

                    cells_html = [f'<td style="border:1px solid #cbd5e1; padding:8px 12px; text-align:left; font-weight:600; background-color:#ffffff; color:#0f172a;">{metric}</td>']
                    
                    for stat in stat_cols:
                        stat_k = _norm(stat)
                        val = row.get(stat, np.nan)
                        
                        if pd.isna(val) or not np.isfinite(val):
                            cells_html.append('<td style="border:1px solid #cbd5e1; padding:8px 12px; text-align:right; background-color:#ffffff; color:#94a3b8;">-</td>')
                            continue

                        val_float = float(val)
                        formatted = f"{val_float:.1f}%" if is_fraction else f"{val_float:.1f} d"

                        # Apply bolding if minimum value across models
                        best_v = best_vals.get((win_k, met_k, stat_k))
                        if best_v is not None and abs(val_float - best_v) <= 1e-5:
                            formatted = f"<b>{formatted}</b>"

                        # Explicit color coding vs baseline
                        base_v = baseline_vals.get((win_k, met_k, stat_k))
                        
                        if is_clim or base_v is None:
                            bg_color = "#f1f5f9"
                            text_color = "#334155"
                            font_weight = "normal"
                        else:
                            diff = val_float - base_v
                            if abs(diff) <= 1e-5:
                                bg_color = "#f1f5f9"
                                text_color = "#334155"
                                font_weight = "normal"
                            elif diff < 0:
                                # LOWER than baseline = BETTER (Green)
                                bg_color = "#dcfce7"
                                text_color = "#15803d"
                                font_weight = "600"
                            else:
                                # HIGHER than baseline = WORSE (Red)
                                bg_color = "#fee2e2"
                                text_color = "#b91c1c"
                                font_weight = "normal"

                        cell_style = f"border:1px solid #cbd5e1; padding:8px 12px; text-align:right; background-color:{bg_color} !important; color:{text_color} !important; font-weight:{font_weight};"
                        cells_html.append(f'<td style="{cell_style}">{formatted}</td>')
                        
                    unit_str = "fraction" if is_fraction else "days"
                    cells_html.append(f'<td style="border:1px solid #cbd5e1; padding:8px 12px; text-align:center; color:#64748b; background-color:#ffffff;">{unit_str}</td>')
                    rows_html.append(f"<tr>{''.join(cells_html)}</tr>")
                    
                card_html = (
                    f'<div style="margin-bottom:1.75rem; background:#ffffff; border:1px solid #e2e8f0; border-radius:8px; padding:16px; box-shadow:0 1px 3px rgba(0,0,0,0.05);">'
                    f'<div style="font-size:1rem; font-weight:700; color:#0f172a; text-transform:uppercase;">{display_model} - {display_window}</div>'
                    f'<div style="font-size:0.8rem; color:#64748b; margin-bottom:12px;">+/-{tolerance_days} day tolerance</div>'
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
            fig_options = ["Show All Figures in Selection"]
            for p in filtered_paths:
                lbl = format_figure_label(p)
                if lbl in label_to_path:
                    lbl = f"{lbl} ({Path(p).name})"
                label_to_path[lbl] = p
                fig_options.append(lbl)

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
                if not filtered_paths:
                    info_box("No figures match the selected dropdown filters.", "amber")
                else:
                    st.caption(f"Displaying {len(filtered_paths)} figure(s)")
                    
                    if view_mode == "Full Width":
                        for idx, p in enumerate(filtered_paths):
                            render_benchmark_figure_with_download(p, key_suffix=f"full_{idx}")
                            st.divider()
                    else:  # Grid View (2 Columns)
                        for i in range(0, len(filtered_paths), 2):
                            cols = st.columns(2)
                            for j, col in enumerate(cols):
                                idx = i + j
                                if idx < len(filtered_paths):
                                    with col:
                                        render_benchmark_figure_with_download(
                                            filtered_paths[idx], key_suffix=f"grid_{idx}"
                                        )

        st.divider()
        section_header("NetCDF Data Explorer", "sky")
        nc_files = sorted(str(p.resolve()) for p in ROMP_DEMO_OUT_DIR.rglob("*.nc"))
        
        if nc_files:
            sel = st.selectbox("Dataset", nc_files, format_func=format_nc_label, key="nc_explorer_sel")
            try:
                ds = _safe_open_dataset(sel)
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
    # Tab 4: Onset Analysis
    if OTS_AVAILABLE and "Onset Analysis" in tab_labels:
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
                    st.image(img_bytes, use_container_width=True)

    # Tab 5: Skill vs Baseline
    if PHASE_4_AVAILABLE and "Skill vs Baseline" in tab_labels:
        with res_tabs[tab_labels.index("Skill vs Baseline")]:
            section_header("Skill versus Climatology", "green")
            try:
                scores = collect_scores_from_momp_outputs(ROMP_DEMO_OUT_DIR)
                if not scores.empty:
                    skill_df = compute_skill_vs_baseline(scores, baseline_model="Climatology")
                    if not skill_df.empty:
                        st.dataframe(skill_df, hide_index=True, use_container_width=True)
                    else:
                        info_box("No climatology baseline found for skill comparison.", "amber")
                else:
                    info_box("Run a benchmark to generate verification scores.", "amber")
            except Exception as e:
                info_box(f"Verification engine error: {e}", "rose")

    # Tab 6: Uncertainty
    if PHASE_5_AVAILABLE and "Uncertainty" in tab_labels:
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
                st.dataframe(st.session_state["unc_df"], hide_index=True, use_container_width=True)
                st.markdown("**Significance vs Baseline**")
                st.dataframe(st.session_state["sig_df"], hide_index=True, use_container_width=True)

    # Tab 7: Benchmark Report & Export
    if (PDF_AVAILABLE or PHASE_6_AVAILABLE) and "Benchmark Report & Export" in tab_labels:
        with res_tabs[tab_labels.index("Benchmark Report & Export")]:
            section_header("Benchmark Report & Export", "teal")
            st.caption("Generate executive summary reports and inspect workflow provenance metadata.")

            if PHASE_6_AVAILABLE:
                st.markdown("### Provenance Metadata")
                try:
                    prov = build_provenance()
                    st.json(prov)
                except Exception as e:
                    info_box(f"Could not build provenance metadata: {e}", "amber")

            if PDF_AVAILABLE:
                st.markdown("### Executive Summary Export")
                st.info("PDF engine ready. You can export current benchmark graphics and metrics into a standardized report.")


if __name__ == "__main__":
    page()
