# utils.py
"""
Benchmark utilities for AI weather model evaluation.
Pure Python module without Streamlit dependencies.
"""

from __future__ import annotations

import base64
import importlib
import inspect
import io
import json
import os
import re
import subprocess
import sys
import warnings
from calendar import monthrange
from contextlib import contextmanager
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from PIL import Image, ImageDraw

import config

# ======================================================================
# Safe config access
# ======================================================================
# config.py in this project exposes PROJ_ROOT (older Streamlit configs used PROJECT_ROOT)
PROJECT_ROOT = Path(
    getattr(config, "PROJECT_ROOT", getattr(config, "PROJ_ROOT", Path.cwd()))
).resolve()
ROMP_DEMO_FIG_DIR = Path(
    getattr(
        config,
        "ROMP_FIG_DIR",
        PROJECT_ROOT / "data" / "ROMP_OUT" / "et" / "figure",
    )
)
ROMP_DEMO_OUT_DIR = Path(
    getattr(
        config,
        "ROMP_OUT_DIR",
        PROJECT_ROOT / "data" / "ROMP_OUT" / "et" / "output",
    )
)
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
def _default_obs_catalog() -> Dict[str, Any]:
    """Build the observation catalog from config.ENACTS_DIR / CHIRPS_DIR when config.py has none."""
    cat: Dict[str, Any] = {}
    if hasattr(config, "ENACTS_DIR"):
        cat["ENACTS"] = {"dir": str(config.ENACTS_DIR), "file_pattern": "{}_0p25.nc", "var": "precip"}
    if hasattr(config, "CHIRPS_DIR"):
        cat["CHIRPS"] = {"dir": str(config.CHIRPS_DIR), "file_pattern": "{}.nc", "var": "RAINFALL"}
    return cat


BENCHMARK_OBS_CATALOG = getattr(config, "BENCHMARK_OBS_CATALOG", None) or _default_obs_catalog()
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
# Phase 2-6 & Onset Safe Imports
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
# Constants
# ======================================================================
MODEL_DISPLAY_LABELS = {
    "AIFS": "AIFS Single v2",
    "AIFS_ENS": "AIFS Ensemble v2",
    "FuXi": "FuXi",
    "GraphCast": "GraphCast",
    "GenCast": "GenCast",
    # config.py uses lower-case keys / folder names for some models
    "aifs": "AIFS Single v2",
    "aifs_ens": "AIFS Ensemble v2",
    "fuxi": "FuXi",
    "graphcast": "GraphCast",
    "gencast": "GenCast",
}

LIVE_FORECAST_RECENT_YEARS = 2

REFERENCE_CATEGORY_ORDER = [
    "Deterministic AI",
    "Probabilistic AI",
    "Climatology / Baseline",
    "Other",
]

MAP_METRICS = {
    "False Alarm Rate": {
        "keywords": ["false_alarm_rate", "false_alarm", "far"],
        "fraction": True,
        "unit": "%",
        "threshold": 35.0,
        "cmap": "Reds",
        "description": (
            "Fraction of forecast onset grid cells that were not observed "
            "onset cells. Lower is better."
        ),
    },
    "Miss Rate": {
        "keywords": ["miss_rate", "miss", "mr"],
        "fraction": True,
        "unit": "%",
        "threshold": 60.0,
        "cmap": "Reds",
        "description": (
            "Fraction of observed onset grid cells that the forecast missed. "
            "Lower is better."
        ),
    },
    "Mean Absolute Error": {
        "keywords": ["mean_absolute_error", "mean_mae", "mae"],
        "fraction": False,
        "unit": "days",
        "threshold": 7.0,
        "cmap": "YlOrRd",
        "description": (
            "Average absolute error in onset timing, measured in days. "
            "Lower is better."
        ),
    },
}

MAP_ADMIN_COLUMN_CANDIDATES = {
    "Admin1": [
        "ADM1_EN",
        "ADM1_NAME",
        "ADM1_NM",
        "NAME_1",
        "ADM1",
        "admin1",
        "Region",
    ],
    "Admin2": [
        "ADM2_EN",
        "ADM2_NAME",
        "ADM2_NM",
        "NAME_2",
        "ADM2",
        "admin2",
        "Zone",
    ],
    "Admin3": [
        "ADM3_EN",
        "ADM3_NAME",
        "ADM3_NM",
        "NAME_3",
        "ADM3",
        "adm3_name",
        "admin3",
        "Woreda",
    ],
}

FIGURE_TYPE_TOKENS = {
    "Spatial Metrics (MAE / FAR / Miss Rate)": ["spatial_metrics"],
    "Portrait Panel": ["panel_portrait", "portrait"],
    "Reliability Diagram": ["reliability"],
    "Skill Scores Heatmap": ["skill_scores", "heatmap"],
    "Climatology Onset": ["climatology_onset"],
}

FIGURE_HUMAN_LABELS = {
    "panel_portrait_mae_far_mr_AIFS_30day.png": (
        "Panel Portrait: MAE / FAR / MR - AIFS (30-day)"
    ),
    "panel_portrait_mae_far_mr_prob_30day.png": (
        "Panel Portrait: MAE / FAR / MR - Probabilistic (30-day)"
    ),
    "spatial_metrics_climatology_1-15.png": (
        "Spatial Metrics: Climatology (Days 1-15)"
    ),
    "spatial_metrics_climatology_16-30.png": (
        "Spatial Metrics: Climatology (Days 16-30)"
    ),
    "spatial_metrics_prob_1-15.png": (
        "Spatial Metrics: Probabilistic (Days 1-15)"
    ),
    "spatial_metrics_prob_16-30.png": (
        "Spatial Metrics: Probabilistic (Days 16-30)"
    ),
}

WINDOW_TOLERANCE_DAYS = {
    "Days 1–15": 3,
    "Days 16–30": 5,
}

# ======================================================================
# Generic helpers
# ======================================================================
def _bm(key: str, default: Any) -> Any:
    if key in BENCHMARK_METHOD:
        return BENCHMARK_METHOD[key]
    spaced = f"{key} "
    if spaced in BENCHMARK_METHOD:
        return BENCHMARK_METHOD[spaced]
    return default


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


def _get_obs_meta(obs_key: str) -> Dict[str, Any]:
    catalog = BENCHMARK_OBS_CATALOG or {}
    if obs_key in catalog:
        return catalog[obs_key]
    target = str(obs_key).strip().upper()
    for key, meta in catalog.items():
        if str(key).strip().upper() == target:
            return meta
    return {}


def _find_var(ds: xr.Dataset, keywords: List[str]) -> Optional[str]:
    for kw in keywords:
        for v in ds.data_vars:
            if kw in str(v).lower():
                return str(v)
    return None


def _safe_open_dataset(nc_path: str | Path) -> xr.Dataset:
    ds = xr.open_dataset(Path(nc_path))
    if "day" in ds.dims:
        if "time" in ds.coords and "time" not in ds.dims:
            ds = ds.swap_dims({"day": "time"})
        elif "time" not in ds.coords and "time" not in ds.dims:
            ds = ds.rename({"day": "time"})
    return ds


def format_nc_label(path_str: str) -> str:
    return Path(path_str).stem.replace("_", " ").title()


def _filter_temporal_vars(vars_list: List[str]) -> List[str]:
    return [
        v
        for v in vars_list
        if not any(k in v.lower() for k in ["dayofyear", "doy", "onset"])
    ]


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


def save_benchmark_figure(path_str: str, output_dir: str | Path) -> Path:
    """
    Copy a benchmark figure to an output directory.
    
    Parameters
    ----------
    path_str : str
        Path to the source figure
    output_dir : str or Path
        Directory to copy the figure to
        
    Returns
    -------
    Path
        Path to the copied figure
    """
    import shutil
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dest = output_dir / Path(path_str).name
    shutil.copy2(path_str, dest)
    return dest


# ======================================================================
# Auto-detection helpers
# ======================================================================
def _extract_time_range_from_nc(nc_path: Path):
    try:
        ds = xr.open_dataset(nc_path, decode_cf=True, decode_times=True)
    except Exception:
        try:
            ds = xr.open_dataset(nc_path, decode_cf=False, decode_times=False)
        except Exception:
            return None, None
    try:
        time_name = None
        for name in ds.coords:
            lname = str(name).lower()
            if (
                lname in {"time", "t", "date", "datetime", "valid_time"}
                or "time" in lname
            ):
                time_name = str(name)
                break
        if time_name is None:
            for name in ds.coords:
                try:
                    if np.issubdtype(ds[name].dtype, np.datetime64):
                        time_name = str(name)
                        break
                except Exception:
                    pass
        if time_name is None:
            return None, None
        da = ds[time_name]
        if np.issubdtype(da.dtype, np.number):
            return None, None
        values = np.asarray(da.values).ravel()
        if values.size == 0:
            return None, None
        dates = []
        for v in values[:5000]:
            try:
                ts = pd.Timestamp(v)
                if not pd.isna(ts):
                    dates.append(ts.date())
            except Exception:
                try:
                    dates.append(date(int(v.year), int(v.month), int(v.day)))
                except Exception:
                    pass
        if not dates:
            return None, None
        return min(dates), max(dates)
    except Exception:
        return None, None
    finally:
        try:
            ds.close()
        except Exception:
            pass


def _scan_filename_date_range(nc_files: List[Path]):
    pairs: List[tuple] = []
    years: set = set()
    for path in nc_files:
        for match in re.finditer(
            r"((?:19|20)\d{2})(?:[-_]?(\d{2}))?",
            path.stem,
        ):
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


def _detect_model_data_date_range(model_dir: Path):
    if not model_dir or str(model_dir).strip() in {"", "."}:
        return None, None
    model_dir = Path(model_dir)
    if not model_dir.exists():
        return None, None
    try:
        nc_files = list(model_dir.rglob("*.nc"))
    except Exception:
        return None, None
    if not nc_files:
        return None, None
    
    def file_year(p: Path) -> int:
        m = re.search(r"(?:19|20)\d{2}", p.stem)
        return int(m.group(0)) if m else 0
    
    files_sorted = sorted(nc_files, key=lambda p: (file_year(p), p.name))
    sample = (
        files_sorted
        if len(files_sorted) <= 10
        else files_sorted[:5] + files_sorted[-5:]
    )
    starts, ends = [], []
    for nc_path in sample:
        s, e = _extract_time_range_from_nc(nc_path)
        if s and e:
            starts.append(s)
            ends.append(e)
    if starts and ends:
        return min(starts), max(ends)
    return _scan_filename_date_range(files_sorted)


@lru_cache(maxsize=1)
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
        start_date, end_date = _detect_model_data_date_range(Path(dir_raw))
        if start_date is None or end_date is None:
            continue
        probabilistic = bool(_meta_get(meta, "probabilistic", False))
        category = (
            _clean(_meta_get(meta, "category", "Deterministic"))
            or "Deterministic"
        )
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
    preferred = ["aifs", "aifs_ens", "fuxi", "graphcast", "gencast"]
    detected.sort(
        key=lambda r: preferred.index(r["_row_key"].lower())
        if r["_row_key"].lower() in preferred
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
    if any(
        t in s
        for t in ["1-15", "1_15", "days_1_15", "d1_15", "lead_1_15"]
    ):
        return "Days 1–15"
    if any(
        t in s
        for t in ["16-30", "16_30", "days_16_30", "d16_30", "lead_16_30"]
    ):
        return "Days 16–30"
    return "Days 1–15"


def _map_infer_model_label(nc_path: Path) -> Optional[str]:
    s = str(nc_path).lower()
    if "climatology" in s or "traditional" in s:
        return "Traditional Climatology"
    parent = nc_path.parent
    if (
        parent
        and parent != ROMP_DEMO_OUT_DIR
        and parent.name != ROMP_DEMO_OUT_DIR.name
    ):
        return MODEL_DISPLAY_LABELS.get(
            parent.name,
            parent.name.replace("_", " ").title(),
        )
    if "aifs_ens" in s:
        return "AIFS Ensemble v2"
    if "gencast" in s:
        return "GenCast"
    if "graphcast" in s:
        return "GraphCast"
    if "fuxi" in s:
        return "FuXi"
    if "aifs" in s:
        return "AIFS Single v2"
    return None


@lru_cache(maxsize=1)
def discover_map_outputs(force_refresh: bool = False) -> List[Dict[str, Any]]:
    outputs: List[Dict[str, Any]] = []
    if not ROMP_DEMO_OUT_DIR.exists():
        return outputs
    for nc_path in sorted(ROMP_DEMO_OUT_DIR.rglob("*.nc")):
        first_var = None
        metrics_found: List[str] = []
        try:
            with xr.open_dataset(nc_path) as ds:
                for metric_key, mm in MAP_METRICS.items():
                    var = _find_var(ds, mm["keywords"])
                    if var:
                        metrics_found.append(metric_key)
                        if first_var is None:
                            first_var = var
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
                "metrics": metrics_found,
            }
        )
    return outputs


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


@lru_cache(maxsize=1)
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


@lru_cache(maxsize=32)
def map_admin_options(al: str) -> List[str]:
    gdf = _map_load_admin_shapefile()
    if gdf is None:
        return []
    col = _map_admin_column(gdf, al)
    if not col:
        return []
    vals = gdf[col].dropna().astype(str).str.strip()
    return sorted(vals[vals != ""].unique().tolist())


@lru_cache(maxsize=64)
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


def _map_clip_bbox(da: xr.DataArray, bbox) -> xr.DataArray:
    lon, lat = _map_lon_lat_names(da)
    if not lon or not lat:
        return da
    minx, miny, maxx, maxy = bbox
    lats = da[lat].values
    lat_slice = (
        slice(maxy, miny)
        if len(lats) > 1 and lats[0] > lats[-1]
        else slice(miny, maxy)
    )
    try:
        return da.sel({lon: slice(minx, maxx), lat: lat_slice})
    except Exception:
        return da


@lru_cache(maxsize=128)
def load_map_da_cached(
    nc_path_str: str,
    ml: str,
    wl: str,
    al: str = "",
    an: str = "",
) -> Optional[xr.DataArray]:
    try:
        ds = xr.open_dataset(Path(nc_path_str))
    except Exception:
        return None
    mm = MAP_METRICS.get(ml)
    if not mm:
        ds.close()
        return None
    var = _find_var(ds, mm["keywords"])
    if not var:
        ds.close()
        return None
    da = _map_select_window(ds[var], wl)
    while len(da.dims) > 2:
        da = da.isel({da.dims[0]: 0})
    if mm.get("fraction", False):
        arr = np.asarray(da.values, dtype=float)
        if np.isfinite(arr).any() and float(np.nanmax(arr)) <= 1.5:
            da = da * 100.0
    if al and an and an != "All":
        bbox = map_admin_bounds(al, an)
        if bbox:
            da = _map_clip_bbox(da, bbox)
    da = da.load()
    ds.close()
    return da


def plot_map(
    da,
    title,
    cmap="viridis",
    unit="",
    vmin=None,
    vmax=None,
    diverging=False,
    save_path: Optional[str | Path] = None,
):
    """
    Plot a metric map.
    
    Parameters
    ----------
    save_path : str or Path, optional
        If provided, save the figure to this path instead of displaying it
        
    Returns
    -------
    matplotlib.figure.Figure
        The figure object
    """
    arr = np.asarray(da.values, dtype=float)
    if not np.isfinite(arr).any():
        return None
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if diverging:
        lim = float(np.nanmax(np.abs(arr))) or 1.0
        vmin, vmax, cmap = -lim, lim, "RdBu_r"
    lon, lat = _map_lon_lat_names(da)
    try:
        if lon and lat:
            da.plot.imshow(
                ax=ax,
                x=lon,
                y=lat,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                cbar_kwargs={"label": unit or title},
            )
        else:
            im = ax.imshow(
                arr,
                origin="lower",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )
            fig.colorbar(im, ax=ax, label=unit or title)
    except Exception:
        im = ax.imshow(
            arr,
            origin="lower",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        fig.colorbar(im, ax=ax, label=unit or title)
    ax.set_title(title, fontsize=10)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    
    return fig


def plot_map_image(da, cmap="viridis", vmin=None, vmax=None):
    arr = np.asarray(da.values, dtype=float)
    if not np.isfinite(arr).any():
        return None
    fig = plt.figure(figsize=(6, 4), dpi=140)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(
        arr,
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_aspect("auto")
    ax.axis("off")
    return fig


def fig_to_bytes(fig, tight: bool = True) -> bytes:
    buf = io.BytesIO()
    fig.savefig(
        buf,
        format="png",
        dpi=140,
        bbox_inches="tight" if tight else None,
        pad_inches=0 if not tight else 0.1,
    )
    buf.seek(0)
    plt.close(fig)
    return buf.read()


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


@lru_cache(maxsize=128)
def compute_model_comparison_cached(ml, wl, al, an):
    outs = discover_map_outputs()
    ual = al if al and al != "All" else ""
    uan = an if an and an != "All" else ""
    recs: Dict[str, List[float]] = {}
    for item in outs:
        if wl not in item.get("windows", []):
            continue
        da = load_map_da_cached(item["path"], ml, wl, ual, uan)
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
# All Metrics table helpers
# ======================================================================
def _extract_spatial_mean(da, convert_fraction=False):
    while len(da.dims) > 2:
        da = da.isel({da.dims[0]: 0})
    arr = np.asarray(da.values, dtype=float)
    if not np.isfinite(arr).any():
        return None
    val = float(np.nanmean(arr))
    return val / 100.0 if convert_fraction and val > 1.5 else val


@lru_cache(maxsize=128)
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


@lru_cache(maxsize=1)
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
            da = load_map_da_cached(path, metric_key, wl)
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


# ======================================================================
# Reference / Benchmark Outputs helpers
# ======================================================================
def classify_reference_asset(path: Path) -> str:
    s = str(path).lower()
    prob_keys = [
        "aifs_ens",
        "ens",
        "ensemble",
        "probabilistic",
        "gencast",
        "brier",
        "rps",
        "reliability",
        "spread",
        "rank",
    ]
    clim_keys = [
        "climatology",
        "baseline",
        "traditional",
    ]
    det_keys = [
        "aifs",
        "fuxi",
        "graphcast",
        "deterministic",
        "mae",
        "far",
        "mr",
    ]
    if any(k in s for k in prob_keys):
        return "Probabilistic AI"
    if any(k in s for k in clim_keys):
        return "Climatology / Baseline"
    if any(k in s for k in det_keys):
        return "Deterministic AI"
    return "Other"


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
class _FilteredStream:
    def __init__(self, stream, pattern):
        self.stream = stream
        self.pattern = pattern
        self._skip_nl = False

    def write(self, text):
        if self.pattern in text:
            self._skip_nl = True
            return len(text)
        if self._skip_nl and text == "\n":
            self._skip_nl = False
            return 1
        self._skip_nl = False
        return self.stream.write(text)

    def __getattr__(self, name):
        return getattr(self.stream, name)


@contextmanager
def suppress_print_pattern(pattern):
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = _FilteredStream(old_out, pattern)
    sys.stderr = _FilteredStream(old_err, pattern)
    try:
        yield
    finally:
        sys.stdout = old_out
        sys.stderr = old_err


def _get_onset_base_kwargs():
    obs_key = os.getenv("ROMP_OBS", getattr(config, "obs", "CHIRPS")).upper()
    meta = _get_obs_meta(obs_key) or _get_obs_meta("CHIRPS")
    obs_dir = Path(
        _meta_get(
            meta,
            "dir",
            PROJECT_ROOT / "data" / "external" / "CHIRPS_IMERG",
        )
    )
    pattern = _meta_get(meta, "file_pattern", "{}.nc")
    CHIRPS_DIR = obs_dir  # kept under this name for the callers below
    base_kwargs = {
        "obs": obs_key,
        "obs_dir": str(obs_dir),
        "obs_file_pattern": (pattern,),
        "obs_var": _meta_get(meta, "var", "RAINFALL"),
        "ref_model_dir": str(obs_dir),
        "ref_model_file_pattern": pattern,
    }
    if OTS_AVAILABLE:
        sig = inspect.signature(ots.obs_onset_analysis)
        if not any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in sig.parameters.values()
        ):
            base_kwargs = {
                k: v
                for k, v in base_kwargs.items()
                if k in sig.parameters
            }
    return base_kwargs, CHIRPS_DIR


@lru_cache(maxsize=1)
def _available_years():
    base_kwargs, CHIRPS_DIR = _get_onset_base_kwargs()
    if not CHIRPS_DIR.is_dir():
        return list(range(2015, 2024))
    pattern = base_kwargs.get("obs_file_pattern", ("{}.nc",))
    pattern = pattern[0] if isinstance(pattern, (tuple, list)) else pattern
    years = [
        y
        for y in range(1980, 2031)
        if (CHIRPS_DIR / pattern.format(y)).is_file()
    ]
    return years or list(range(2015, 2024))


@lru_cache(maxsize=128)
def _snap_to_grid(lat, lon, data_dir_str):
    data_dir = Path(data_dir_str)
    sample = next(data_dir.glob("*.nc"), None)
    if not sample:
        return lat, lon
    try:
        with xr.open_dataset(sample) as ds:
            lat_key = next(
                (
                    k
                    for k in ds.variables
                    if str(k).lower() in ("lat", "latitude", "y")
                ),
                None,
            )
            lon_key = next(
                (
                    k
                    for k in ds.variables
                    if str(k).lower() in ("lon", "longitude", "x")
                ),
                None,
            )
            if not lat_key or not lon_key:
                return lat, lon
            nlat = ds[lat_key].sel({lat_key: lat}, method="nearest").values.item()
            nlon = ds[lon_key].sel({lon_key: lon}, method="nearest").values.item()
        return float(nlat), float(nlon)
    except Exception:
        return lat, lon


@lru_cache(maxsize=1)
def _find_shapefile_path():
    hint = (
        getattr(config, "WOREDA_SHP", None)
        or getattr(config, "SHAPEFILE_PATH", None)
        or getattr(config, "ZONE_WOREDA_SHP", None)
    )
    if hint and Path(hint).is_file():
        return str(Path(hint).resolve())
    root = Path(
        getattr(
            config,
            "DATA_DIR",
            PROJECT_ROOT / "data",
        )
    )
    found = list(root.rglob("*.shp")) if root.is_dir() else []
    preferred = [
        p
        for p in found
        if any(k in p.name.lower() for k in ("woreda", "wereda", "adm3"))
    ]
    best = (preferred or found or [None])[0]
    return str(best.resolve()) if best else None


@lru_cache(maxsize=1)
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
        NAME_COLS = (
            "woreda",
            "w_name",
            "wname",
            "woreda_name",
            "wor_name",
            "adm3_en",
            "adm3_name",
            "admin3name",
            "name",
        )
        ZONE_COLS = (
            "zone",
            "z_name",
            "zone_name",
            "adm2_en",
            "admin2name",
        )
        lower = {c.lower(): c for c in gdf.columns}
        name_col = next(
            (lower[c] for c in NAME_COLS if c in lower),
            None,
        )
        if name_col is None:
            name_col = next(
                (
                    c
                    for c in gdf.columns
                    if c != gdf.geometry.name and gdf[c].dtype == object
                ),
                None,
            )
        if name_col is None:
            return {}
        zone_col = next(
            (lower[c] for c in ZONE_COLS if c in lower),
            None,
        )
        pts = gdf.geometry.representative_point()
        out = {}
        for i, name in enumerate(gdf[name_col].astype(str)):
            label = (
                f"{name} ({gdf[zone_col].iloc[i]})"
                if zone_col
                else name
            )
            if label in out:
                label = f"{label} #{i}"
            out[label] = (
                float(pts.iloc[i].y),
                float(pts.iloc[i].x),
            )
        return dict(sorted(out.items()))
    except Exception:
        return {}


def _fig_to_png(fig):
    buf = io.BytesIO()
    fig.savefig(
        buf,
        format="png",
        dpi=110,
        bbox_inches="tight",
    )
    buf.seek(0)
    return buf.getvalue()


def analyze_onset(
    year: int,
    lat: float,
    lon: float,
    save_dir: Optional[str | Path] = None,
) -> Tuple[List[bytes], float, float]:
    """
    Run onset time series analysis.
    
    Parameters
    ----------
    year : int
        Year to analyze
    lat : float
        Latitude
    lon : float
        Longitude
    save_dir : str or Path, optional
        If provided, save PNG figures to this directory
        
    Returns
    -------
    tuple
        (list of PNG bytes, snapped latitude, snapped longitude)
    """
    base_kwargs, CHIRPS_DIR = _get_onset_base_kwargs()
    grid_lat, grid_lon = _snap_to_grid(lat, lon, str(CHIRPS_DIR))
    pngs = []
    plt.close("all")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with suppress_print_pattern(
                "specified region is not in cartopy"
            ), patch.object(
                plt,
                "show",
                lambda *a, **k: None,
            ), patch.object(
                plt,
                "close",
                lambda *a, **k: None,
            ):
                ots.obs_onset_analysis(
                    year=year,
                    lat_select=grid_lat,
                    lon_select=grid_lon,
                    **base_kwargs,
                )
        figs = [
            plt.figure(n)
            for n in plt.get_fignums()
            if plt.figure(n).get_axes()
        ]
        pngs = [_fig_to_png(f) for f in figs]
        
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            for i, png_bytes in enumerate(pngs):
                output_path = save_dir / f"onset_{year}_{i+1}.png"
                output_path.write_bytes(png_bytes)
    except Exception as e:
        print(f"Error during onset analysis: {e}")
    finally:
        plt.close("all")
    return pngs, grid_lat, grid_lon


# ======================================================================
# Execution bridge
# ======================================================================
def _locate_run_momp() -> Optional[Path]:
    for c in [PROJECT_ROOT / "run_momp.py", Path.cwd() / "run_momp.py"]:
        if c.exists():
            return c.resolve()
    return None


def run_momp_benchmark(
    mode: str,
    model_key: Optional[str] = None,
    run_config: Optional[Dict[str, Any]] = None,
    spec_path: Optional[str | Path] = None,
    timeout_minutes: float = 30,
    stream: bool = False,
    log_dir: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """
    Run the MOMP benchmark pipeline (``run_momp.py``) in a subprocess.

    Parameters
    ----------
    mode : "det" or "prob"
    model_key : run a single catalog model (e.g. "AIFS"); None = all of that mode
    run_config : legacy dict -> ROMP_* environment variables. NOTE: the current
        ``run_momp.py`` ignores these; prefer ``spec_path``.
    spec_path : benchmark spec JSON, passed as ``--spec`` (dates, onset rules,
        mask, baseline ... are applied by run_momp.apply_spec_to_cfg)
    timeout_minutes : kill the run after this many minutes
    stream : print the output live and write it to a log file
    log_dir : where the log goes when ``stream`` is True (default: parent of the output dir)

    Returns
    -------
    dict with keys: mode, returncode, ok, stdout, stderr, runtime_dir, command[, log]
    """
    mode = mode.lower()
    if mode not in ("det", "deterministic", "prob", "probabilistic"):
        return {
            "mode": mode,
            "returncode": 1,
            "ok": False,
            "stdout": "",
            "stderr": f"Invalid mode: {mode}",
            "runtime_dir": str(ROMP_DEMO_OUT_DIR),
        }
    rmp = _locate_run_momp()
    if rmp is None:
        return {
            "mode": mode,
            "returncode": 1,
            "ok": False,
            "stdout": "",
            "stderr": "run_momp.py not found (looked in the project root and the current folder).",
            "runtime_dir": str(ROMP_DEMO_OUT_DIR),
        }
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(PROJECT_ROOT.resolve()), env.get("PYTHONPATH", "")]
    )
    env["MPLCONFIGDIR"] = str((PROJECT_ROOT / ".mplcache").resolve())
    env.setdefault("MPLBACKEND", "Agg")
    Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    if run_config:
        if run_config.get("eval_start"):
            env["ROMP_EVAL_START"] = str(run_config["eval_start"])
        if run_config.get("eval_end"):
            env["ROMP_EVAL_END"] = str(run_config["eval_end"])
        if run_config.get("baseline_start_year"):
            env["ROMP_BASELINE_START_YEAR"] = str(
                run_config["baseline_start_year"]
            )
        if run_config.get("baseline_end_year"):
            env["ROMP_BASELINE_END_YEAR"] = str(
                run_config["baseline_end_year"]
            )
        if "initialization_days" in run_config:
            env["ROMP_INIT_DAYS"] = ",".join(
                str(int(x))
                for x in run_config.get("initialization_days", [])
            )
        env["ROMP_RUN_YEARS_CONCURRENTLY"] = (
            "1"
            if run_config.get("run_years_concurrently", True)
            else "0"
        )
        env["ROMP_ENSEMBLE_FORECAST"] = (
            "1"
            if run_config.get("ensemble_forecast", False)
            else "0"
        )
        if run_config.get("forecast_window"):
            env["ROMP_FORECAST_WINDOW"] = str(
                run_config["forecast_window"]
            ).replace(" days", "")
        if run_config.get("ground_truth"):
            env["ROMP_GROUND_TRUTH"] = str(run_config["ground_truth"])
        if run_config.get("benchmark_coverage"):
            env["ROMP_BENCHMARK_COVERAGE"] = str(
                run_config["benchmark_coverage"]
            )
    cmd = [sys.executable, "-u", str(rmp), "--mode", mode]
    if model_key:
        cmd.extend(["--model", model_key])
    if spec_path:
        cmd.extend(["--spec", str(spec_path)])
    command = " ".join(cmd)
    base = {"mode": mode, "runtime_dir": str(ROMP_DEMO_OUT_DIR), "command": command}

    if not stream:
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(PROJECT_ROOT.resolve()),
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout_minutes * 60,
            )
            clear_caches()
            return {
                **base,
                "returncode": proc.returncode,
                "stdout": proc.stdout or "",
                "stderr": proc.stderr or "",
                "ok": proc.returncode == 0,
            }
        except subprocess.TimeoutExpired:
            return {**base, "returncode": 124, "ok": False, "stdout": "", "stderr": "Timed out."}
        except Exception as e:
            return {**base, "returncode": 1, "ok": False, "stdout": "", "stderr": str(e)}

    # ---- streaming mode: live output + log file + time-out ----
    import threading
    import time

    log_root = Path(log_dir) if log_dir else ROMP_DEMO_OUT_DIR.parent
    log_root.mkdir(parents=True, exist_ok=True)
    log_path = log_root / f"run_{mode}_{model_key or 'all'}_{datetime.now():%Y%m%d_%H%M%S}.log"
    print("COMMAND:", command)
    print("LOG    :", log_path)
    t0 = time.time()
    lines: List[str] = []
    timed_out = {"flag": False}
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT.resolve()),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except Exception as e:
        return {**base, "returncode": 1, "ok": False, "stdout": "", "stderr": str(e), "log": str(log_path)}

    def _kill():
        timed_out["flag"] = True
        proc.kill()

    timer = threading.Timer(timeout_minutes * 60, _kill)
    timer.start()
    try:
        with open(log_path, "w", encoding="utf-8") as fh:
            for line in proc.stdout:
                fh.write(line)
                lines.append(line)
                print(line, end="")
        proc.wait()
    finally:
        timer.cancel()
    rc = 124 if timed_out["flag"] else proc.returncode
    ok = rc == 0
    print(f"\n{'finished' if ok else 'FAILED'} (exit code {rc}) in {(time.time() - t0) / 60:.1f} min")
    if not ok:
        print("---- last lines ----\n" + "".join(lines[-25:]))
    clear_caches()
    return {
        **base,
        "returncode": rc,
        "ok": ok,
        "stdout": "".join(lines),
        "stderr": "Timed out." if timed_out["flag"] else "",
        "log": str(log_path),
    }


def clear_caches() -> None:
    """Forget cached discovery/metric results (call after a new MOMP run)."""
    for fn in (
        detect_benchmark_models,
        discover_map_outputs,
        load_map_da_cached,
        load_metric_values_by_window_cached,
        compute_model_comparison_cached,
        compute_metric_distribution_stats,
    ):
        cc = getattr(fn, "cache_clear", None)
        if cc:
            cc()


# ======================================================================
# PDF Report helpers
# ======================================================================
def _pdf_safe_text(value: Any) -> str:
    text = str(value)
    replacements = {
        "\u2013": "-",
        "\u2014": "-",
        "\u00b7": "-",
        "\u2713": "Yes",
        "\u2019": "'",
        "\u2018": "'",
        "\u201c": '"',
        "\u201d": '"',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text.encode("latin-1", "ignore").decode("latin-1")


def _normalize_dataframe_for_pdf(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = pd.DataFrame(index=df.index)
    clean_cols = []
    seen = set()
    for col in df.columns:
        base = str(col).strip() or "column"
        name = base
        suffix = 1
        while name in seen:
            suffix += 1
            name = f"{base}_{suffix}"
        seen.add(name)
        clean_cols.append(name)
        
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
        return getattr(
            self,
            "epw",
            self.w - self.l_margin - self.r_margin,
        )

    def header(self):
        if self.page_no() > 1:
            self.set_font("Helvetica", "B", 9)
            self.set_text_color(15, 23, 42)
            self.set_x(self.l_margin)
            self.cell(
                0,
                8,
                _pdf_safe_text("Forecast Benchmarking Report"),
            )
            self.ln(4)
            self.set_draw_color(226, 232, 240)
            self.line(
                self.l_margin,
                self.get_y(),
                self.w - self.r_margin,
                self.get_y(),
            )
            self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(100, 116, 139)
        self.set_x(self.l_margin)
        self.cell(
            0,
            10,
            f"Page {self.page_no()}",
            align="C",
        )

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
        self.line(
            self.l_margin,
            line_y,
            self.l_margin + 45,
            line_y,
        )
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
            self.multi_cell(
                0,
                6,
                _pdf_safe_text(f"{key}: {value}"),
            )

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
            if not text:
                return ""
            return text + ".."
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
        max_cols_per_table = max(
            1,
            int(usable_width // self.MIN_COL_WIDTH),
        )
        col_chunks = [
            all_cols[i:i + max_cols_per_table]
            for i in range(0, len(all_cols), max_cols_per_table)
        ]
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
                self.cell(
                    0,
                    5,
                    _pdf_safe_text(
                        f"Table continued - columns {start_col} to {end_col}"
                    ),
                )
                self.ln(6)
            n_cols = len(cols)
            widths = [usable_width / n_cols] * n_cols
            row_height = 6
            if n_cols <= 3:
                header_font = 8
                body_font = 7
            elif n_cols <= 5:
                header_font = 7
                body_font = 6
            else:
                header_font = 6
                body_font = 5
            
            def draw_header():
                self.set_x(self.l_margin)
                self.set_font("Helvetica", "B", header_font)
                self.set_fill_color(241, 245, 249)
                self.set_text_color(15, 23, 42)
                for i, col_name in enumerate(cols):
                    cell_width = widths[i]
                    if cell_width > 6:
                        label = self._fit_text(col_name, cell_width)
                    else:
                        label = ""
                    self.cell(
                        cell_width,
                        row_height + 1,
                        label,
                        border=1,
                        align="C",
                        fill=True,
                    )
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
                    if cell_width > 6:
                        txt = self._fit_text(
                            row.get(col_name, ""),
                            cell_width,
                        )
                    else:
                        txt = ""
                    self.cell(
                        cell_width,
                        row_height,
                        txt,
                        border=1,
                    )
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
            max_w = usable_width - 10
            if max_w < 20:
                max_w = 20
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
            self.image(
                str(image_path),
                x=x,
                y=start_y,
                w=target_w,
                h=target_h,
            )
            self.set_y(start_y + target_h + 2)
            self.set_x(self.l_margin)
            if caption:
                self.set_font("Helvetica", "I", 8)
                self.set_text_color(71, 85, 105)
                self.multi_cell(
                    0,
                    5,
                    _pdf_safe_text(caption),
                    align="C",
                )
                self.ln(4)
        except Exception:
            self.paragraph(f"Could not embed image: {image_path.name}")


def generate_benchmark_pdf_report(context: Dict[str, Any]) -> bytes:
    """
    Generate a complete PDF benchmark report.
    
    Parameters
    ----------
    context : dict
        Report context with keys:
        - generated_at, ground_truth, benchmark_coverage, forecast_window
        - shared_settings, active_rows, fair_plan, map_outputs
        - scores_df, distribution_stats, uncertainty_df, significance_df
        - figures
        
    Returns
    -------
    bytes
        PDF file content
    """
    pdf = BenchmarkReportPDF(orientation="L", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()
    
    # Cover
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(15, 23, 42)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(
        0,
        12,
        "Forecast Benchmarking Report",
        align="C",
    )
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(71, 85, 105)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(
        0,
        7,
        _pdf_safe_text(context.get("generated_at", "")),
        align="C",
    )
    pdf.ln(6)
    
    # Executive summary
    pdf.section_title("Executive Summary")
    active_rows = context.get("active_rows", [])
    shared_settings = context.get("shared_settings", {})
    det_count = sum(
        1
        for r in active_rows
        if r.get("Category") == "Deterministic"
    )
    prob_count = sum(
        1
        for r in active_rows
        if r.get("Category") == "Probabilistic"
    )
    pdf.key_value_lines(
        [
            ("Ground Truth", context.get("ground_truth", "")),
            ("Benchmark Coverage", context.get("benchmark_coverage", "")),
            ("Forecast Window", context.get("forecast_window", "")),
            (
                "Baseline Forecast",
                shared_settings.get("baseline_forecast", "climatology"),
            ),
            ("Models Evaluated", len(active_rows)),
            ("Deterministic Models", det_count),
            ("Probabilistic Models", prob_count),
        ]
    )
    
    # Models evaluated
    if active_rows:
        models_df = pd.DataFrame(
            [
                {
                    "Model": r.get("Model", ""),
                    "Category": r.get("Category", ""),
                    "Type": r.get("Type", ""),
                    "Coverage": r.get("Coverage", ""),
                    "Date Range": r.get("Date Range", ""),
                }
                for r in active_rows
            ]
        )
        if not models_df.empty:
            pdf.section_title("Models Evaluated")
            pdf.dataframe_table(models_df)
    
    # Benchmark inputs
    pdf.section_title("Benchmark Inputs")
    pdf.key_value_lines(
        [
            ("Ground Truth", context.get("ground_truth", "")),
            ("Benchmark Coverage", context.get("benchmark_coverage", "")),
            ("Forecast Window", context.get("forecast_window", "")),
        ]
    )
    
    # Shared settings
    if shared_settings:
        pdf.section_title("Shared Settings")
        pdf.key_value_lines(
            [
                (str(k).replace("_", " ").title(), v)
                for k, v in shared_settings.items()
            ]
        )
    
    # Fair comparison plan
    fair_plan = context.get("fair_plan")
    if fair_plan:
        pdf.section_title("Fair Comparison Plan")
        eval_years = fair_plan.get("evaluation_years", [])
        pdf.key_value_lines(
            [
                ("Can Run", fair_plan.get("can_run", "")),
                ("Ground Truth", fair_plan.get("ground_truth", "")),
                (
                    "Benchmark Coverage",
                    fair_plan.get("benchmark_coverage", ""),
                ),
                (
                    "Climatology Policy",
                    fair_plan.get("climatology_policy", ""),
                ),
                ("Common Year Min", fair_plan.get("common_year_min", "")),
                ("Common Year Max", fair_plan.get("common_year_max", "")),
                (
                    "Evaluation Years",
                    ", ".join(str(y) for y in eval_years),
                ),
            ]
        )
        issues = fair_plan.get("issues", [])
        warnings_list = fair_plan.get("warnings", [])
        if issues:
            pdf.sub_title("Blocking Issues")
            for issue in issues:
                pdf.paragraph(f"- {issue}")
        if warnings_list:
            pdf.sub_title("Warnings")
            for warning in warnings_list:
                pdf.paragraph(f"- {warning}")
    
    # Available model outputs
    map_outputs = context.get("map_outputs", [])
    if map_outputs:
        outputs_df = pd.DataFrame(
            [
                {
                    "Model": o.get("model_label", ""),
                    "Windows": ", ".join(o.get("windows", [])),
                    "File": Path(o.get("path", "")).name,
                }
                for o in map_outputs
            ]
        )
        if not outputs_df.empty:
            pdf.section_title("Available Model Outputs")
            pdf.dataframe_table(outputs_df)
    
    # Verification scores
    scores_df = context.get("scores_df")
    if scores_df is not None and not scores_df.empty:
        pdf.section_title("Verification Scores")
        pdf.dataframe_table(scores_df)
    
    # Detailed metric distributions
    distribution_stats = context.get("distribution_stats")
    if distribution_stats is not None and not distribution_stats.empty:
        pdf.section_title("Detailed Metric Distributions")
        pdf.dataframe_table(distribution_stats)
    
    # Uncertainty
    uncertainty_df = context.get("uncertainty_df")
    if uncertainty_df is not None and not uncertainty_df.empty:
        pdf.section_title("Metric Uncertainty")
        pdf.dataframe_table(uncertainty_df)
    
    significance_df = context.get("significance_df")
    if significance_df is not None and not significance_df.empty:
        pdf.section_title("Significance vs Baseline")
        pdf.dataframe_table(significance_df)
    
    # Benchmark figures
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
                    pdf.add_image_path(
                        path_str,
                        format_figure_label(path_str),
                    )
    
    # Notes
    pdf.section_title("Notes")
    pdf.paragraph(
        "This report was generated automatically from the benchmark dashboard. "
        "Metrics are computed from model output files available at generation time."
    )
    
    return bytes(pdf.output())

import runpy
import os
import sys
import importlib.metadata
from importlib.metadata import PackageNotFoundError
import pyproj
from contextlib import contextmanager

correct_proj_data = pyproj.datadir.get_data_dir()
os.environ["PROJ_DATA"] = correct_proj_data
os.environ["PROJ_LIB"] = correct_proj_data

from momp.lib.loader import apply_overrides, reset_cfg, get_cfg
reset_cfg()                      # clear cached singleton before switching modes/files
apply_overrides(mode="prob")     # forces mode-derived overrides, same as --mode
cfg = get_cfg()
@contextmanager
def sanitized_driver_argv(driver_path, mode):
    """
    Temporarily replace sys.argv so driver.py's argparse sees only:
        driver.py --mode det/prob
    
    This prevents Jupyter's kernel arguments from interfering.
    """
    original_argv = sys.argv.copy()
    
    # Normalize mode for the driver
    driver_mode = mode.lower()
    if driver_mode == "deterministic":
        driver_mode = "det"
    elif driver_mode == "probabilistic":
        driver_mode = "prob"
    
    # Set sys.argv to what the driver expects
    sys.argv = [
        str(driver_path),
        "--mode",
        driver_mode,
    ]
    
    try:
        yield
    finally:
        # Always restore sys.argv
        sys.argv = original_argv


def execute_momp_benchmark(mode="det"):
    """Execute the full MOMP driver pipeline for the given mode."""
    print("=" * 60)
    print(f"Initializing MOMP Benchmark Execution [{mode.upper()} MODE]")
    print("=" * 60)

    # Setup Earthdata authentication credentials
    os.environ.setdefault("EARTHDATA_AUTH", "netrc")
    try:
        config.setup_earthdata_auth()
    except Exception as e:
        print(f"Authentication warning: {e}")

    # Patch importlib metadata for uninstalled local package imports
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
        # Load configuration settings
        cfg = get_cfg()

        # Resolve dataset and path settings against project root
        cfg.pkg_dir = str(MOMP_PKG_DIR)
        cfg.work_dir = resolve_path(getattr(cfg, "work_dir", str(ROMP_ROOT)))
        cfg.obs_dir = resolve_path(getattr(cfg, "obs_dir", "data/external/ENACTS_regridded_025"))
        cfg.ref_model_dir = resolve_path(getattr(cfg, "ref_model_dir", "data/external/ENACTS"))
        cfg.nc_mask = resolve_path(getattr(cfg, "nc_mask", "data/external/jjas_100mm_rainfall_mask_0p25.nc"))

        if hasattr(cfg, "shpfile_dir") and cfg.shpfile_dir:
            cfg.shpfile_dir = resolve_path(cfg.shpfile_dir)

        cfg.dir_out = str(config.ROMP_OUT_DIR)
        cfg.dir_fig = str(config.ROMP_FIG_DIR)

        # Apply mode configuration
        configure_benchmark_mode(cfg, mode=mode)
        cfg.model_dir_list = tuple(resolve_path(d) for d in cfg.model_dir_list)

        # Disable parallel execution to prevent Jupyter deadlock issues
        cfg.parallel = True

        # Synchronize configuration into momp.params module state
        for attr in dir(cfg):
            if not attr.startswith("_"):
                setattr(params, attr, getattr(cfg, attr))
        params.parallel = True

        os.makedirs(cfg.dir_out, exist_ok=True)
        os.makedirs(cfg.dir_fig, exist_ok=True)

        print(f"Work Directory : {cfg.work_dir}")
        print(f"Package Dir    : {cfg.pkg_dir}")
        print(f"Obs Directory  : {cfg.obs_dir}")
        print(f"Model List     : {cfg.model_list}")
        print(f"Parallel Execution : {cfg.parallel}")
        print("-" * 60)

        # Execute driver pipeline script
        driver_path = MOMP_PKG_DIR / "driver.py"
        if not driver_path.exists():
            raise FileNotFoundError(f"Driver not found at: {driver_path}")

        sys.stdout.flush()
        
        # CRITICAL FIX: Sanitize sys.argv before running the driver
        with sanitized_driver_argv(driver_path, mode):
            runpy.run_path(str(driver_path), run_name="__main__")
        
        print(f"\nMOMP [{mode.upper()}] benchmark completed successfully.")

    except SystemExit as e:
        # Handle SystemExit(0) as success
        if e.code == 0 or e.code is None:
            print(f"\nMOMP [{mode.upper()}] benchmark completed successfully.")
        else:
            print(f"\nExecution failed with exit code: {e.code}")
            import traceback
            traceback.print_exc()
    except Exception as e:
        print(f"\nExecution failed: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Restore original importlib hooks
        importlib.metadata.version = original_version
        importlib.metadata.distribution = original_distribution


