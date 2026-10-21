"""
ui/pages/short_medium_benchmarking.py
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
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import xarray as xr
from PIL import Image, ImageDraw

import config

# ======================================================================
# Environment as it was at startup (after config.py loaded .env).
# Subprocesses are launched with this copy so nothing that runs inside
# the Streamlit process can leak into them.
# ======================================================================
_STARTUP_ENV: Dict[str, str] = dict(os.environ)

# ======================================================================
# Safe config access
# ======================================================================
PROJECT_ROOT = Path(getattr(config, "PROJECT_ROOT", Path.cwd())).resolve()

ROMP_DEMO_FIG_DIR = Path(
    getattr(
        config,
        "ROMP_DEMO_FIG_DIR",
        getattr(
            config,
            "ROMP_FIG_DIR",
            PROJECT_ROOT / "data" / "ROMP_OUT" / "et" / "figure",
        ),
    )
)
ROMP_DEMO_OUT_DIR = Path(
    getattr(
        config,
        "ROMP_DEMO_OUT_DIR",
        getattr(
            config,
            "ROMP_OUT_DIR",
            PROJECT_ROOT / "data" / "ROMP_OUT" / "et" / "output",
        ),
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
        "ADM1_EN", "ADM1_NAME", "ADM1_NM", "NAME_1", "ADM1", "admin1", "Region",
    ],
    "Admin2": [
        "ADM2_EN", "ADM2_NAME", "ADM2_NM", "NAME_2", "ADM2", "admin2", "Zone",
    ],
    "Admin3": [
        "ADM3_EN", "ADM3_NAME", "ADM3_NM", "NAME_3", "ADM3", "adm3_name",
        "admin3", "Woreda",
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


def render_benchmark_figure(path_str: str) -> None:
    st.image(path_str, width="stretch")
    st.caption(format_figure_label(path_str))


# ======================================================================
# Auto-detection helpers
# ======================================================================
def _extract_time_range_from_nc(nc_path):
    try:
        ds = config.open_dataset(nc_path, decode_cf=True, decode_times=True)   # local or gs://
    except Exception:
        try:
            ds = config.open_dataset(nc_path, decode_cf=False, decode_times=False)
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


def _detect_model_data_date_range(model_dir):
    if not model_dir or str(model_dir).strip() in {"", "."}:
        return None, None

    model_dir = str(model_dir)                 # keep gs:// intact (Path() would make it gs:/)
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
    n_each = 2 if remote else 5                # fewer files opened over the network
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


@st.cache_data(show_spinner=False, ttl=120)
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


@st.cache_data(show_spinner=False, ttl=120)
def discover_map_outputs(force_refresh: bool = False) -> List[Dict[str, Any]]:
    outputs: List[Dict[str, Any]] = []
    if not ROMP_DEMO_OUT_DIR.exists():
        return outputs

    for nc_path in sorted(ROMP_DEMO_OUT_DIR.rglob("*.nc")):
        first_var = None
        try:
            with xr.open_dataset(nc_path) as ds:
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


@st.cache_data(show_spinner=False, ttl=120)
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
):
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


@st.cache_data(show_spinner=False, ttl=120)
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


# ======================================================================
# Reference / Benchmark Outputs helpers
# ======================================================================
def classify_reference_asset(path: Path) -> str:
    s = str(path).lower()
    prob_keys = [
        "aifs_ens", "ens", "ensemble", "probabilistic", "gencast",
        "brier", "rps", "reliability", "spread", "rank",
    ]
    clim_keys = [
        "climatology", "baseline", "traditional",
    ]
    det_keys = [
        "aifs", "fuxi", "graphcast", "deterministic", "mae", "far", "mr",
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


def _onset_config_kwargs() -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in vars(config).items():
        if key.startswith("_") or not key.islower():
            continue
        if inspect.ismodule(value) or inspect.isclass(value) or callable(value):
            continue
        out[key] = str(value) if isinstance(value, Path) else value
    return out


def _get_onset_base_kwargs():
    _chirps = getattr(
        config,
        "CHIRPS_DIR",
        Path(config.PROJECT_ROOT) / "data" / "external" / "CHIRPS_IMERG",
    )
    CHIRPS_DIR = str(_chirps) if config.is_uri(_chirps) else Path(_chirps)   # gs:// stays a string
    base_kwargs = {
        "obs": "CHIRPS",
        "obs_dir": str(CHIRPS_DIR),
        "obs_file_pattern": ("{}.nc",),
        "obs_var": "RAINFALL",
        "ref_model_dir": str(CHIRPS_DIR),
        "ref_model_file_pattern": "{}.nc",
        "ref_model_var": "RAINFALL",
    }
    base_kwargs = {**_onset_config_kwargs(), **base_kwargs}
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


@st.cache_data(show_spinner=False)
def _available_years():
    base_kwargs, CHIRPS_DIR = _get_onset_base_kwargs()
    if not config.exists(CHIRPS_DIR):
        return list(range(2015, 2024))
    pattern = base_kwargs.get("obs_file_pattern", ("{}.nc",))
    pattern = pattern[0] if isinstance(pattern, (tuple, list)) else pattern
    present = {Path(f).name for f in config.glob(str(CHIRPS_DIR).rstrip("/") + "/*.nc")}   # one listing
    years = [y for y in range(1980, 2031) if pattern.format(y) in present]
    return years or list(range(2015, 2024))


@st.cache_data(show_spinner=False)
def _snap_to_grid(lat, lon, data_dir_str):
    files = config.glob(str(data_dir_str).rstrip("/") + "/*.nc")
    sample = files[0] if files else None
    if not sample:
        return lat, lon
    try:
        with config.open_dataset(sample) as ds:
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


@st.cache_data(show_spinner=False)
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
            Path(config.PROJECT_ROOT) / "data",
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
        NAME_COLS = (
            "woreda", "w_name", "wname", "woreda_name", "wor_name",
            "adm3_en", "adm3_name", "admin3name", "name",
        )
        ZONE_COLS = (
            "zone", "z_name", "zone_name", "adm2_en", "admin2name",
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


# Code run in a separate process for the onset analysis (started with `python -c`, so no extra
# file is needed). A separate process keeps anything the analysis changes (working folder,
# sys.argv, environment, matplotlib state) away from the Streamlit server.
_ONSET_WORKER_SOURCE = r"""
import os, pickle, sys, warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.getcwd())  # project root: the page starts this process there

payload = pickle.loads(Path(sys.argv[1]).read_bytes())
out_dir = Path(sys.argv[2])
out_dir.mkdir(parents=True, exist_ok=True)
sys.argv = [sys.argv[0]]  # ROMP may parse sys.argv

import config  # noqa: F401  (puts the ROMP folders on sys.path)
import momp.app.onset_time_series as ots

real_close = plt.close
plt.show = lambda *a, **k: None
plt.close = lambda *a, **k: None  # keep figures alive until they are saved

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    ots.obs_onset_analysis(
        year=payload["year"],
        lat_select=payload["lat_select"],
        lon_select=payload["lon_select"],
        **payload["kwargs"],
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
    """
    Run the onset analysis in a separate process and return its figures.
    """
    import pickle
    import tempfile

    base_kwargs, CHIRPS_DIR = _get_onset_base_kwargs()
    grid_lat, grid_lon = _snap_to_grid(lat, lon, str(CHIRPS_DIR))
    pngs: List[bytes] = []

    # Minimal environment isolation
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
    env["PYTHONFAULTHANDLER"] = "1"
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    env["HDF5_USE_FILE_LOCKING"] = "FALSE"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    env["VECLIB_MAXIMUM_THREADS"] = "1"
    env["BLIS_NUM_THREADS"] = "1"
    env["GOTO_NUM_THREADS"] = "1"
    env["MPLBACKEND"] = "Agg"
    env["MPLCONFIGDIR"] = str((PROJECT_ROOT / ".mplcache").resolve())
    Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    env["PYTHONDONTWRITEBYTECODE"] = "1"



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
            st.error(f"Could not start the onset analysis: {type(e).__name__}: {e}")
            return pngs, grid_lat, grid_lon

        if out_dir.exists():
            pngs = [f.read_bytes() for f in sorted(out_dir.glob("*.png"))]

        if proc.returncode != 0:
            err_lines = (proc.stderr or proc.stdout or "").strip().splitlines()
            st.error(
                "Error during onset analysis: "
                + (err_lines[-1] if err_lines else f"exit code {proc.returncode}")
            )
            with st.expander("Technical details"):
                st.code("\n".join(((proc.stdout or "") + "\n" + (proc.stderr or "")).splitlines()[-60:]))
        elif not pngs:
            st.warning("The onset analysis finished but produced no figures.")

    return pngs, grid_lat, grid_lon


# ======================================================================
# Execution bridge
# ======================================================================
def _locate_run_momp() -> Optional[Path]:
    for c in [PROJECT_ROOT / "run_momp.py", Path.cwd() / "run_momp.py"]:
        if c.exists():
            return c.resolve()
    return None


def _spawn_no_fork(argv, env, cwd, timeout, on_line=None):
    """
    Run `argv` in `cwd` and stream its combined stdout/stderr, line by line, to `on_line`.
    Returns (returncode, timed_out); returncode is negative when the process was killed by
    a signal (-11 = segmentation fault) and 124 on timeout.

    Why not subprocess.run/Popen: they fork() the Streamlit process and then exec(). On macOS a
    fork from this large multi-threaded process (numpy, xarray, matplotlib, ... loaded) can crash
    in the CHILD before exec - exit code -11 and no output at all, even for a plain shell - while
    the same command works in a terminal. Environment variables cannot help because they only
    take effect after exec. os.posix_spawn starts the new program directly and never runs code in
    a forked copy of this process.
    """
    import select
    import signal
    import time

    started = time.time()

    if not hasattr(os, "posix_spawn"):  # e.g. Windows, which has no fork problem
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
        (os.POSIX_SPAWN_OPEN, 0, os.devnull, os.O_RDONLY, 0),  # never inherit Streamlit's stdin
        (os.POSIX_SPAWN_DUP2, write_fd, 1),
        (os.POSIX_SPAWN_DUP2, write_fd, 2),
        (os.POSIX_SPAWN_CLOSE, read_fd),
        (os.POSIX_SPAWN_CLOSE, write_fd),
    ]
    # posix_spawn has no working-directory option, so let sh do `cd` and then exec the program
    # (the program replaces sh, so it runs exactly as if started from a terminal in `cwd`)
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
            if not chunk:  # EOF: the process (and anything it started) closed its output
                break

            buf += chunk
            *done, buf = buf.split(b"\n")
            for raw in done:
                if on_line:
                    on_line(raw.decode("utf-8", "replace").rstrip("\r"))

        if buf and on_line:
            on_line(buf.decode("utf-8", "replace").rstrip("\r"))
    except BaseException:
        # e.g. Streamlit interrupting the script because the user touched the page
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

    # ==================================================================
    # Same environment as the working version of this page. The extra band-aids
    # (thread limits, KMP_DUPLICATE_LIB_OK, HDF5 locking, forced MPLBACKEND, fork-safety
    # flags, faulthandler) are NOT set: the child gets exactly what the terminal run gets.
    env = os.environ.copy()

    env["PYTHONPATH"] = os.pathsep.join(
        [str(PROJECT_ROOT.resolve()), env.get("PYTHONPATH", "")]
    )

    env["MPLCONFIGDIR"] = str((PROJECT_ROOT / ".mplcache").resolve())
    Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    env["PYTHONUNBUFFERED"] = "1"  # only so the live log updates while the run is going

    # ==================================================================
    # Pass run_config as environment variables
    # ==================================================================
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
            env["ROMP_INIT_DAYS"] = ",".join(
                str(int(x)) for x in run_config.get("initialization_days", [])
            )
        env["ROMP_RUN_YEARS_CONCURRENTLY"] = (
            "1" if run_config.get("run_years_concurrently", True) else "0"
        )
        env["ROMP_ENSEMBLE_FORECAST"] = (
            "1" if run_config.get("ensemble_forecast", False) else "0"
        )
        if run_config.get("forecast_window"):
            env["ROMP_FORECAST_WINDOW"] = str(
                run_config["forecast_window"]
            ).replace(" days", "")
        if run_config.get("ground_truth"):
            env["ROMP_GROUND_TRUTH"] = str(run_config["ground_truth"])
        if run_config.get("benchmark_coverage"):
            env["ROMP_BENCHMARK_COVERAGE"] = str(run_config["benchmark_coverage"])

        # Advanced temporal & verification parameters (read by run_momp.apply_env_vars_to_cfg)
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
        if run_config.get("mok"):
            env["ROMP_MOK"] = str(run_config["mok"])
        if run_config.get("years"):
            env["ROMP_YEARS"] = str(run_config["years"])
        if run_config.get("years_clim"):
            env["ROMP_YEARS_CLIM"] = str(run_config["years_clim"])
        if run_config.get("fallback_date"):
            env["ROMP_FALLBACK_DATE"] = str(run_config["fallback_date"])

    cmd = [sys.executable, str(rmp), "--mode", mode]
    if model_key:
        cmd.extend(["--model", model_key])

    try:
        import time
        live = st.empty()
        lines: List[str] = []
        started = time.time()
        last_draw = 0.0
        timed_out = False

        def _on_line(line: str) -> None:
            nonlocal last_draw
            lines.append(line)
            now = time.time()
            if now - last_draw > 0.5:
                live.code(
                    f"[{int(now - started)}s elapsed]  {' '.join(cmd[-4:])}\n"
                    + "\n".join(lines[-20:])
                )
                last_draw = now

        returncode, timed_out = _spawn_no_fork(
            cmd, env, str(PROJECT_ROOT.resolve()), 1800, _on_line
        )

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
            "stderr": f"Could not run the benchmark process: {type(e).__name__}: {e!r}\n"
            + traceback.format_exc(),
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
    st.markdown(
        f'<div class="ibox {kind}">{text}</div>',
        unsafe_allow_html=True,
    )


def section_header(text: str, accent: str = "sky") -> None:
    st.markdown(
        f'<div class="sec-hd {accent}">'
        f'<span class="sec-hd-accent"></span>{text}'
        f"</div>",
        unsafe_allow_html=True,
    )


# Countries offered under "Benchmark coverage" as (flag, name), in display order.
BENCHMARK_COVERAGE_COUNTRIES = (
    ("🇸🇳", "Senegal"),
    ("🇷🇼", "Rwanda"),
    ("🇨🇴", "Colombia"),
    ("🇵🇪", "Peru"),
    ("🇧🇴", "Bolivia"),
    ("🇵🇭", "Philippines"),
    ("🇮🇩", "Indonesia"),
    ("🇵🇰", "Pakistan"),
    ("🇳🇬", "Nigeria"),
    ("🇪🇹", "Ethiopia"),
    ("🇰🇪", "Kenya"),
    ("🇸🇱", "Sierra Leone"),
    ("🇨🇱", "Chile"),
    ("🇧🇩", "Bangladesh"),
)
DEFAULT_BENCHMARK_COVERAGE = "Ethiopia"

# Countries that have data and a mask configured. config.py currently sets up Ethiopia only
# (region, JJAS mask, CHIRPS grid). Add a country here once its data is in place. Run
# Benchmark is disabled for the others; otherwise the run would silently use Ethiopia's data
# under another country's name.
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
                <div class="kpi-value" style="font-size:0.85rem;word-break:break-all;">
                    {runtime}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_variable_plot(ds: xr.Dataset, var_name: str, title_prefix: str) -> bool:
    if var_name not in ds:
        return False
    try:
        da = ds[var_name].compute()
        arr = np.asarray(da.values, dtype=float)
        if arr.ndim != 2 or not np.isfinite(arr).any():
            return False
        fig, ax = plt.subplots(figsize=(4, 3))
        im = ax.imshow(arr, origin="lower", cmap="viridis")
        ax.set_title(f"{title_prefix} - {var_name}", fontsize=10)
        fig.colorbar(im, ax=ax, shrink=0.8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        return True
    except Exception:
        return False


def render_shared_settings(ground_truth: str = "CHIRPS") -> Dict[str, Any]:
    section_header("Shared settings", "sky")
    st.markdown(
        "Optional settings applied to every selected model in this benchmark."
    )

    st.markdown("**EVENT DETECTION**")
    c1, c2, c3 = st.columns(3)
    wet_day_threshold = c1.number_input(
        "Wet-day threshold",
        min_value=0.0,
        value=20.0,
        step=0.5,
        format="%.1f",
        help="Default: 20 mm",
        key="shared_wet_day_threshold",
    )
    minimum_wet_day_rainfall = c2.number_input(
        "Minimum wet-day rainfall",
        min_value=0.0,
        value=1.0,
        step=0.1,
        format="%.1f",
        help="Default: 1 mm",
        key="shared_min_wet_day",
    )
    wet_spell_length = c3.number_input(
        "Wet spell length",
        min_value=1,
        value=3,
        format="%d",
        help="Default: 3 days",
        key="shared_wet_spell",
    )

    c4, c5, _ = st.columns(3)
    dry_spell_limit = c4.number_input(
        "Dry spell limit",
        min_value=0,
        value=7,
        format="%d",
        help="Default: 7 days",
        key="shared_dry_spell",
    )
    dry_spell_search_extension = c5.number_input(
        "Dry spell search extension",
        min_value=0,
        value=0,
        format="%d",
        help="Default: 0 days",
        key="shared_dry_ext",
    )

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
    area_mask_choice = m1.selectbox(
        "Area mask file",
        list(mask_options.keys()),
        index=0,
        help="Default: No mask",
        key="shared_area_mask",
    )
    area_mask_file = mask_options[area_mask_choice]
    if area_mask_choice == "Custom path":
        area_mask_file = m2.text_input(
            "Area mask file path",
            value="",
            key="shared_area_mask_custom",
        )

    t1, t2 = st.columns(2)
    default_threshold = ONSET_REFERENCE_DIR / "thresholds_df.csv"
    threshold_options = {
        "Dataset default": "",
        "Reference thresholds CSV": str(default_threshold),
        "Custom path": "__custom__",
    }
    onset_threshold_choice = t1.selectbox(
        "Onset threshold file",
        list(threshold_options.keys()),
        index=0,
        help="Default: Dataset default",
        key="shared_onset_thresh",
    )
    onset_threshold_file = threshold_options[onset_threshold_choice]
    if onset_threshold_choice == "Custom path":
        onset_threshold_file = t2.text_input(
            "Onset threshold file path",
            value="",
            key="shared_onset_custom",
        )

    b1, b2 = st.columns(2)
    baseline_forecast_choice = b1.selectbox(
        "Baseline forecast",
        ["climatology", "custom"],
        index=0,
        help="Default: climatology",
        key="shared_baseline_fc",
    )
    if baseline_forecast_choice == "custom":
        baseline_forecast = b2.text_input(
            "Baseline forecast name",
            value="climatology",
            key="shared_baseline_custom",
        )
    else:
        baseline_forecast = "climatology"

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

    pdf.section_title("Benchmark Inputs")
    pdf.key_value_lines([
        ("Ground Truth", context.get("ground_truth", "")),
        ("Benchmark Coverage", context.get("benchmark_coverage", "")),
        ("Forecast Window", context.get("forecast_window", "")),
    ])

    if shared_settings:
        pdf.section_title("Shared Settings")
        pdf.key_value_lines([
            (str(k).replace("_", " ").title(), v)
            for k, v in shared_settings.items()
        ])

    fair_plan = context.get("fair_plan")
    if fair_plan:
        pdf.section_title("Fair Comparison Plan")
        eval_years = fair_plan.get("evaluation_years", [])
        pdf.key_value_lines([
            ("Can Run", fair_plan.get("can_run", "")),
            ("Ground Truth", fair_plan.get("ground_truth", "")),
            ("Benchmark Coverage", fair_plan.get("benchmark_coverage", "")),
            ("Climatology Policy", fair_plan.get("climatology_policy", "")),
            ("Common Year Min", fair_plan.get("common_year_min", "")),
            ("Common Year Max", fair_plan.get("common_year_max", "")),
            ("Evaluation Years", ", ".join(str(y) for y in eval_years)),
        ])
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

    uncertainty_df = context.get("uncertainty_df")
    if uncertainty_df is not None and not uncertainty_df.empty:
        pdf.section_title("Metric Uncertainty")
        pdf.dataframe_table(uncertainty_df)
    significance_df = context.get("significance_df")
    if significance_df is not None and not significance_df.empty:
        pdf.section_title("Significance vs Baseline")
        pdf.dataframe_table(significance_df)

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


# ======================================================================
# Main Page
# ======================================================================
def page() -> None:
    _inject_theme_css()

    presets = get_benchmark_presets()
    st.markdown(
        '<div class="page-title">Forecast Benchmarking</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="page-subtitle">'
        "Evaluate AI weather model skill against observational references "
        "across the Ethiopian highlands."
        "</div>",
        unsafe_allow_html=True,
    )

    last_run = st.session_state.get("benchmark_last_run")
    status_text = (
        "Completed"
        if last_run and last_run.get("ok")
        else ("Failed" if last_run else "Idle")
    )
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
        bi2.caption(
            f"⚠️ No data or mask is configured for {benchmark_coverage} yet, "
            "so Run Benchmark is disabled."
        )
    forecast_window = bi3.selectbox(
        "Forecast window",
        ["1-15 days", "16-30 days"],
    )

    # Models
    section_header("Models", "teal")
    model_registry = detect_benchmark_models()
    active_rows: List[Dict[str, Any]] = []

    if not model_registry:
        info_box("No forecast model data was detected.", "amber")
    else:
        mrr = {r["_row_key"]: r for r in model_registry}
        category_options = ["All", "Deterministic", "Probabilistic"]
        selected_category_filter = st.radio(
            "Filter by category",
            category_options,
            horizontal=True,
            key="model_category_filter",
        )
        if selected_category_filter == "All":
            visible_registry = model_registry
        else:
            visible_registry = [
                r
                for r in model_registry
                if r.get("Category") == selected_category_filter
            ]

        if not visible_registry:
            info_box(
                f"No {selected_category_filter} models detected in the data.",
                "amber",
            )
            active_rows = []
        else:
            table = st.dataframe(
                pd.DataFrame(visible_registry)[
                    ["Model", "Category", "Type", "Coverage", "Date Range"]
                ],
                hide_index=True,
                selection_mode="multi-row",
                on_select="rerun",
                key="model_table_selection",
                width="stretch",
            )
            try:
                sel_idx = table.selection.rows
            except Exception:
                sel_idx = (
                    st.session_state.get("model_table_selection", {})
                    .get("selection", {})
                    .get("rows", [])
                )
            sel_keys = [
                visible_registry[int(i)]["_row_key"]
                for i in sel_idx
                if 0 <= int(i) < len(visible_registry)
            ]
            active_rows = [mrr[k] for k in sel_keys if k in mrr]
            st.session_state.selected_model_rows = sel_keys
            st.caption(
                f"**{len(active_rows)}** selected. "
                "Selection is applied automatically."
            )
            det_count = sum(
                1 for r in active_rows if r.get("Category") == "Deterministic"
            )
            prob_count = sum(
                1 for r in active_rows if r.get("Category") == "Probabilistic"
            )
            if active_rows:
                st.markdown(
                    f"""
                    <div style="display:flex; flex-wrap:wrap; gap:0.5rem; margin:0.5rem 0;">
                        <span class="metric-pill">
                            Deterministic: <strong>{det_count}</strong>
                        </span>
                        <span class="metric-pill">
                            Probabilistic: <strong>{prob_count}</strong>
                        </span>
                        <span class="metric-pill">
                            Total: <strong>{len(active_rows)}</strong>
                        </span>
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
            st.info(
                "Fair-comparison window automatically set to "
                f"{cs:%Y-%m-%d} to {ce:%Y-%m-%d}."
            )
        for row in active_rows:
            rk = row["_row_key"]
            avs, ave = row["_start_date"], row["_end_date"]
            des, dee = (
                (max(cs, avs), min(ce, ave))
                if cs and ce
                else (avs, ave)
            )
            if des > dee:
                des, dee = avs, ave
            with st.expander(
                f"{row['Model']} - available {row['Date Range']}"
            ):
                c1, c2 = st.columns(2)
                eval_start = c1.date_input(
                    "Evaluation Start",
                    value=des,
                    min_value=avs,
                    max_value=ave,
                    key=f"es_{rk}",
                )
                eval_end = c2.date_input(
                    "Evaluation End",
                    value=dee,
                    min_value=avs,
                    max_value=ave,
                    key=f"ee_{rk}",
                )
                c3, c4 = st.columns(2)
                bsy = c3.number_input(
                    "Baseline Start Year",
                    min_value=avs.year,
                    max_value=ave.year,
                    value=des.year,
                    key=f"bsy_{rk}",
                )
                bey = c4.number_input(
                    "Baseline End Year",
                    min_value=avs.year,
                    max_value=ave.year,
                    value=dee.year,
                    key=f"bey_{rk}",
                )
                init_days = st.multiselect(
                    "Initialization Days",
                    list(range(16)),
                    default=[0, 3],
                    key=f"id_{rk}",
                )
                # Advanced temporal & verification parameters
                with st.expander("Advanced Temporal & Verification Settings"):
                    st.markdown("**Temporal Settings**")
                    c5, c6 = st.columns(2)
                    init_type = c5.selectbox(
                        "Init Type",
                        ["weekly", "daily", "monthly"],
                        index=0,
                        key=f"init_type_{rk}",
                    )
                    date_filter_year = c6.number_input(
                        "Date Filter Year",
                        min_value=2000,
                        max_value=2030,
                        value=2024,
                        key=f"date_filter_year_{rk}",
                    )
                    
                    st.markdown("**Evaluation & Verification Parameters**")
                    c7, c8 = st.columns(2)
                    verification_window_list = c7.text_input(
                        "Verification Window List",
                        value="((1, 15), (16, 30))",
                        help="Format: tuple of tuples, e.g., ((1, 15), (16, 30))",
                        key=f"vwl_{rk}",
                    )
                    tolerance_days_list = c8.text_input(
                        "Tolerance Days List",
                        value="(3, 5)",
                        help="Format: tuple of ints, e.g., (3, 5)",
                        key=f"tdl_{rk}",
                    )
                    
                    c9, c10 = st.columns(2)
                    max_forecast_day = c9.number_input(
                        "Max Forecast Day",
                        min_value=1,
                        max_value=60,
                        value=30,
                        key=f"max_forecast_day_{rk}",
                    )
                    
                    st.markdown("**Reserved for Future Versions**")
                    c11, c12 = st.columns(2)
                    mok = c11.text_input(
                        "MOK",
                        value="(5, 15)",
                        key=f"mok_{rk}",
                    )
                    years = c12.text_input(
                        "Years",
                        value="None",
                        key=f"years_{rk}",
                    )
                    
                    c13, c14 = st.columns(2)
                    years_clim = c13.text_input(
                        "Years Clim",
                        value="None",
                        key=f"years_clim_{rk}",
                    )
                    fallback_date = c14.text_input(
                        "Fallback Date",
                        value="None",
                        key=f"fallback_date_{rk}",
                    )
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
                    "mok": mok,
                    "years": years,
                    "years_clim": years_clim,
                    "fallback_date": fallback_date,
                }

    # Shared settings
    shared_settings = render_shared_settings(ground_truth)

    # Phase 2 & 3
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
                    st.dataframe(
                        inventory_to_dataframe(m_inv + o_inv),
                        hide_index=True,
                        width="stretch",
                    )
                except Exception as e:
                    st.warning(f"Could not load inventory: {e}")

        if PHASE_3_AVAILABLE and active_rows:
            with st.expander("Fair Comparison Plan", expanded=False):
                try:
                    sel_keys = [r["_row_key"] for r in active_rows]
                    plan = build_fair_comparison_plan(
                        selected_model_keys=sel_keys,
                        ground_truth=ground_truth,
                        forecast_window=forecast_window,
                        benchmark_coverage=benchmark_coverage,
                    )
                    if plan.can_run:
                        st.success("Fair comparison plan is valid.")
                    else:
                        st.error("Fair comparison plan has blocking issues.")
                    for issue in plan.issues:
                        st.error(issue)
                    for warning in plan.warnings:
                        st.warning(warning)
                    if plan.common_year_min and plan.common_year_max:
                        st.info(
                            "Common Evaluation Window: "
                            f"{plan.common_year_min} - {plan.common_year_max}"
                        )
                    st.dataframe(
                        plan_to_dataframe(plan),
                        hide_index=True,
                        width="stretch",
                    )
                except Exception as e:
                    st.warning(f"Could not build fair plan: {e}")

    # Run section
    st.markdown('<div class="run-section">', unsafe_allow_html=True)
    c1, c2 = st.columns([1, 2])
    run_clicked = c1.button(
        "Run Benchmark",
        key="btn_run",
        type="primary",
        width="stretch",
        disabled=not coverage_ready,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if run_clicked and not active_rows:
        st.warning(
            "No model is selected. Click one or more rows in the model table above, "
            "then press Run Benchmark."
        )

    if run_clicked and active_rows:
        runs: List[Dict[str, Any]] = []
        with st.spinner("Running benchmark... (this can take several minutes per model)"):
            for row in active_rows:
                rk, ek = row["_row_key"], row["_exec_key"]
                rc = dict(model_run_windows.get(rk, {}))
                rc.update(shared_settings)
                rc.update(
                    {
                        "ground_truth": ground_truth,
                        "benchmark_coverage": benchmark_coverage,
                        "forecast_window": forecast_window,
                    }
                )
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
            "stdout": "",
            "stderr": "",
            "runs": runs,
        }
        st.rerun()

    if last_run:
        _runs = last_run.get("runs") or [last_run]
        _failed = [r for r in _runs if not r.get("ok")]
        if not _failed:
            st.success(f"Benchmark finished for {len(_runs)} model(s).")
        else:
            st.error(
                f"{len(_failed)} of {len(_runs)} model run(s) failed. "
                "Open the log below for details."
            )
        for _i, _r in enumerate(_runs):
            if _r.get("ok"):
                continue  # successful runs print nothing; only failures show a log

            _name = _r.get("model") or _r.get("mode", "run")
            _icon = "✅" if _r.get("ok") else "❌"
            with st.expander(f"{_icon} {_name} - log", expanded=not _r.get("ok")):
                _text = "\n".join(
                    part for part in (_r.get("stdout", ""), _r.get("stderr", "")) if part
                )
                _lines = _text.splitlines()
                if _r.get("command"):
                    st.caption(_r["command"])
                _rc = _r.get("returncode")
                if _rc is not None:
                    st.caption(f"Exit code: {_rc}")
                if not _r.get("ok") and not _text.strip():
                    _hint = {
                        -9: "The process was killed (SIGKILL), usually out of memory or stopped by the system.",
                        -11: "The process crashed (segmentation fault) before printing anything.",
                        -15: "The process was terminated. This happens when you click or change something "
                             "on the page while a run is in progress, so wait for it to finish.",
                        124: "The run exceeded the 30 minute limit.",
                    }.get(_rc, "The process ended without printing anything.")
                    st.warning(
                        f"{_hint} Run the command above in a terminal to see the real error, "
                        "and check the terminal where Streamlit is running."
                    )
                if not _r.get("ok"):
                    _pat = re.compile(
                        r"error|traceback|exception|failed|not found|no such file|"
                        r"cannot|unable|invalid|missing",
                        re.IGNORECASE,
                    )
                    _key_lines = list(dict.fromkeys(l for l in _lines if _pat.search(l)))[:40]
                    if _key_lines:
                        st.markdown("**Key lines**")
                        st.code("\n".join(_key_lines))
                st.markdown("**Last 80 lines**")
                st.code("\n".join(_lines[-80:]) or "(no output)")
                if _text:
                    st.download_button(
                        "Download full log",
                        data=_text,
                        file_name=f"benchmark_{_name}.log",
                        key=f"dl_log_{_i}",
                    )

    # Results
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
            wl = c2.selectbox(
                "Lead Time",
                [w for w in ["Days 1–15", "Days 16–30"] if w in all_w]
                + sorted(all_w - {"Days 1–15", "Days 16–30"}),
                key="m_wl",
            )
            m_opts = sorted({o["model_label"] for o in map_outputs if wl in o["windows"]})
            model_label = c3.selectbox("Model", m_opts, key="m_mod")
            sf_avail = _map_load_admin_shapefile() is not None
            al = c4.selectbox(
                "Subregion level",
                ["All", "Admin1", "Admin2", "Admin3"] if sf_avail else ["All"],
                disabled=not sf_avail,
                key="m_al",
            )
            an = "All"
            if al != "All":
                ao = map_admin_options(al)
                if ao:
                    an = st.selectbox("Subregion", ["All"] + ao, key="m_an")

            sel_out = next(
                (o for o in map_outputs if o["model_label"] == model_label and wl in o["windows"]),
                None,
            )
            if sel_out:
                ual = "" if al == "All" else al
                uan = "" if an == "All" else an
                base_da = load_map_da_cached(sel_out["path"], ml, wl, ual, uan)
                if base_da is not None:
                    mm = MAP_METRICS[ml]
                    sub_tabs = st.tabs(["Values", "Skill", "Difference", "Swipe Compare"])
                    with sub_tabs[0]:
                        fig = plot_map(base_da, f"{ml} - {model_label} - {wl}", cmap=mm["cmap"], unit=mm["unit"])
                        if fig:
                            st.pyplot(fig)
                            plt.close(fig)
                        df = compute_model_comparison_cached(ml, wl, ual, uan)
                        if not df.empty:
                            if ml in {"False Alarm Rate", "Miss Rate"}:
                                df[ml] = df[ml].round(1)
                                df = df.rename(columns={ml: f"{ml} (%)"})
                            else:
                                df[ml] = df[ml].round(1)
                                df = df.rename(columns={ml: f"{ml} (days)"})
                            df["Selected"] = np.where(df["Model"] == model_label, "Yes", "")
                            st.dataframe(
                                df[["Selected", "Model"] + [c for c in df.columns if c not in ["Selected", "Model"]]],
                                width="stretch",
                                hide_index=True,
                            )
                    with sub_tabs[1]:
                        thr = mm["threshold"]
                        skill_da = (base_da <= thr).astype(float).where(base_da.notnull())
                        fig = plot_map(skill_da, f"Skill - {ml}", cmap="RdYlGn", vmin=0, vmax=1)
                        if fig:
                            st.pyplot(fig)
                            plt.close(fig)
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
                                cda = load_map_da_cached(cout["path"], ml, cw, ual, uan)
                                if cda is not None:
                                    dda = subtract_maps(base_da, cda)
                                    if dda is not None:
                                        fig = plot_map(dda, f"Difference - {model_label} minus {cm}", diverging=True)
                                        if fig:
                                            st.pyplot(fig)
                                            plt.close(fig)
                    with sub_tabs[3]:
                        sc1, sc2, sc3 = st.columns(3)
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
                                rda = load_map_da_cached(rout["path"], ml, rw, ual, uan)
                                if rda is not None:
                                    sp = st.slider("Drag to swipe", 0, 100, 50, key="s_sp")
                                    vmin, vmax = _shared_vmin_vmax(base_da.values, rda.values)
                                    lf = plot_map_image(base_da, cmap=mm["cmap"], vmin=vmin, vmax=vmax)
                                    rf = plot_map_image(rda, cmap=mm["cmap"], vmin=vmin, vmax=vmax)
                                    if lf and rf:
                                        lb = fig_to_bytes(lf, False)
                                        rb = fig_to_bytes(rf, False)
                                        comp = create_swipe_comparison_image(lb, rb, sp)
                                        if comp:
                                            lc1, lc2 = st.columns(2)
                                            lc1.markdown(f"**Left:** {model_label} - {wl}")
                                            lc2.markdown(f"**Right:** {rm} - {rw}")
                                            st.image(comp, width="stretch")

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
            "Green = better (lower), Red = worse (higher). Bold = best value."
        )
        stats_df = compute_metric_distribution_stats()
        if stats_df.empty:
            info_box("No distribution statistics could be computed from the available outputs.", "amber")
        else:
            metric_order = {"False Alarm Rate": 0, "Miss Rate": 1, "Mean Absolute Error": 2}
            model_order = {"AIFS Single v2": 0, "FuXi": 1, "GraphCast": 2, "AIFS Ensemble v2": 3, "GenCast": 4, "Traditional Climatology": 5}
            window_order = {"Days 1–15": 0, "Days 16–30": 1}
            stats_df = stats_df.copy()
            stats_df["metric_sort"] = stats_df["Metric"].map(lambda m: metric_order.get(m, 999))
            stats_df["model_sort"] = stats_df["Model"].map(lambda m: model_order.get(m, 999))
            stats_df["window_sort"] = stats_df["Lead Window"].map(lambda w: window_order.get(w, 999))
            stats_df = stats_df.sort_values(["model_sort", "window_sort", "metric_sort"])
            stat_cols = ["Mean", "Min", "Median", "P75", "P90", "Max"]
            ranges = {}
            for (metric, window), g in stats_df.groupby(["Metric", "Lead Window"], sort=False):
                for stat in stat_cols:
                    vals = pd.to_numeric(g[stat], errors="coerce").dropna()
                    if not vals.empty:
                        ranges[(metric, window, stat)] = (float(vals.min()), float(vals.max()))
            html_parts = []
            for (model_label, window_label), group_df in stats_df.groupby(["Model", "Lead Window"], sort=False):
                display_model = "Climatology" if model_label == "Traditional Climatology" else model_label
                title_model = display_model.replace(" ", "_").upper()
                title_window = window_label.upper()
                html_parts.append("<div style='margin-bottom:1.75rem;'>")
                html_parts.append(
                    "<h4 style='margin:0 0 0.25rem 0; font-size:1rem; font-weight:700; color:#0F172A;'>"
                    f"{title_model} - {title_window}</h4>"
                )
                tolerance_days = WINDOW_TOLERANCE_DAYS.get(window_label)
                if tolerance_days is not None:
                    html_parts.append(
                        "<div style='font-size:0.8rem; color:#64748B; margin-bottom:0.5rem;'>"
                        f"+/-{tolerance_days} day tolerance</div>"
                    )
                html_parts.append(
                    '<table class="all-metrics-table"><thead><tr>'
                    "<th>METRIC</th><th>MEAN</th><th>MIN</th><th>MEDIAN</th>"
                    "<th>P75</th><th>P90</th><th>MAX</th><th>UNIT</th>"
                    "</tr></thead><tbody>"
                )
                for _, row in group_df.iterrows():
                    metric = row["Metric"]
                    is_fraction = bool(row["fraction"])
                    html_parts.append(f"<tr><td>{metric}</td>")
                    for stat in stat_cols:
                        value = row[stat]
                        style = ""
                        if pd.notna(value):
                            range_key = (metric, window_label, stat)
                            if range_key in ranges:
                                best, worst = ranges[range_key]
                                if worst == best:
                                    style = "background-color:rgba(148,163,184,0.12);"
                                else:
                                    normalized = max(0.0, min(1.0, (float(value) - best) / (worst - best)))
                                    r = int(34 + (239 - 34) * normalized)
                                    g = int(197 + (68 - 197) * normalized)
                                    b = int(94 + (68 - 94) * normalized)
                                    alpha = 0.10 + (0.30 * normalized)
                                    style = f"background-color:rgba({r},{g},{b},{alpha:.2f});"
                                    if abs(float(value) - best) <= 1e-6:
                                        style += "font-weight:700;"
                        formatted = _format_distribution_value(value, is_fraction)
                        html_parts.append(f"<td style='{style}'>{formatted}</td>")
                    unit_label = "fraction" if is_fraction else "days"
                    html_parts.append(f"<td>{unit_label}</td></tr>")
                html_parts.append("</tbody></table></div>")
            st.markdown("".join(html_parts), unsafe_allow_html=True)

    # Tab 3: Benchmark Outputs
    with res_tabs[2]:
        st.markdown("MOMP benchmark output figures and datasets.")
        momp_figures = collect_reference_figures()
        if not momp_figures:
            info_box("No MOMP output figures found. Run a benchmark first.", "amber")
        else:
            cat_accent = {"Deterministic AI": "teal", "Probabilistic AI": "violet", "Climatology / Baseline": "amber", "Other": "sky"}
            for cat in REFERENCE_CATEGORY_ORDER:
                paths = momp_figures.get(cat, [])
                if not paths:
                    continue
                type_groups: Dict[str, List[str]] = {}
                for path_str in paths:
                    ftype = classify_figure_type(Path(path_str).name)
                    type_groups.setdefault(ftype, []).append(path_str)
                section_header(f"{cat} ({len(paths)} figures)", cat_accent.get(cat, "sky"))
                for ftype, type_paths in type_groups.items():
                    st.markdown(
                        f"""<div style="font-weight:600;font-size:0.9rem;color:var(--color-slate-700);margin:0.75rem 0 0.5rem;">{ftype} ({len(type_paths)})</div>""",
                        unsafe_allow_html=True,
                    )
                    if len(type_paths) == 1:
                        render_benchmark_figure(type_paths[0])
                    elif len(type_paths) == 2:
                        cols = st.columns(2)
                        for i, col in enumerate(cols):
                            with col:
                                render_benchmark_figure(type_paths[i])
                    else:
                        for i in range(0, len(type_paths), 2):
                            cols = st.columns(2)
                            for j, col in enumerate(cols):
                                idx = i + j
                                if idx < len(type_paths):
                                    with col:
                                        render_benchmark_figure(type_paths[idx])

        section_header("NetCDF Data Explorer", "sky")
        nc_files = sorted(str(p.resolve()) for p in ROMP_DEMO_OUT_DIR.rglob("*.nc"))
        if nc_files:
            sel = st.selectbox("Dataset", nc_files, format_func=format_nc_label)
            try:
                ds = _safe_open_dataset(sel)
                vars_list = [v for v in ds.data_vars if "time" not in v.lower()]
                if vars_list:
                    vn = st.selectbox("Variable", vars_list)
                    _render_variable_plot(ds, vn, format_nc_label(sel))
            except Exception:
                info_box("Could not open dataset.", "amber")
        else:
            info_box("No reference datasets available.", "amber")

    # Tab 4: Onset Analysis
    if OTS_AVAILABLE:
        with res_tabs[tab_labels.index("Onset Analysis")]:
            section_header("Onset Time Series Analysis", "teal")
            mode = st.radio("Location Selection Mode", ["Coordinates", "Woreda"], horizontal=True, key="ots_mode")
            years = _available_years()
            year = st.selectbox("Year", years, index=years.index(2021) if 2021 in years else len(years) - 1, key="ots_year")
            lat, lon, label = None, None, None
            if mode == "Coordinates":
                c1, c2 = st.columns(2)
                lat = c1.number_input("Latitude", value=8.5, step=0.25, format="%.4f", key="ots_lat")
                lon = c2.number_input("Longitude", value=39.2, step=0.25, format="%.4f", key="ots_lon")
            else:
                shp_path = _find_shapefile_path()
                if shp_path:
                    woredas = _load_woredas_cached(shp_path)
                    if woredas:
                        search_q = st.text_input("Filter Woreda", "", key="ots_search")
                        filtered_woredas = [w for w in woredas.keys() if search_q.lower() in w.lower()]
                        selected_woreda = st.selectbox("Select Woreda", filtered_woredas, key="ots_woreda")
                        if selected_woreda:
                            lat, lon = woredas[selected_woreda]
                            label = selected_woreda
                    else:
                        st.warning("No woredas found in the shapefile.")
                else:
                    st.warning("Shapefile not found. Please set config.WOREDA_SHP or ensure a .shp file is in the data directory.")

            run_ots_clicked = st.button("Run Onset Analysis", type="primary", key="btn_run_ots")
            if run_ots_clicked and lat is not None and lon is not None:
                with st.spinner("Running onset analysis..."):
                    pngs, grid_lat, grid_lon = _analyze_onset_streamlit(year, lat, lon)
                    st.session_state["onset_pngs"] = pngs
                    st.session_state["onset_meta"] = {
                        "year": year, "lat": lat, "lon": lon,
                        "label": label, "grid_lat": grid_lat, "grid_lon": grid_lon,
                    }
            if "onset_pngs" in st.session_state and st.session_state["onset_pngs"]:
                meta = st.session_state["onset_meta"]
                st.info(
                    f"**Results for {meta.get('label') or 'Selected Location'}** | "
                    f"Year: {meta['year']} | "
                    f"Snapped to Grid: Lat {meta['grid_lat']:.4f}, Lon {meta['grid_lon']:.4f}"
                )
                for png_bytes in st.session_state["onset_pngs"]:
                    st.image(png_bytes, width="stretch")

    # Tab 5: Skill vs Baseline
    if PHASE_4_AVAILABLE:
        with res_tabs[tab_labels.index("Skill vs Baseline")]:
            section_header("Skill versus Climatology", "green")
            try:
                scores = collect_scores_from_momp_outputs(ROMP_DEMO_OUT_DIR)
                if not scores.empty:
                    skill_df = compute_skill_vs_baseline(scores, baseline_model="Climatology")
                    if not skill_df.empty:
                        st.dataframe(skill_df, hide_index=True, width="stretch")
                    else:
                        info_box("No climatology baseline found for skill comparison.", "amber")
                else:
                    info_box("Run a benchmark to generate verification scores.", "amber")
            except Exception as e:
                info_box(f"Verification engine error: {e}", "rose")

    # Tab 6: Uncertainty
    if PHASE_5_AVAILABLE:
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
                st.dataframe(st.session_state["unc_df"], hide_index=True, width="stretch")
                st.markdown("**Significance vs Baseline**")
                st.dataframe(st.session_state["sig_df"], hide_index=True, width="stretch")

    # Tab 7: Benchmark Report & Export
    if "Benchmark Report & Export" in tab_labels:
        with res_tabs[tab_labels.index("Benchmark Report & Export")]:
            section_header("Benchmark Report & Export", "violet")
            st.markdown(
                "Generate a complete scientific PDF report including benchmark inputs, "
                "model configuration, fair-comparison plan, verification scores, "
                "uncertainty results, and benchmark output figures."
            )
            st.markdown("### Executive Summary")
            c1, c2, c3, c4 = st.columns(4)
            det_count = sum(1 for r in active_rows if r.get("Category") == "Deterministic")
            prob_count = sum(1 for r in active_rows if r.get("Category") == "Probabilistic")
            c1.metric("Models Evaluated", len(active_rows))
            c2.metric("Ground Truth", ground_truth)
            c3.metric("Forecast Window", forecast_window)
            c4.metric("Baseline", shared_settings.get("baseline_forecast", "climatology"))

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
            if not PDF_AVAILABLE:
                st.error("PDF generation requires `fpdf2`. Install it with: `pip install fpdf2`")
            else:
                st.markdown("### Download Complete Report")
                generate_clicked = st.button(
                    "Generate Complete PDF Report", type="primary", width="stretch",
                    key="btn_generate_pdf_report",
                )
                if generate_clicked:
                    with st.spinner("Generating complete PDF report..."):
                        try:
                            scores_df = pd.DataFrame()
                            if PHASE_4_AVAILABLE:
                                try:
                                    scores_df = collect_scores_from_momp_outputs(ROMP_DEMO_OUT_DIR)
                                except Exception:
                                    scores_df = pd.DataFrame()
                            try:
                                distribution_stats = compute_metric_distribution_stats()
                            except Exception:
                                distribution_stats = pd.DataFrame()
                            try:
                                report_figures = collect_reference_figures()
                            except Exception:
                                report_figures = {}
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
                            pdf_bytes = generate_benchmark_pdf_report(report_context)
                            st.session_state["benchmark_pdf_bytes"] = pdf_bytes
                            st.session_state["benchmark_pdf_file"] = (
                                f"benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                            )
                            st.success("PDF report generated successfully.")
                        except Exception as e:
                            st.error(f"PDF generation failed: {e}")

                pdf_bytes = st.session_state.get("benchmark_pdf_bytes")
                pdf_file = st.session_state.get("benchmark_pdf_file")
                if pdf_bytes and pdf_file:
                    st.download_button(
                        "Download Complete PDF Report",
                        data=pdf_bytes,
                        file_name=pdf_file,
                        mime="application/pdf",
                        key="btn_download_pdf_report",
                    )


if __name__ == "__main__":
    page()