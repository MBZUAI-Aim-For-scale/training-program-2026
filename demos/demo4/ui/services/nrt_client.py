# ui/services/nrt_client.py
"""
Near‑Real‑Time Monitoring Page
===============================
Split into two main tabs: Wind (MJO + wind diagnostics) and RF (Rainfall).
"""
from __future__ import annotations

import re
import json
import logging
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import requests
import xarray as xr
import geopandas as gpd
from shapely.ops import unary_union
from shapely.geometry import Point

from config import (
    STATION_KEY, MJO_ETH_IMPACT,
    earthdata_auth_status,
    APP_DIR, WRF_MASK_PATH, WRF_SHAPEFILE_PATH
)
from data.loader import DS_BY_KEY, COORDS_BY_KEY

# ---- IMERG fallback ----
try:
    from data.imerg import fetch_imerg_nrt
    IMERG_AVAILABLE = True
except ImportError:
    IMERG_AVAILABLE = False

# ---- NRT cards & sources ----
try:
    from nrt.cards import nrt_card, render_image_pair
    from nrt.sources import NRT_SOURCES
except ImportError:
    def nrt_card(col, key, source):
        col.info(f"NRT card for {key} not available")
    def render_image_pair(*args, **kwargs):
        st.warning("Image pair rendering unavailable")
    NRT_SOURCES = {}

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------
ETHIOPIA_BOUNDS = (33.0, 3.0, 48.0, 15.0)
MSWEP_DATA_DIR = APP_DIR / "data" / "MSWEP"

# ---------------------------------------------------------------------
# Cached resource loaders (shapefile, mask) – safe with @st.cache_resource
# ---------------------------------------------------------------------
@st.cache_resource
def load_ethiopia_shapefile():
    if not WRF_SHAPEFILE_PATH.exists():
        return None
    try:
        gdf = gpd.read_file(WRF_SHAPEFILE_PATH)
        return gdf.to_crs("EPSG:4326") if gdf.crs else gdf.set_crs("EPSG:4326")
    except Exception as e:
        st.error(f"Failed to load shapefile: {e}")
        return None

@st.cache_resource
def load_ethiopia_mask():
    if not WRF_MASK_PATH.exists():
        return None
    try:
        with xr.open_dataset(WRF_MASK_PATH, engine="netcdf4") as ds:
            mask_var = next((v for v in ds.data_vars if v.lower() == "mask"), list(ds.data_vars)[0])
            return ds[mask_var].load()
    except Exception as e:
        logger.warning(f"Could not load mask: {e}")
        return None

# ---------------------------------------------------------------------
# Cached MSWEP utilities – only hashable arguments
# ---------------------------------------------------------------------
@st.cache_data(ttl=3600)
def get_mswep_files_cached(data_dir: Path) -> List[Dict[str, Any]]:
    if not data_dir.exists():
        return []
    files = [f for f in data_dir.glob("*.nc") if not f.name.endswith('.partial')]
    info_list = []
    for f in files:
        stem = f.stem
        parts = stem.split('.')
        if len(parts) == 2:
            yyddd, hour_str = parts
            hour = int(hour_str)
        else:
            yyddd = parts[0]
            hour = 0
        year = int(yyddd[:4])
        doy = int(yyddd[4:7])
        date = datetime(year, 1, 1) + timedelta(days=doy-1, hours=hour)
        info_list.append({
            "year": year, "doy": doy, "hour": hour,
            "datetime": date, "filename": f.name, "path": f
        })
    info_list.sort(key=lambda x: x["datetime"])
    return info_list

get_mswep_files = get_mswep_files_cached

@st.cache_data(ttl=3600)
def load_mswep_metadata():
    files_info = get_mswep_files(MSWEP_DATA_DIR)
    if not files_info:
        return None, None, None, None, None, None
    with xr.open_dataset(files_info[-1]["path"], engine="h5netcdf") as sample_ds:
        precip_var = next((v for v in ["precip", "precipitation", "rain", "tp"] if v in sample_ds.data_vars), None)
    day_groups = {}
    for info in files_info:
        day_groups.setdefault((info["year"], info["doy"]), []).append(info)
    sorted_days = sorted(day_groups.keys())
    latest_day_files = day_groups[sorted_days[-1]]
    latest_date = latest_day_files[0]["datetime"].strftime("%Y-%m-%d")
    return files_info, precip_var, day_groups, sorted_days, latest_day_files, latest_date

@st.cache_data(ttl=3600)
def compute_recent_mswep_signal_cached(lat0: float, lon0: float, days: int = 7):
    files_info = get_mswep_files(MSWEP_DATA_DIR)
    if not files_info:
        return None
    with xr.open_dataset(files_info[-1]["path"], engine="h5netcdf") as sample_ds:
        precip_var = next((v for v in ["precip", "precipitation", "rain", "tp"] if v in sample_ds.data_vars), None)
    if precip_var is None:
        return None
    end_date = files_info[-1]["datetime"]
    start_date = end_date - timedelta(days=days - 1)
    day_groups = {}
    for info in files_info:
        day_groups.setdefault((info["year"], info["doy"]), []).append(info)
    selected_files = []
    for (year, doy), file_list in day_groups.items():
        file_date = datetime(year, 1, 1) + timedelta(days=doy - 1)
        if start_date <= file_date <= end_date:
            selected_files.extend(file_list)
    if not selected_files:
        return None
    # Load shapefile/mask here (cached via resources)
    gdf_eth = load_ethiopia_shapefile()
    eth_poly = unary_union(gdf_eth.geometry) if gdf_eth is not None else None
    mask_da = load_ethiopia_mask()
    total_rain = 0.0
    valid_steps = 0
    snapped_lat = snapped_lon = None
    for file_info in selected_files:
        try:
            da = load_hourly(file_info, precip_var, eth_poly, mask_da)
            lat_name = next((c for c in ["lat", "latitude", "y", "Lat"] if c in da.coords), None)
            lon_name = next((c for c in ["lon", "longitude", "x", "Lon"] if c in da.coords), None)
            if lat_name is None or lon_name is None:
                continue
            lat_vals = np.asarray(da.coords[lat_name].values, dtype=float)
            lon_vals = np.asarray(da.coords[lon_name].values, dtype=float)
            ilat = int(np.argmin(np.abs(lat_vals - lat0)))
            ilon = int(np.argmin(np.abs(lon_vals - lon0)))
            point_val = float(da.isel({lat_name: ilat, lon_name: ilon}).values)
            if np.isfinite(point_val):
                total_rain += point_val
                valid_steps += 1
            if snapped_lat is None:
                snapped_lat = float(lat_vals[ilat])
                snapped_lon = float(lon_vals[ilon])
        except Exception as e:
            logger.warning(f"MSWEP file skipped ({file_info['filename']}): {e}")
            continue
    return {
        "dataset": "MSWEP NRT",
        "label": "MSWEP NRT",
        "total_mm": total_rain,
        "valid_steps": valid_steps,
        "start": start_date,
        "end": end_date,
        "snapped_lat": snapped_lat if snapped_lat is not None else lat0,
        "snapped_lon": snapped_lon if snapped_lon is not None else lon0,
    }

# ---------------------------------------------------------------------
# Helper functions (layout, signal, MJO, wind, plots, IMERG, etc.)
# ---------------------------------------------------------------------

def render_wf_bar(active: int) -> None:
    steps = ["NRT + MJO", "Observational Timeseries", "Onset Detection",
             "Region Assessment", "ML Model Setup", "Forecast Ensemble"]
    items = []
    for i, s in enumerate(steps):
        cls = "wf-step done" if i < active else ("wf-step active" if i == active else "wf-step")
        items.append(f'<div class="{cls}"><span class="wf-num">{i+1}</span>{s}</div>')
        if i < len(steps) - 1:
            items.append('<span class="wf-arrow">›</span>')
    st.markdown(f'<div class="wf-bar">{"".join(items)}</div>', unsafe_allow_html=True)

def info_box(text: str, kind: str = "") -> None:
    st.markdown(f'<div class="ibox {kind}">{text}</div>', unsafe_allow_html=True)

def section_header(text: str, icon: str = "", accent: str = "") -> None:
    ico = f"<span style='font-size:.95rem'>{icon}</span> " if icon else ""
    cls = f"sec-hd {accent}".strip()
    st.markdown(f'<div class="{cls}"><span class="sec-hd-accent"></span>{ico}{text}</div>', unsafe_allow_html=True)

def kpi(label: str, value: str, unit: str = "", color: str = "") -> str:
    cls = f"kpi {color}".strip()
    u = f"<div class='kpi-unit'>{unit}</div>" if unit else ""
    return (f"<div class='{cls}'>"
            f"<div class='kpi-label'>{label}</div>"
            f"<div class='kpi-value'>{value}</div>{u}</div>")

def render_source_status(title: str, status: str, detail: str, kind: str = "sky") -> None:
    info_box(f"<b>{title}</b><br>{status}<br><span style='font-size:.74rem'>{detail}</span>", kind)

def render_likelihood_meter(value: float) -> None:
    pct = int(value * 100)
    color, verdict = (("#DC2626", "Low") if pct < 35
                      else ("#F59E0B", "Moderate") if pct < 60
                      else ("#22C55E", "High") if pct < 80
                      else ("#15803D", "Very High"))
    st.markdown(f"""
    <div class="card" style="padding:1.25rem 1.5rem;">
      <div style="font-size:.72rem;font-weight:700;text-transform:uppercase;
                  letter-spacing:.1em;color:#718096;margin-bottom:.6rem;">
         Onset Likelihood
      </div>
      <div style="display:flex;align-items:baseline;gap:10px;margin-bottom:12px;">
        <span style="font-family:'Playfair Display',serif;font-size:2.6rem;
                     font-weight:900;color:{color};line-height:1;">{pct}%</span>
        <span class="badge"
              style="background:{color}22;color:{color};font-size:.78rem;">{verdict}</span>
      </div>
      <div style="height:12px;border-radius:6px;
                  background:linear-gradient(90deg,#DC2626,#F59E0B 35%,#22C55E 65%,#15803D 100%);
                  position:relative;">
        <div style="position:absolute;top:-5px;left:{pct}%;transform:translateX(-50%);
                    width:5px;height:22px;background:#0D1B2A;border-radius:3px;
                    box-shadow:0 0 0 2px #fff;"></div>
      </div>
      <div style="display:flex;justify-content:space-between;
                  font-size:.6rem;color:#718096;margin-top:6px;">
        <span>Low</span><span>Moderate</span><span>High</span><span>Very High</span>
      </div>
    </div>""", unsafe_allow_html=True)

def render_mjo_ethiopia_guide(mjo_context):
    section_header("MJO → Ethiopia Rainfall Teleconnection", "🌀", "violet")
    if mjo_context.get("phase") in MJO_ETH_IMPACT:
        label, _, explanation = MJO_ETH_IMPACT[mjo_context["phase"]]
        date_label = (mjo_context["date"].strftime("%Y-%m-%d")
                      if isinstance(mjo_context.get("date"), pd.Timestamp) else "n/a")
        render_source_status("Realtime MJO status",
                             f"{mjo_context['source']} · P{mjo_context['phase']} · {label}",
                             f"Amplitude {mjo_context.get('amplitude', np.nan):.2f} · "
                             f"{mjo_context.get('strength','Unknown')} · valid {date_label}. {explanation}",
                             "violet")
    else:
        render_source_status("Realtime MJO status", "Automatic MJO state unavailable",
                             "Use the manual override only if the operational source is unavailable or stale.",
                             "amber")
    cols = st.columns(4)
    for phase, (label, color, explanation) in MJO_ETH_IMPACT.items():
        with cols[(phase - 1) % 4]:
            st.markdown(f"""
            <div class="mjo-phase-box">
              <div class="mjo-phase-num" style="color:{color};">P{phase}</div>
              <div class="mjo-phase-label">{label}</div>
              <div style="font-size:.66rem;margin-top:6px;">{explanation[:80]}…</div>
            </div>""", unsafe_allow_html=True)

def render_wind_diagnostics(selected_grids):
    section_header("Operational Wind Diagnostics", "💨")
    info_box("Displaying external CORe wind and divergence products (NOAA/CPC).", "sky")
    wind_tabs = st.tabs(["850 hPa (Somali Jet)", "700 hPa (Highlands)", "500 hPa (Trough)",
                         "200 hPa (TEJ)", "Divergence"])
    patterns = ["850.*Anom", "700.*Anom", "500.*Anom", "200.*Anom", "Divergence 200"]
    for i, pat in enumerate(patterns):
        with wind_tabs[i]:
            key = get_source_key_by_pattern(pat)
            if pat == "Divergence 200":
                key_conv = get_source_key_by_pattern("Convergence 925")
                render_image_pair(key, key, key_conv)
            else:
                key_avg = get_source_key_by_pattern(pat.replace("Anom", "Avg"))
                render_image_pair(key_avg, key_avg, key)

def compute_recent_grid_signal(dataset_key, lat0, lon0, days=7):
    if dataset_key not in DS_BY_KEY:
        return None
    ds = DS_BY_KEY[dataset_key]
    lat_name, lon_name, time_name = COORDS_BY_KEY[dataset_key]
    from config import RAIN_VAR_BY_KEY, RAIN_VAR
    var = RAIN_VAR_BY_KEY.get(dataset_key, RAIN_VAR)
    try:
        end_dt = pd.Timestamp(ds[time_name].max().values)
        start_dt = end_dt - pd.Timedelta(days=max(days-1, 0))
        da = (ds[var]
              .sel({time_name: slice(str(start_dt.date()), str(end_dt.date()))})
              .sel({lat_name: lat0, lon_name: lon0}, method="nearest")
              .compute())
        values = np.asarray(da.values, dtype=float)
        return {
            "dataset": dataset_key,
            "label": dataset_key,
            "total_mm": float(np.nansum(values)),
            "valid_steps": int(np.isfinite(values).sum()),
            "start": start_dt,
            "end": end_dt,
            "snapped_lat": float(da[lat_name].values),
            "snapped_lon": float(da[lon_name].values),
        }
    except Exception:
        return None

def compute_recent_station_signal(lat0, lon0, emi_meta, emi_ts, days=7):
    if emi_meta is None or emi_ts is None or len(emi_meta) == 0:
        return None
    try:
        dist = np.sqrt((emi_meta["lat"] - lat0)**2 + (emi_meta["lon"] - lon0)**2)
        near_st = emi_meta.index[dist.argmin()]
        series = emi_ts[near_st].dropna()
        if series.empty:
            return None
        end_dt = pd.Timestamp(series.index.max())
        start_dt = end_dt - pd.Timedelta(days=max(days-1, 0))
        recent = series.loc[start_dt:end_dt]
        return {
            "dataset": STATION_KEY,
            "label": near_st,
            "station_id": near_st,
            "distance_deg": float(dist.min()),
            "total_mm": float(recent.sum()),
            "valid_steps": int(recent.notna().sum()),
            "start": start_dt,
            "end": end_dt,
            "snapped_lat": float(emi_meta.loc[near_st, "lat"]),
            "snapped_lon": float(emi_meta.loc[near_st, "lon"]),
        }
    except Exception:
        return None

def get_primary_rainfall_key(selected_grids):
    for key in ["CHIRPS", "ENACTS", "IMERG", "ERA5"]:
        if key in selected_grids:
            return key
    return selected_grids[0] if selected_grids else None

def get_source_key_by_pattern(pattern: str) -> str:
    regex = re.compile(pattern, re.IGNORECASE)
    for key in NRT_SOURCES.keys():
        normalized_key = re.sub(r'[\s\-\.\·]+', '', key)
        normalized_pattern = re.sub(r'[\s\-\.\·]+', '', pattern)
        if regex.search(key) or normalized_pattern in normalized_key:
            return key
    return pattern

def parse_nrt_info(filename: str) -> Dict[str, Any]:
    # Kept for compatibility, but we use the cached version.
    pass

def _points_in_polygon(polygon: Any, lon2d: np.ndarray, lat2d: np.ndarray) -> np.ndarray:
    try:
        from shapely import contains_xy
        return contains_xy(polygon, lon2d.ravel(), lat2d.ravel()).reshape(lon2d.shape)
    except ImportError:
        from shapely.geometry import Point
        from shapely.prepared import prep
        prepared = prep(polygon)
        flat_mask = np.array([prepared.contains(Point(x, y)) for x, y in zip(lon2d.ravel(), lat2d.ravel())])
        return flat_mask.reshape(lon2d.shape)

def _get_lat_lon_coords(da_2d: Any) -> Tuple[np.ndarray, np.ndarray]:
    LAT_NAMES = ("Lat", "lat", "latitude", "XLAT", "LAT")
    LON_NAMES = ("Lon", "lon", "longitude", "XLONG", "LON")
    lat_coord = next((c for c in LAT_NAMES if c in da_2d.coords), None)
    lon_coord = next((c for c in LON_NAMES if c in da_2d.coords), None)
    if lat_coord is None or lon_coord is None:
        raise KeyError(f"Unable to parse lat/lon coordinates: {list(da_2d.coords)}")
    return da_2d.coords[lat_coord].values, da_2d.coords[lon_coord].values

def mask_to_ethiopia(da: Any, eth_poly: Optional[Any], mask_da: Optional[Any]) -> Any:
    lat_values, lon_values = _get_lat_lon_coords(da)
    lon_mesh, lat_mesh = np.meshgrid(lon_values, lat_values) if lon_values.ndim == 1 else (lon_values, lat_values)
    if eth_poly is not None:
        mask = _points_in_polygon(eth_poly, lon_mesh, lat_mesh)
        da = da.where(mask)
    if mask_da is not None:
        try:
            mask_interp = mask_da.interp_like(da, method="nearest")
            da = da.where(mask_interp > 0.5)
        except Exception as e:
            logger.error(f"Failed to apply mask interpolation: {e}")
    return da

def load_hourly(file_info: Dict[str, Any], precip_var: str, eth_poly: Optional[Any], mask_da: Optional[Any]) -> Any:
    with xr.open_dataset(file_info["path"], engine="h5netcdf") as ds:
        da = ds[precip_var]
        if "time" in da.dims:
            da = da.isel(time=0)
        return mask_to_ethiopia(da, eth_poly, mask_da).load()

def load_daily_accumulation(file_infos: List[Dict[str, Any]], precip_var: str, eth_poly: Optional[Any], mask_da: Optional[Any]) -> Optional[Any]:
    total = None
    for info in file_infos:
        with xr.open_dataset(info["path"], engine="h5netcdf") as ds:
            da = ds[precip_var]
            if "time" in da.dims:
                da = da.isel(time=0)
            da_masked = mask_to_ethiopia(da, eth_poly, mask_da).load()
            total = da_masked if total is None else total + da_masked
    return total

def plot_rainfall(
    da: Any,
    title: str,
    gdf_eth: Optional[Any],
    bounds: Optional[Tuple] = None,
    show_contours: bool = True,
) -> None:
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter

    PRECIP_LEVELS  = [0, 1, 5, 10, 20, 30, 50, 75, 100]
    PRECIP_COLORS  = [
        "#D9D9D9", "#FFA500", "#FFFF00", "#CAFF70", "#A2CD5A",
        "#7FFF00", "#66CD00", "#008000", "#228B22",
    ]
    CONTOUR_LEVELS = [5, 10, 20, 30, 50, 75]

    lat_name = next((c for c in ["lat", "latitude", "y", "Lat"] if c in da.coords), None)
    lon_name = next((c for c in ["lon", "longitude", "x", "Lon"] if c in da.coords), None)
    if not lat_name or not lon_name:
        st.error("Coordinate schema mapping breakdown.")
        return

    lon  = da.coords[lon_name].values
    lat  = da.coords[lat_name].values
    data = np.asarray(da.values, dtype=float)

    cmap = mcolors.ListedColormap(PRECIP_COLORS)
    norm = mcolors.BoundaryNorm(PRECIP_LEVELS, len(PRECIP_COLORS), extend="max")

    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    mesh = ax.pcolormesh(lon, lat, data, cmap=cmap, norm=norm, shading="auto")

    if show_contours:
        data_filled = np.where(np.isfinite(data), data, 0.0)
        smoothed = gaussian_filter(data_filled, sigma=1.0)
        valid_mask = np.isfinite(data)
        if valid_mask.any() and smoothed.max() >= CONTOUR_LEVELS[0]:
            if lon.ndim == 1 and lat.ndim == 1:
                lon2d, lat2d = np.meshgrid(lon, lat)
            else:
                lon2d, lat2d = lon, lat
            cs = ax.contour(lon2d, lat2d, smoothed, levels=CONTOUR_LEVELS,
                            colors="black", linewidths=0.55, alpha=0.65)
            ax.clabel(cs, inline=True, fontsize=5.5, fmt="%g mm", inline_spacing=2)

    if gdf_eth is not None:
        gdf_eth.boundary.plot(ax=ax, edgecolor="#333333", linewidth=0.4, alpha=0.8)

    cbar = fig.colorbar(mesh, ax=ax, ticks=PRECIP_LEVELS, shrink=0.75, aspect=25, pad=0.03)
    cbar.ax.set_yticklabels([str(l) for l in PRECIP_LEVELS], fontsize=7)
    cbar.ax.set_ylabel("Precipitation (mm)", fontsize=8, rotation=90, labelpad=6)

    ax.set_title(title, fontsize=8, weight="bold", pad=8)
    ax.set_xlabel("Longitude", fontsize=7)
    ax.set_ylabel("Latitude", fontsize=7)
    ax.tick_params(labelsize=7)

    effective_bounds = bounds if bounds is not None else ETHIOPIA_BOUNDS
    ax.set_xlim(effective_bounds[0], effective_bounds[2])
    ax.set_ylim(effective_bounds[1], effective_bounds[3])

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

def render_imerg_timeseries(daily_df: pd.DataFrame, latest: dict, half_hourly_df: pd.DataFrame | None = None) -> None:
    df = daily_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    total_7d   = df["rain_mm"].sum()
    peak_row   = df.loc[df["rain_mm"].idxmax()]
    peak_day   = peak_row["date"].strftime("%d %b")
    peak_mm    = peak_row["rain_mm"]
    latest_mm  = latest.get("rain_mm", np.nan)
    n_days     = df["date"].nunique()
    coverage   = f"{n_days} / {len(df)} days"
    df["roll3"]  = df["rain_mm"].rolling(3, min_periods=1).mean()
    df["cumul"]  = df["rain_mm"].cumsum()
    kpi_html = '<div class="kpi-row">'
    kpi_html += kpi("7-day total",   f"{total_7d:.1f}",  "mm",     "teal")
    kpi_html += kpi("Peak day",      f"{peak_mm:.1f}",   peak_day, "amber")
    kpi_html += kpi("Latest 24 h",   f"{latest_mm:.1f}", "mm",     "sky")
    kpi_html += kpi("Data coverage", coverage,            "",       "")
    kpi_html += "</div>"
    st.markdown(kpi_html, unsafe_allow_html=True)

    agg_col, smooth_col, _ = st.columns([1.6, 1.6, 3])
    agg_label = agg_col.radio("Aggregation", ["Daily", "3-hourly"] if half_hourly_df is not None else ["Daily"],
                              horizontal=True, key="imerg_agg_radio")
    show_cumul = smooth_col.checkbox("Show cumulative", value=True, key="imerg_show_cumul")

    if agg_label == "3-hourly" and half_hourly_df is not None:
        plot_df = half_hourly_df.copy()
        plot_df["time"] = pd.to_datetime(plot_df["time"])
        plot_df["roll3"] = plot_df["rain_mmhr"].rolling(6, min_periods=1).mean()
        plot_df["cumul"] = (plot_df["rain_mmhr"] * 0.5).cumsum()
        x_col, y_col, roll_col, y_label = "time", "rain_mmhr", "roll3", "Rain rate (mm/hr)"
    else:
        plot_df = df
        x_col, y_col, roll_col, y_label = "date", "rain_mm", "roll3", "Rainfall (mm/day)"

    fig = go.Figure()
    fig.add_trace(go.Bar(x=plot_df[x_col], y=plot_df[y_col], name="Rainfall",
                         marker_color="#1A6B9A", marker_opacity=0.72, yaxis="y1",
                         hovertemplate="%{x|%d %b %H:%M}<br><b>%{y:.2f}</b> "+y_label.split("(")[1].rstrip(")")+"<extra></extra>"))
    fig.add_trace(go.Scatter(x=plot_df[x_col], y=plot_df[roll_col],
                             name="3-day rolling mean" if agg_label == "Daily" else "3-hr rolling mean",
                             line=dict(color="#F97316", width=2.2), mode="lines", yaxis="y1",
                             hovertemplate="%{x|%d %b}<br>Mean: <b>%{y:.2f}</b><extra></extra>"))
    if show_cumul:
        fig.add_trace(go.Scatter(x=plot_df[x_col], y=plot_df["cumul"], name="Cumulative (mm)",
                                 line=dict(color="#7C3AED", width=1.8, dash="dot"), mode="lines", yaxis="y2",
                                 hovertemplate="%{x|%d %b}<br>Cumul: <b>%{y:.1f} mm</b><extra></extra>"))
    target_x = peak_row["date"] if agg_label == "Daily" else plot_df[x_col].iloc[0]
    if hasattr(target_x, "strftime"):
        target_x = target_x.strftime("%Y-%m-%d %H:%M:%S")
    fig.add_vline(x=target_x, line_width=1.2, line_dash="dash", line_color="#F59E0B",
                  annotation_text=f"Peak {peak_mm:.1f} mm", annotation_position="top right",
                  annotation_font_size=11, annotation_font_color="#B45309")
    current_run = st.session_state.get("imerg_active_run", "unknown")
    fig.update_layout(height=360, paper_bgcolor="#F7F9FC", plot_bgcolor="#FFFFFF",
                      margin=dict(l=52, r=52, t=48, b=44),
                      title=dict(text=f"IMERG {current_run.upper()} Run — recent rainfall evolution",
                                 font_size=13, x=0, xanchor="left"),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0), hovermode="x unified",
                      xaxis=dict(title="Date (UTC)", showgrid=True, gridcolor="#EEF2F7", tickformat="%d %b"),
                      yaxis=dict(title=y_label, showgrid=True, gridcolor="#EEF2F7", rangemode="tozero"),
                      yaxis2=dict(title="Cumulative (mm)", overlaying="y", side="right", showgrid=False,
                                  rangemode="tozero", tickfont=dict(color="#7C3AED"), titlefont=dict(color="#7C3AED")) if show_cumul else {},
                      barmode="overlay")
    st.plotly_chart(fig, width='stretch')  # use width instead of use_container_width

# ---------------------------------------------------------------------
# IMERG FETCHING WITH CACHE AND FALLBACK
# ---------------------------------------------------------------------
IMERG_CACHE_DIR = APP_DIR / "data" / "IMERG_cache"
IMERG_CACHE_DIR.mkdir(parents=True, exist_ok=True)

def _get_imerg_cache_path(lat: float, lon: float, run: str, days_back: int) -> Path:
    date_str = datetime.utcnow().strftime("%Y%m%d_%H")
    return IMERG_CACHE_DIR / f"imerg_{lat}_{lon}_{run}_{days_back}_{date_str}.parquet"

def _save_imerg_cache(data: dict, cache_path: Path):
    data["daily"].to_parquet(cache_path.with_suffix(".daily.parquet"))
    if data.get("half_hourly") is not None:
        data["half_hourly"].to_parquet(cache_path.with_suffix(".hourly.parquet"))
    metadata = {k: v for k, v in data.items() if k not in ("daily", "half_hourly")}
    with open(cache_path.with_suffix(".meta.json"), "w") as f:
        json.dump(metadata, f, default=str)

def _load_imerg_cache(cache_path: Path) -> Optional[dict]:
    daily_path = cache_path.with_suffix(".daily.parquet")
    meta_path  = cache_path.with_suffix(".meta.json")
    if daily_path.exists() and meta_path.exists():
        age = datetime.utcnow() - datetime.fromtimestamp(daily_path.stat().st_mtime)
        if age.total_seconds() < 6 * 3600:
            daily   = pd.read_parquet(daily_path)
            hourly  = None
            hourly_path = cache_path.with_suffix(".hourly.parquet")
            if hourly_path.exists():
                hourly = pd.read_parquet(hourly_path)
            with open(meta_path, "r") as f:
                metadata = json.load(f)
            return {"status": "ok", "daily": daily, "half_hourly": hourly, **metadata}
    return None

def fetch_imerg_mock(lat: float, lon: float, days_back: int = 7, run: str = "late") -> dict:
    end_date   = datetime.utcnow()
    start_date = end_date - timedelta(days=days_back)
    dates = pd.date_range(start_date, end_date, freq="D")
    np.random.seed(hash((lat, lon, run)) % 2**32)
    rain = np.random.gamma(shape=2, scale=2, size=len(dates))
    daily  = pd.DataFrame({"date": dates, "rain_mm": rain})
    latest = {"rain_mm": float(rain[-1]), "snapped_lat": lat, "snapped_lon": lon}
    return {
        "status":           "ok",
        "daily":            daily,
        "half_hourly":      None,
        "latest":           latest,
        "message":          "Mock IMERG data (real data not available)",
        "point_in_ethiopia": True,
    }

def fetch_imerg_nrt_cached(lat: float, lon: float, days_back: int = 7, run: str = "late") -> dict:
    cache_path = _get_imerg_cache_path(lat, lon, run, days_back)
    cached = _load_imerg_cache(cache_path)
    if cached:
        st.info("Using cached IMERG data (less than 6 hours old).")
        return cached

    if not IMERG_AVAILABLE:
        st.info("Original fetch_imerg_nrt not found. Using synthetic IMERG data.")
        return fetch_imerg_mock(lat, lon, days_back, run)

    auth = earthdata_auth_status()
    if auth.get("ok") != "true":
        st.warning(f"Earthdata authentication required: {auth.get('message', 'credentials missing')}. Falling back to synthetic IMERG data.")
        return fetch_imerg_mock(lat, lon, days_back, run)

    runs_to_try = [run, "early" if run == "late" else "late"]
    result: Optional[dict] = None
    active_run = run

    for attempt_run in runs_to_try:
        try:
            st.info(f"Fetching IMERG **{attempt_run.upper()}** run for ({lat:.2f}°N, {lon:.2f}°E) …")
            result = fetch_imerg_nrt(lat, lon, days_back=days_back, run=attempt_run)
            if result.get("status") == "ok":
                active_run = attempt_run
                break
        except Exception as exc:
            st.warning(f"IMERG {attempt_run.upper()} run raised an exception: {exc}. Trying alternate run…" if attempt_run == runs_to_try[0] else "")
            result = None

    if result is None or result.get("status") != "ok":
        st.warning("Both IMERG run attempts failed. Using synthetic data as fallback.")
        result = fetch_imerg_mock(lat, lon, days_back, run)
        active_run = run

    st.session_state["imerg_active_run"] = active_run

    if result.get("status") == "ok" and result.get("message", "") != "Mock IMERG data (real data not available)":
        try:
            _save_imerg_cache(result, cache_path)
        except Exception as e:
            logger.warning(f"Could not write IMERG cache: {e}")

    return result

# ---------------------------------------------------------------------
# MJO FETCHING
# ---------------------------------------------------------------------
MJO_VPI_URL = "https://www.cpc.ncep.noaa.gov/products/precip/CWlink/daily_mjo_index/proj_norm_order.ascii"
MJO_RMM_URL = "https://www.cpc.ncep.noaa.gov/products/precip/CWlink/daily_mjo_index/mjo_index.html"

def parse_mjo_date(date_str: str) -> Optional[datetime]:
    s = date_str.strip()
    if len(s) == 8 and s.isdigit():
        try:
            return datetime.strptime(s, "%Y%m%d")
        except:
            return None
    elif len(s) == 7 and s.isdigit():
        yy = int(s[:2])
        mm = int(s[2:4])
        dd = int(s[4:7])
        if 1 <= mm <= 12 and 1 <= dd <= 31:
            year = 2000 + yy
            try:
                return datetime(year, mm, dd)
            except:
                return None
    return None

def fetch_mjo_from_vpi() -> Optional[Dict]:
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(MJO_VPI_URL, timeout=15, headers=headers)
        resp.raise_for_status()
        lines = resp.text.strip().split('\n')
        data = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 11:
                continue
            date = parse_mjo_date(parts[0])
            if date is None:
                continue
            indices = []
            ok = True
            for x in parts[1:11]:
                if x == '*****':
                    ok = False
                    break
                try:
                    indices.append(float(x))
                except:
                    ok = False
                    break
            if not ok or len(indices) != 10:
                continue
            arr = np.array(indices)
            minv, maxv = arr.min(), arr.max()
            if abs(minv) > abs(maxv):
                vpi_phase = np.argmin(arr) + 1
                amp = abs(minv)
            else:
                vpi_phase = np.argmax(arr) + 1
                amp = abs(maxv)
            phase_map = {1:2,2:3,3:4,4:5,5:6,6:6,7:7,8:8,9:8,10:1}
            rmm_phase = phase_map.get(vpi_phase, 1)
            data.append({
                "date": date,
                "phase": rmm_phase,
                "amplitude": amp,
                "vpi_phase": vpi_phase,
                "indices": indices
            })
        if not data:
            return None
        return data[-1]
    except Exception as e:
        logger.warning(f"VPI fetch failed: {e}")
        return None

def fetch_mjo_from_rmm_html() -> Optional[Dict]:
    try:
        tables = pd.read_html(MJO_RMM_URL)
        for tbl in tables:
            cols_lower = [str(c).lower() for c in tbl.columns]
            if any('rmm1' in c for c in cols_lower) and any('phase' in c for c in cols_lower):
                df = tbl.copy()
                df.columns = cols_lower
                if 'year' in df.columns and 'mon' in df.columns and 'day' in df.columns:
                    df['date'] = pd.to_datetime(df[['year', 'mon', 'day']])
                elif 'yr' in df.columns and 'mon' in df.columns and 'day' in df.columns:
                    df['date'] = pd.to_datetime(df[['yr', 'mon', 'day']])
                else:
                    continue
                latest = df.iloc[-1]
                if 'phase' in latest and 'amp' in latest:
                    phase = int(latest['phase'])
                    amplitude = float(latest['amp'])
                    strength = "Active" if amplitude >= 1.0 else "Weak"
                    return {
                        "phase": phase,
                        "amplitude": amplitude,
                        "date": latest['date'],
                        "strength": strength,
                        "source": "NOAA CPC RMM (HTML)",
                        "source_url": MJO_RMM_URL,
                    }
        return None
    except Exception as e:
        logger.warning(f"RMM HTML fetch failed: {e}")
        return None

@st.cache_data(ttl=3600)
def fetch_mjo_status() -> Optional[Dict]:
    data = fetch_mjo_from_vpi()
    source = "NOAA CPC VPI" if data else None
    if not data:
        data = fetch_mjo_from_rmm_html()
        source = "NOAA CPC RMM (HTML)" if data else None
    if not data:
        return None

    latest_date = data["date"]
    age_days = (datetime.utcnow() - latest_date).days
    strength = data.get("strength", "Active" if data["amplitude"] >= 1.0 else "Weak")

    month = latest_date.month
    if 6 <= month <= 9:
        eth_season = "Kiremt (Main Rainy Season)"
    elif 3 <= month <= 5:
        eth_season = "Belg (Short/Spring Rains)"
    else:
        eth_season = "Bega (Dry / Harvest Season)"

    phase = data["phase"]
    if strength == "Active":
        if phase in (1, 2, 8):
            regional_impact = "Suppressed convection over East Africa"
        elif phase in (3, 4, 5):
            regional_impact = "Enhanced moisture influx, elevated rainfall risk"
        else:
            regional_impact = "Neutral / transitional"
    else:
        regional_impact = "Weak MJO, low predictability"

    return {
        "phase": phase,
        "amplitude": round(data["amplitude"], 2),
        "date": latest_date,
        "strength": strength,
        "source": source,
        "source_url": MJO_VPI_URL if source == "NOAA CPC VPI" else MJO_RMM_URL,
        "age_days": age_days,
        "ethiopia_climate_season": eth_season,
        "regional_impact": regional_impact,
        "impact_mechanism": f"MJO Phase {phase} with amplitude {data['amplitude']:.2f}.",
    }

def resolve_mjo_context() -> Dict:
    auto = fetch_mjo_status()
    with st.sidebar.expander("MJO control", expanded=False):
        use_manual = st.checkbox("Use manual MJO override", value=False)
        manual_phase = st.number_input("Manual phase", 1, 8, 2, key="manual_mjo_phase")
        manual_amplitude = st.number_input("Manual amplitude", 0.0, 4.0, 1.0, 0.1, key="manual_mjo_amplitude")
    if use_manual:
        return {"phase": int(manual_phase), "amplitude": float(manual_amplitude),
                "date": None, "strength": "Manual override", "source": "Manual", "source_url": ""}
    if auto:
        return auto
    return {"phase": None, "amplitude": None, "date": None, "strength": "Unavailable",
            "source": "Unavailable", "source_url": ""}

# ---------------------------------------------------------------------
# WIND SYNOPTICS (mock)
# ---------------------------------------------------------------------
@st.cache_data(ttl=3600)
def fetch_wind_indices(lat: float, lon: float) -> Dict[str, float]:
    import random
    random.seed(hash((lat, lon, datetime.utcnow().date())))
    somali = 8.0 + 4.0 * random.random()
    tej = -12.0 - 6.0 * random.random()
    div = 1e-5 + 2e-5 * random.random()
    somali_score = min(max((somali - 5) / (15 - 5), 0), 1)
    tej_score = min(max((-tej - 10) / (20 - 10), 0), 1)
    div_score = min(max((div - 0.5e-5) / (3e-5 - 0.5e-5), 0), 1)
    wind_score = 0.5 * somali_score + 0.3 * tej_score + 0.2 * div_score
    return {"somali_jet_ms": round(somali, 1), "tej_index": round(tej, 1),
            "divergence_200": f"{div:.1e}", "wind_score": round(wind_score, 2)}

# ---------------------------------------------------------------------
# ONSET LIKELIHOOD
# ---------------------------------------------------------------------
def compute_onset_likelihood_full(gauge_7d, gridded_7d, mswep_7d, accum_thresh,
                                  mjo_phase, mjo_amplitude, wind_score):
    sig_gauge   = min(gauge_7d   / max(accum_thresh, 1.0), 1.0)
    sig_gridded = min(gridded_7d / max(accum_thresh, 1.0), 1.0)
    sig_mswep   = min(mswep_7d   / max(accum_thresh, 1.0), 1.0)
    mjo_weights = {1: 0.60, 2: 0.85, 3: 0.75, 4: 0.50, 5: 0.30, 6: 0.20, 7: 0.25, 8: 0.55}
    sig_mjo = mjo_weights.get(mjo_phase, 0.50) if mjo_phase else 0.50
    if mjo_amplitude is not None:
        amp_factor = min(max(mjo_amplitude, 0.0), 1.5) / 1.5
        sig_mjo = 0.50 + (sig_mjo - 0.50) * amp_factor
    likelihood = (0.25 * sig_gauge + 0.20 * sig_gridded + 0.25 * sig_mswep +
                  0.15 * sig_mjo + 0.15 * wind_score)
    likelihood = round(min(likelihood, 1.0), 3)
    components = {"Gauge 7d": sig_gauge, "Gridded 7d": sig_gridded,
                  "MSWEP 7d": sig_mswep, "MJO signal": sig_mjo, "Wind synoptic": wind_score}
    return likelihood, components

# ---------------------------------------------------------------------
# MAIN PAGE FUNCTION – split into Wind and RF tabs
# ---------------------------------------------------------------------
def page():
    try:
        from utils.helpers import load_css
        load_css("page_styles/nrt.css")
    except Exception:
        pass

    # Optional workflow bar
    render_wf_bar(0)

    # Get session state
    selected_grids = st.session_state.get("selected_grids", [])
    lat0 = st.session_state.get("lat0", 9.0)
    lon0 = st.session_state.get("lon0", 39.0)
    onset_params = st.session_state.get("onset_params", {})
    emi_meta = st.session_state.get("emi_meta")
    emi_ts = st.session_state.get("emi_ts")

    # ---- Compute all signals once (cached) ----
    with st.spinner("Loading data..."):
        mjo_context = resolve_mjo_context()
        wind_indices = fetch_wind_indices(lat0, lon0)
        wind_score = wind_indices["wind_score"]

        station_signal = compute_recent_station_signal(lat0, lon0, emi_meta, emi_ts, days=7)
        grid_keys = [k for k in selected_grids if k != "ERA5"] or selected_grids
        primary_key = get_primary_rainfall_key(grid_keys)
        primary_grid_signal = compute_recent_grid_signal(primary_key, lat0, lon0, days=7) if primary_key else None
        mswep_signal = compute_recent_mswep_signal_cached(lat0, lon0, days=7)

    # ---- KPI row (overview) ----
    summary = '<div class="kpi-row">'
    summary += kpi("Location", f"{lat0:.2f} / {lon0:.2f}", "lat / lon", "sky")
    if station_signal:
        summary += kpi("Nearest Gauge 7d", f"{station_signal['total_mm']:.1f}", "mm", "green")
    if primary_grid_signal:
        summary += kpi(f"{primary_grid_signal['dataset']} 7d", f"{primary_grid_signal['total_mm']:.1f}", "mm", "teal")
    if mswep_signal:
        summary += kpi("MSWEP 7d", f"{mswep_signal['total_mm']:.1f}", "mm", "blue")
    if mjo_context.get("phase"):
        summary += kpi("MJO Phase", f"P{mjo_context['phase']}", mjo_context.get("strength", ""), "violet")
    if mjo_context.get("amplitude") is not None:
        summary += kpi("MJO Amplitude", f"{mjo_context['amplitude']:.2f}", "RMM", "amber")
    summary += kpi("Wind Score", f"{wind_score:.2f}", "", "teal")
    summary += "</div>"
    st.markdown(summary, unsafe_allow_html=True)

    # ---- Two main tabs: Wind and RF ----
    tab_wind, tab_rf = st.tabs(["Wind", "RF"])

    # ---- TAB 1: WIND ----
    with tab_wind:
        render_mjo_ethiopia_guide(mjo_context)
        render_wind_diagnostics(selected_grids)

    # ---- TAB 2: RF (Rainfall) ----
    with tab_rf:
        # Observed Rainfall Snapshot
        section_header("Observed Rainfall Snapshot", "🌧")
        if station_signal or primary_grid_signal or mswep_signal:
            rows = []
            if station_signal:
                rows.append({"Source": f"Gauge · {station_signal['label']}", "7d total (mm)": station_signal["total_mm"], "Window end": station_signal["end"].strftime("%Y-%m-%d")})
            if primary_grid_signal:
                rows.append({"Source": primary_grid_signal["dataset"], "7d total (mm)": primary_grid_signal["total_mm"], "Window end": primary_grid_signal["end"].strftime("%Y-%m-%d")})
            if mswep_signal:
                rows.append({"Source": "MSWEP NRT", "7d total (mm)": mswep_signal["total_mm"], "Window end": mswep_signal["end"].strftime("%Y-%m-%d")})
            snap_df = pd.DataFrame(rows)
            c1, c2 = st.columns([1.05, 1.4])
            c1.dataframe(snap_df, width='stretch', hide_index=True)
            fig_snap = px.bar(snap_df, x="Source", y="7d total (mm)", color="Source", title="Latest 7-day rainfall comparison")
            fig_snap.update_layout(height=320, showlegend=False, paper_bgcolor="#F7F9FC", plot_bgcolor="#FFFFFF")
            c2.plotly_chart(fig_snap, width='stretch')

        # CPC RFE / ARC2 images
        c1, c2 = st.columns(2)
        for i, target_pattern in enumerate(["CPC RFE", "CPC ARC2"]):
            true_key = get_source_key_by_pattern(target_pattern)
            nrt_card([c1, c2][i], true_key, NRT_SOURCES.get(true_key, {"url": "", "label": target_pattern, "name": target_pattern}))

        # ---- Sub‑tabs for MSWEP and IMERG ----
        sub_tabs = st.tabs(["MSWEP National Gridded Field Engine", "NASA IMERG NRT Point Verification"])

        # MSWEP
        with sub_tabs[0]:
            section_header("MSWEP National Gridded Field Engine", "🗺️")
            with st.spinner("Loading MSWEP data..."):
                gdf_eth = load_ethiopia_shapefile()
                bounds = tuple(gdf_eth.total_bounds) if gdf_eth is not None else ETHIOPIA_BOUNDS
                eth_poly = unary_union(gdf_eth.geometry) if gdf_eth is not None else None
                mask_da = load_ethiopia_mask()
                files_info, precip_var_ms, day_groups, sorted_days, latest_day_files, latest_date = load_mswep_metadata()

            if not files_info:
                st.warning(f"No valid data layers discovered in `{MSWEP_DATA_DIR.name}`")
            else:
                vis_col, contour_col = st.columns([3, 1])
                with vis_col:
                    mode = st.radio(
                        "Visualization Mode",
                        ["7‑day Accumulation", "Daily Accumulation Total", "Latest Operational Hour"],
                        horizontal=True, key="mswep_vis_strategy"
                    )
                with contour_col:
                    show_contours = st.checkbox(
                        "Show contour lines", value=True, key="mswep_show_contours",
                        help="Overlay labelled contour lines (5, 10, 20, 30, 50, 75 mm) on the rainfall map."
                    )

                with st.spinner("Generating MSWEP map..."):
                    if mode == "Latest Operational Hour":
                        latest_file = files_info[-1]
                        da_ms = load_hourly(latest_file, precip_var_ms, eth_poly, mask_da)
                        plot_rainfall(
                            da_ms,
                            f"MSWEP NRT — {latest_file['datetime'].strftime('%Y-%m-%d %H:%M')}",
                            gdf_eth, bounds, show_contours=show_contours,
                        )
                    elif mode == "7‑day Accumulation":
                        last_7_days_keys = sorted_days[-7:] if len(sorted_days) >= 7 else sorted_days
                        seven_day_total = None
                        for day_key in last_7_days_keys:
                            day_files = day_groups[day_key]
                            # Compute accumulation directly – no caching here (unhashable args)
                            da_day = load_daily_accumulation(day_files, precip_var_ms, eth_poly, mask_da)
                            if da_day is not None:
                                seven_day_total = da_day.copy() if seven_day_total is None else seven_day_total + da_day
                        if seven_day_total is not None:
                            start_date_label = (datetime(sorted_days[-7][0], 1, 1) + timedelta(days=sorted_days[-7][1]-1)
                                                if len(sorted_days) >= 7 else files_info[0]["datetime"])
                            end_date_label = files_info[-1]["datetime"]
                            title = f"MSWEP NRT — 7‑day accumulation ({start_date_label.strftime('%Y-%m-%d')} to {end_date_label.strftime('%Y-%m-%d')})"
                            plot_rainfall(seven_day_total, title, gdf_eth, bounds, show_contours=show_contours)
                        else:
                            st.warning("Not enough daily data to compute 7‑day accumulation.")
                    else:  # Daily Accumulation Total
                        da_ms = load_daily_accumulation(latest_day_files, precip_var_ms, eth_poly, mask_da)
                        plot_rainfall(
                            da_ms,
                            f"Total Accumulated Rainfall from MSWEP — {latest_date}",
                            gdf_eth, bounds, show_contours=show_contours,
                        )

        # IMERG
        with sub_tabs[1]:
            section_header("NASA IMERG NRT Point Verification", "🛰️")
            auth_state = earthdata_auth_status()
            auth_ok = auth_state.get("ok") == "true"

            if auth_ok:
                render_source_status("IMERG preflight", "Authenticated — ready to fetch.", auth_state["message"], "teal")
            else:
                render_source_status(
                    "IMERG preflight",
                    "Not authenticated – will use synthetic data",
                    f"Earthdata credentials missing: {auth_state.get('message', '')}",
                    "amber"
                )

            ctrl_cols = st.columns([1.2, 1.4, 1.6])
            days_back = ctrl_cols[0].slider("Days back", 3, 30, 7, 1, key="imerg_days_back")
            preferred_run = ctrl_cols[1].selectbox("Preferred NRT Run Type", ["late", "early"], index=0, key="imerg_preferred_run_select")
            load_imerg = ctrl_cols[2].checkbox("Fetch NASA IMERG NRT for selected point", value=False, key="load_imerg_nrt")

            if load_imerg:
                with st.spinner(f"Fetching IMERG {preferred_run.upper()} Run …"):
                    imerg_data = fetch_imerg_nrt_cached(lat0, lon0, days_back=days_back, run=preferred_run)

                active_run = st.session_state.get("imerg_active_run", preferred_run)

                if imerg_data.get("status") != "ok":
                    st.error(f"IMERG fetch failed: {imerg_data.get('message', 'Unknown error')}")
                else:
                    if not imerg_data.get("point_in_ethiopia", True):
                        st.warning(f"Point ({lat0:.2f}°N, {lon0:.2f}°E) falls outside the land mask.")

                    snapped = imerg_data.get("latest", {})
                    render_source_status(
                        "Ethiopia mask",
                        f"Active run: **{active_run.upper()}** · "
                        f"Snapped to ({snapped.get('snapped_lat', lat0):.2f}°N, {snapped.get('snapped_lon', lon0):.2f}°E).",
                        imerg_data.get("message", ""),
                        "teal"
                    )
                    daily_df = imerg_data.get("daily", pd.DataFrame())
                    if not daily_df.empty:
                        render_imerg_timeseries(daily_df, snapped, imerg_data.get("half_hourly"))
                    else:
                        st.warning("IMERG returned an empty daily time series.")

        # ---- Onset Readiness Summary (belongs to Rainfall tab) ----
        section_header("Onset Readiness Summary", "🎯")
        gauge_7d = station_signal["total_mm"] if station_signal else 0.0
        gridded_7d = primary_grid_signal["total_mm"] if primary_grid_signal else 0.0
        mswep_7d = mswep_signal["total_mm"] if mswep_signal else 0.0

        likelihood, components = compute_onset_likelihood_full(
            gauge_7d, gridded_7d, mswep_7d, onset_params.get("accum_mm", 20.0),
            mjo_context.get("phase"), mjo_context.get("amplitude"), wind_score
        )

        mjo_note = (f"P{mjo_context['phase']} · {MJO_ETH_IMPACT[mjo_context['phase']][0]}"
                    if mjo_context.get("phase") in MJO_ETH_IMPACT else "MJO unavailable")
        readiness_text = ("Signals are aligned for onset support." if likelihood >= 0.70
                          else "Signals are mixed; monitor synoptic support and persistence." if likelihood >= 0.45
                          else "Readiness remains weak; accumulation and large-scale support are limited.")

        like_cols = st.columns([2.0, 1.0])
        with like_cols[0]:
            render_likelihood_meter(likelihood)
            comp_fig = go.Figure(go.Bar(
                x=list(components.values()), y=list(components.keys()), orientation="h",
                marker=dict(color=["#1A6B9A", "#0D7377", "#6D28D9", "#D97706", "#E11D48"]),
                text=[f"{v:.0%}" for v in components.values()], textposition="outside"
            ))
            comp_fig.update_layout(height=280, plot_bgcolor="#FFFFFF", paper_bgcolor="#F7F9FC",
                                   xaxis=dict(range=[0, 1.2], tickformat=".0%"), showlegend=False)
            st.plotly_chart(comp_fig, width='stretch')
        with like_cols[1]:
            mswep_note = f"{mswep_7d:.1f} mm / 7 days" if mswep_signal else "No data"
            st.markdown(f"""
            <div class='card'>
              <b>Gauge signal:</b> {gauge_7d:.1f} mm / 7 days<br>
              <b>{primary_key or 'Grid'} signal:</b> {gridded_7d:.1f} mm / 7 days<br>
              <b>MSWEP NRT signal:</b> {mswep_note}<br>
              <b>MJO:</b> {mjo_note}<br>
              <b>Wind synoptic:</b> Somali Jet {wind_indices['somali_jet_ms']} m/s, TEJ {wind_indices['tej_index']} m/s<br>
              <b>Wind score:</b> {wind_score:.2f}<br>
              <b>Interpretation:</b> {readiness_text}
            </div>
            """, unsafe_allow_html=True)

    # ---- Clear cache button (outside tabs) ----
    if st.button("🔄 Clear Cache and Refresh Data Streams", key="nrt_refresh", width='stretch'):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

# ---------------------------------------------------------------------
# Sidebar (optional)
# ---------------------------------------------------------------------
def sidebar():
    st.markdown("### NRT Controls")
    # Add any sidebar controls here if needed.
    pass