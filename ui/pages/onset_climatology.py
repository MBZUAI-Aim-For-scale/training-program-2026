"""
ui/pages/onset_climatology.py
"""
import hashlib
import io
import json
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from config import APP_DIR
from plotly.subplots import make_subplots
import streamlit as st

# Updated timeseries renderer import
from ui.pages.woreda_rainfal_ts import show as render_timeseries
from ui.utils.concurrent_processor import run_concurrent_tasks
from config import APP_DIR, DATA_DIR, ZONE_WOREDA_SHP, SHAPEFILE_PATH

try:
    from apps.onset_obs.climatology_engine import (
        BOOTSTRAP_WIDEN_DAYS,
        CLIM_END_YEAR,
        CLIM_START_YEAR,
        CRITICAL_DRY_RUN,
        DEFAULT_SELECTED_WOREDAS,
        DRY_RUN,
        MIN_VALID_YEARS_CONFIDENCE,
        POST_DAYS,
        WET_MM,
        WET_RUN,
        WET_TOTAL_MM,
        run_climatology_engine,
    )
except ImportError as e:
    st.error(f"Failed to import climatology engine: {e}")
    st.stop()

# ============================================================================
# Date Conversion Helper
# ============================================================================
def offset_to_date(offset_days):
    """
    NOTE: 2001 is an arbitrary anchor year used only for date arithmetic.
    It is NOT a real calendar year.
    """
    base_date = datetime(2001, 6, 1)
    return base_date + timedelta(days=float(offset_days))

# ============================================================================
# Disk Cache for Heavy Computation
# ============================================================================
CACHE_DIR = APP_DIR / "data" / "climatology_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_FILE = CACHE_DIR / "climatology_result.pkl"

@st.cache_data(ttl=3600)
def load_cached_result():
    """Load climatology result from disk cache."""
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None
    return None

def save_cached_result(result):
    """Save climatology result to disk cache."""
    try:
        with open(CACHE_FILE, "wb") as f:
            pickle.dump(result, f)
    except Exception:
        pass

def clear_cached_result():
    """Wipes both the in-memory session result and the on-disk cache."""
    st.session_state.pop("clim_result", None)
    try:
        if CACHE_FILE.exists():
            CACHE_FILE.unlink()
    except Exception:
        pass
    load_cached_result.clear()

# ============================================================================
# GeoJSON Serialization Handlers (Cached)
# ============================================================================
def safe_serialize_timestamp(obj):
    """
    Fallback encoder for converting Timestamps and Datetimes into JSON-serializable
    string formats instead of failing.
    """
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} is not serializable")

@st.cache_data(ttl=3600)
def safe_to_json_cached(gdf_json_str: str) -> str:
    """
    Cached wrapper for GeoDataFrame to JSON conversion.
    Takes a pre-serialized string representation to enable caching.
    """
    return gdf_json_str

def safe_to_json(gdf) -> str:
    """
    Safely converts a GeoDataFrame to a JSON dictionary string even if it
    contains pandas Timestamp or Datetime columns.
    """
    return gdf.to_json(default=safe_serialize_timestamp)

# ============================================================================
# Shapefile Loaders
# ============================================================================
@st.cache_resource
def load_shapefile():
    """Load and cache the woreda shapefile."""
    shapefile_path = ZONE_WOREDA_SHP
    if not shapefile_path.exists():
        return None
    try:
        import geopandas as gpd
        gdf = gpd.read_file(shapefile_path)
        # Use newer pyproj API
        if gdf.crs is None:
            gdf = gdf.set_crs(epsg=4326)
        elif gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(epsg=4326)
        return gdf
    except Exception:
        return None

@st.cache_resource
def load_national_geojson():
    """Load the official Ethiopia Admin 3 GeoJSON overlay."""
    geojson_path = SHAPEFILE_PATH
    if not geojson_path.exists():
        return None
    try:
        import geopandas as gpd
        gdf = gpd.read_file(geojson_path)
        #Use newer pyproj API
        if gdf.crs is None:
            gdf = gdf.set_crs(epsg=4326)
        elif gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(epsg=4326)
        return gdf
    except Exception:
        return None

# ============================================================================
# Data Hash Computation (Cached)
# ============================================================================
@st.cache_data(ttl=3600)
def compute_data_hash_cached(series_bytes: bytes) -> str:
    """Cached hash computation from serialized bytes."""
    if not series_bytes:
        return "empty"
    return hashlib.md5(series_bytes).hexdigest()

def compute_data_hash(series) -> str:
    """Compute hash of a pandas series for cache invalidation."""
    if series is None or series.empty:
        return "empty"
    series_bytes = pd.util.hash_pandas_object(series).values.tobytes()
    return compute_data_hash_cached(series_bytes)

# ============================================================================
# Map Creation Functions (Module-Level Cached)
# ============================================================================
@st.cache_data(ttl=3600)
def cached_create_choropleth_map(
    data_hash: str,
    column: str,
    title: str,
    color_continuous_scale: str = "Viridis",
    reverse: bool = False,
    _zones_gdf_json: str = None,
    centroid_y: float = 9.145,
    centroid_x: float = 40.4896,
    zoom: float = 5,
    hover_data_json: str = None,
    z_values_json: str = None,
    locations_json: str = None,
):
    """
    Create a choropleth map with proper caching.
    All GeoDataFrame data is passed as JSON strings to enable caching.
    """
    if _zones_gdf_json is None:
        return None
    try:
        geojson_dict = json.loads(_zones_gdf_json)
        hover_texts = json.loads(hover_data_json) if hover_data_json else []
        z_values = json.loads(z_values_json) if z_values_json else []
        locations = json.loads(locations_json) if locations_json else []
        
        fig = go.Figure(
            go.Choroplethmap(
                geojson=geojson_dict,
                locations=locations,
                z=z_values,
                colorscale=color_continuous_scale,
                reversescale=reverse,
                text=hover_texts,
                hovertemplate="%{text}<extra></extra>",
                marker=dict(opacity=0.7),
                colorbar=dict(title=""),
            )
        )
        fig.update_layout(
            map=dict(
                style="open-street-map",
                center=dict(lat=centroid_y, lon=centroid_x),
                zoom=zoom,
            ),
            margin=dict(l=0, r=0, t=30, b=0),
            height=500,
            paper_bgcolor="#F8FAFC",
            plot_bgcolor="#FFFFFF",
            font_color="#334155",
        )
        return fig
    except Exception as e:
        st.warning(f"Map creation failed: {e}")
        return None

@st.cache_data(ttl=3600)
def cached_create_portfolio_overlay_map(
    data_hash: str,
    _national_gdf_json: str,
    risk_values_json: str,
    pilot_status_json: str,
    hover_texts_json: str,
    centroid_y: float,
    centroid_x: float,
    zoom: float,
):
    """
    Create portfolio overlay map with proper caching.
    All data passed as JSON strings.
    """
    if _national_gdf_json is None:
        return None
    try:
        geojson_dict = json.loads(_national_gdf_json)
        risk_values = json.loads(risk_values_json)
        pilot_status = json.loads(pilot_status_json)
        hover_texts = json.loads(hover_texts_json)
        
        fig = go.Figure(
            go.Choroplethmap(
                geojson=geojson_dict,
                locations=list(range(len(risk_values))),
                z=risk_values,
                colorscale="Reds",
                text=hover_texts,
                hovertemplate="%{text}<extra></extra>",
                marker=dict(opacity=0.85),
                colorbar=dict(title="Risk %"),
            )
        )
        fig.update_layout(
            map=dict(
                style="open-street-map",
                center=dict(lat=centroid_y, lon=centroid_x),
                zoom=zoom,
            ),
            margin=dict(l=0, r=0, t=30, b=0),
            height=500,
            paper_bgcolor="#F8FAFC",
            plot_bgcolor="#FFFFFF",
            font_color="#334155",
            title={
                "text": "Ethiopia National Overlay - Portfolio False-Start Risk Map",
                "y": 0.95,
                "x": 0.5,
                "xanchor": "center",
                "yanchor": "top",
                "font": dict(size=14, color="#334155"),
            },
        )
        return fig
    except Exception as e:
        st.warning(f"Portfolio map creation failed: {e}")
        return None

# ============================================================================
# Plotly Figure Creators (Module-Level Cached)
# ============================================================================
@st.cache_data(ttl=3600)
def create_zone_timeline(zone_name: str, ts_hash: str, zone_ts_json: str):
    """Create zone timeline plot with proper caching."""
    try:
        zone_ts = pd.read_json(io.StringIO(zone_ts_json))
        fig = go.Figure()
        for woreda in sorted(zone_ts["Woreda"].unique()):
            w_df = zone_ts[zone_ts["Woreda"] == woreda].sort_values("Year")
            fig.add_trace(
                go.Scatter(
                    x=w_df["Year"],
                    y=w_df["OnsetOffset"],
                    mode="lines+markers",
                    name=woreda,
                    hovertemplate="Year: %{x}<br>Offset: %{y:.1f} days<extra>%{fullData.name}</extra>",
                )
            )
        fig.update_layout(
            title=f"Onset Offsets - Zone: {zone_name}",
            xaxis_title="Year",
            yaxis_title="Onset offset (days after June 1)",
            template="plotly_white",
            hovermode="x unified",
            height=500,
            font_color="#334155",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
            ),
        )
        return fig
    except Exception as e:
        st.warning(f"Zone timeline creation failed: {e}")
        return None

@st.cache_data(ttl=3600)
def create_guardrail_plot(
    ts_df_hash: str,
    guardrail_woredas_tuple: tuple,
    ts_df_json: str,
):
    """
    Standard horizontal box plot of onset-offset climatology per woreda.
    Woredas are sorted from EARLIEST to LATEST median onset date (top to bottom).
    Shows Q1, median, Q3 as the box; whiskers to 1.5×IQR; outliers as points.
    The mean is overlaid as a dashed line with ±1 SD band.
    """
    try:
        ts_df = pd.read_json(io.StringIO(ts_df_json))
        guardrail_woredas_tuple = tuple(guardrail_woredas_tuple)
        
        # ── Compute median onset offset per woreda for sorting ──────────────
        median_by_woreda = {}
        for woreda in guardrail_woredas_tuple:
            w_data = (
                ts_df[ts_df["Woreda"] == woreda]["OnsetOffset"]
                .dropna()
                .values
            )
            if len(w_data) > 0:
                median_by_woreda[woreda] = float(np.median(w_data))
                
        # Sort woredas by median onset offset (ascending = earliest → latest)
        # Woredas without data are placed at the end
        sorted_woredas = sorted(
            guardrail_woredas_tuple,
            key=lambda w: median_by_woreda.get(w, float("inf")),
        )
        
        fig = go.Figure()
        colors = px.colors.qualitative.Plotly
        for i, woreda in enumerate(sorted_woredas):
            w_data = (
                ts_df[ts_df["Woreda"] == woreda]["OnsetOffset"]
                .dropna()
                .values
            )
            if len(w_data) == 0:
                continue
                
            base_color = colors[i % len(colors)]
            # Parse hex → rgba for the translucent box fill
            r = int(base_color[1:3], 16)
            g = int(base_color[3:5], 16)
            b = int(base_color[5:7], 16)
            fill_color = f"rgba({r}, {g}, {b}, 0.35)"
            
            fig.add_trace(
                go.Box(
                    x=w_data,
                    name=woreda,
                    orientation="h",
                    marker=dict(
                        color=base_color,
                        outliercolor=base_color,
                        size=5,
                        opacity=0.7,
                    ),
                    line=dict(color=base_color, width=1.8),
                    fillcolor=fill_color,
                    boxmean="sd",           # dashed line = mean, dotted band = ±1 SD
                    boxpoints="outliers",   # only show outlier points
                    quartilemethod="linear",
                    hovertemplate=(
                        "Woreda: %{name}<br>"
                        "Value: %{x:.1f} days<br>"
                        f"Median: {median_by_woreda.get(woreda, float('nan')):.1f} days<br>"
                        "<extra></extra>"
                    ),
                )
            )
            
        # Build x-axis tick labels as calendar dates (offset from June 1, 2001)
        tick_offsets = list(range(-30, 181, 30))  # every ~30 days
        tick_labels = [offset_to_date(v).strftime("%b %d") for v in tick_offsets]
        
        # FIX: Combine subtitle into the title string using HTML <br> tag
        # to avoid Plotly's "Invalid property: subtitle" error.
        fig.update_layout(
            title="Guardrails — Selected Woredas (2000–2022 Climatology)<br><span style='font-size: 12px; color: #64748B;'>Sorted by median onset date (earliest at top, latest at bottom)</span>",
            xaxis_title="Climatological Onset Date",
            yaxis_title="Woreda",
            template="plotly_white",
            showlegend=False,
            margin=dict(l=160, r=20, t=80, b=40), # Increased top margin for the 2-line title
            height=max(340, len(sorted_woredas) * 55),
            font_color="#334155",
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#F8FAFC",
        )
        fig.update_xaxes(
            tickvals=tick_offsets,
            ticktext=tick_labels,
            gridcolor="#E2E8F0",
            zeroline=True,
            zerolinecolor="#CBD5E1",
            zerolinewidth=1,
        )
        fig.update_yaxes(
            gridcolor="#F1F5F9",
            automargin=True,
            # Reverse y-axis so earliest onset appears at the TOP
            autorange="reversed",
        )
        return fig
    except Exception as e:
        st.warning(f"Guardrail plot creation failed: {e}")
        return None

@st.cache_data(ttl=3600)
def build_portfolio_timelines_figure(selected_tuple: tuple, ts_hash: str, portfolio_ts_json: str):
    """Build portfolio timelines figure with proper caching."""
    try:
        portfolio_ts = pd.read_json(io.StringIO(portfolio_ts_json))
        selected_tuple = tuple(selected_tuple)
        
        sub = portfolio_ts[
            portfolio_ts["Woreda"].isin(selected_tuple)
        ].sort_values(["Zone", "Woreda"])
        woredas_sorted = (
            sub.drop_duplicates(subset=["Woreda"])
            .sort_values(["Zone", "Woreda"])["Woreda"]
            .tolist()
        )
        n = len(woredas_sorted)
        n_cols = 3 if n > 1 else 1
        n_rows = int(np.ceil(n / n_cols))
        
        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=woredas_sorted,
            vertical_spacing=0.12 / max(n_rows, 1),
            horizontal_spacing=0.06,
        )
        for idx, woreda in enumerate(woredas_sorted):
            r = idx // n_cols + 1
            c = idx % n_cols + 1
            w_df = sub[sub["Woreda"] == woreda].sort_values("Year")
            w_mean = w_df["OnsetOffset"].mean()
            date_strs = [
                offset_to_date(v).strftime("%b %d") if np.isfinite(v) else "N/A"
                for v in w_df["OnsetOffset"]
            ]
            fig.add_trace(
                go.Scatter(
                    x=w_df["Year"],
                    y=w_df["OnsetOffset"],
                    mode="lines+markers",
                    line=dict(color="#059669", width=1.8),
                    marker=dict(size=4),
                    customdata=date_strs,
                    hovertemplate="Year: %{x}<br>Onset: %{customdata}<extra></extra>",
                    showlegend=False,
                ),
                row=r,
                col=c,
            )
            if np.isfinite(w_mean):
                fig.add_hline(
                    y=w_mean,
                    line=dict(color="#F97316", dash="dot", width=1.5),
                    row=r,
                    col=c,
                )
        fig.update_layout(
            height=max(280, 240 * n_rows),
            template="plotly_white",
            margin=dict(l=10, r=10, t=40, b=10),
            font_color="#334155",
        )
        fig.update_yaxes(title_text="Days after Jun 1", title_font_size=10)
        return fig
    except Exception as e:
        st.warning(f"Portfolio timelines creation failed: {e}")
        return None

@st.cache_data(ttl=3600)
def build_stability_figure(selected_tuple: tuple, ts_hash: str, portfolio_ts_json: str):
    """Build stability figure with proper caching."""
    try:
        portfolio_ts = pd.read_json(io.StringIO(portfolio_ts_json))
        selected_tuple = tuple(selected_tuple)
        
        rows = []
        for woreda in selected_tuple:
            w_ts = portfolio_ts[portfolio_ts["Woreda"] == woreda].sort_values("Year")
            epoch1 = w_ts[w_ts["Year"] <= 2001]["OnsetOffset"].dropna().values
            epoch2 = w_ts[w_ts["Year"] > 2001]["OnsetOffset"].dropna().values
            if len(epoch1) == 0 or len(epoch2) == 0:
                continue
            rows.append({
                "Woreda": woreda,
                "DecadalShift_Days": float(np.median(epoch2) - np.median(epoch1)),
                "InterannualVolatility_Days": float(np.nanstd(w_ts["OnsetOffset"].values)),
            })
            
        stability_df = pd.DataFrame(rows).sort_values("DecadalShift_Days")
        if stability_df.empty:
            return None, stability_df
            
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        bar_colors = [
            "#d7191c" if v > 0 else "#2c7bb6"
            for v in stability_df["DecadalShift_Days"]
        ]
        fig.add_trace(
            go.Bar(
                x=stability_df["Woreda"],
                y=stability_df["DecadalShift_Days"],
                marker_color=bar_colors,
                name="Decadal Shift (days)",
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=stability_df["Woreda"],
                y=stability_df["InterannualVolatility_Days"],
                mode="lines+markers",
                line=dict(color="#F97316", width=2.5),
                marker=dict(symbol="diamond", size=8),
                name="Interannual Volatility (σ days)",
            ),
            secondary_y=True,
        )
        fig.update_layout(
            template="plotly_white",
            height=420,
            margin=dict(l=10, r=10, t=20, b=60),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
            ),
        )
        fig.update_yaxes(
            title_text="Decadal Shift (days, +later / -earlier)",
            secondary_y=False,
        )
        fig.update_yaxes(
            title_text="Interannual Volatility (σ days)", secondary_y=True
        )
        fig.update_xaxes(tickangle=40)
        return fig, stability_df
    except Exception as e:
        st.warning(f"Stability figure creation failed: {e}")
        return None, pd.DataFrame()

# ============================================================================
# Visual Theme & Summary Helpers
# ============================================================================
def inject_theme_css():
    """Inject theme CSS for the onset climatology page."""
    st.markdown(
        """
        <style>
        div[data-testid="stRadio"] > div{flex-direction:row !important; gap:0.4rem;}
        div[data-testid="stRadio"] label{
            background: #F1F5F9; border: 1px solid #CBD5E1; padding: 0.55rem 1.4rem;
            border-radius: 8px; cursor: pointer; transition: all 0.15s ease;
            font-weight: 500;
        }
        div[data-testid="stRadio"] label:hover{
            border-color: #10B981; background: #ECFDF5;
        }
        div[data-testid="stRadio"] label[data-checked="true"]{
            background: #10B981 !important; color: white !important; border-color: #10B981 !important;
            box-shadow: 0 2px 6px rgba(16,185,129,0.35);
        }
        div[data-testid="stMetric"]{
            background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px;
            padding: 0.9rem 1rem; box-shadow: 0 1px 2px rgba(15,23,42,0.04);
        }
        div[data-testid="stMetricLabel"]{ color: #64748B; font-weight: 600; }
        .confidence-high{
            background:#DCFCE7; color:#166534; padding:2px 10px; border-radius:999px;
            font-weight:600; font-size:0.8rem;
        }
        .confidence-low{
            background:#FEF3C7; color:#92400E; padding:2px 10px; border-radius:999px;
            font-weight:600; font-size:0.8rem;
        }
        .clim-header-sub{ color:#64748B; font-size:1rem; margin-top:-0.6rem; }
        .decision-stop{
            background: linear-gradient(135deg, #10B981 0%, #059669 100%);
            color: white; padding: 1.2rem 1.5rem; border-radius: 14px;
            box-shadow: 0 4px 14px rgba(16,185,129,0.35);
            text-align: center; font-weight: 700;
        }
        .decision-continue{
            background: linear-gradient(135deg, #F59E0B 0%, #D97706 100%);
            color: white; padding: 1.2rem 1.5rem; border-radius: 14px;
            box-shadow: 0 4px 14px rgba(245,158,11,0.35);
            text-align: center; font-weight: 700;
        }
        .decision-uncertain{
            background: linear-gradient(135deg, #EF4444 0%, #B91C1C 100%);
            color: white; padding: 1.2rem 1.5rem; border-radius: 14px;
            box-shadow: 0 4px 14px rgba(239,68,68,0.35);
            text-align: center; font-weight: 700;
        }
        .dataset-ok{
            background:#DCFCE7; color:#166534; padding:4px 12px; border-radius:999px;
            font-weight:600; font-size:0.8rem; display:inline-block; margin:2px;
        }
        .dataset-missing{
            background:#FEE2E2; color:#991B1B; padding:4px 12px; border-radius:999px;
            font-weight:600; font-size:0.8rem; display:inline-block; margin:2px;
        }
        .dataset-delayed{
            background:#FEF3C7; color:#92400E; padding:4px 12px; border-radius:999px;
            font-weight:600; font-size:0.8rem; display:inline-block; margin:2px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

def render_analytical_summary(header: str, description: str, metrics: list):
    """
    Renders a standardized two-column summary block for analytical plots.
    Left column: Contextual description.
    Right column: Key dynamic metrics.
    Collapsed by default to reduce visual clutter.
    """
    with st.expander(f"View Analysis Details: {header}", expanded=False):
        col_text, col_metrics = st.columns([2, 1])
        with col_text:
            st.markdown(
                f"<p style='color: #475569; font-size: 0.9rem; line-height: 1.5; margin-top: -0.5rem;'>{description}</p>",
                unsafe_allow_html=True,
            )
        with col_metrics:
            for m in metrics:
                st.metric(label=m["label"], value=m["value"], delta=m.get("delta"))
            st.markdown("<br>", unsafe_allow_html=True)

def render_controls_panel():
    """Render the sidebar controls panel."""
    with st.sidebar:
        st.markdown("### Climatology Configuration")
        dataset = st.selectbox(
            "Dataset", ["ENACTS", "CHIRPS", "IMERG"], key="onset_dataset_select"
        )
        threshold = st.slider(
            "Wet threshold (mm)", 0, 50, 20, key="onset_threshold_slider"
        )
        st.session_state["clim_dataset"] = dataset
        st.session_state["clim_threshold"] = threshold
        
        st.divider()
        st.markdown("### Cache Management")
        if st.button("Recompute Climatology", width="stretch"):
            clear_cached_result()
            st.rerun()

# ============================================================================
# NRT Data Loader & AI Decision Engine (New)
# ============================================================================
@st.cache_data(ttl=900)  # 15-minute cache for NRT data
def _load_nrt_dataset(dataset_name: str, woreda: str):
    """
    Attempt to load near-real-time rainfall data for a given dataset and woreda.
    Returns a dict with keys: {status, data, last_updated, message}
    - status: 'ok' | 'missing' | 'delayed' | 'error'
    - data: DataFrame with columns [Date, Rainfall_mm] or None
    """
    # Expected file paths (flexible - adapt to your actual data layout)
    dataset_paths = {
        "MSWEP": DATA_DIR / "nrt" / "mswep" / f"{woreda.replace(' ', '_')}.csv",
        "CHIRPS": DATA_DIR / "nrt" / "chirps" / f"{woreda.replace(' ', '_')}.csv",
        "ENACTS": DATA_DIR / "nrt" / "enacts" / f"{woreda.replace(' ', '_')}.csv",
    }
    
    result = {
        "status": "missing",
        "data": None,
        "last_updated": None,
        "message": "",
        "dataset": dataset_name,
    }
    
    path = dataset_paths.get(dataset_name)
    if path is None or not path.exists():
        result["message"] = f"{dataset_name} file not found at expected path."
        return result
    
    try:
        df = pd.read_csv(path, parse_dates=["Date"])
        if df.empty:
            result["status"] = "missing"
            result["message"] = f"{dataset_name} file is empty."
            return result
        
        # Check for staleness (data older than 7 days = delayed)
        last_date = df["Date"].max()
        days_old = (datetime.now() - last_date).days
        result["last_updated"] = last_date
        
        if days_old > 7:
            result["status"] = "delayed"
            result["message"] = f"{dataset_name} data is {days_old} days old."
        else:
            result["status"] = "ok"
            result["message"] = f"{dataset_name} data current (last: {last_date.strftime('%Y-%m-%d')})."
        
        # Normalize columns
        if "Rainfall_mm" not in df.columns:
            # Try common alternatives
            for alt in ["rain", "precip", "prcp", "RF"]:
                if alt in df.columns:
                    df = df.rename(columns={alt: "Rainfall_mm"})
                    break
        
        result["data"] = df[["Date", "Rainfall_mm"]].dropna().sort_values("Date")
        return result
    
    except Exception as e:
        result["status"] = "error"
        result["message"] = f"Error loading {dataset_name}: {str(e)[:80]}"
        return result


def _compute_weekly_climatology(ts_df: pd.DataFrame, woreda: str):
    """
    Compute weekly climatological rainfall statistics for a given woreda.
    Returns a DataFrame with columns: [Week_Label, Week_Start_Offset, Mean, Std, P10, P90]
    """
    w_ts = ts_df[ts_df["Woreda"] == woreda].copy()
    if w_ts.empty:
        return pd.DataFrame()
    
    # Define weekly bins starting May 25 (offset -7 from June 1)
    bin_edges = list(range(-7, 150, 7))
    bin_labels = []
    bin_starts = []
    for i in range(len(bin_edges) - 1):
        start_date = offset_to_date(bin_edges[i])
        bin_labels.append(f"W{i}: {start_date.strftime('%b %d')}")
        bin_starts.append(bin_edges[i])
    
    w_ts["Week_Bin"] = pd.cut(
        w_ts["OnsetOffset"],
        bins=bin_edges,
        labels=bin_labels,
        right=False,
        include_lowest=True,
    )
    w_ts = w_ts.dropna(subset=["Week_Bin"])
    
    # For each week, compute stats across years
    rows = []
    for label, start_off in zip(bin_labels, bin_starts):
        week_data = w_ts[w_ts["Week_Bin"] == label]["OnsetOffset"].dropna()
        if len(week_data) == 0:
            continue
        rows.append({
            "Week_Label": label,
            "Week_Start_Offset": start_off,
            "Mean": float(week_data.mean()),
            "Std": float(week_data.std()) if len(week_data) > 1 else 0.0,
            "P10": float(week_data.quantile(0.10)),
            "P50": float(week_data.median()),
            "P90": float(week_data.quantile(0.90)),
            "Sample_Years": len(week_data),
        })
    
    return pd.DataFrame(rows)


def _evaluate_forecast_against_climatology(forecast_offset: float, clim_df: pd.DataFrame):
    """
    Compare a forecast onset offset against the weekly climatology.
    Returns a dict with:
      - decision: 'STOP' | 'CONTINUE' | 'UNCERTAIN'
      - reason: str
      - confidence: float (0-100)
      - within_range: bool
    """
    if clim_df.empty or not np.isfinite(forecast_offset):
        return {
            "decision": "UNCERTAIN",
            "reason": "Insufficient climatology or invalid forecast.",
            "confidence": 0.0,
            "within_range": False,
            "nearest_week": None,
        }
    
    # Find the week where the forecast offset falls
    clim_df = clim_df.sort_values("Week_Start_Offset").reset_index(drop=True)
    
    # Find nearest week
    clim_df["distance"] = (clim_df["Week_Start_Offset"] - forecast_offset).abs()
    nearest_idx = clim_df["distance"].idxmin()
    nearest_week = clim_df.loc[nearest_idx]
    
    # Check if forecast is within the climatological range (P10 - P90)
    p10 = nearest_week["P10"]
    p90 = nearest_week["P90"]
    within_range = bool(p10 <= forecast_offset <= p90)
    
    # Confidence: higher if forecast is near the median and within range
    std = nearest_week["Std"] if nearest_week["Std"] > 0 else 1.0
    z_score = abs(forecast_offset - nearest_week["P50"]) / std
    confidence = max(0, min(100, 100 * (1 - z_score / 3)))  # Cap at 3 SD
    
    if within_range and confidence >= 60:
        decision = "STOP"
        reason = (
            f"Forecast onset ({forecast_offset:+.1f} days) falls within the "
            f"expected climatological range for {nearest_week['Week_Label']} "
            f"(P10={p10:.1f}, P90={p90:.1f}). Confidence: {confidence:.0f}%."
        )
    elif within_range:
        decision = "CONTINUE"
        reason = (
            f"Forecast is within range but confidence is low ({confidence:.0f}%). "
            f"Consider additional verification before finalizing."
        )
    else:
        decision = "UNCERTAIN"
        reason = (
            f"Forecast onset ({forecast_offset:+.1f} days) falls OUTSIDE the "
            f"expected climatological range for {nearest_week['Week_Label']} "
            f"(P10={p10:.1f}, P90={p90:.1f}). Further forecasting recommended."
        )
    
    return {
        "decision": decision,
        "reason": reason,
        "confidence": confidence,
        "within_range": within_range,
        "nearest_week": nearest_week["Week_Label"],
    }


# ============================================================================
# Tab Renderers
# ============================================================================
def render_interactive_maps_tab(zones_gdf, all_woredas, ts_df):
    """Render interactive maps tab with weekly onset intelligence and woreda highlighting."""
    st.subheader("Interactive Overview Maps")
    
    # Metrics cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            f"""
            <div style='background: #F0F9FF; padding: 1rem; border-radius: 10px; 
                        border-left: 4px solid #0EA5E9; text-align: center;'>
                <div style='font-size: 2rem; font-weight: 700; color: #0369A1;'>{len(all_woredas)}</div>
                <div style='color: #64748B; font-size: 0.85rem; font-weight: 500;'>Total Woredas</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div style='background: #F0FDF4; padding: 1rem; border-radius: 10px; 
                        border-left: 4px solid #10B981; text-align: center;'>
                <div style='font-size: 2rem; font-weight: 700; color: #047857;'>4</div>
                <div style='color: #64748B; font-size: 0.85rem; font-weight: 500;'>Active Layers</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """
            <div style='background: #FFFBEB; padding: 1rem; border-radius: 10px; 
                        border-left: 4px solid #F59E0B; text-align: center;'>
                <div style='font-size: 2rem; font-weight: 700; color: #B45309;'>ENACTS</div>
                <div style='color: #64748B; font-size: 0.85rem; font-weight: 500;'>Dataset</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ── Weekly Onset Window Selector ───────────────────────────────────────
    st.markdown("##### Weekly Onset Analysis (Starting May 25)")
    
    # Build week bins
    bin_edges = list(range(-7, 150, 7))
    week_info = []
    for i in range(len(bin_edges) - 1):
        start_date = offset_to_date(bin_edges[i])
        end_date = offset_to_date(bin_edges[i+1] - 1)
        week_info.append({
            "idx": i,
            "label": f"W{i}",
            "display": f"W{i}: {start_date.strftime('%b %d')} - {end_date.strftime('%b %d')}",
            "start_offset": bin_edges[i],
            "end_offset": bin_edges[i+1] - 1,
        })
    
    week_labels_internal = [w["label"] for w in week_info]
    
    # Compute weekly probabilities per woreda
    plot_df = ts_df.copy()
    plot_df["Week_Bin"] = pd.cut(
        plot_df["OnsetOffset"],
        bins=bin_edges,
        labels=week_labels_internal,
        right=False,
        include_lowest=True,
    )
    plot_df = plot_df.dropna(subset=["Week_Bin"])
    
    week_counts = plot_df.groupby(["Woreda", "Week_Bin"]).size().reset_index(name="Count")
    total_years = plot_df.groupby("Woreda").size().reset_index(name="Total")
    prob_df = week_counts.merge(total_years, on="Woreda")
    prob_df["Probability"] = (prob_df["Count"] / prob_df["Total"]) * 100
    
    # Week selector
    col_week, col_woreda = st.columns(2)
    with col_week:
        selected_week_display = st.selectbox(
            "Select onset week",
            [w["display"] for w in week_info],
            key="map_week_select"
        )
        selected_week_internal = [w["label"] for w in week_info if w["display"] == selected_week_display][0]
    
    # Get probabilities for selected week
    week_prob = prob_df[prob_df["Week_Bin"] == selected_week_internal][["Woreda", "Probability"]].copy()
    all_woredas_df = pd.DataFrame({"Woreda": all_woredas})
    week_prob = all_woredas_df.merge(week_prob, on="Woreda", how="left").fillna(0)
    week_prob = week_prob.sort_values("Probability", ascending=False).reset_index(drop=True)
    
    # Filter to only show woredas that actually have onset in this week (Probability > 0)
    week_prob_active = week_prob[week_prob["Probability"] > 0].copy()
    if week_prob_active.empty:
        week_prob_active = week_prob.copy()
        st.info("No historical onset recorded for any woreda in this specific week.")

    # Woreda selector (ranked by probability)
    with col_woreda:
        dropdown_labels = [f"{row['Woreda']} ({row['Probability']:.1f}%)" for _, row in week_prob_active.iterrows()]
        
        selected_woreda_label = st.selectbox(
            "Select woreda to highlight",
            dropdown_labels,
            key="map_woreda_select"
        )
        
        selected_woreda = selected_woreda_label.rsplit(" (", 1)[0]
        
        prob_match = week_prob[week_prob["Woreda"] == selected_woreda]["Probability"]
        selected_woreda_prob = float(prob_match.iloc[0]) if not prob_match.empty else 0.0
    
    # ── Map Layer Selector ─────────────────────────────────────────────────
    map_options = {
        "Median Onset Date": ("clim_median_offset", "Viridis", False, "days", 
                              "Average onset timing across the historical record"),
        "Interannual Volatility": ("clim_std_offset", "Plasma", False, "days",
                                  "Year-to-year variability in onset timing"),
        "Probability of Onset": ("clim_pct_onset", "RdYlGn", False, "%",
                                "Likelihood of onset detection in historical record"),
        "False-Start Risk": ("false_start_prob", "Reds", False, "%",
                            "Risk of premature onset followed by dry spell"),
    }
    
    selected_map_label = st.selectbox(
        "Select a map layer to visualize",
        list(map_options.keys())
    )
    
    col_name, cmap, reverse, unit, description = map_options[selected_map_label]
    
    st.markdown(
        f"<p style='color: #475569; font-size: 0.9rem; background: #F8FAFC; "
        f"padding: 0.8rem; border-radius: 8px; border-left: 3px solid #6366F1;'>"
        f"<strong>{selected_map_label}:</strong> {description}</p>",
        unsafe_allow_html=True,
    )
    
    col_map, col_info = st.columns([2.5, 1])
    
    with col_map:
        if col_name in zones_gdf.columns:
            # Prepare data for caching
            data_hash = compute_data_hash(zones_gdf[col_name])
            zones_gdf_json = safe_to_json(zones_gdf)
            
            centroid_y = zones_gdf.geometry.centroid.y.mean()
            centroid_x = zones_gdf.geometry.centroid.x.mean()
            zoom = 6
            
            hover_texts = []
            z_values = []
            for idx, row in zones_gdf.iterrows():
                val = row[col_name]
                val_str = f"{val:.1f}" if not pd.isna(val) else "N/A"
                hover_texts.append(
                    f"<b style='font-size: 1.1em;'>{row['adm3_name']}</b><br>"
                    f"<span style='color: #64748B;'>Zone:</span> {row['adm2_name']}<br>"
                    f"<span style='color: #64748B;'>{col_name}:</span> <b>{val_str} {unit}</b>"
                )
                z_values.append(float(val) if not pd.isna(val) else None)
                
            fig_map = cached_create_choropleth_map(
                data_hash=data_hash,
                column=col_name,
                title=selected_map_label,
                color_continuous_scale=cmap,
                reverse=reverse,
                _zones_gdf_json=zones_gdf_json,
                centroid_y=centroid_y,
                centroid_x=centroid_x,
                zoom=zoom,
                hover_data_json=json.dumps(hover_texts),
                z_values_json=json.dumps(z_values),
                locations_json=json.dumps(list(zones_gdf.index)),
            )
            
            if fig_map:
                # Add highlight marker for selected woreda
                selected_row = zones_gdf[zones_gdf["adm3_name"] == selected_woreda]
                if not selected_row.empty:
                    fig_map = go.Figure(fig_map)
                    centroid = selected_row.geometry.centroid.iloc[0]
                    fig_map.add_trace(go.Scattermap(
                        lat=[centroid.y],
                        lon=[centroid.x],
                        mode="markers+text",
                        marker=dict(size=22, color="#EF4444", symbol="star", opacity=0.9),
                        text=[selected_woreda],
                        textposition="top center",
                        textfont=dict(size=12, color="#EF4444"),
                        name="Selected Woreda",
                        hoverinfo="text",
                        hovertext=f"<b>{selected_woreda}</b><br>Weekly Probability: {selected_woreda_prob:.1f}%",
                    ))
                st.plotly_chart(fig_map, width='stretch')
                
                # ── NEW: Weekly Probability Distribution Chart ────────────────
                st.markdown("##### Weekly Onset Probability Distribution")
                st.caption(f"Historical onset probability by week for **{selected_woreda}** (starting May 25)")
                
                # Get all weekly probabilities for selected woreda
                woreda_weekly_prob = prob_df[prob_df["Woreda"] == selected_woreda].copy()
                woreda_weekly_prob = woreda_weekly_prob.sort_values("Week_Bin")
                
                # Create bar chart
                colors = []
                for week_label in woreda_weekly_prob["Week_Bin"]:
                    if week_label == selected_week_internal:
                        colors.append("#EF4444")  # Highlight selected week
                    else:
                        colors.append("#3B82F6")  # Normal color
                
                fig_weekly_dist = go.Figure(data=[
                    go.Bar(
                        x=woreda_weekly_prob["Week_Bin"],
                        y=woreda_weekly_prob["Probability"],
                        marker_color=colors,
                        hovertemplate="Week: %{x}<br>Probability: %{y:.1f}%<extra></extra>",
                    )
                ])
                
                # Add median onset line
                woreda_data = zones_gdf[zones_gdf["adm3_name"] == selected_woreda]
                if not woreda_data.empty:
                    median_offset = woreda_data.iloc[0].get("clim_median_offset", np.nan)
                    if np.isfinite(median_offset):
                        # Find which week the median falls in
                        for i, (start, end) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
                            if start <= median_offset < end:
                                median_week_label = week_labels_internal[i]
                                median_idx = list(woreda_weekly_prob["Week_Bin"]).index(median_week_label) if median_week_label in woreda_weekly_prob["Week_Bin"].values else -1
                                if median_idx >= 0:
                                    fig_weekly_dist.add_vline(
                                        x=median_idx,
                                        line_dash="dash",
                                        line_color="#10B981",
                                        annotation_text=f"Median: {offset_to_date(median_offset).strftime('%b %d')}",
                                        annotation_position="top right",
                                    )
                                break
                
                fig_weekly_dist.update_layout(
                    template="plotly_white",
                    height=300,
                    margin=dict(l=40, r=20, t=20, b=80),
                    xaxis=dict(tickangle=45, title="Week"),
                    yaxis=dict(title="Probability (%)", range=[0, max(woreda_weekly_prob["Probability"].max() * 1.1, 10)]),
                    font_color="#334155",
                    showlegend=False,
                )
                
                st.plotly_chart(fig_weekly_dist, width='stretch')
        else:
            st.warning(f"Column '{col_name}' not found. Try running the engine again.")
            
    with col_info:
        st.markdown(
            """
            <div style='background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%); 
                        padding: 1rem; border-radius: 10px; margin-bottom: 1rem;'>
                <h4 style='color: white; margin: 0; font-size: 1.1rem;'>
                    Map Intelligence
                </h4>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        # ── Selected Woreda Details ────────────────────────────────────────
        st.markdown(
            f"""
            <div style='background: #FFFFFF; padding: 1rem; border-radius: 8px; 
                        border: 2px solid #EF4444; margin-bottom: 0.8rem;'>
                <div style='color: #EF4444; font-size: 0.8rem; font-weight: 700; 
                            text-transform: uppercase; margin-bottom: 0.5rem;'>
                    Selected Woreda
                </div>
                <div style='font-size: 1.3rem; font-weight: 700; color: #0F172A; margin-bottom: 0.5rem;'>
                    {selected_woreda}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        # Get selected woreda details
        woreda_data = zones_gdf[zones_gdf["adm3_name"] == selected_woreda]
        if not woreda_data.empty:
            woreda_row = woreda_data.iloc[0]
            median_offset = woreda_row.get("clim_median_offset", np.nan)
            median_date = offset_to_date(median_offset).strftime("%b %d") if np.isfinite(median_offset) else "N/A"
            
            st.markdown(
                f"""
                <div style='background: #FFFFFF; padding: 1rem; border-radius: 8px; 
                            border: 1px solid #E2E8F0; margin-bottom: 0.8rem;'>
                    <div style='color: #64748B; font-size: 0.8rem; font-weight: 600; margin-bottom: 0.5rem;'>
                        ONSET CRITERIA
                    </div>
                    <div style='display: flex; justify-content: space-between; margin: 0.3rem 0;'>
                        <span style='color: #475569;'>Median Onset:</span>
                        <span style='font-weight: 600; color: #0F172A;'>{median_date}</span>
                    </div>
                    <div style='display: flex; justify-content: space-between; margin: 0.3rem 0;'>
                        <span style='color: #475569;'>Selected Week Probability:</span>
                        <span style='font-weight: 600; color: #0F172A;'>{selected_woreda_prob:.1f}%</span>
                    </div>
                    <div style='display: flex; justify-content: space-between; margin: 0.3rem 0;'>
                        <span style='color: #475569;'>Zone:</span>
                        <span style='font-weight: 600; color: #0F172A;'>{woreda_row.get('adm2_name', 'N/A')}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            
            # Onset detection criteria
            st.markdown(
                f"""
                <div style='background: #F0F9FF; padding: 0.8rem; border-radius: 8px; 
                            border-left: 3px solid #0EA5E9; margin-bottom: 0.8rem;'>
                    <div style='color: #0369A1; font-size: 0.8rem; font-weight: 600; margin-bottom: 0.3rem;'>
                        DETECTION CRITERIA
                    </div>
                    <div style='color: #475569; font-size: 0.85rem; line-height: 1.5;'>
                        <b>Wet threshold:</b> {WET_MM} mm<br>
                        <b>Wet run:</b> {WET_RUN} days<br>
                        <b>Dry run:</b> {DRY_RUN} days<br>
                        <b>Total wet:</b> {WET_TOTAL_MM} mm in {POST_DAYS} days
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        
        # ── Ranked Woredas for Selected Week ───────────────────────────────
        st.markdown(
            """
            <div style='background: #FFFFFF; padding: 1rem; border-radius: 8px; 
                        border: 1px solid #E2E8F0;'>
                <div style='color: #64748B; font-size: 0.8rem; font-weight: 600; margin-bottom: 0.5rem;'>
                    TOP WOREDAS FOR SELECTED WEEK
                </div>
            """,
            unsafe_allow_html=True,
        )
        
        top_n = min(10, len(week_prob_active))
        if top_n == 0:
            st.markdown("<p style='color: #64748B; font-size: 0.85rem; padding: 0.5rem;'>No woredas have historical onset recorded in this specific week.</p>", unsafe_allow_html=True)
        else:
            for rank, (_, row) in enumerate(week_prob_active.head(top_n).iterrows(), 1):
                is_selected = row["Woreda"] == selected_woreda
                bg_color = "#ECFDF5" if is_selected else "#F8FAFC"
                border_color = "#10B981" if is_selected else "transparent"
                
                st.markdown(
                    f"""
                    <div style='padding: 0.6rem; background: {bg_color}; border-radius: 6px; 
                                margin: 0.3rem 0; border-left: 3px solid {border_color};'>
                        <div style='display: flex; justify-content: space-between; align-items: center;'>
                            <div>
                                <span style='font-weight: 600; color: #0F172A;'>#{rank}</span>
                                <span style='margin-left: 0.5rem; font-weight: 600; color: #0F172A;'>
                                    {row['Woreda']}
                                </span>
                            </div>
                            <div style='font-weight: 700; color: #059669; font-size: 1.1rem;'>
                                {row['Probability']:.1f}%
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown(
            "<p style='font-size: 0.75rem; color: #94A3B8; margin-top: 1rem; "
            "padding: 0.5rem; background: #F1F5F9; border-radius: 6px;'>"
            "<b>Tip:</b> Select different weeks to see how onset probabilities shift geographically. "
            "The dropdown automatically filters to show only woredas active in the selected week."
            "</p>",
            unsafe_allow_html=True,
        )
def render_zone_timelines_tab(ts_df):
    """Render zone timelines tab with proper caching."""
    st.subheader("Zone Timelines")
    zones = sorted(ts_df["Zone"].unique())
    selected_zone = st.selectbox("Select a zone", zones)
    zone_ts = ts_df[ts_df["Zone"] == selected_zone]
    
    if zone_ts.empty:
        st.warning(f"No data for zone '{selected_zone}'")
        return
        
    n_woredas = zone_ts["Woreda"].nunique()
    zone_mean = zone_ts["OnsetOffset"].mean()
    zone_std = zone_ts["OnsetOffset"].std()
    
    render_analytical_summary(
        header="Temporal Variability Analysis",
        description=f"Displays the interannual onset offset trends for individual woredas within the **{selected_zone}** zone. This view helps identify synchronized shifts or localized anomalies in rainy season onset over the historical record.",
        metrics=[
            {"label": "Woredas in Zone", "value": n_woredas},
            {
                "label": "Mean Onset",
                "value": offset_to_date(zone_mean).strftime("%b %d") if np.isfinite(zone_mean) else "N/A",
            },
            {
                "label": "Volatility (σ)",
                "value": f"{zone_std:.1f} days" if np.isfinite(zone_std) else "N/A",
            },
        ]
    )
    
    # Prepare data for caching
    ts_hash = hashlib.md5(
        pd.util.hash_pandas_object(zone_ts).values.tobytes()
    ).hexdigest()
    zone_ts_json = zone_ts.to_json()
    
    fig_zone = create_zone_timeline(selected_zone, ts_hash, zone_ts_json)
    if fig_zone:
        st.plotly_chart(fig_zone, width='stretch')

def _style_confidence_column(styler):
    """Style the confidence column in the guardrails table."""
    def _fmt(val):
        if val == "HIGH":
            return "background-color:#DCFCE7; color:#166534; font-weight:600; text-align:center; border-radius:4px;"
        if val == "LOW":
            return "background-color:#FEF3C7; color:#92400E; font-weight:600; text-align:center; border-radius:4px;"
        return ""
        
    if hasattr(styler, "map"):
        return styler.map(_fmt, subset=["Confidence"])
    return styler.applymap(_fmt, subset=["Confidence"])

def render_guardrails_tab(guardrail_df, selected_woredas, all_woredas, ts_df):
    """Render guardrails tab with proper caching."""
    st.subheader("Operational Guardrails")
    guardrail_woredas = selected_woredas if selected_woredas else all_woredas
    table_df = guardrail_df[guardrail_df["Woreda"].isin(guardrail_woredas)]
    
    n_high = int((table_df["Confidence"] == "HIGH").sum()) if not table_df.empty else 0
    n_low = int((table_df["Confidence"] == "LOW").sum()) if not table_df.empty else 0
    total_analyzed = len(table_df)
    high_pct = f"{(n_high/total_analyzed*100):.0f}%" if total_analyzed > 0 else "0%"
    
    render_analytical_summary(
        header="Climatological Confidence Bounds",
        description=f"The horizontal box plots illustrate the 5th to 95th percentile climatological range, with the median onset marked. Woredas with fewer than {MIN_VALID_YEARS_CONFIDENCE} valid years are flagged as LOW confidence and widened by ±{BOOTSTRAP_WIDEN_DAYS:.0f} days as a bootstrap safety margin.",
        metrics=[
            {"label": "Woredas Analyzed", "value": total_analyzed},
            {"label": "High Confidence", "value": n_high, "delta": high_pct},
            {"label": "Low Confidence", "value": n_low},
        ]
    )
    
    missing_from_guardrails = sorted(
        set(guardrail_woredas) - set(guardrail_df["Woreda"])
    )
    if missing_from_guardrails:
        st.warning(
            f"No guardrail could be computed for {len(missing_from_guardrails)} selected "
            f"woreda(s) - insufficient valid onset detections across the record: "
            f"{', '.join(missing_from_guardrails)}"
        )
        
    st.markdown("##### Guardrails Data Table")
    if not table_df.empty:
        display_cols = [
            "Woreda",
            "Zone",
            "Median_Valid_Date",
            "Earliest_Valid_Date",
            "Latest_Valid_Date",
            "Confidence",
            "SampleSize_Years",
        ]
        styled = _style_confidence_column(table_df[display_cols].style)
        st.dataframe(styled, width='stretch', hide_index=True)
        
        csv_bytes = table_df[display_cols].to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Guardrails (CSV)",
            data=csv_bytes,
            file_name="operational_sms_woreda_guardrails.csv",
            mime="text/csv",
        )
    else:
        st.warning("No guardrail data for selected woredas.")
        
    st.markdown("##### Guardrails Comparison Plot")
    # Pass raw onset offsets per woreda so Plotly computes quartiles natively
    guardrail_subset = ts_df[ts_df["Woreda"].isin(guardrail_woredas)]
    if not guardrail_subset.empty:
        # Hash the subset for cache key
        subset_hash = hashlib.md5(
            pd.util.hash_pandas_object(guardrail_subset).values.tobytes()
        ).hexdigest()
        ts_df_json = guardrail_subset.to_json()
        
        fig_guard = create_guardrail_plot(
            ts_df_hash=subset_hash,
            guardrail_woredas_tuple=tuple(guardrail_woredas),
            ts_df_json=ts_df_json,
        )
        if fig_guard is not None:
            st.plotly_chart(fig_guard, width='stretch')
        else:
            st.info("Insufficient data to generate the guardrails comparison plot.")
    else:
        st.info("Insufficient data to generate the guardrails comparison plot.")

def render_portfolio_diagnostics_tab(ts_df, zones_gdf, selected_woredas):
    """Render portfolio diagnostics tab with proper caching."""
    st.subheader("Portfolio Diagnostics")
    if not selected_woredas:
        st.info("Select one or more woredas above (or click 'Apply Pilot Portfolio') to see diagnostics here.")
        return
        
    portfolio_ts = ts_df[ts_df["Woreda"].isin(selected_woredas)]
    if portfolio_ts.empty:
        st.warning("No onset data available for the selected woredas.")
        return
        
    avg_risk = zones_gdf[zones_gdf["adm3_name"].isin(selected_woredas)]["false_start_prob"].mean()
    
    render_analytical_summary(
        header="Aggregated Risk and Stability Assessment",
        description="Evaluates the selected woreda portfolio for false-start probabilities, historical decadal shifts in onset timing, and interannual volatility. This informs operational readiness and resource allocation strategies.",
        metrics=[
            {"label": "Portfolio Size", "value": len(selected_woredas)},
            {"label": "Avg. False-Start Risk", "value": f"{avg_risk:.1f}%" if np.isfinite(avg_risk) else "N/A"},
        ]
    )
    
    ts_hash = hashlib.md5(
        pd.util.hash_pandas_object(portfolio_ts).values.tobytes()
    ).hexdigest()
    portfolio_ts_json = portfolio_ts.to_json()
        
    st.markdown("##### False-Start Risk Assessment")
    risk_cols = ["adm3_name", "adm2_name", "false_start_prob"]
    if all(c in zones_gdf.columns for c in risk_cols):
        risk_view = (
            zones_gdf[zones_gdf["adm3_name"].isin(selected_woredas)][risk_cols]
            .drop_duplicates(subset=["adm3_name"])
            .dropna(subset=["false_start_prob"])
            .sort_values("false_start_prob", ascending=True)
        )
        if not risk_view.empty:
            map_col, chart_col = st.columns(2)
            with map_col:
                st.markdown("**Spatial False-Start Risk Map**")
                national_gdf = load_national_geojson()
                if national_gdf is not None:
                    # Prepare data for caching
                    map_hash = compute_data_hash(
                        zones_gdf[zones_gdf["adm3_name"].isin(selected_woredas)]["false_start_prob"]
                    )
                    national_map_gdf = national_gdf.copy()
                    selected_woredas_lower = [w.lower().strip() for w in selected_woredas]
                    risk_mapping = dict(
                        zip(
                            zones_gdf["adm3_name"].str.lower().str.strip(),
                            zones_gdf["false_start_prob"],
                        )
                    )
                    name_col = None
                    for candidate in ["ADM3_EN", "adm3_en", "ADM3_NAME", "adm3_name"]:
                        if candidate in national_map_gdf.columns:
                            name_col = candidate
                            break
                            
                    def map_risk(row):
                        if name_col:
                            w_name = str(row[name_col]).lower().strip()
                            if w_name in selected_woredas_lower:
                                return risk_mapping.get(w_name, np.nan)
                        return np.nan
                        
                    national_map_gdf["risk_value"] = national_map_gdf.apply(map_risk, axis=1)
                    if name_col:
                        national_map_gdf["pilot_status"] = national_map_gdf[name_col].apply(
                            lambda x: "Active Pilot" if str(x).lower().strip() in selected_woredas_lower else "Other Woreda"
                        )
                        # Prepare data for caching
                        portfolio_only_gdf = zones_gdf[zones_gdf["adm3_name"].isin(selected_woredas)]
                        if not portfolio_only_gdf.empty:
                            centroid_y = portfolio_only_gdf.geometry.centroid.y.mean()
                            centroid_x = portfolio_only_gdf.geometry.centroid.x.mean()
                            zoom = 6
                        else:
                            centroid_y = 9.145
                            centroid_x = 40.4896
                            zoom = 5
                            
                        hover_texts = []
                        risk_values = []
                        for idx, row in national_map_gdf.iterrows():
                            status = row["pilot_status"]
                            risk = row["risk_value"]
                            risk_str = f"{risk:.1f}%" if not pd.isna(risk) else "N/A"
                            hover_texts.append(
                                f"<b>{row[name_col]}</b><br>Status: {status}<br>Risk: {risk_str}"
                            )
                            risk_values.append(float(risk) if not pd.isna(risk) else None)
                            
                        fig_map = cached_create_portfolio_overlay_map(
                            data_hash=map_hash,
                            _national_gdf_json=safe_to_json(national_map_gdf),
                            risk_values_json=json.dumps(risk_values),
                            pilot_status_json=json.dumps(national_map_gdf["pilot_status"].tolist()),
                            hover_texts_json=json.dumps(hover_texts),
                            centroid_y=centroid_y,
                            centroid_x=centroid_x,
                            zoom=zoom,
                        )
                        if fig_map:
                            st.plotly_chart(fig_map, width='stretch')
                        else:
                            st.info("Map rendering is currently unavailable.")
                    else:
                        st.info("Map rendering is currently unavailable.")
                else:
                    st.info("Map rendering is currently unavailable.")
                    
            with chart_col:
                st.markdown("**Risk Distribution Ranking**")
                fig_risk = go.Figure(
                    go.Bar(
                        x=risk_view["false_start_prob"],
                        y=risk_view["adm3_name"],
                        orientation="h",
                        marker=dict(
                            color=risk_view["false_start_prob"],
                            colorscale=[
                                [0, "#2b83ba"],
                                [0.5, "#fdae61"],
                                [1, "#d7191c"],
                            ],
                            cmin=0,
                            cmax=100,
                        ),
                        customdata=risk_view["adm2_name"],
                        hovertemplate="%{y} (%{customdata})<br>False-start risk: %{x:.1f}%<extra></extra>",
                    )
                )
                fig_risk.update_layout(
                    template="plotly_white",
                    height=500,
                    xaxis_title="False-Start Probability (%)",
                    margin=dict(l=10, r=10, t=30, b=0),
                    xaxis_range=[0, 100],
                )
                st.plotly_chart(fig_risk, width='stretch')
        else:
            st.info("No false-start risk data available for the selected woredas.")
    else:
        st.info("False-start risk columns not found.")
        
    st.markdown("##### Decadal Shift vs. Historical Volatility")
    fig_stability, stability_df = build_stability_figure(
        tuple(sorted(selected_woredas)), ts_hash, portfolio_ts_json
    )
    if fig_stability is not None:
        st.plotly_chart(fig_stability, width='stretch')


# ============================================================================
# NEW TAB: AI Forecast Validation & Weekly Onset Monitor
# ============================================================================
def render_ai_forecast_validation_tab(ts_df, all_woredas):
    """
    Weekly rainfall-onset climatology tab with multi-dataset NRT comparison
    and AI forecast decision mechanism.
    """
    st.subheader("AI Forecast Validation & Weekly Onset Monitor")
    render_analytical_summary(
        header="Multi-Dataset NRT Rainfall vs. Climatology",
        description=(
            "Compares near-real-time rainfall observations (MSWEP, CHIRPS, ENACTS) "
            "against the historical weekly onset climatology for a selected woreda. "
            "Provides a **STOP / CONTINUE** decision signal for the AI forecasting workflow: "
            "when the forecast falls within the expected climatological range with high confidence, "
            "further forecasting iterations can be safely stopped."
        ),
        metrics=[
            {"label": "Start Window", "value": "May 25"},
            {"label": "Bin Width", "value": "7 Days"},
            {"label": "Datasets", "value": "MSWEP, CHIRPS, ENACTS"},
        ]
    )
    
    # ── 1. Woreda Selection ────────────────────────────────────────────────
    st.markdown("##### 1. Select Woreda")
    selected_woreda = st.selectbox(
        "Choose a woreda for weekly onset analysis",
        options=sorted(all_woredas),
        key="ai_validation_woreda_select",
    )
    
    if not selected_woreda:
        st.info("Select a woreda to begin analysis.")
        return
    
    # ── 2. Dataset Availability Panel ──────────────────────────────────────
    st.markdown("##### 2. Dataset Availability & Status")
    dataset_names = ["MSWEP", "CHIRPS", "ENACTS"]
    nrt_results = {}
    status_cols = st.columns(len(dataset_names))
    
    for i, ds_name in enumerate(dataset_names):
        with status_cols[i]:
            result = _load_nrt_dataset(ds_name, selected_woreda)
            nrt_results[ds_name] = result
            
            if result["status"] == "ok":
                st.markdown(f'<span class="dataset-ok">✓ {ds_name}: Available</span>', unsafe_allow_html=True)
                if result["last_updated"]:
                    st.caption(f"Last updated: {result['last_updated'].strftime('%Y-%m-%d')}")
            elif result["status"] == "delayed":
                st.markdown(f'<span class="dataset-delayed">⚠ {ds_name}: Delayed</span>', unsafe_allow_html=True)
                st.caption(result["message"])
            else:
                st.markdown(f'<span class="dataset-missing">✗ {ds_name}: Unavailable</span>', unsafe_allow_html=True)
                st.caption(result["message"])
    
    available_datasets = [ds for ds, r in nrt_results.items() if r["status"] in ("ok", "delayed")]
    
    if not available_datasets:
        st.warning(
            "No near-real-time datasets are currently available for this woreda. "
            "Please ensure NRT data files are placed in `data/nrt/{mswep,chirps,enacts}/` "
            "with columns `[Date, Rainfall_mm]`."
        )
        # Still show climatology even without NRT
        st.info("Displaying historical weekly climatology only.")
    
    # ── 3. Weekly Climatology for Selected Woreda ──────────────────────────
    st.markdown("##### 3. Weekly Onset Climatology (Historical)")
    clim_df = _compute_weekly_climatology(ts_df, selected_woreda)
    
    if clim_df.empty:
        st.warning(f"No historical onset data available for **{selected_woreda}**.")
        return
    
    # Build climatology visualization
    fig_clim = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Climatological range band (P10 - P90)
    fig_clim.add_trace(go.Scatter(
        x=clim_df["Week_Label"].tolist() + clim_df["Week_Label"].tolist()[::-1],
        y=clim_df["P90"].tolist() + clim_df["P10"].tolist()[::-1],
        fill="toself",
        fillcolor="rgba(16, 185, 129, 0.15)",
        line=dict(color="rgba(255,255,255,0)"),
        name="Climatological Range (P10–P90)",
        hovertemplate="Week: %{x}<br>Range: %{y:.1f} days<extra></extra>",
    ), secondary_y=False)
    
    # Median line
    fig_clim.add_trace(go.Scatter(
        x=clim_df["Week_Label"],
        y=clim_df["P50"],
        mode="lines+markers",
        line=dict(color="#059669", width=3),
        marker=dict(size=7),
        name="Median Onset",
        hovertemplate="Week: %{x}<br>Median: %{y:.1f} days<extra></extra>",
    ), secondary_y=False)
    
    # Sample size as bar on secondary axis
    fig_clim.add_trace(go.Bar(
        x=clim_df["Week_Label"],
        y=clim_df["Sample_Years"],
        name="Sample Years",
        marker_color="rgba(100, 116, 139, 0.3)",
        yaxis="y2",
        hovertemplate="Week: %{x}<br>Years: %{y}<extra></extra>",
    ), secondary_y=True)
    
    fig_clim.update_layout(
        template="plotly_white",
        height=420,
        margin=dict(l=50, r=50, t=30, b=100),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        font_color="#334155",
        xaxis=dict(tickangle=45, title="Weekly Window (starting May 25)"),
    )
    fig_clim.update_yaxes(title_text="Onset Offset (days from Jun 1)", secondary_y=False)
    fig_clim.update_yaxes(title_text="Sample Size (years)", secondary_y=True)
    
    st.plotly_chart(fig_clim, width='stretch')
    
    # ── 4. NRT vs Climatology Comparison ───────────────────────────────────
    st.markdown("##### 4. Current / Forecast Rainfall vs. Climatology")
    
    if not available_datasets:
        st.info("Skipping NRT comparison — no datasets available.")
        return
    
    # Let user input or simulate a forecast onset offset
    st.markdown("**AI Forecast Input**")
    forecast_col1, forecast_col2 = st.columns(2)
    with forecast_col1:
        forecast_offset = st.number_input(
            "Forecast onset offset (days from Jun 1)",
            min_value=-30.0,
            max_value=180.0,
            value=float(clim_df["P50"].iloc[0]) if not clim_df.empty else 0.0,
            step=1.0,
            help="Enter the AI model's predicted onset offset in days relative to June 1.",
            key="ai_forecast_offset_input",
        )
    with forecast_col2:
        forecast_date = offset_to_date(forecast_offset)
        st.metric(
            label="Forecast Onset Date (anchor year)",
            value=forecast_date.strftime("%B %d"),
            delta=f"{forecast_offset:+.1f} days",
        )
    
    # Build comparison chart
    fig_compare = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Climatology range
    fig_compare.add_trace(go.Scatter(
        x=clim_df["Week_Label"].tolist() + clim_df["Week_Label"].tolist()[::-1],
        y=clim_df["P90"].tolist() + clim_df["P10"].tolist()[::-1],
        fill="toself",
        fillcolor="rgba(59, 130, 246, 0.15)",
        line=dict(color="rgba(255,255,255,0)"),
        name="Clim Range (P10–P90)",
    ), secondary_y=False)
    
    # Median line
    fig_compare.add_trace(go.Scatter(
        x=clim_df["Week_Label"],
        y=clim_df["P50"],
        mode="lines",
        line=dict(color="#3B82F6", width=2, dash="dash"),
        name="Clim Median",
    ), secondary_y=False)
    
    # NRT dataset observations (weekly accumulated rainfall)
    colors_nrt = {"MSWEP": "#EF4444", "CHIRPS": "#F59E0B", "ENACTS": "#8B5CF6"}
    for ds_name in available_datasets:
        ds_data = nrt_results[ds_name]["data"]
        if ds_data is None or ds_data.empty:
            continue
        
        # Aggregate to weekly bins aligned with climatology
        ds_data = ds_data.copy()
        ds_data["Week"] = ds_data["Date"].dt.isocalendar().week.astype(int)
        weekly_rain = ds_data.groupby("Week")["Rainfall_mm"].sum().reset_index()
        
        # Map to our bin labels (simplified: use week number offset from May 25)
        may25_week = datetime(2001, 5, 25).isocalendar()[1]
        weekly_rain["Bin_Index"] = ((weekly_rain["Week"] - may25_week) % 52).astype(int)
        
        # Only plot weeks that align with our climatology bins
        valid_bins = list(range(len(clim_df)))
        weekly_rain = weekly_rain[weekly_rain["Bin_Index"].isin(valid_bins)]
        
        if not weekly_rain.empty:
            x_labels = [clim_df.iloc[i]["Week_Label"] for i in weekly_rain["Bin_Index"].values]
            fig_compare.add_trace(go.Bar(
                x=x_labels,
                y=weekly_rain["Rainfall_mm"].values,
                name=f"{ds_name} (NRT)",
                marker_color=colors_nrt.get(ds_name, "#64748B"),
                opacity=0.7,
                yaxis="y2",
                hovertemplate="Week: %{x}<br>Rain: %{y:.1f} mm<extra></extra>",
            ), secondary_y=True)
    
    # Forecast marker
    # Find nearest week for the forecast
    clim_df_sorted = clim_df.sort_values("Week_Start_Offset").reset_index(drop=True)
    clim_df_sorted["distance"] = (clim_df_sorted["Week_Start_Offset"] - forecast_offset).abs()
    nearest_idx = clim_df_sorted["distance"].idxmin()
    nearest_week_label = clim_df_sorted.loc[nearest_idx, "Week_Label"]
    
    fig_compare.add_trace(go.Scatter(
        x=[nearest_week_label],
        y=[forecast_offset],
        mode="markers",
        marker=dict(size=16, color="#10B981" if forecast_offset <= clim_df_sorted.loc[nearest_idx, "P90"] and forecast_offset >= clim_df_sorted.loc[nearest_idx, "P10"] else "#EF4444",
                    symbol="star", line=dict(width=2, color="white")),
        name="AI Forecast",
        hovertemplate=f"Forecast: {forecast_offset:+.1f} days<br>Week: {nearest_week_label}<extra></extra>",
    ), secondary_y=False)
    
    fig_compare.update_layout(
        template="plotly_white",
        height=450,
        margin=dict(l=50, r=50, t=30, b=100),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        font_color="#334155",
        xaxis=dict(tickangle=45, title="Weekly Window"),
        barmode="group",
    )
    fig_compare.update_yaxes(title_text="Onset Offset (days)", secondary_y=False)
    fig_compare.update_yaxes(title_text="Weekly Rainfall (mm)", secondary_y=True)
    
    st.plotly_chart(fig_compare, width='stretch')
    
    # ── 5. Decision Mechanism ──────────────────────────────────────────────
    st.markdown("##### 5. AI Forecast Decision Signal")
    
    evaluation = _evaluate_forecast_against_climatology(forecast_offset, clim_df)
    
    # Render decision banner
    if evaluation["decision"] == "STOP":
        st.markdown(
            f'<div class="decision-stop">'
            f'✓ STOP — Forecast is Reliable<br>'
            f'<span style="font-size: 0.9rem; font-weight: 400;">'
            f'Confidence: {evaluation["confidence"]:.0f}% | '
            f'Nearest Week: {evaluation["nearest_week"]}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.success(
            f"**Recommendation:** The AI forecast ({forecast_offset:+.1f} days) falls within the "
            f"expected climatological range. **Further forecasting iterations may be stopped.** "
            f"The current prediction is considered reliable for operational use."
        )
    elif evaluation["decision"] == "CONTINUE":
        st.markdown(
            f'<div class="decision-continue">'
            f'⟳ CONTINUE — Low Confidence<br>'
            f'<span style="font-size: 0.9rem; font-weight: 400;">'
            f'Confidence: {evaluation["confidence"]:.0f}% | '
            f'Nearest Week: {evaluation["nearest_week"]}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.warning(
            f"**Recommendation:** The forecast is within range but confidence is low "
            f"({evaluation['confidence']:.0f}%). **Continue forecasting** with additional "
            f"ensemble members or higher-resolution input data before finalizing."
        )
    else:
        st.markdown(
            f'<div class="decision-uncertain">'
            f'✗ UNCERTAIN — Out of Range<br>'
            f'<span style="font-size: 0.9rem; font-weight: 400;">'
            f'Confidence: {evaluation["confidence"]:.0f}% | '
            f'Nearest Week: {evaluation["nearest_week"]}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.error(
            f"**Recommendation:** The AI forecast ({forecast_offset:+.1f} days) falls **outside** "
            f"the expected climatological range. **Do not stop forecasting.** "
            f"Re-evaluate model inputs, consider bias correction, or run additional ensemble members."
        )
    
    st.caption(f"**Reasoning:** {evaluation['reason']}")
    
    # ── 6. Detailed Comparison Table ───────────────────────────────────────
    st.markdown("##### 6. Detailed Weekly Comparison Table")
    
    # Build a comparison table
    table_rows = []
    for _, row in clim_df.iterrows():
        table_rows.append({
            "Week": row["Week_Label"],
            "Clim Mean (days)": f"{row['Mean']:.1f}",
            "Clim Median (days)": f"{row['P50']:.1f}",
            "P10 (days)": f"{row['P10']:.1f}",
            "P90 (days)": f"{row['P90']:.1f}",
            "Std (days)": f"{row['Std']:.1f}",
            "Sample Years": int(row["Sample_Years"]),
            "Forecast In Range": "✓" if row["P10"] <= forecast_offset <= row["P90"] else "—",
        })
    
    comparison_table = pd.DataFrame(table_rows)
    
    # Highlight the nearest week
    def highlight_nearest(row):
        if row["Week"] == evaluation["nearest_week"]:
            return ["background-color: #ECFDF5"] * len(row)
        return [""] * len(row)
    
    styled_table = comparison_table.style.apply(highlight_nearest, axis=1)
    st.dataframe(styled_table, width='stretch', hide_index=True)
    
    # Download button
    csv_bytes = comparison_table.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Weekly Comparison (CSV)",
        data=csv_bytes,
        file_name=f"weekly_onset_comparison_{selected_woreda.replace(' ', '_')}.csv",
        mime="text/csv",
    )


def render_weekly_explorer_tab(ts_df, selected_woredas, all_woredas):
    """
    State-of-the-art weekly onset probability explorer.
    Computes empirical weekly onset probabilities starting from May 25 
    and visualizes them via a heatmap and cumulative probability curves (Ogive).
    """
    st.subheader("Weekly Onset Explorer")
    render_analytical_summary(
        header="Weekly Onset Probability & Cumulative Risk",
        description="Analyzes the empirical probability of rainy season onset occurring in specific weekly windows starting from May 25. The cumulative probability curve (Ogive) is a state-of-the-art operational tool for determining the likelihood of onset by a given week, crucial for agricultural planning and early warning systems.",
        metrics=[
            {"label": "Start Window", "value": "May 25"},
            {"label": "Bin Width", "value": "7 Days"},
        ]
    )
    
    # Filter data based on selection
    if selected_woredas:
        plot_df = ts_df[ts_df["Woreda"].isin(selected_woredas)].copy()
    else:
        plot_df = ts_df.copy()
        
    if plot_df.empty:
        st.info("No onset data available for the selected woredas.")
        return

    # Define weekly bins starting from May 25 (offset -7 from June 1)
    # W0: May 25 - May 31 (offset -7 to -1)
    # W1: Jun 01 - Jun 07 (offset 0 to 6)
    bin_edges = list(range(-7, 150, 7))
    bin_labels = []
    for i in range(len(bin_edges) - 1):
        start_date = offset_to_date(bin_edges[i])
        end_date = offset_to_date(bin_edges[i+1] - 1)
        bin_labels.append(f"W{i}: {start_date.strftime('%b %d')}")
        
    plot_df["Week_Bin"] = pd.cut(
        plot_df["OnsetOffset"], 
        bins=bin_edges, 
        labels=bin_labels, 
        right=False,
        include_lowest=True
    )
    
    # Drop NaNs that fall outside the bin range (e.g., extreme outliers)
    plot_df = plot_df.dropna(subset=["Week_Bin"])
    
    if plot_df.empty:
        st.warning("No onset dates fall within the defined weekly window (May 25 onwards).")
        return

    # Calculate probabilities per woreda
    week_counts = plot_df.groupby(["Woreda", "Week_Bin"]).size().reset_index(name="Count")
    total_years = plot_df.groupby("Woreda").size().reset_index(name="Total")
    
    prob_df = week_counts.merge(total_years, on="Woreda")
    prob_df["Probability"] = (prob_df["Count"] / prob_df["Total"]) * 100
    
    # 1. Heatmap of Weekly Probabilities
    st.markdown("##### Weekly Onset Probability Heatmap")
    st.caption("Color intensity represents the historical probability (%) of onset occurring in that specific week. Woredas are sorted by median onset date.")
    
    # Pivot for heatmap (Safe: 'values' restricts the DF to only the float 'Probability' column)
    heatmap_data = prob_df.pivot(index="Woreda", columns="Week_Bin", values="Probability").fillna(0)
    
    # Sort woredas by median onset offset
    median_offsets = plot_df.groupby("Woreda")["OnsetOffset"].median().sort_values()
    sorted_woredas = median_offsets.index.tolist()
    
    # Ensure all bins are present in columns
    heatmap_data = heatmap_data.reindex(columns=bin_labels, fill_value=0)
    heatmap_data = heatmap_data.reindex(sorted_woredas)
    
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns.tolist(),
        y=heatmap_data.index.tolist(),
        colorscale="YlOrRd",
        colorbar=dict(title="Prob (%)", ticksuffix="%"),
        hovertemplate="Woreda: %{y}<br>Week: %{x}<br>Probability: %{z:.1f}%<extra></extra>"
    ))
    
    fig_heatmap.update_layout(
        template="plotly_white",
        height=max(400, len(sorted_woredas) * 35),
        margin=dict(l=150, r=20, t=20, b=100),
        xaxis=dict(tickangle=45, title="Weekly Window"),
        yaxis=dict(title="Woreda", autorange="reversed"),
        font_color="#334155"
    )
    st.plotly_chart(fig_heatmap, width='stretch')
    
    # 2. Cumulative Probability Curve (Ogive)
    st.markdown("##### Cumulative Probability of Onset (Ogive)")
    st.caption("Operational early warning tool: Shows the likelihood that the onset has occurred *by* a given week. Thresholds indicate median (50%) and reliable (80%) onset timing.")
    
    selected_woreda_ogive = st.selectbox(
        "Select Woreda for Cumulative Analysis", 
        sorted_woredas, 
        key="ogive_woreda_select"
    )
    
    woreda_prob = prob_df[prob_df["Woreda"] == selected_woreda_ogive].copy()
    
    # FIX: Reindex without fill_value=0 to avoid PyArrow dtype errors on string columns
    woreda_prob = woreda_prob.set_index("Week_Bin").reindex(bin_labels)
    
    # Fill missing values appropriately per column type
    woreda_prob["Probability"] = woreda_prob["Probability"].fillna(0)
    woreda_prob["Count"] = woreda_prob["Count"].fillna(0)
    woreda_prob["Total"] = woreda_prob["Total"].fillna(0)
    woreda_prob["Woreda"] = woreda_prob["Woreda"].fillna(selected_woreda_ogive)
    
    woreda_prob = woreda_prob.reset_index()
    woreda_prob["Cumulative_Prob"] = woreda_prob["Probability"].cumsum()
    
    # FIX: Use make_subplots to properly support secondary_y=True for dual-axis charts
    fig_ogive = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Bar chart for individual week probability (Primary Y-Axis)
    fig_ogive.add_trace(go.Bar(
        x=woreda_prob["Week_Bin"],
        y=woreda_prob["Probability"],
        name="Weekly Probability",
        marker_color="#3B82F6",
        opacity=0.6,
    ), secondary_y=False)
    
    # Line chart for cumulative probability (Secondary Y-Axis)
    fig_ogive.add_trace(go.Scatter(
        x=woreda_prob["Week_Bin"],
        y=woreda_prob["Cumulative_Prob"],
        name="Cumulative Probability",
        mode="lines+markers",
        line=dict(color="#EF4444", width=3),
        marker=dict(size=6),
    ), secondary_y=True)
    
    # Add 50% and 80% threshold lines on the Secondary Y-Axis
    fig_ogive.add_hline(y=50, line_dash="dash", line_color="#10B981", 
                        annotation_text="50% (Median)", annotation_position="bottom right",
                        secondary_y=True)
    fig_ogive.add_hline(y=80, line_dash="dot", line_color="#F59E0B", 
                        annotation_text="80% (Reliable)", annotation_position="top right",
                        secondary_y=True)
    
    fig_ogive.update_layout(
        template="plotly_white",
        height=450,
        margin=dict(l=50, r=50, t=40, b=100),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        font_color="#334155",
        xaxis=dict(tickangle=45, title="Weekly Window")
    )
    fig_ogive.update_yaxes(title_text="Weekly Probability (%)", secondary_y=False, range=[0, 100])
    fig_ogive.update_yaxes(title_text="Cumulative Probability (%)", secondary_y=True, range=[0, 100])
    
    st.plotly_chart(fig_ogive, width='stretch')

def render_daily_timeseries_tab():
    """Render daily timeseries tab."""
    st.subheader("Daily Timeseries Analysis")
    render_analytical_summary(
        header="High-Resolution Rainfall Tracking",
        description="Provides daily accumulation analysis to validate onset detection algorithms against raw precipitation records. This view allows for granular inspection of wet/dry spell sequences that trigger the onset definitions.",
        metrics=[
            {"label": "Analysis Type", "value": "Daily Accumulation"},
            {"label": "Purpose", "value": "Algorithm Validation"},
        ]
    )
    render_timeseries()

def _explain_engine_error(exc: Exception) -> str:
    """Explain engine errors in user-friendly terms."""
    msg = str(exc)
    if isinstance(exc, FileNotFoundError):
        if "Missing ENACTS files" in msg or "Could not find a complete set" in msg:
            return "Missing ENACTS files - please check file paths."
        return f"File not found: {msg}"
    return f"Computation failed: {msg}"

def load_custom_css():
    """Load custom CSS for the onset climatology page."""
    css_path = Path(__file__).parent.parent / "styles" / "page_styles" / "onset_climatology.css"
    if css_path.exists():
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ============================================================================
# Main Page Entry Point
# ============================================================================
def page():
    """Main onset climatology page."""
    load_custom_css()
    inject_theme_css()
    render_controls_panel()
    
    st.title("Onset Climatology Analysis")
    st.markdown(
        f'<p class="clim-header-sub">Rainy-season onset detection across Ethiopia\'s pilot '
        f"woredas, built from the {CLIM_START_YEAR}–{CLIM_END_YEAR} ENACTS record.</p>",
        unsafe_allow_html=True,
    )
    
    shapefile = load_shapefile()
    if shapefile is None:
        st.error("Base shapefile not found. Please verify manual zones paths.")
        st.stop()
        
    cached_result = load_cached_result()
    if cached_result is not None:
        st.session_state["clim_result"] = cached_result
        
    if "clim_result" not in st.session_state:
        with st.container(border=True):
            col1, col2 = st.columns([2, 1])
            with col1:
                st.info("No climatology result found. Run the climatology engine to generate results.")
            with col2:
                if st.button("Run Climatology Engine", type="primary", width='stretch'):
                    with st.spinner("Processing gridded dataset..."):
                        try:
                            result = run_climatology_engine(selected_woredas=None)
                            st.session_state["clim_result"] = result
                            save_cached_result(result)
                            st.rerun()
                        except Exception as e:
                            st.error(_explain_engine_error(e))
                            st.stop()
        st.stop()
        
    result = st.session_state["clim_result"]
    guardrail_df = result["guardrail_df"]
    ts_df = result["ts_df"]
    zones_gdf = result["zones_gdf"]
    years_used = result["years_used"]
    all_woredas = result["all_woredas"]
    
    # Cache the zones_gdf modifications
    if "clim_median_offset" not in zones_gdf.columns:
        with st.spinner("Building vector layers..."):
            zones_gdf["clim_median_offset"] = zones_gdf["uid"].map(
                ts_df.groupby("uid")["OnsetOffset"].median()
            )
            zones_gdf["clim_std_offset"] = zones_gdf["uid"].map(
                ts_df.groupby("uid")["OnsetOffset"].std()
            )
            zones_gdf["clim_pct_onset"] = (
                zones_gdf["uid"].map(
                    ts_df.groupby("uid")["OnsetOffset"].count() / len(years_used) * 100
                )
            )
            # Update session state with modified zones_gdf
            st.session_state["clim_result"]["zones_gdf"] = zones_gdf
            
    st.session_state["_zones_gdf_for_map"] = zones_gdf
    
    # KPI SUMMARY
    n_high_conf = int((guardrail_df["Confidence"] == "HIGH").sum())
    n_woredas_with_guardrails = guardrail_df["Woreda"].nunique()
    completeness_pct = (
        (n_woredas_with_guardrails / len(all_woredas) * 100.0)
        if all_woredas
        else 0.0
    )
    
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Years of Record", f"{years_used[0]}–{years_used[-1]}", f"{len(years_used)} years")
    k2.metric("Woredas Mapped", len(all_woredas))
    k3.metric("High Confidence", f"{n_high_conf}/{n_woredas_with_guardrails}")
    k4.metric("Data Completeness", f"{completeness_pct:.0f}%")
    
    st.divider()
    st.subheader("Interactive Woreda Analysis")
    
    if "_onset_selected_woredas" not in st.session_state:
        st.session_state["_onset_selected_woredas"] = (
            [all_woredas[0]] if all_woredas else []
        )
        
    def _apply_pilot_portfolio():
        st.session_state["_onset_selected_woredas"] = [
            w for w in DEFAULT_SELECTED_WOREDAS if w in all_woredas
        ]
        
    sel_col1, sel_col2 = st.columns([4, 1])
    with sel_col1:
        selected_woredas = st.multiselect(
            "Select one or more woredas",
            options=all_woredas,
            key="_onset_selected_woredas",
        )
    with sel_col2:
        st.markdown("<br>", unsafe_allow_html=True)
        st.button("Apply Pilot Portfolio", width='stretch', on_click=_apply_pilot_portfolio)
        
    st.divider()
    
    chosen_tab = st.radio(
        label="Select Analytical Viewport Profile",
        options=[
            "Interactive Maps",
            "Zone Timelines",
            "Guardrails",
            "Portfolio Diagnostics",
            "Weekly Onset Explorer",
            "Daily Timeseries",
        ],
        label_visibility="collapsed",
    )
    

    if chosen_tab == "Interactive Maps":
        render_interactive_maps_tab(zones_gdf, all_woredas, ts_df)
    elif chosen_tab == "Zone Timelines":
        render_zone_timelines_tab(ts_df)
    elif chosen_tab == "Guardrails":
        render_guardrails_tab(guardrail_df, selected_woredas, all_woredas, ts_df)
    elif chosen_tab == "Portfolio Diagnostics":
        render_portfolio_diagnostics_tab(ts_df, zones_gdf, selected_woredas)
    elif chosen_tab == "Weekly Onset Explorer":
        render_weekly_explorer_tab(ts_df, selected_woredas, all_woredas)
    elif chosen_tab == "Daily Timeseries":
        render_daily_timeseries_tab()

if __name__ == "__main__":
    page()