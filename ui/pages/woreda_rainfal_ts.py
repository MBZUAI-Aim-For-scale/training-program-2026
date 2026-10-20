# ui/pages/woreda_rainfal_ts.py
"""
Streamlit interactive UI page for woreda rainfall time series analysis and onset classification.
(Decoupled Frontend: Relies on data pre-computed by woreda_onset_rainfall.py)
"""

import os
from datetime import date, datetime
from pathlib import Path

import geopandas as gpd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, LogLocator
import numpy as np
import pandas as pd
import streamlit as st

from apps.onset_obs.woreda_aggregate import aggregate_daily
from apps.onset_obs.woreda_weights import load_weights

# Configuration and helper imports
from config import (
    BUCKET_COLORS,
    BUCKET_LABELS,
    BUCKET_ORDER,
    DRY_MM,
    CHRIPS_FEATURES_CSV,
    LOG_BASE,
    LOG_FLOOR,
    LOG_SCALE,
    CHRIPS_MAP_PNG,
    MSWEP_CSV,
    MSWEP_FEATURE,
    MSWEP_MAP_PNG,
    PLOT_DAYS,
    WET_MM,
    WET_RUN,
    WET_TOTAL_MM,
    STUDYAREA_SHAPEFILE,
)


@st.cache_data
def get_woreda_zone_map():
    """Load shapefile to get the direct mapping of woreda (pcode & name) to zone."""
    gdf = gpd.read_file(STUDYAREA_SHAPEFILE)
    cols = [c for c in ["adm3_pcode", "adm3_name", "adm2_name"] if c in gdf.columns]
    df = gdf[cols].drop_duplicates()
    if "adm3_pcode" in df.columns:
        df["adm3_pcode"] = df["adm3_pcode"].astype(str).str.strip()
    return df


@st.cache_data(show_spinner="Loading spatial weights and daily aggregates...")
def load_aggregated_daily():
    """Load spatial weights and compute or retrieve daily aggregated precipitation."""
    W, lats, lons, pcodes, hw, bounds_xy = load_weights()
    daily = aggregate_daily(W, pcodes, hw, bounds_xy)
    if "adm3_pcode" in daily.columns:
        daily["adm3_pcode"] = daily["adm3_pcode"].astype(str).str.strip()
    return daily


@st.cache_data(show_spinner="Loading MSWEP overlay data...")
def load_mswep_data():
    """Load MSWEP dataset if present on disk."""
    if MSWEP_CSV.exists():
        df = pd.read_csv(MSWEP_CSV)
        if "adm3_pcode" in df.columns:
            df["adm3_pcode"] = df["adm3_pcode"].astype(str).str.strip()
        return df
    return None


@st.cache_data(show_spinner="Loading onset features...")
def load_features_data():
    """Strictly read pre-computed features instead of computing on the fly."""
    if CHRIPS_FEATURES_CSV.exists():
        df = pd.read_csv(CHRIPS_FEATURES_CSV)
        if "adm3_pcode" in df.columns:
            df["adm3_pcode"] = df["adm3_pcode"].astype(str).str.strip()
        return df
    return None


# =====================================================================
# 1. WOREDA TIME SERIES PLOT GENERATION
# =====================================================================
def generate_woreda_plot(daily, pcode, name, mswep=None):
    """Generate and return a Matplotlib figure for a single woreda rainfall time series."""
    df = daily[daily["adm3_pcode"] == pcode].dropna(subset=["precip_mm"]).copy()
    if df.empty:
        return None

    src_rank = {"final": 0, "prelim": 1, "forecast": 2}
    df["rank"] = df["source"].map(src_rank)
    df["date_dt"] = pd.to_datetime(df["date"])
    df = (
        df.sort_values(["date_dt", "rank"])
        .drop_duplicates("date_dt", keep="first")
        .sort_values("date_dt")
        .reset_index(drop=True)
    )
    df = df.tail(PLOT_DAYS).reset_index(drop=True)

    roll = df["precip_mm"][::-1].rolling(WET_RUN, min_periods=1).sum()[::-1]
    roll = roll[:-2]

    m = None
    if mswep is not None:
        m = mswep[mswep["adm3_pcode"] == pcode].dropna(subset=["precip_mm"]).copy()
        m["date_dt"] = pd.to_datetime(m["date"])
        m = m[
            (m["date_dt"] >= df["date_dt"].min())
            & (m["date_dt"] <= df["date_dt"].max())
        ].sort_values("date_dt")
        if m.empty:
            m = None

    floor = LOG_FLOOR if LOG_FLOOR > 0 else 0.1

    fig, ax = plt.subplots(figsize=(10, 4.5))
    half_day = pd.Timedelta(hours=12)

    for src, color in (("prelim", "#d9d9d9"), ("forecast", "#fff59d")):
        sub = df[df["source"] == src]
        if not sub.empty:
            x0 = sub["date_dt"].min() - half_day
            x1 = sub["date_dt"].max() + half_day
            ax.axvspan(x0, x1, color=color, alpha=0.9, zorder=0)

    if LOG_SCALE:
        ax.set_yscale("log", base=LOG_BASE)
        ax.yaxis.set_major_locator(
            LogLocator(base=LOG_BASE, subs=(1.0,), numticks=15)
        )
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _pos: f"{v:g}"))

    heights = (
        df["precip_mm"].where(df["precip_mm"] > 0) if LOG_SCALE else df["precip_mm"]
    )
    ax.bar(df["date_dt"], heights, width=0.9, color="steelblue", zorder=3)

    def for_log(series):
        return series.clip(lower=floor) if LOG_SCALE else series

    ax.plot(
        df["date_dt"][:-2],
        for_log(roll),
        color="#d95f02",
        lw=1.6,
        zorder=4,
        solid_capstyle="round",
    )

    if m is not None:
        ax.plot(
            m["date_dt"],
            for_log(m["precip_mm"]),
            color="#e41add",
            lw=2.0,
            zorder=4,
            solid_capstyle="round",
        )

    handles = [
        Patch(facecolor="#d9d9d9", edgecolor="black", label="Prelim CHIRPS"),
        Patch(facecolor="#fff59d", edgecolor="black", label="Forecast CHIRPS GEFS"),
        Line2D(
            [0],
            [0],
            color="#d95f02",
            lw=1.6,
            label=f"{WET_RUN}-day accumulation",
        ),
    ]
    if m is not None:
        handles.append(
            Line2D([0], [0], color="#e41add", lw=2.0, label="MSWEP daily")
        )

    def hline(y, color, label):
        ax.axhline(y, ls=":", lw=1.3, color=color, zorder=5)
        handles.append(Line2D([0], [0], color=color, ls=":", lw=1.3, label=label))

    hline(DRY_MM, "navy", f"Daily threshold ({DRY_MM:g} mm)")
    hline(
        WET_TOTAL_MM,
        "#d95f02",
        f"{WET_RUN}-day total threshold ({WET_TOTAL_MM:g} mm)",
    )

    top_vals = [df["precip_mm"].max(), roll.max(), WET_MM, WET_TOTAL_MM]
    if m is not None:
        top_vals.append(m["precip_mm"].max())
    top = float(np.nanmax(top_vals))
    if not np.isfinite(top) or top <= 0:
        top = 1.0

    if LOG_SCALE:
        ax.set_ylim(floor, top * 1.6)
    else:
        ax.set_ylim(0, top * 1.1)

    scale_note = f"  (log{LOG_BASE:g})" if LOG_SCALE else ""
    ax.set_title(
        f"{name} ({pcode}) - Daily Rainfall [mm]{scale_note}\nGenerated: {date.today():%Y-%m-%d}"
    )
    ax.set_ylabel("Rainfall (mm)")
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.grid(axis="y", which="major", linestyle="-", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(handles=handles, fontsize=8, ncol=2, loc="upper left")
    ax.set_xlim(df["date_dt"].min() - half_day, df["date_dt"].max() + half_day)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def show():
    """Render the Streamlit page layout."""
    st.set_page_config(
        page_title="Woreda Onset Rainfall Context",
        page_icon="W",
        layout="wide",
    )

    st.title("Woreda Rainfall Time Series & Onset Analysis")
    st.markdown(
        "This dashboard provides a comprehensive view of daily rainfall accumulation, onset classification, "
        "and comparative analysis between CHIRPS (Final/Prelim/Forecast) and MSWEP datasets at the woreda level."
    )

    # Load Base Data
    with st.spinner("Initializing data sources..."):
        daily = load_aggregated_daily()
        mswep = load_mswep_data()
        features_df = load_features_data()
        zone_map = get_woreda_zone_map()

    # Block the UI gracefully if the background pipeline hasn't run yet
    if features_df is None:
        st.warning(
            "**Data Pipeline Incomplete:** The backend data pipeline has not yet generated the required classification features. "
            f"Please execute the data processing script to generate `{CHRIPS_FEATURES_CSV.name}`."
        )
        st.stop()

    # Display Last Updated Time in Sidebar
    last_modified = datetime.fromtimestamp(os.path.getmtime(CHRIPS_FEATURES_CSV))
    st.sidebar.markdown("### System Status")
    st.sidebar.info(f"**Data Last Updated:**\n{last_modified.strftime('%Y-%m-%d %H:%M:%S')}")
    st.sidebar.markdown("---")
    
    st.sidebar.header("Filters & Configuration")

    # Reset Filters Button
    if st.sidebar.button("Reset All Filters", width='stretch'):
        st.session_state["selected_zone"] = "ALL"
        st.session_state["status_category"] = "ALL"
        st.session_state["selected_bucket"] = "ALL"
        st.rerun()

    # 1. Zone Selection directly from the Zone Shapefile
    all_zones = sorted(zone_map["adm2_name"].dropna().unique())
    selected_zone = st.sidebar.selectbox(
        "Select Administrative Zone",
        ["ALL"] + all_zones,
        key="selected_zone",
    )

    # Extract Woredas belonging to the selected zone directly from shapefile mapping
    if selected_zone != "ALL":
        zone_woredas_df = zone_map[zone_map["adm2_name"] == selected_zone]
    else:
        zone_woredas_df = zone_map.copy()

    # 2. Quick Onset Category Filter
    status_category = st.sidebar.radio(
        "Onset Status Filter",
        ["ALL", "None / No Onset", "Onset Met"],
        key="status_category",
    )

    # 3. Specific Bucket Filter
    buckets_list = list(BUCKET_ORDER)
    if "none" not in buckets_list:
        buckets_list.append("none")

    available_buckets = ["ALL"] + buckets_list
    selected_bucket = st.sidebar.selectbox(
        "Detailed Classification Bucket",
        available_buckets,
        format_func=lambda x: "All Buckets" if x == "ALL" else BUCKET_LABELS.get(x, "None / No Onset" if x == "none" else x),
        key="selected_bucket",
    )

    # Apply Zone Filter to features_df using the P-codes from the shapefile
    zone_pcodes = zone_woredas_df["adm3_pcode"].dropna().unique()
    filtered_df = features_df[features_df["adm3_pcode"].isin(zone_pcodes)].copy()

    # Apply Quick Category Filter
    if status_category == "None / No Onset":
        filtered_df = filtered_df[filtered_df["bucket"] == "none"]
    elif status_category == "Onset Met":
        filtered_df = filtered_df[filtered_df["bucket"] != "none"]

    # Apply Specific Bucket Filter
    if selected_bucket != "ALL":
        filtered_df = filtered_df[filtered_df["bucket"] == selected_bucket]

    # Combine Shapefile Woreda metadata with current Onset Status
    woreda_options = filtered_df[["adm3_pcode", "adm3_name", "bucket"]].drop_duplicates("adm3_pcode")

    st.sidebar.caption(f"**{len(woreda_options)}** woreda(s) match the current filter criteria.")

    woreda_dict = {}
    woreda_bucket_dict = {}
    all_pcodes = []

    if not woreda_options.empty:
        woreda_dict = dict(zip(woreda_options["adm3_pcode"], woreda_options["adm3_name"]))
        woreda_bucket_dict = dict(zip(woreda_options["adm3_pcode"], woreda_options["bucket"]))
        all_pcodes = list(woreda_dict.keys())

    def format_woreda_label(code):
        name = woreda_dict.get(code, code)
        b_val = woreda_bucket_dict.get(code, "none")
        label = BUCKET_LABELS.get(b_val, "None / No Onset" if b_val == "none" else b_val)
        return f"{name} ({code}) — Status: {label}"

    # Main Dashboard Tabs
    tab_ts, tab_maps, tab_summary = st.tabs(
        ["Time Series Analysis", "Spatial Onset Maps", "Classification Summary"]
    )

    with tab_ts:
        st.subheader("Woreda Daily Rainfall Context")
        
        with st.expander("About this Chart", expanded=False):
            st.markdown(
                f"The chart displays daily rainfall (bars), the {WET_RUN}-day rolling accumulation (orange line), "
                f"and comparative MSWEP daily values (purple line, if available). Shaded regions indicate Preliminary "
                f"and Forecast data sources. Horizontal dashed lines represent the daily dry threshold ({DRY_MM} mm) "
                f"and the {WET_RUN}-day total threshold ({WET_TOTAL_MM} mm)."
            )

        if all_pcodes:
            default_index = all_pcodes.index("ET030101") if "ET030101" in all_pcodes else 0

            selected_pcode = st.selectbox(
                "Select Woreda for Time Series",
                options=all_pcodes,
                index=default_index,
                format_func=format_woreda_label,
                key="main_woreda_selectbox",
            )

            if selected_pcode:
                woreda_name = woreda_dict.get(selected_pcode, selected_pcode)
                fig = generate_woreda_plot(daily, selected_pcode, woreda_name, mswep=mswep)
                if fig is not None:
                    st.pyplot(fig, width='stretch')
                    plt.close(fig)
                else:
                    st.warning(f"No time series data found for {woreda_name} ({selected_pcode}).")
        else:
            st.info("No woredas match the active zone and filter options in the sidebar.")

    # =====================================================================
    # 2. ONSET MAPS TAB
    # =====================================================================
    with tab_maps:
        st.subheader("Spatial Onset Classification Maps")
        st.markdown(
            "Visual representation of onset status across the study area. "
            "Maps are generated by the backend processing pipeline."
        )
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Composite Onset Status (CHIRPS)**")
            if CHRIPS_MAP_PNG.exists():
                st.image(str(CHRIPS_MAP_PNG), width='stretch')
            else:
                st.info("Map PNG not generated yet. Please run the backend pipeline.")

        with col2:
            st.markdown("**MSWEP Onset Feature**")
            if MSWEP_MAP_PNG.exists():
                st.image(str(MSWEP_MAP_PNG), width='stretch')
            else:
                st.info("MSWEP Feature Map PNG not generated yet.")

    # =====================================================================
    # 3. CLASSIFICATION SUMMARY TAB
    # =====================================================================
    with tab_summary:
        st.subheader("Onset Classification Summary")
        
        col_metric1, col_metric2, col_metric3 = st.columns(3)
        col_metric1.metric("Total Woredas Analyzed", len(filtered_df))
        col_metric2.metric(
            "Onset Criteria Met",
            int((filtered_df["bucket"] != "none").sum()),
        )
        col_metric3.metric(
            "Onset Criteria Not Met",
            int((filtered_df["bucket"] == "none").sum()),
        )

        st.markdown("---")
        st.markdown("**Detailed Woreda Classification**")
        
        cols_to_show = [col for col in ["adm3_pcode", "adm3_name", "bucket", MSWEP_FEATURE] if col in filtered_df.columns]
        
        # Create a display dataframe with renamed columns for better readability
        display_df = filtered_df[cols_to_show].copy()
        column_mapping = {
            "adm3_pcode": "Woreda Code",
            "adm3_name": "Woreda Name",
            "bucket": "Classification Bucket",
        }
        if MSWEP_FEATURE in display_df.columns:
            column_mapping[MSWEP_FEATURE] = "MSWEP Status"
            
        display_df = display_df.rename(columns=column_mapping)

        st.dataframe(
            display_df,
            width='stretch',
            hide_index=True,
        )


if __name__ == "__main__":
    show()