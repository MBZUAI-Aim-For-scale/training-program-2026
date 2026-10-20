# ui/pages/timeseries.py
from __future__ import annotations

import os
import glob
import streamlit as st
import pandas as pd
import numpy as np
import xarray as xr
from datetime import datetime, timedelta
import plotly.graph_objects as go

from config import STATION_KEY, DATASET_COLORS
from utils.plotting import (
    build_ts_figure, add_rain_trace, add_onset_markers, add_threshold_line
)

# Import climatology engine
try:
    from apps.onset_obs.climatology_engine import (
        run_climatology_engine,
        DEFAULT_SELECTED_WOREDAS,
    )
except ImportError:
    run_climatology_engine = None
    DEFAULT_SELECTED_WOREDAS = []

# Directory for local CHIRPS_IMERG files
CHIRPS_IMERG_DIR = "data/CHIRPS_IMERG"


def offset_to_date(offset_days: float) -> datetime:
    """Helper to convert climatology day offsets back to readable calendar dates."""
    base_date = datetime(2001, 6, 1)
    return base_date + timedelta(days=float(offset_days))


def get_woreda_centroid(woreda_name: str, gdf=None, guardrail_df=None) -> tuple[float, float]:
    """
    Calculates the exact lat/lon centroid for a given Woreda dynamically 
    from the shapefile GeoDataFrame.
    """
    if gdf is not None and not gdf.empty:
        name_col = None
        for col in ["Woreda", "ADM3_EN", "woreda", "WOREDANAME", "WOREDA"]:
            if col in gdf.columns:
                name_col = col
                break

        if name_col:
            match = gdf[gdf[name_col] == woreda_name]
            if not match.empty:
                try:
                    geom_gdf = match.to_crs(epsg=4326) if hasattr(match, "crs") and match.crs and match.crs.to_epsg() != 4326 else match
                    centroid = geom_gdf.geometry.centroid.iloc[0]
                    return float(centroid.y), float(centroid.x)
                except Exception:
                    pass

    if guardrail_df is not None and not guardrail_df.empty and "Woreda" in guardrail_df.columns:
        match = guardrail_df[guardrail_df["Woreda"] == woreda_name]
        if not match.empty:
            w_row = match.iloc[0]
            lat_col = next((c for c in ["Lat", "lat", "Latitude", "latitude"] if c in w_row), None)
            lon_col = next((c for c in ["Lon", "lon", "Longitude", "longitude"] if c in w_row), None)
            if lat_col and lon_col:
                return float(w_row[lat_col]), float(w_row[lon_col])

    if "lat0" in st.session_state and "lon0" in st.session_state:
        return float(st.session_state["lat0"]), float(st.session_state["lon0"])

    return 9.0, 39.0


def extract_chirps_imerg_ts(lat: float, lon: float, start_year: int, end_year: int) -> pd.DataFrame:
    """
    Extracts daily time series from NetCDF files inside data/CHIRPS_IMERG/
    for specified range of years near lat/lon.
    """
    records = []
    
    for yr in range(start_year, end_year + 1):
        nc_path = os.path.join(CHIRPS_IMERG_DIR, f"{yr}.nc")
        if not os.path.exists(nc_path):
            continue

        try:
            with xr.open_dataset(nc_path) as ds:
                # Detect Latitude / Longitude coordinates
                lat_name = next((c for c in ["lat", "latitude", "y"] if c in ds.coords or c in ds.dims), None)
                lon_name = next((c for c in ["lon", "longitude", "x"] if c in ds.coords or c in ds.dims), None)
                
                # Detect rainfall variable name
                var_name = next((v for v in ["precip", "precipitation", "PRCP", "rain", "r"] if v in ds.data_vars), None)
                if not var_name:
                    var_name = list(ds.data_vars.keys())[0]

                if lat_name and lon_name:
                    point_ds = ds.sel({lat_name: lat, lon_name: lon}, method="nearest")[var_name]
                else:
                    point_ds = ds[var_name]

                df = point_ds.to_dataframe().reset_index()
                
                # Identify time column
                time_col = next((c for c in ["time", "date"] if c in df.columns), df.columns[0])
                df = df.rename(columns={time_col: "time", var_name: "rain"})
                records.append(df[["time", "rain"]])
        except Exception as e:
            st.warning(f"Error reading NetCDF file {nc_path}: {e}")

    if records:
        full_df = pd.concat(records, ignore_index=True)
        full_df["time"] = pd.to_datetime(full_df["time"])
        return full_df.sort_values("time")
    
    return pd.DataFrame(columns=["time", "rain"])


def info_box(text: str, kind: str = "") -> None:
    st.markdown(f'<div class="ibox {kind}">{text}</div>', unsafe_allow_html=True)


def section_header(text: str, icon: str = "", accent: str = "") -> None:
    ico = f"<span style='font-size:.95rem'>{icon}</span> " if icon else ""
    cls = f"sec-hd {accent}".strip()
    st.markdown(f'<div class="{cls}"><span class="sec-hd-accent"></span>{ico}{text}</div>', unsafe_allow_html=True)


def render():
    # Detect available NetCDF files in data/CHIRPS_IMERG
    nc_files = glob.glob(os.path.join(CHIRPS_IMERG_DIR, "*.nc"))
    available_nc_years = sorted([
        int(os.path.basename(f).replace(".nc", "")) 
        for f in nc_files if os.path.basename(f).replace(".nc", "").isdigit()
    ])

    emi_meta = st.session_state.get("emi_meta")
    emi_ts = st.session_state.get("emi_ts")

    # Available data sources
    all_sources = ["CHIRPS_IMERG (Combined NetCDF)"]
    if emi_meta is not None or STATION_KEY in st.session_state.get("selected_datasets", []):
        all_sources.append("Ground Gauge Stations")

    selected_sources = st.session_state.get("selected_datasets", all_sources)
    y0 = st.session_state.get("y0", available_nc_years[0] if available_nc_years else 2010)
    y1 = st.session_state.get("y1", available_nc_years[-1] if available_nc_years else 2024)
    time_mode = st.session_state.get("time_mode", "Single Year")
    show_markers = st.session_state.get("show_markers", True)
    onset_params = st.session_state.get("onset_params", {})

    info_box(
        "Compare daily rainfall time series from local CHIRPS_IMERG NetCDF files (1998–2024) and Ground Gauge Stations.", 
        "sky"
    )

    clim_result = st.session_state.get("clim_result", {})
    gdf = st.session_state.get("gdf") or clim_result.get("gdf")
    all_woredas = clim_result.get("all_woredas", DEFAULT_SELECTED_WOREDAS)
    guardrail_df = clim_result.get("guardrail_df", pd.DataFrame())

    if gdf is not None and not gdf.empty:
        for col in ["Woreda", "ADM3_EN", "woreda", "WOREDANAME"]:
            if col in gdf.columns:
                all_woredas = sorted(gdf[col].dropna().unique().tolist())
                break

    st.markdown("---")

    tab_ts, tab_onset_clim = st.tabs(["Woreda Daily Time Series", "Woreda Onset Climatology"])

    # =========================================================================
    # TAB 1: Daily Time Series (CHIRPS_IMERG NetCDF Data + Ground Gauge Stations)
    # =========================================================================
    with tab_ts:
        section_header("Woreda Daily Rainfall Time Series Comparison")

        ctrl_col1, ctrl_col2 = st.columns([1, 2])

        with ctrl_col1:
            woreda_options = all_woredas if all_woredas else ["Addis Ababa", "Adama", "Hawassa"]
            selected_woreda = st.selectbox(
                "Select Woreda",
                options=woreda_options,
                index=0
            )

        with ctrl_col2:
            selected_datasets_ts = st.multiselect(
                "Select Data Sources",
                options=all_sources,
                default=[s for s in selected_sources if s in all_sources] or all_sources
            )

        # Dynamic centroid calculation
        lat0, lon0 = get_woreda_centroid(selected_woreda, gdf=gdf, guardrail_df=guardrail_df)

        st.caption(
            f"Extracting time series near **{selected_woreda}** shapefile centroid "
            f"(**{lat0:.2f}°N, {lon0:.2f}°E**)"
        )

        ts_title = (
            f"Rainfall Time Series for {selected_woreda} — {y0}" 
            if time_mode == "Single Year"
            else f"Rainfall Time Series for {selected_woreda} — {y0}–{y1}"
        )
        
        fig_ts = build_ts_figure(ts_title, height=500)
        last_times = None
        export_ts_dict = {}

        # 1. Load CHIRPS_IMERG NetCDF dataset
        if "CHIRPS_IMERG (Combined NetCDF)" in selected_datasets_ts:
            start_yr = y0 if time_mode == "Single Year" else y0
            end_yr = y0 if time_mode == "Single Year" else y1
            
            df_nc = extract_chirps_imerg_ts(lat0, lon0, start_yr, end_yr)
            
            if not df_nc.empty:
                clr = DATASET_COLORS.get("CHIRPS", {}).get("rain", "#1E88E5")
                add_rain_trace(fig_ts, df_nc["time"], df_nc["rain"], label="CHIRPS_IMERG", color=clr, markers=show_markers)
                add_onset_markers(fig_ts, df_nc, "CHIRPS_IMERG", "CHIRPS_IMERG", onset_params)
                last_times = df_nc["time"]
                export_ts_dict["CHIRPS_IMERG"] = df_nc.set_index("time")["rain"]
            else:
                st.warning(f"No NetCDF data found in `{CHIRPS_IMERG_DIR}` for the selected period {start_yr}–{end_yr}.")

        # 2. Load Ground Gauge Station Data
        if "Ground Gauge Stations" in selected_datasets_ts:
            if emi_ts is not None and emi_meta is not None:
                dist = np.sqrt((emi_meta["lat"] - lat0)**2 + (emi_meta["lon"] - lon0)**2)
                near_st = emi_meta.index[dist.argmin()]
                try:
                    s = emi_ts[near_st].loc[f"{y0}-01-01":f"{y1 if time_mode != 'Single Year' else y0}-12-31"]
                    df_st = s.rename("rain").to_frame().reset_index()
                    df_st.columns = ["time", "rain"]
                    
                    st_color = DATASET_COLORS.get(STATION_KEY, {}).get("rain", "#D81B60")
                    add_rain_trace(
                        fig_ts, df_st["time"], df_st["rain"],
                        label=f"{near_st} (Station)",
                        color=st_color,
                        markers=show_markers
                    )
                    add_onset_markers(fig_ts, df_st, STATION_KEY, near_st, onset_params)
                    last_times = df_st["time"]
                    export_ts_dict[f"{near_st}_station"] = df_st.set_index("time")["rain"]
                    st.info(
                        f"Nearest ground gauge station to **{selected_woreda}** centroid ({lat0:.2f}°N, {lon0:.2f}°E): "
                        f"**{near_st}** (distance: {dist.min():.2f}°)"
                    )
                except Exception as e:
                    st.warning(f"Station data unavailable for {near_st}: {e}")

        # 3. Add Wet Day Threshold Line
        if last_times is not None:
            wet_day_mm = onset_params.get("wet_day_mm", 1.0)
            add_threshold_line(fig_ts, wet_day_mm, last_times)

        fig_ts.update_yaxes(title_text="Rainfall (mm/day)")
        st.plotly_chart(fig_ts, width="stretch")

        # 4. CSV Export
        if export_ts_dict:
            df_export = pd.DataFrame(export_ts_dict)
            st.download_button(
                label=f"Export Daily Timeseries for {selected_woreda} (CSV)",
                data=df_export.to_csv().encode("utf-8"),
                file_name=f"chirps_imerg_timeseries_{selected_woreda}_{y0}_{y1}.csv",
                mime="text/csv"
            )

    # =========================================================================
    # TAB 2: Woreda Onset Climatology
    # =========================================================================
    with tab_onset_clim:
        section_header("Woreda Onset Climatology & Multi-Year Trends")

        if "clim_result" not in st.session_state or st.session_state["clim_result"] is None:
            st.warning("Climatology data has not been generated yet. Execute the climatology engine to proceed.")
            if st.button("Compute Woreda Climatology", type="primary"):
                with st.spinner("Processing Woreda climatology datasets..."):
                    if run_climatology_engine is not None:
                        res = run_climatology_engine(selected_woredas=None)
                        st.session_state["clim_result"] = res
                        st.rerun()
                    else:
                        st.error("Climatology engine module could not be loaded.")
            return

        ts_df = clim_result.get("ts_df", pd.DataFrame())
        guardrail_df = clim_result.get("guardrail_df", pd.DataFrame())

        if ts_df.empty:
            st.error("No woreda climatology timeseries data is available.")
            return

        default_woredas = DEFAULT_SELECTED_WOREDAS if DEFAULT_SELECTED_WOREDAS else all_woredas[:1]
        selected_woredas = st.multiselect(
            "Select Woreda(s) for Climatology View",
            options=all_woredas,
            default=[w for w in default_woredas if w in all_woredas]
        )

        if not selected_woredas:
            st.info("Please select at least one Woreda above to view the climatology time series.")
            return

        filtered_ts = ts_df[ts_df["Woreda"].isin(selected_woredas)]

        fig_woreda_clim = go.Figure()
        for idx, woreda_name in enumerate(selected_woredas):
            w_df = filtered_ts[filtered_ts["Woreda"] == woreda_name].sort_values("Year")
            if w_df.empty:
                continue

            date_labels = [
                offset_to_date(val).strftime("%b %d") if np.isfinite(val) else "N/A"
                for val in w_df["OnsetOffset"]
            ]

            fig_woreda_clim.add_trace(go.Scatter(
                x=w_df["Year"],
                y=w_df["OnsetOffset"],
                mode="lines+markers",
                name=woreda_name,
                line=dict(width=2),
                marker=dict(size=6),
                text=date_labels,
                hovertemplate=(
                    "<b>%{fullData.name}</b><br>"
                    "Year: %{x}<br>"
                    "Offset: %{y:.1f} days after June 1<br>"
                    "Date: %{text}<extra></extra>"
                )
            ))

            mean_offset = w_df["OnsetOffset"].mean()
            if np.isfinite(mean_offset):
                fig_woreda_clim.add_trace(go.Scatter(
                    x=w_df["Year"],
                    y=[mean_offset] * len(w_df),
                    mode="lines",
                    name=f"{woreda_name} Mean",
                    line=dict(dash="dash", width=1.5),
                    showlegend=False,
                    hoverinfo="skip"
                ))

        fig_woreda_clim.update_layout(
            title="Woreda Rainy-Season Onset Offset Trend (Days after June 1)",
            xaxis=dict(title="Year", tickmode="linear", dtick=2),
            yaxis=dict(title="Onset Offset (Days)"),
            height=480,
            hovermode="x unified",
            template="plotly_white"
        )
        st.plotly_chart(fig_woreda_clim, width="stretch")

        st.markdown("#### Climatology Summary & Guardrails")
        metric_cols = st.columns(min(len(selected_woredas), 4))
        for i, woreda_name in enumerate(selected_woredas):
            col_idx = i % 4
            w_guard = guardrail_df[guardrail_df["Woreda"] == woreda_name] if not guardrail_df.empty else pd.DataFrame()
            w_ts = filtered_ts[filtered_ts["Woreda"] == woreda_name]

            if not w_guard.empty:
                med_date = w_guard["Median_Valid_Date"].values[0]
                conf = w_guard["Confidence"].values[0]
                std_days = w_ts["OnsetOffset"].std()
            else:
                med_date = "N/A"
                conf = "N/A"
                std_days = np.nan

            with metric_cols[col_idx]:
                st.metric(
                    label=woreda_name,
                    value=f"{med_date}",
                    delta=f"±{std_days:.1f} days" if pd.notna(std_days) else None,
                    delta_color="off"
                )
                st.caption(f"Confidence: **{conf}**")

        st.markdown("---")

        st.markdown("#### Export Woreda Climatology Time Series")
        st.download_button(
            label="Download Woreda Onset Climatology CSV",
            data=filtered_ts.to_csv(index=False).encode("utf-8"),
            file_name=f"woreda_onset_climatology_{'_'.join(selected_woredas)}.csv",
            mime="text/csv"
        )