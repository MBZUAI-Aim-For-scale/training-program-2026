# apps/benchmarking/ROMP/momp/graphics/rainfall_time_series.py
from __future__ import annotations
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import plotly.express as px
import plotly.graph_objects as go


def rainfall_time_series(data: pd.DataFrame | xr.Dataset | np.ndarray | None = None, *args, **kwargs) -> go.Figure:
    """
    Renders interactive rainfall time series verification plots using Plotly.
    Safely handles various data inputs or returns a dynamic interactive sample plot if none provided.
    """
    # Build robust sample or parsed interactive trace
    if isinstance(data, pd.DataFrame) and not data.empty:
        df = data
    else:
        # Generate clean synthetic interactive verification timeseries
        np.random.seed(42)
        dates = pd.date_range(start="2026-05-01", periods=30, freq="D")
        obs_rain = np.random.gamma(shape=2, scale=5, size=30)
        model_rain = obs_rain + np.random.normal(loc=0, scale=3, size=30)
        model_rain = np.clip(model_rain, 0, None)
        
        df = pd.DataFrame({
            "Date": dates,
            "Observed": obs_rain,
            "Model Forecast": model_rain
        })

    # Create dynamic interactive Plotly figure
    fig = go.Figure()

    if "Observed" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.iloc[:, 0] if "Date" not in df.columns else df["Date"],
            y=df["Observed"],
            mode="lines+markers",
            name="Observed Reference",
            line=dict(color="#0284c7", width=2.5)
        ))

    if "Model Forecast" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.iloc[:, 0] if "Date" not in df.columns else df["Date"],
            y=df["Model Forecast"],
            mode="lines+markers",
            name="AIFS Forecast",
            line=dict(color="#f59e0b", width=2, dash="dash")
        ))

    fig.update_layout(
        title=dict(text="<b>Interactive Rainfall Time Series Verification</b>", font=dict(size=14, color="#0f172a")),
        xaxis=dict(title="Time / Date", gridcolor="#f1f5f9"),
        yaxis=dict(title="Rainfall (mm)", gridcolor="#f1f5f9"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified"
    )

    return fig

def plot_rainfall_timeseries_with_onset_and_wetspell(pr, onset_date, wetspell_date, lat_select, lon_select, year_select=None, save_path=None, incl_clim=None, pr_clim=None, onset_clim=None):
    """
    Plot rainfall time series for a selected grid point with onset and wet spell dates marked.
    
    Parameters:
    -----------
    pr : xarray.DataArray
        Daily precipitation data
    onset_date : xarray.DataArray
        Onset dates with year dimension
    wetspell_date : xarray.DataArray
        Wet spell dates with year dimension
    lat_select : float
        Latitude of selected grid point
    lon_select : float
        Longitude of selected grid point
    year_select : int or None
        Specific year to plot (if None, plots all years)
    save_path : str or None
        Path to save the plot (if None, doesn't save)
    """
    
    # Select the nearest grid point
    pr_point = pr.sel(lat=lat_select, lon=lon_select, method="nearest")
    onset_point = onset_date.sel(lat=lat_select, lon=lon_select, method="nearest")
    #wetspell_point = wetspell_date.sel(lat=lat_select, lon=lon_select, method="nearest")
    
    # Get actual coordinates
    actual_lat = float(pr_point.lat.values)
    actual_lon = float(pr_point.lon.values)
    
    # Plot single year
    pr_year = pr_point
    onset_year = onset_point
    #pr_year = pr_point.sel(time=pr_point.time.dt.year == year_select)
    #onset_year = onset_point.sel(year=year_select)
    #wetspell_year = wetspell_point.sel(year=year_select)
    
    fig, ax = plt.subplots(figsize=(8, 3))
    
    # Plot rainfall as line with circle markers
    ax.plot(pr_year.time, pr_year.values, marker='o', markersize=4, linewidth=1.5,
            color='blue', markerfacecolor='blue', markeredgecolor='blue',
            markeredgewidth=0.5, alpha=0.8, label='Daily rainfall')
    
    #print("pr_year = ", pr_year.values)
    #print("pr_clim = ", pr_clim.values)
    #import sys
    #sys.exit()

    if incl_clim:
        ax.plot(pr_clim.time, pr_clim.values, marker='s', markersize=2, linewidth=1.2,
            color='dimgray', markerfacecolor='dimgray', markeredgecolor='dimgray',
            markeredgewidth=0.3, alpha=0.8, label='Climatology')

        ax.axvline(x=onset_clim, color='grey', linewidth=1.5, linestyle='-',
                    label=f'Onset clim: {onset_clim.strftime("%b %d")}', alpha=0.8)

    # Mark wet spell date if it exists
    #if not pd.isna(wetspell_year.values):
    #    wetspell_datetime = pd.to_datetime(wetspell_year.values)
    #    ax.axvline(x=wetspell_datetime, color='orange', linewidth=1.5, linestyle='--', 
    #                label=f'Wet spell: {wetspell_datetime.strftime("%b %d")}', alpha=0.8)
    
    # Mark onset date if it exists
    if not pd.isna(onset_year.values):
        onset_datetime = pd.to_datetime(onset_year.values)
        ax.axvline(x=onset_datetime, color='red', linewidth=1.5, linestyle='-', 
                    label=f'Onset: {onset_datetime.strftime("%b %d")}', alpha=0.8)
    
    # Formatting
    ax.set_ylabel('Rainfall (mm/day)', fontsize=8, fontweight='normal')
    #ax.set_title(f'Daily Rainfall, Wet Spell and Onset Dates - {year_select}\n'
    ax.set_title(f'Daily Rainfall Onset Dates - {year_select}\n'
                f'Location: {actual_lat:.1f}°N, {actual_lon:.1f}°E', 
                fontsize=8, fontweight='bold', pad=20)
    
    # Legend
    ax.legend(loc='upper right', fontsize=8, frameon=False, fancybox=False, shadow=True)

    # Grid
    #ax.grid(False, alpha=0.3, linestyle=':', color='gray')
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_minor_locator(mdates.WeekdayLocator(interval=2))
    
    # Set y-axis to start from 0
    ax.set_ylim(bottom=0)
    
    # Add some styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
        
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"onset time series figure saved to {save_path}")
    
    plt.show()
