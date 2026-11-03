"""Time-series comparison of the loaded models: one chart of area-mean daily rainfall
(on the CHIRPS grid, with CHIRPS) or daily maximum 2 m temperature (with ERA5 on
request), plus each model's onset share for the Onset use case."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import xarray as xr

from . import catalog, obs, onset, theme
from .regions import Box
from .results import crop

COLORS = ["#2563eb", "#dc2626", "#059669", "#d97706"]


@st.cache_data(max_entries=32, show_spinner=False)
def _area_series(path: str, mtime: float, box_t: tuple, init: pd.Timestamp, lead_days: int,
                 var: str) -> xr.DataArray:
    """Daily area-mean series (ensemble, time): rain (mm/day) over CHIRPS land points, or daily
    maximum 2 m temperature (°C, from the 6-hourly values)."""
    box = Box(*box_t)
    with xr.open_dataset(path) as ds:
        sub = crop(ds, Box(box.lat_min - 1, box.lat_max + 1, box.lon_min - 1, box.lon_max + 1),
                   lead_days * 24).load()
    if var == "tp":
        fc = onset.forecast_daily(sub, init)
        try:
            last = obs.last_day()
            like = obs.daily(box, last, last).isel(time=0, drop=True)
            fc = onset.to_grid(fc, like)
        except (OSError, FileNotFoundError):
            fc = fc.sel(lat=slice(box.lat_min, box.lat_max), lon=slice(box.lon_min, box.lon_max))
        return onset.area_mean(fc)
    t = sub.t2m.sel(lat=slice(box.lat_min, box.lat_max), lon=slice(box.lon_min, box.lon_max)) - 273.15
    t = t.sel(lead_time=t.lead_time > 0)
    day = init.normalize() + pd.to_timedelta((t.lead_time.values - 1) // 24, unit="D")
    t = t.assign_coords(day=("lead_time", day)).groupby("day").max("lead_time").rename(day="time")
    return onset.area_mean(t)


@st.cache_data(max_entries=16, show_spinner="Getting observations…")
def _obs_series(box_t: tuple, init: pd.Timestamp, lead_days: int, var: str) -> xr.DataArray | None:
    box = Box(*box_t)
    end = init.normalize() + pd.Timedelta(days=lead_days - 1)
    if var == "tp":
        try:
            return onset.area_mean(obs.daily(box, init.normalize(), end))
        except (OSError, FileNotFoundError):
            return None
    # ERA5 2 m temperature from Google's copy (as in the event movie), daily max of 6-hourly values
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent / "event_movie"))
    import run_event_movie as rem
    da = rem._era5("2m_temperature", init.normalize(), end + pd.Timedelta(hours=18),
                   (box.lon_min, box.lon_max, box.lat_min, box.lat_max))
    da = (da - 273.15).resample(time="1D").max()
    return onset.area_mean(da)


def chart(paths, keys, init, lead_days, box, region_name, var):
    box_t = (box.lat_min, box.lat_max, box.lon_min, box.lon_max)
    fig = go.Figure()
    unit = "mm/day" if var == "tp" else "°C"
    for i, k in enumerate(k for k in keys if k in paths):
        p = Path(paths[k])
        ser = _area_series(str(p), p.stat().st_mtime, box_t, init, lead_days, var)
        name, color = catalog.BY_KEY[k].name, COLORS[i % len(COLORS)]
        mean = ser.mean("ensemble") if "ensemble" in ser.dims else ser
        if "ensemble" in ser.dims and ser.sizes["ensemble"] > 1:
            lo, hi = ser.min("ensemble"), ser.max("ensemble")
            fig.add_scatter(x=np.concatenate([ser.time.values, ser.time.values[::-1]]),
                            y=np.concatenate([hi.values, lo.values[::-1]]), fill="toself",
                            line=dict(width=0), fillcolor=color, opacity=0.15, showlegend=False,
                            hoverinfo="skip", name=f"{name} range")
            name += f" (mean of {ser.sizes['ensemble']})"
        fig.add_scatter(x=mean.time.values, y=mean.values, mode="lines+markers", name=name,
                        line=dict(color=color, width=2))
    obs_name = "CHIRPS" if var == "tp" else "ERA5"
    want_obs = var == "tp" or st.checkbox("Add ERA5 observations", key="cmp_obs",
                                          help="Fetched from the cloud: a minute or two the first time.")
    if want_obs:
        o = _obs_series(box_t, init, lead_days, var)
        if o is not None and o.sizes.get("time", 0):
            fig.add_scatter(x=o.time.values, y=o.values, mode="lines+markers", name=f"Observed ({obs_name})",
                            line=dict(color="black", width=2, dash="dash"))
        elif var == "tp":
            st.caption("No CHIRPS observations for these dates yet.")
    what = {"tp": "area-mean daily rainfall (land)", "t2m": "area-mean daily maximum 2 m temperature"}[var]
    fig.update_layout(title=f"{region_name}: {what}", yaxis_title=unit, height=420,
                      margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig, width="stretch", key="compare_chart")
    theme.hint("Shaded: range across members",
               "Each model's rainfall is put on the CHIRPS grid and averaged over land, so it compares "
               "like for like with the observations." if var == "tp" else
               "Daily maximum of the 6-hourly values (models and ERA5 alike).")



def onset_table(paths, keys, init, lead_days, box, region_name, defn, search_from):
    from .onset_view import _day_field, result
    rows, obs_row = [], None
    for k in keys:
        if k not in paths:
            continue
        try:
            r = result(paths[k], box, init, lead_days, defn, search_from)
        except (OSError, FileNotFoundError):
            continue
        land = r["land"]
        n = max(1, int(land.sum()))
        f = r["f_on"]
        fday, _ = _day_field(f, r["init_day"], land)
        during = fday >= 1
        share = lambda m: 100 * float((m & land).sum()) / n / (f.sizes.get("ensemble", 1))
        rows.append((catalog.BY_KEY[k].name, f"{share(during & (f.status == onset.CONFIRMED)):.0f}%",
                     f"{share(during & (f.status == onset.POSSIBLE)):.0f}%"))
        if obs_row is None and r["o_on"] is not None:
            oday, _ = _day_field(r["o_on"], r["init_day"], land)
            n_days = (r["fc_end"] - r["init_day"]).days + 1
            obs_row = ("Observed (CHIRPS)",
                       f"{100 * float(((oday >= 1) & (oday <= n_days) & land).sum()) / n:.0f}%", "")
    if obs_row:
        rows.append(obs_row)
    if not rows:
        return
    st.markdown("**Onset during the forecast**",
                help=f"{defn.name}, searching from {search_from:%-d %b %Y}. Share of land points "
                     "(averaged over members for ensembles).")
    st.dataframe(pd.DataFrame(rows, columns=["", "Onset (confirmed)", "Possible onset"]),
                 hide_index=True, width="stretch")
