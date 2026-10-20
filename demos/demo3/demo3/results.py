"""Result panels: how the forecast was made, what the model needs, forecast maps, download.

These cover the Demo 3 learning goals beside the event movie: the stages of AI
inference and their cost on the Spark, the practical requirements of each model,
maps of z500 / 2 m temperature / rainfall, and saving the forecast as NetCDF.
"""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import xarray as xr

from . import catalog
from .regions import Box

SPARK_MEMORY_GB = 121  # GB10 unified memory, shared by CPU and GPU
JAX_ENVS = ("graphcast", "neuralgcm", "fgn")  # compiled on the first step (XLA)


def spark_now(path: Path) -> tuple[float | None, float | None, float]:
    """(memory available now, memory total, disk free) in GB on this machine."""
    info = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, val = line.split(":", 1)
            info[key] = int(val.split()[0]) / 1e6
    except (OSError, ValueError):
        pass
    disk = shutil.disk_usage(path if path.exists() else "/").free / 1e9
    return info.get("MemAvailable"), info.get("MemTotal"), disk


def _size(gb: float) -> str:
    return f"{gb * 1000:.0f} MB" if gb < 1 else f"{gb:g} GB"


def requirements(m: catalog.Model, timings: dict, output_dir: Path) -> None:
    """What the selected model needs, plus what this Spark has free right now."""
    t = timings.get(m.key, {})
    peak = t.get("peak_gpu_gb")
    members = t.get("members", m.default_members)
    cols = st.columns(4)
    cols[0].metric("Model weights", _size(m.weights_gb), help="Downloaded once, then read from disk")
    cols[1].metric("Peak GPU memory", f"{peak:g} GB" if peak else "—",
                   help=f"Measured for {t.get('lead_days', 10)} days, {members} member(s); the same on "
                        f"every Spark. The Spark has {SPARK_MEMORY_GB} GB shared by CPU and GPU.")
    cols[2].metric("Output (10 days)", f"{m.output_mb_10d:g} MB",
                   help=f"NetCDF for {m.default_members if m.kind == 'ensemble' else 1} member(s)")
    cols[3].metric("Grid", f"{m.grid_deg:g}° (~{round(m.grid_deg * 111, -1):.0f} km)")
    st.caption(f"**Inputs:** {m.inputs}")
    mem, total, disk = spark_now(output_dir)
    st.caption(f"**This Spark right now:** {mem:.0f} GB of {total or SPARK_MEMORY_GB:.0f} GB memory free · "
               f"{disk:,.0f} GB disk free. (This changes with whatever else is running or stored; "
               "the figures above don't.)" if mem is not None else
               f"**This Spark right now:** {disk:,.0f} GB disk free.")


def _stages(path: str, m: catalog.Model, timings: dict) -> tuple[dict, bool]:
    """Stage timings for this run (written next to the output), else the model's
    typical timings measured on the Spark."""
    side = Path(path).with_suffix(".json")
    if side.exists():
        try:
            return json.loads(side.read_text()), True
        except json.JSONDecodeError:
            pass
    return dict(timings.get(m.key, {})), False


def pipeline(path: str, full: xr.Dataset, m: catalog.Model, timings: dict) -> None:
    """The four stages of AI inference for this run, with their time on the Spark."""
    s, measured = _stages(path, m, timings)
    lead_h = int(full.lead_time.max())
    members = int(full.sizes["ensemble"])
    steps = lead_h // 6
    fmt = lambda v: f"{v:.0f} s" if isinstance(v, (int, float)) else "—"
    run_s, fetch_s = s.get("run_s"), s.get("fetch_s")
    per_step = (f"≈ {run_s / max(1, steps * members):.1f} s per 6 h step"
                + (" per member" if members > 1 else "") if isinstance(run_s, (int, float)) else "")
    out_mb = s.get("output_mb") or round(Path(path).stat().st_size / 1e6, 1)
    peak = s.get("peak_gpu_gb")
    c = st.columns(4)
    c[0].metric("1 · Load model weights", fmt(s.get("load_s")))
    c[0].caption(f"Read {_size(m.weights_gb)} of trained weights from the Spark's disk into GPU "
                 "memory (downloaded from the developer once, on the first run).")
    c[1].metric("2 · Get and prepare inputs", fmt(fetch_s) if fetch_s is not None else "in step 3")
    c[1].caption(f"Download the starting conditions and put them in the exact form the model "
                 f"expects. {m.inputs}")
    c[2].metric("3 · Run the model", fmt(run_s))
    c[2].caption(f"{steps} steps of 6 h" + (f" × {members} members" if members > 1 else "")
                 + (f", {per_step}" if per_step else "") + "; each step feeds the next."
                 + (f" Peak GPU memory {peak:g} GB of {SPARK_MEMORY_GB} GB." if peak else "")
                 + (" The first step includes compiling the model for the GPU (often over a "
                    "minute), so short forecasts look slower per step." if m.env in JAX_ENVS else ""))
    c[3].metric("4 · Post-process and save", fmt(s.get("post_s")))
    c[3].caption(f"Convert units (rain into 6-hour totals in mm, temperature in K, z500 in "
                 f"m² s⁻²), add metadata and save as NetCDF ({out_mb:g} MB).")
    if not measured:
        st.caption("Timings are this model's typical values on the Spark (this saved run was "
                   "made before per-run timings were recorded).")


# ── forecast maps ────────────────────────────────────────────────────────────

def crop(ds: xr.Dataset, box: Box, lead_hours: int) -> xr.Dataset:
    return ds.sel(lat=slice(box.lat_min, box.lat_max), lon=slice(box.lon_min, box.lon_max),
                  lead_time=slice(0, lead_hours))


def daily_precip(ds: xr.Dataset) -> xr.DataArray:
    """24 h totals [mm/day] per forecast day (day 1 = hours 6–24)."""
    tp = ds.tp.sel(lead_time=ds.lead_time > 0)
    day = ((tp.lead_time - 1) // 24 + 1).rename("day")
    return tp.groupby(day).sum("lead_time", min_count=4).rename("tp_daily")


def _map(da: xr.DataArray, box: Box, title: str, cmap: str, units: str, contours=None):
    import cartopy.crs as ccrs
    import cartopy.feature as cf
    fig, ax = plt.subplots(figsize=(7, 5), subplot_kw=dict(projection=ccrs.PlateCarree()))
    ax.set_extent([box.lon_min, box.lon_max, box.lat_min, box.lat_max], crs=ccrs.PlateCarree())
    pc = dict(transform=ccrs.PlateCarree())
    mesh = ax.pcolormesh(da.lon, da.lat, da, cmap=cmap, shading="auto", **pc)
    if contours is not None:
        cs = ax.contour(da.lon, da.lat, contours, colors="k", linewidths=0.6, **pc)
        ax.clabel(cs, fontsize=7, fmt="%.0f")
    ax.add_feature(cf.COASTLINE, linewidth=0.7)
    ax.add_feature(cf.BORDERS, linewidth=0.5, linestyle=":")
    fig.colorbar(mesh, ax=ax, shrink=0.8, label=units)
    ax.set_title(title, fontsize=11)
    return fig


def _series(x, ys: dict, title: str, unit: str, kind: str = "lines"):
    fig = go.Figure()
    for name, y in ys.items():
        if kind == "bars":
            fig.add_bar(x=x, y=y, name=name)
        else:
            fig.add_scatter(x=x, y=y, mode="lines+markers", name=name)
    fig.update_layout(title=title, yaxis_title=unit, height=380, barmode="group",
                      margin=dict(l=10, r=10, t=40, b=10))
    return fig


def field_maps(ds: xr.Dataset, m: catalog.Model, box: Box, region_name: str) -> None:
    """Maps of z500, 2 m temperature and daily rainfall over the region, by lead time."""
    missing = catalog.missing_in_run(ds)
    options = {"Rainfall (daily total)": "tp", "2 m temperature": "t2m",
               "500 hPa geopotential height": "z500"}
    c1, c2 = st.columns([2, 2])
    label = c1.radio("Field", list(options), horizontal=True, key="map_field")
    var = options[label]
    if var in missing:
        st.info(f"{m.name} does not forecast {catalog.VAR_LABELS[var]}, so this map isn't "
                "available for this run.")
        return
    members = [int(e) for e in ds.ensemble.values]
    pick = (c2.selectbox("Ensemble", ["mean"] + [f"member {e + 1}" for e in members],
                         key="map_member") if len(members) > 1 else "mean")
    sel = (lambda da: da.mean("ensemble")) if pick == "mean" else \
        (lambda da: da.sel(ensemble=int(pick.split()[1]) - 1))
    which = "ensemble mean" if pick == "mean" and len(members) > 1 else pick if len(members) > 1 else ""

    if var == "tp":
        daily = daily_precip(ds)
        days = [int(d) for d in daily.day.values]
        if not days:
            st.info("The forecast is too short for a full day of rainfall.")
            return
        d = st.select_slider("Forecast day", days, value=days[0], key="map_day")
        field = sel(daily.sel(day=d))
        title = f"Daily rainfall, day {d}" + (f" ({which})" if which else "")
        fig = _map(field, box, title, "YlGnBu", "mm/day")
        area = daily.weighted(np.cos(np.deg2rad(daily.lat))).mean(["lat", "lon"])
        series = _series(area.day.values, {f"member {e + 1}": area.sel(ensemble=e).values
                                           for e in members},
                         f"{region_name}: area-mean daily rainfall", "mm/day", "bars")
    else:
        leads = [int(h) for h in ds.lead_time.values]
        h = st.select_slider("Forecast hour", leads, value=leads[min(len(leads) - 1, 4)],
                             key="map_lead", format_func=lambda h: f"+{h} h (day {h / 24:g})")
        if var == "t2m":
            da = ds.t2m - 273.15
            field, cmap, unit, contours = sel(da.sel(lead_time=h)), "RdYlBu_r", "°C", None
            name = "2 m temperature"
        else:
            da = ds.z500 / 9.80665
            field, cmap, unit = sel(da.sel(lead_time=h)), "viridis", "m"
            contours, name = field, "500 hPa geopotential height"
        fig = _map(field, box, f"{name}, +{h} h" + (f" ({which})" if which else ""),
                   cmap, unit, contours)
        area = da.weighted(np.cos(np.deg2rad(da.lat))).mean(["lat", "lon"])
        series = _series(area.valid_time.values, {f"member {e + 1}": area.sel(ensemble=e).values
                                                  for e in members},
                         f"{region_name}: area mean", unit)
    a, b = st.columns([3, 2])
    a.pyplot(fig)
    b.plotly_chart(series, width="stretch")
    st.caption(f"Shown on {m.name}'s own {m.grid_deg:g}° grid.")


# ── download ─────────────────────────────────────────────────────────────────

@st.cache_data(max_entries=8, show_spinner=False)
def netcdf_bytes(path: str, mtime: float, box: tuple, lead_hours: int) -> bytes:
    with xr.open_dataset(path) as ds:
        sub = crop(ds, Box(*box), lead_hours).load()
    with tempfile.NamedTemporaryFile(suffix=".nc") as f:
        sub.to_netcdf(f.name)
        return Path(f.name).read_bytes()
