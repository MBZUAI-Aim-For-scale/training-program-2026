"""Onset results tab: definition picker, onset maps, rainfall series, onset movie."""
from __future__ import annotations

import hashlib
import io
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import xarray as xr

from . import catalog, obs, onset
from .regions import Box
from .results import crop
from .store import OUTPUT_DIR

MOVIES = OUTPUT_DIR / "_movies"
GREY, NONE_C = "#9ca3af", "#ffffff"


def pick_definition(region_name: str, key: str = "onset") -> onset.Definition:
    """Definition selector plus an expander to adjust its settings."""
    keys = list(onset.DEFINITIONS)
    default = onset.default_definition(region_name)
    c1, c2 = st.columns([2, 3])
    k = c1.selectbox("Onset definition", keys, index=keys.index(default), key=f"{key}_def",
                     format_func=lambda k: f"{onset.DEFINITIONS[k].name}  ·  {onset.DEFINITIONS[k].home}")
    d = onset.DEFINITIONS[k]
    if k != default:
        c2.caption(f"ⓘ The usual choice for {region_name} is "
                   f"{onset.DEFINITIONS[default].name}.")
    if d.spell_mm is None and region_name not in onset.MR_REGIONS:
        c2.warning(f"{d.name} uses local wet-spell amounts that ROMP provides for India only; "
                   f"points outside that grid get no onset. Set a fixed amount below to use it "
                   f"for {region_name}.")
    with st.expander("Adjust the definition"):
        a, b, c = st.columns(3)
        wet = a.number_input("Wet day: at least (mm)", 0.1, 10.0, d.wet_day_mm, 0.1, key=f"{key}_wet_{k}")
        n = a.number_input("Wet spell length (days)", 1, 10, d.spell_days, key=f"{key}_n_{k}")
        local = d.spell_mm is None and b.checkbox("Local climatological amount (MWset)", True,
                                                   key=f"{key}_local_{k}")
        amt = None if local else b.number_input("Wet spell total: more than (mm)", 1.0, 200.0,
                                                d.spell_mm or 25.0, 1.0, key=f"{key}_amt_{k}")
        dry_on = b.checkbox("Check for a dry spell afterwards", d.dry_kind is not None,
                            key=f"{key}_dry_on_{k}")
        dry_days = c.slider("Dry spell length (days)", 3, 20, d.dry_days if d.dry_kind else 7,
                            key=f"{key}_dd_{k}", disabled=not dry_on)
        dry_win = c.slider("Searched for within (days after the wet spell)", 5, 45,
                           d.dry_window if d.dry_kind else 21, key=f"{key}_dw_{k}", disabled=not dry_on)
        kind = (d.dry_kind or "days") if dry_on else None
        dry_mm = (c.number_input("Dry spell total: less than (mm)", 0.0, 50.0, d.dry_mm, 0.5,
                                 key=f"{key}_dmm_{k}") if kind == "total" else d.dry_mm)
    out = onset.custom(d, wet_day_mm=wet, spell_days=int(n), dry_kind=kind,
                       dry_days=int(dry_days), dry_window=int(dry_win), dry_mm=dry_mm)
    if amt is not None:
        out = onset.custom(out, spell_mm=amt)
    if out != d:
        out = onset.custom(out, name=f"{d.name} (adjusted)")
    st.caption(f"**{out.name}:** {out.describe()} {d.note}")
    return out


@st.cache_data(max_entries=16, show_spinner="Working out onset dates…")
def compute(path: str, mtime: float, box_t: tuple, init: pd.Timestamp, lead_days: int,
            defn: onset.Definition, search_from: pd.Timestamp) -> dict:
    box = Box(*box_t)
    with xr.open_dataset(path) as ds:
        sub = crop(ds, Box(box.lat_min - 1, box.lat_max + 1, box.lon_min - 1, box.lon_max + 1),
                   lead_days * 24).load()
    fc = onset.forecast_daily(sub, init)
    fc_end = pd.Timestamp(fc.time.values[-1])
    obs_last = obs.last_day()
    # observations: the season so far, the forecast days, and enough after them to confirm onset
    want_end = fc_end + pd.Timedelta(days=defn.needs_after)
    end = min(want_end, obs_last) if obs_last is not None else want_end
    try:
        o = obs.daily(box, min(search_from, end), end)
    except FileNotFoundError:
        o = obs.daily(box, obs_last, obs_last)
    template = o.isel(time=-1, drop=True)
    fc_g = onset.to_grid(fc, template)
    thresh = onset.threshold_for(defn, o)
    series = onset.joined(o, fc_g, init, search_from)
    f_on = onset.onset_dates(series, defn, search_from, thresh)
    o_win = o.sel(time=slice(search_from, None))
    o_on = onset.onset_dates(o_win, defn, search_from, thresh) if o_win.sizes["time"] else None
    # one area-mean series for the chart (threshold: area mean of the local amounts)
    land = template.notnull() & thresh.notnull()  # points where the definition can apply
    a_thr = float(onset.area_mean(thresh.where(land))) if defn.spell_mm is None else defn.spell_mm
    a_ser = onset.area_mean(series)
    a_obs = onset.area_mean(o_win) if o_win.sizes["time"] else None
    a_f = onset.onset_dates(a_ser.expand_dims(lat=[0.0], lon=[0.0]), defn, search_from,
                            xr.DataArray(a_thr))
    a_o = (onset.onset_dates(a_obs.expand_dims(lat=[0.0], lon=[0.0]), defn, search_from,
                             xr.DataArray(a_thr)) if a_obs is not None else None)
    return dict(series=series, obs=o_win, f_on=f_on, o_on=o_on, land=land, thresh=thresh,
                a_ser=a_ser, a_obs=a_obs, a_f=a_f, a_o=a_o, a_thr=a_thr,
                obs_last=obs_last, fc_end=fc_end, init_day=init.normalize())


def _day_field(on: xr.Dataset, init_day: pd.Timestamp, land: xr.DataArray):
    """Onset as forecast day (1 = start date), 0 = before the start, NaN = none; plus a mask of
    possible (unconfirmed) onsets during the forecast days."""
    d = (on.onset - np.datetime64(init_day)) / np.timedelta64(1, "D") + 1
    d = d.where(on.status > 0).clip(min=0)
    return d.where(land), (on.status == onset.POSSIBLE) & (d >= 1) & land


def _onset_map(ax, day, possible, box, n_days, title):
    import cartopy.crs as ccrs
    import cartopy.feature as cf
    from matplotlib.colors import BoundaryNorm, ListedColormap
    ax.set_extent([box.lon_min, box.lon_max, box.lat_min, box.lat_max], crs=ccrs.PlateCarree())
    colors = [GREY] + list(plt.get_cmap("viridis_r")(np.linspace(0.05, 0.95, n_days)))
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, n_days + 1.5), cmap.N)
    mesh = ax.pcolormesh(day.lon, day.lat, day, cmap=cmap, norm=norm, shading="auto",
                         transform=ccrs.PlateCarree())
    if bool(possible.any()):
        # hatch possible (unconfirmed) onsets, leaving their colours visible
        plt.rcParams["hatch.linewidth"] = 0.4
        ax.contourf(possible.lon, possible.lat, possible.astype(float), levels=[0.5, 1.5],
                    colors="none", hatches=["////"], transform=ccrs.PlateCarree())
    ax.add_feature(cf.COASTLINE, linewidth=0.7)
    ax.add_feature(cf.BORDERS, linewidth=0.5, linestyle=":")
    ax.set_title(title, fontsize=10)
    return mesh


def _maps_figure(left, right, box, n_days, titles, init_day):
    import cartopy.crs as ccrs
    panels = [p for p in (left, right) if p is not None]
    fig, axes = plt.subplots(1, len(panels), figsize=(6.2 * len(panels), 5),
                             subplot_kw=dict(projection=ccrs.PlateCarree()), squeeze=False)
    mesh = None
    for ax, (day, poss), t in zip(axes[0], panels, titles):
        mesh = _onset_map(ax, day, poss, box, n_days, t)
    cb = fig.colorbar(mesh, ax=axes[0].tolist(), shrink=0.8, ticks=range(0, n_days + 1))
    cb.ax.set_yticklabels(["before start"] + [f"{(init_day + pd.Timedelta(days=i - 1)):%-d %b}"
                                              for i in range(1, n_days + 1)])
    cb.set_label("Onset date (white: no onset yet)")
    return fig


def _prob_figure(f_on, init_day, land, box, n_days):
    import cartopy.crs as ccrs
    import cartopy.feature as cf
    started = (f_on.status > 0)
    p = (started.mean("ensemble") * 100).where(land)
    fig, ax = plt.subplots(figsize=(6.2, 5), subplot_kw=dict(projection=ccrs.PlateCarree()))
    ax.set_extent([box.lon_min, box.lon_max, box.lat_min, box.lat_max], crs=ccrs.PlateCarree())
    mesh = ax.pcolormesh(p.lon, p.lat, p, cmap="YlGnBu", vmin=0, vmax=100, shading="auto",
                         transform=ccrs.PlateCarree())
    ax.add_feature(cf.COASTLINE, linewidth=0.7)
    ax.add_feature(cf.BORDERS, linewidth=0.5, linestyle=":")
    fig.colorbar(mesh, ax=ax, shrink=0.8, label="% of members")
    ax.set_title(f"Chance onset has happened by {init_day + pd.Timedelta(days=n_days - 1):%-d %b} "
                 f"({f_on.sizes['ensemble']} members)", fontsize=10)
    return fig


def _series_figure(r, defn, search_from, region_name, point_label):
    ser, a_obs = r["a_ser"], r["a_obs"]
    init_day = r["init_day"]
    fig = go.Figure()
    times = pd.DatetimeIndex(ser.time.values)
    past = times < init_day
    first = ser.isel(ensemble=0) if "ensemble" in ser.dims else ser
    fig.add_bar(x=times[past], y=first.values[past], name="Observed (CHIRPS)", marker_color=GREY)
    if "ensemble" in ser.dims and ser.sizes["ensemble"] > 1:
        for e in ser.ensemble.values:
            fig.add_scatter(x=times[~past], y=ser.sel(ensemble=e).values[~past], mode="lines+markers",
                            name=f"Forecast member {int(e) + 1}", line=dict(width=1.5))
    else:
        fig.add_bar(x=times[~past], y=first.values[~past], name="Forecast", marker_color="#2563eb")
    if a_obs is not None:
        after = a_obs.sel(time=slice(init_day, None))
        if after.sizes["time"]:
            fig.add_scatter(x=after.time.values, y=after.values, mode="markers", name="Observed (after start)",
                            marker=dict(color="black", symbol="circle-open", size=8))
    if np.isfinite(r["a_thr"]):
        fig.add_hline(y=r["a_thr"] / defn.spell_days, line_dash="dot", line_color="#6b7280",
                      annotation_text=f"wet-spell amount ÷ {defn.spell_days} days", annotation_position="top left")
    marks = [(search_from, "search starts", "#6b7280"), (init_day, "forecast starts", "#111827")]
    for name, on, color in (("observed onset", r["a_o"], "black"), ("forecast onset", r["a_f"], "#2563eb")):
        if on is None:
            continue
        vals = on.onset.values.ravel()
        stats = on.status.values.ravel()
        for v, s in zip(vals, stats):
            if s > 0:
                marks.append((pd.Timestamp(v), name + (" (possible)" if s == onset.POSSIBLE else ""), color))
    seen = set()
    for x, text, color in marks:
        label = text if text not in seen else None
        seen.add(text)
        fig.add_vline(x=x, line_dash="dash" if "onset" in text else "solid", line_color=color, line_width=1)
        if label:
            fig.add_annotation(x=x, y=1.02, yref="paper", text=label, showarrow=False,
                               font=dict(size=10, color=color), textangle=-30, xanchor="left")
    fig.update_layout(title=f"{region_name}: {point_label} daily rainfall", yaxis_title="mm/day",
                      height=420, barmode="overlay", margin=dict(l=10, r=10, t=70, b=10),
                      legend=dict(orientation="h", y=-0.15))
    return fig


def _land_share(mask, land) -> float:
    n = int(land.sum())
    return 100 * float((mask & land).sum()) / n if n else 0.0


def usable(full, m, box) -> str | None:
    """Why this run can't show onset, or None."""
    if "tp" in catalog.missing_in_run(full):
        return f"{m.name} does not forecast rainfall, so it can't forecast onset."
    if not obs.covers(box):
        return "Onset needs observed rainfall from CHIRPS, which covers 50°S–50°N only."
    return None


def controls(region_name: str, init: pd.Timestamp):
    """Definition and search start, shared by every model on the page."""
    defn = pick_definition(region_name)
    auto_from = onset.search_start(defn, region_name, init)
    c1, c2 = st.columns([1, 3])
    search_from = pd.Timestamp(c1.date_input(
        "Search for onset from", auto_from, max_value=init.date(), key=f"onset_from_{defn.key}_{init:%Y%m%d}"))
    c2.caption(f"ⓘ {'Fixed by the definition' if defn.search_from else f'Start of the main rainy season for {region_name}'}"
               f": {auto_from:%-d %b %Y}. Observed rainfall (CHIRPS) is used from here to the day "
               f"before the forecast starts; the forecast's rainfall after that.")
    return defn, search_from


def result(path, box: Box, init, lead_days, defn, search_from):
    p = Path(path)
    return compute(path, p.stat().st_mtime, (box.lat_min, box.lat_max, box.lon_min, box.lon_max),
                   init, lead_days, defn, search_from)


def view(path, full, m, init, box: Box, region_name: str, lead_days: int, defn, search_from) -> None:
    """Onset maps, shares and rainfall series for one model's run."""
    why = usable(full, m, box)
    if why:
        st.info(why)
        return
    try:
        r = result(path, box, init, lead_days, defn, search_from)
    except (OSError, FileNotFoundError) as e:
        st.error(f"Couldn't get CHIRPS observations: {e}")
        return
    key = m.key
    init_day, fc_end, obs_last = r["init_day"], r["fc_end"], r["obs_last"]
    n_days = (fc_end - init_day).days + 1
    if obs_last is not None and obs_last < init_day - pd.Timedelta(days=1):
        st.warning(f"CHIRPS currently ends on {obs_last:%-d %b %Y}, so the days from then until the "
                   f"forecast starts have no rainfall; an onset in that gap can't be found.")
    if defn.needs_after > n_days:
        st.caption(f"ⓘ Confirming onset needs {defn.needs_after} days of rain after a wet spell starts "
                   f"(the {defn.spell_days}-day spell plus the {defn.dry_window}-day dry-spell check); "
                   f"the forecast covers {n_days}. Wet spells whose check runs past the forecast are "
                   "shown as **possible onset**.")

    f_on, o_on, land = r["f_on"], r["o_on"], r["land"]
    ens = "ensemble" in f_on.dims and f_on.sizes["ensemble"] > 1
    members = [int(e) for e in f_on.ensemble.values] if "ensemble" in f_on.dims else [0]
    pick = (st.selectbox("Forecast shown", ["chance of onset"] + [f"member {e + 1}" for e in members],
                         key=f"onset_member_{key}") if ens else "member 1")
    one = f_on.isel(ensemble=members.index(int(pick.split()[1]) - 1)) if pick.startswith("member") \
        and "ensemble" in f_on.dims else (f_on.isel(ensemble=0) if "ensemble" in f_on.dims else f_on)
    fday, fposs = _day_field(one, init_day, land)

    before = fday == 0
    during = (fday >= 1) & (one.status == onset.CONFIRMED) & land
    cols = st.columns(4)
    cols[0].metric("Onset before the forecast", f"{_land_share(before, land):.0f}%",
                   help="Share of land points where observed rain already met the definition")
    cols[1].metric("Forecast onset (confirmed)", f"{_land_share(during, land):.0f}%",
                   help="Onset during the forecast with the full dry-spell check passed")
    cols[2].metric("Possible onset", f"{_land_share(fposs, land):.0f}%",
                   help="Wet spell during the forecast; its dry-spell check runs past the data")
    if o_on is not None:
        oday, oposs = _day_field(o_on, init_day, land)
        o_during = (oday >= 1) & (oday <= n_days)
        cols[3].metric("Observed onset in these days", f"{_land_share(o_during, land):.0f}%",
                       help=f"From CHIRPS (up to {obs_last:%-d %b %Y})")
    st.caption("Shares are of land points in the plot area (the country plus a margin)"
               + (", where the local wet-spell amounts exist." if defn.spell_mm is None else "."))

    if ens and pick == "chance of onset":
        a, b = st.columns(2)
        a.pyplot(_prob_figure(f_on, init_day, land, box, n_days))
        if o_on is not None:
            b.pyplot(_maps_figure((oday.where(oday <= n_days), oposs), None, box, n_days,
                                  ["Observed onset (CHIRPS)"], init_day))
    else:
        right = (oday.where(oday <= n_days), oposs & (oday <= n_days)) if o_on is not None else None
        st.pyplot(_maps_figure((fday, fposs), right, box, n_days,
                               [f"Forecast onset: {m.name}" + (f" ({pick})" if ens else ""),
                                f"Observed onset (CHIRPS, to {obs_last:%-d %b})"], init_day))
    st.caption("Grey: onset (or a wet spell still awaiting its dry-spell check) before the forecast "
               "started. Hatched: possible onset during the forecast (its dry-spell check runs past the "
               "data). White: no onset yet.")

    st.plotly_chart(_series_figure(r, defn, search_from, region_name, "area-mean"), width="stretch",
                    key=f"onset_series_{key}")
    st.caption("Onset on the area-mean rainfall, for illustration; the maps apply the definition "
               "at every grid point.")


def movie_path(path, box: Box, defn, init, lead_days) -> Path:
    key = json.dumps([path, (box.lat_min, box.lat_max, box.lon_min, box.lon_max), asdict(defn),
                      str(pd.Timestamp(init).normalize()), lead_days], default=str)
    return MOVIES / f"onset_{hashlib.md5(key.encode()).hexdigest()[:12]}.gif"


def make_movie(path, m, box, region_name, init, lead_days, defn, search_from, progress) -> Path:
    """Render one model's onset movie (one frame per forecast day) and return the GIF."""
    gif = movie_path(path, box, defn, init, lead_days)
    if gif.exists():
        return gif
    r = result(path, box, init, lead_days, defn, search_from)
    n_days = (r["fc_end"] - r["init_day"]).days + 1
    frames = render_frames(r, m, box, region_name, n_days, progress)
    MOVIES.mkdir(parents=True, exist_ok=True)
    frames[0].save(gif, save_all=True, append_images=frames[1:], duration=900, loop=0)
    return gif


def render_frames(r, m, box, region_name, n_days, progress):
    from PIL import Image
    f_on = r["f_on"]
    one = f_on.isel(ensemble=0) if "ensemble" in f_on.dims else f_on
    init_day, land = r["init_day"], r["land"]
    fday, fposs = _day_field(one, init_day, land)
    have_obs = r["o_on"] is not None
    if have_obs:
        oday, oposs = _day_field(r["o_on"], init_day, land)
    frames = []
    for i in range(1, n_days + 1):
        date = init_day + pd.Timedelta(days=i - 1)
        left = (fday.where(fday <= i), fposs & (fday <= i))
        right = ((oday.where(oday <= i), oposs & (oday <= i))
                 if have_obs and date <= (r["obs_last"] or date) else None)
        fig = _maps_figure(left, right, box, n_days,
                           [f"Forecast ({m.name}" + (", member 1" if "ensemble" in f_on.dims
                                                      and f_on.sizes["ensemble"] > 1 else "") + ")",
                            "Observed (CHIRPS)"], init_day)
        fig.suptitle(f"{region_name}: onset by {date:%-d %b %Y} (day {i})", fontsize=12)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=90, bbox_inches="tight")
        plt.close(fig)
        frames.append(Image.open(buf).convert("RGB"))
        progress(i, n_days)
    w = max(f.width for f in frames)
    h = max(f.height for f in frames)
    return [f if f.size == (w, h) else f.resize((w, h)) for f in frames]
