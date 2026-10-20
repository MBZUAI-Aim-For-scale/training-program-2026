"""
forecast_event_movie.py - generic obs-vs-forecast event movie + event-track map,
for HEAT or PRECIPITATION cases, ANY model, ANY region.

    python forecast_event_movie.py --kind {heat,precip} --model <name> \
        --domain LON_MIN LON_MAX LAT_MIN LAT_MAX --init YYYY-MM-DD [...]

Four choices, all on the command line:
  * --kind    heat (daily-max 2 m T, degC) | precip (daily total, mm/day)
  * --model   a registry name (AIFS-ENS, AIFS-single, GraphCast, Aurora,
              NeuralGCM) or --fc-path to ANY forecast output: a directory of
              per-init stores, or a single .zarr / .zarr.zip / .nc store
  * --domain  lon_min lon_max lat_min lat_max in -180..180 (meridian-safe)
  * --init / --lead   which forecast run (snapped to the nearest store)

Output per init (one movie each; --init takes several dates):
  <name>_<model>_init<YYYYMMDD>_movie.gif (+ .mp4)
       Same layout as mali_movie.py. TOP obs map | forecast map (ens mean,
       model's NATIVE grid), animating over valid days; BOTTOM time series
       building up to the frame with the legend outside on the right: obs
       (black) vs forecast (median + 10-90 band if ensemble), plus any
       --overlay models as extra lines (nearest init within --overlay-tol,
       default 3 d).
  With --track, also <name>_<model>_init<YYYYMMDD>_track.png: a static
  event footprint + daily threshold contours + centroid track (off by
  default).

HOW "ANY MODEL" WORKS
  Per store, the forecast loader auto-detects instead of assuming AIFS:
    variable     first match from a candidate list per kind
                 (heat: 2t, 2m_temperature, t2m, ...; precip: tp,
                 total_precipitation, total_precipitation_24hr, ...)
    names        latitude/longitude -> lat/lon; any ensemble dim (number,
                 member, sample, ...) -> 'number'; singleton surface/level
                 dims dropped
    init time    'time' / 'init_time' coordinate, else the date in the name
    lead axis    the timedelta-typed dim (6-hourly OR daily), else an
                 integer-day dim (NeuralGCM style, see lead_day_offset)
    grid         0-360 -> -180..180, latitude flipped ascending, BEFORE the
                 box slice; coarse grids (> 0.5 deg) get a one-cell margin
                 so a small box is never empty
    temperature  Kelvin detected from magnitude and converted (-273.15)
    precip       each step binned to a calendar day by the MIDPOINT of its
                 accumulation period and summed; days without a full 24 h
                 of steps are dropped. This one rule is right for 6-hourly
                 and daily axes alike, and reproduces the AIFS off-by-one
                 (tp at ptd=N covers day init+N-1) with no special case.
  Things that cannot be detected safely are per-model flags in MODELS
  (and CLI overrides): precip units (m / mm / auto), cumulative-from-init,
  and the day offset of an integer lead axis. Check a new model with
      python forecast_event_movie.py --kind precip --model X --init ... --peek
  which prints the detected var, axes, step spacing, magnitudes and
  monotonicity for one init and exits (no plotting).

Run on the cluster:

    srun --partition=general --qos=protected --cpus-per-task=8 --mem=64G \
         --time=01:00:00 \
         /home/pchoudhary/miniconda3/envs/pc_env/bin/python -u \
         forecast_event_movie.py --case mali_heat

    # any model / event
    python forecast_event_movie.py --kind precip --model GraphCast \
        --domain 60 78 23 37 --init 2022-08-21 --end 2022-09-05 \
        --reduce mean --threshold 40 --name pakistan

    # a forecast output that is not in the registry
    python forecast_event_movie.py --kind heat --fc-path /path/to/inits \
        --fc-label MyModel --domain 2.5 15 4 14 --init 2025-05-30

    # init N days before the obs peak; overlay other models as lines
    python forecast_event_movie.py --case bangladesh_precip --lead 7 \
        --overlay GraphCast AIFS-single

    # several inits in one run (one movie each), like mali_movie.py
    python forecast_event_movie.py --case mali_heat \
        --init 2024-03-25 2024-03-31 2024-04-01

    --still renders only the last movie frame (fast layout check).

DATA PATHS - set them anywhere, no code edits
  Every path (obs and models) comes from, in increasing priority:
    1. the built-in defaults below (MODELS, ERA5_DIR, IMERG_DIR)
    2. a JSON config file: --config FILE, else $FEM_CONFIG, else
       forecast_event_movie.json next to this script (used if it exists)
    3. --path NAME=PATH on the command line (repeatable; NAME is a model,
       or era5 / imerg)
  Start from a template of the current defaults:
      python forecast_event_movie.py --dump-config forecast_event_movie.json
  Config format (any key you leave out keeps its default):
      {"obs":    {"era5": "/path/to/era5_dir", "imerg": "/path/to/imerg_dir"},
       "models": {"AIFS-ENS": {"path": "/new/path/forecasts_AIFS_ENS_v2"},
                  "MyModel":  {"path": "/path/to/inits",        # NEW model
                               "precip_units": "mm",
                               "vars": {"heat": ["t2m"]}}}}
  Paths may use ~ and $ENV_VARS. Check what is in effect with --list-models.

STORE NOTES (verified Oct 2026)
  * AIFS-ENS 2016+ lives in /net/monsoon/marchakitus/reforecast/
    forecasts_AIFS_ENS_v2 on a ~5th/21st/29th-of-month cadence; the old
    AIFS/v2p0/combined store stops at 2015. Some inits are *.zarr.zip
    (opened via zarr ZipStore); *.complete / *.md5 entries are skipped.
  * Aurora stores have no precipitation; NeuralGCM is ~2.8 deg with an
    integer-day lead axis and UNCONFIRMED tp units/day stamping (--peek it).
  * Daily Tmax from 6-hourly snapshots undercounts the true max on both
    sides (like-for-like); a daily-cadence model gets ONE snapshot per day
    for heat (warned at load). AIFS trains on ERA5, so ERA5-vs-AIFS is not a
    fully independent verification. Caption both.
"""

import argparse
import datetime as dt
import glob
import os
import re
import shutil
import subprocess

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as pe
from matplotlib.ticker import MaxNLocator

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from PIL import Image

# ==========================================================================
# ENGINE - from rossbypalooza/event_movie.py (make_event_movie and helpers).
# One change: each map panel is drawn on ITS OWN lat/lon, so a coarse model
# (NeuralGCM 2.8 deg) shows at native resolution next to 0.25-deg obs
# instead of being interpolated onto the obs grid.
# ==========================================================================

ENSEMBLE_DIMS = ("number", "member", "ens", "realization", "ensemble",
                 "sample")
ENS_DIM = "number"          # every ensemble dim is renamed to this on load


def _ensemble_dim(da):
    for c in ENSEMBLE_DIMS:
        if c in da.dims:
            return c
    return None


def _map_field(da):
    edim = _ensemble_dim(da)
    return da.mean(edim) if edim else da


def _alpha_faded_cmap(name, gamma=0.5):
    base = plt.get_cmap(name)(np.linspace(0, 1, 256))
    base[:, 3] = np.linspace(0.0, 1.0, 256) ** gamma
    return mcolors.ListedColormap(base)


def _map_aspect(lat_min, lat_max):
    """y/x aspect for a PlateCarree map so shapes are true at the box centre
    (1 deg lon is cos(lat) shorter than 1 deg lat; equal-degree axes stretch
    mid-latitude countries east-west)."""
    return 1.0 / np.cos(np.deg2rad(0.5 * (lat_min + lat_max)))


def _basemap_scale():
    from cartopy.io import shapereader
    for scale in ("10m", "110m"):
        try:
            shapereader.natural_earth(resolution=scale, category="physical",
                                      name="coastline")
            shapereader.natural_earth(resolution=scale, category="cultural",
                                      name="admin_0_boundary_lines_land")
            return scale
        except Exception:
            continue
    return None


def country_mask(geom, lon, lat):
    import shapely.vectorized as shpvec
    LON, LAT = np.meshgrid(np.asarray(lon), np.asarray(lat))
    return shpvec.contains(geom, LON, LAT)


def natural_earth_country(name_long):
    import cartopy.io.shapereader as shpreader
    fn = shpreader.natural_earth(resolution="10m", category="cultural",
                                 name="admin_0_countries")
    for rec in shpreader.Reader(fn).records():
        if rec.attributes.get("NAME_LONG") == name_long:
            return rec.geometry
    raise ValueError(f"geometry for {name_long!r} not found")


def _lead_for(da, vt):
    # Single-init fields carry an 'init' attr -> lead grows per frame.
    # Constant-lead fields carry only a scalar 'lead_days' -> fixed lead.
    init = da.attrs.get("init")
    if init is not None:
        return int((pd.Timestamp(vt) - pd.Timestamp(init)).days)
    if "lead_days" in da.attrs:
        try:
            return int(da.attrs["lead_days"])
        except (TypeError, ValueError):
            pass
    return None


def _load_padded(frame_files):
    """Open frame PNGs and pad them onto a common (max) canvas.

    bbox_inches='tight' can make frames differ by a pixel; both the GIF writer
    and ffmpeg need a uniform frame size, so pad every frame onto a white
    canvas of the max width/height. Returns a list of RGB PIL Images.
    """
    imgs = [Image.open(f).convert("RGB") for f in frame_files]
    W = max(im.width for im in imgs)
    H = max(im.height for im in imgs)
    # ffmpeg's yuv420p needs even dimensions
    W += W % 2
    H += H % 2
    out = []
    for im in imgs:
        if im.size == (W, H):
            out.append(im); continue
        canvas = Image.new("RGB", (W, H), (255, 255, 255))
        canvas.paste(im, (0, 0))
        out.append(canvas)
    return out


def _write_gif(imgs, path, fps):
    imgs[0].save(path, save_all=True, append_images=imgs[1:],
                 duration=int(1000 / max(fps, 1e-6)), loop=0, disposal=2)


def _write_mp4(imgs, path, fps, frames_dir):
    """Assemble an MP4 from padded frames via the system ffmpeg. Returns True
    on success, False if ffmpeg is unavailable or errors."""
    ff = shutil.which("ffmpeg")
    if ff is None:
        return False
    norm_dir = os.path.join(frames_dir, "_norm")
    os.makedirs(norm_dir, exist_ok=True)
    for f in glob.glob(os.path.join(norm_dir, "*.png")):
        os.remove(f)
    for i, im in enumerate(imgs):
        im.save(os.path.join(norm_dir, f"n_{i:04d}.png"))
    cmd = [ff, "-y", "-framerate", str(fps),
           "-i", os.path.join(norm_dir, "n_%04d.png"),
           "-pix_fmt", "yuv420p", "-c:v", "libx264", path]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
        return True
    except (subprocess.CalledProcessError, OSError):
        return False


def make_event_movie(
    fields, lines, outfile, *,
    domain, times=None, box=None,
    cmap="Blues", levels=None, vmin=0.0, vmax=80.0, n_levels=17, extend="max",
    cbar_label="Precipitation (mm/day)",
    transparent_low=True, alpha_gamma=0.5,
    threshold=None, threshold_label="Threshold", onset_from=None,
    event=None, elevation=None, elevation_levels=None, mask_geom=None,
    ylabel="Precipitation (mm/day)", ylim=None, title="",
    lead_from=None, init_label=None, map_feature_color="grey",
    cities=None,
    legend_loc="outside",
    fps=1, dpi=170, save_mp4=True, frames_dir=None, still_only=False,
    time_fmt=None,
):
    """Render an animated event map + time series to GIF (and optionally MP4).

    fields : dict {panel_title: DataArray(time[, ensemble], lat, lon)} in panel
             order. Ensemble dim is averaged for the map. Panels may be on
             different grids; each is drawn on its own lat/lon.
    lines  : list of dicts for the bottom series. Keys:
                label, data(DataArray time)  [required]
                color, marker, lw, ms, ls, zorder            [optional]
                band=(lo_DA, hi_DA), band_label              [optional spread]
                static=True  -> draw fully every frame (e.g. climatology)
             non-static series are drawn only up to the current valid time.
    domain : (lon_min, lon_max, lat_min, lat_max)
    box    : (lon0, lat0, width, height) averaging rectangle on each map.
    onset_from : label of a line; vertical marker at its first crossing of
                 `threshold`, appearing once the frame reaches that time.
    lead_from  : field label whose per-time lead is shown in the frame title.
    still_only : render only the last frame as a PNG (fast layout iteration).
    """
    labels = list(fields)
    if not labels:
        raise ValueError("no spatial fields given")
    ref = fields[labels[0]]
    if times is None:
        times = np.asarray(ref["time"].values)
    times = np.atleast_1d(np.asarray(times))
    if still_only:
        times = times[-1:]

    # frame-title format: show the hour when the step is sub-daily
    if time_fmt is None:
        if times.size > 1:
            step_h = np.min(np.abs(np.diff(times).astype("timedelta64[h]")
                                   .astype(int)))
            time_fmt = "%Y-%m-%d %HZ" if step_h < 24 else "%Y-%m-%d"
        else:
            time_fmt = "%Y-%m-%d"

    lon_min, lon_max, lat_min, lat_max = domain

    if levels is None:
        levels = np.linspace(vmin, vmax, n_levels)
    plot_cmap = _alpha_faded_cmap(cmap, alpha_gamma) if transparent_low \
        else plt.get_cmap(cmap)

    # per-panel map field, grid and (optional) country mask
    map_fields = {k: _map_field(v) for k, v in fields.items()}
    grids = {k: (np.asarray(v["lon"].values), np.asarray(v["lat"].values))
             for k, v in map_fields.items()}
    masks = {k: country_mask(mask_geom, *grids[k]) for k in labels} \
        if mask_geom is not None else {}
    scale = _basemap_scale()

    all_line_t = np.concatenate([np.asarray(l["data"]["time"].values)
                                 for l in lines])
    tmin = all_line_t.min() - np.timedelta64(12, "h")
    tmax = all_line_t.max() + np.timedelta64(12, "h")

    onset_t = None
    if onset_from is not None and threshold is not None:
        for l in lines:
            if l["label"] == onset_from:
                over = l["data"].where(l["data"] >= threshold, drop=True)
                if over.sizes.get("time", 0):
                    onset_t = over["time"].min().values
                break

    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    if frames_dir is None:
        base = os.path.splitext(os.path.basename(outfile))[0]
        frames_dir = os.path.join(os.path.dirname(os.path.abspath(outfile)) or ".",
                                  "_frames_" + base)
    os.makedirs(frames_dir, exist_ok=True)
    for f in glob.glob(os.path.join(frames_dir, "*.png")):
        os.remove(f)

    n_map = len(labels)
    frame_files = []

    for i, vt in enumerate(times):
        fig = plt.figure(figsize=(6.0 * n_map, 8), constrained_layout=True)
        gs = gridspec.GridSpec(2, n_map, figure=fig, height_ratios=[1.5, 1])
        map_axes = [fig.add_subplot(gs[0, k], projection=ccrs.PlateCarree())
                    for k in range(n_map)]
        ax_ts = fig.add_subplot(gs[1, :])

        last_mesh = None
        lead_days = None
        for k, label in enumerate(labels):
            ax = map_axes[k]
            ax.set_extent([lon_min, lon_max, lat_min, lat_max],
                          crs=ccrs.PlateCarree())
            ax.set_aspect(_map_aspect(lat_min, lat_max))

            if elevation is not None:
                lv = elevation_levels if elevation_levels is not None \
                    else np.arange(0, 4500, 500)
                ax.contour(np.asarray(elevation["lon"].values),
                           np.asarray(elevation["lat"].values),
                           elevation.values, levels=lv, cmap="Greys",
                           alpha=0.7, zorder=-1, transform=ccrs.PlateCarree())

            fld = map_fields[label]
            lon, lat = grids[label]
            try:
                slab = fld.sel(time=vt)
            except KeyError:
                slab = fld.sel(time=vt, method="nearest")
            arr = slab.values
            if label in masks:
                arr = np.where(masks[label], arr, np.nan)

            last_mesh = ax.contourf(
                lon, lat, arr, levels=levels, vmin=vmin, vmax=vmax,
                cmap=plot_cmap, extend=extend, transform=ccrs.PlateCarree())

            if scale is not None:
                ax.add_feature(cfeature.COASTLINE.with_scale(scale),
                               edgecolor=map_feature_color, lw=0.7)
                ax.add_feature(cfeature.BORDERS.with_scale(scale),
                               edgecolor=map_feature_color, lw=0.9)

            if box is not None:
                x0, y0, w, h = box
                ax.add_patch(plt.Rectangle(
                    (x0, y0), w, h, lw=1.5, edgecolor="grey", facecolor="none",
                    transform=ccrs.PlateCarree()._as_mpl_transform(ax)))

            if cities:
                for cname, clat, clon in cities:
                    if not (lon_min <= clon <= lon_max
                            and lat_min <= clat <= lat_max):
                        continue
                    ax.plot(clon, clat, marker="o", ms=6, mfc="white",
                            mec="black", mew=1.0, zorder=6,
                            transform=ccrs.PlateCarree())
                    ax.text(clon + 0.2, clat + 0.2, cname, fontsize=10,
                            zorder=6, transform=ccrs.PlateCarree(),
                            color="black", path_effects=[
                                pe.withStroke(linewidth=2.5, foreground="white")])

            ax.set_title(label, fontsize=15)
            if label == lead_from:
                lead_days = _lead_for(fields[label], vt)

        cbar = fig.colorbar(last_mesh, ax=map_axes, location="left",
                            shrink=0.92, pad=0.02, extend=extend)
        cbar.set_label(cbar_label, fontsize=13)
        cbar.ax.tick_params(labelsize=11)

        ax = ax_ts
        ax.set_xlim(tmin, tmax)
        ci = 0
        for l in lines:
            s = l["data"]
            col = l.get("color") or colours[ci % len(colours)]
            if l.get("color") is None:
                ci += 1
            static = l.get("static", False)
            s_show = s if static else s.sel(time=slice(None, vt))
            if l.get("band") is not None:
                lo, hi = l["band"]
                lo_s = lo if static else lo.sel(time=slice(None, vt))
                hi_s = hi if static else hi.sel(time=slice(None, vt))
                ax.fill_between(np.asarray(lo_s["time"].values),
                                lo_s.values, hi_s.values, color=col,
                                alpha=0.2, zorder=l.get("zorder", -2),
                                label=l.get("band_label"))
            ax.plot(np.asarray(s_show["time"].values), s_show.values,
                    label=l["label"], color=col, marker=l.get("marker", "o"),
                    lw=l.get("lw", 1.6), ms=l.get("ms", 5),
                    ls=l.get("ls", "-"), zorder=l.get("zorder", 1))

        if threshold is not None:
            # labelled so it shows up in the legend, not as text on the plot
            ax.axhline(threshold, color="k", ls="--", alpha=0.5, zorder=-3,
                       label=threshold_label)

        if onset_t is not None and vt >= onset_t:
            ax.axvline(onset_t, color="black", ls="--", zorder=-1)
            ax.text(onset_t + np.timedelta64(3, "h"), ax.get_ylim()[0],
                    " obs onset", color="black", va="bottom", fontsize=11)

        if event is not None:
            e0 = np.datetime64(event[0])
            e1 = np.datetime64(event[1]) + np.timedelta64(1, "D")
            ax.axvspan(e0, e1, color="crimson", alpha=0.08, zorder=-4)

        ax.set_ylabel(ylabel, fontsize=13)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.grid(alpha=0.3, ls=":")
        # legend outside the axes so it never sits on top of the series;
        # pass legend_loc="upper right" (etc.) to keep it inside instead.
        if legend_loc == "outside":
            ax.legend(fontsize=11, loc="upper left", bbox_to_anchor=(1.01, 1.0),
                      borderaxespad=0.0, ncol=1, frameon=True)
        else:
            ax.legend(fontsize=11, loc=legend_loc, ncol=min(len(lines), 5))

        # rotate date labels so they don't overlap on a daily axis
        for lbl in ax.get_xticklabels():
            lbl.set_rotation(30)
            lbl.set_horizontalalignment("right")

        vt_f = pd.to_datetime(vt).strftime(time_fmt)
        ttl = f"{title}  -  {vt_f}" if title else vt_f
        if lead_days is not None:
            ttl += f"   (lead {lead_days} d)"
        # Title on the top row; init label on its own row just below and
        # left-aligned, so a long centred title can't overlap it.
        fig.suptitle(ttl, y=1.09, fontsize=16)
        if init_label:
            fig.text(0.01, 1.038, init_label, fontsize=12, va="top",
                     ha="left", color="0.35")

        if still_only:
            still = os.path.splitext(outfile)[0] + "_still.png"
            fig.savefig(still, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            print(f"  wrote still {still}")
            return still

        fp = os.path.join(frames_dir, f"frame_{i:03d}.png")
        fig.savefig(fp, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        frame_files.append(fp)

    imgs = _load_padded(frame_files)
    gif = os.path.splitext(outfile)[0] + ".gif"
    _write_gif(imgs, gif, fps)
    print(f"  wrote {gif}  ({len(imgs)} frames)")
    if save_mp4:
        mp4 = os.path.splitext(outfile)[0] + ".mp4"
        if _write_mp4(imgs, mp4, fps, frames_dir):
            print(f"  wrote {mp4}")
        else:
            print("  mp4 skipped (no ffmpeg on PATH); GIF written")
    return gif


# ==========================================================================
# CASE CONFIG
# ==========================================================================

ERA5_DIR = "/net/monsoon/kylehall/ERA5/era5_arco_6h_surface"
IMERG_DIR = "/net/monsoon/aasch/IMERG"

# Same IMERG conventions as cluster_data.py (raw values already mm/12h; each
# 12h stamp labels the period ENDING at that time).
IMERG_SCALE = 1.0
IMERG_END_LABELLED = True

# --------------------------------------------------------------------------
# MODEL REGISTRY. Only what cannot be auto-detected lives here; every key
# falls back to DEFAULT_SPEC. Adding a model = adding a path (then --peek).
#   path            directory of per-init stores (any layout whose file or
#                   folder names carry the init date), or a single store
#   vars            {kind: [candidate variable names]}; [] = not available
#   precip_units    "m" (x1000), "mm", or "auto" (metres if every daily
#                   value < 1, with a warning - confirm with --peek)
#   cumulative      precip accumulates from init (differenced on load)
#   lead_day_offset integer-day lead axes only: precip step d ENDS at
#                   init + (d + offset) days
#   inst_day_offset integer-day lead axes only: instantaneous step d is
#                   VALID at init + (d + offset) days
# --------------------------------------------------------------------------
DEFAULT_SPEC = dict(
    path=None,
    vars={"heat": ["2t", "2m_temperature", "t2m", "tas", "temperature_2m",
                   "air_temperature_2m"],
          "precip": ["tp", "total_precipitation", "total_precipitation_24hr",
                     "total_precipitation_6hr", "precipitation", "pr",
                     "precip"]},
    precip_units="auto", cumulative=False,
    lead_day_offset=1, inst_day_offset=0,
)

_RF = "/net/monsoon/marchakitus/reforecast"
MODELS = {
    "AIFS-ENS":    dict(path=f"{_RF}/forecasts_AIFS_ENS_v2", precip_units="m"),
    "AIFS-single": dict(path=f"{_RF}/forecasts_AIFS_v2", precip_units="m"),
    "GraphCast":   dict(path=f"{_RF}/forecasts_graphcast_e2s",
                        precip_units="m"),
    "Aurora":      dict(path=f"{_RF}/forecasts_aurora_e2s",
                        vars={"precip": []}),          # store has no precip
    "NeuralGCM":   dict(path="/net/monsoon/reforecast/NGCM/climatology",
                        precip_units="auto",            # UNCONFIRMED: --peek
                        lead_day_offset=1),             # UNCONFIRMED: --peek
}

OVERLAY_COLORS = ["tab:blue", "tab:green", "tab:purple", "tab:brown",
                  "tab:olive", "tab:cyan"]

CONFIG_ENV = "FEM_CONFIG"
CONFIG_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "forecast_event_movie.json")
SPEC_KEYS = set(DEFAULT_SPEC)


def _expand(p):
    return os.path.expanduser(os.path.expandvars(str(p)))


def _merge_model(name, entry, source):
    """Merge one model entry over MODELS (new names are added)."""
    if not isinstance(entry, dict):
        raise SystemExit(f"{source}: model {name!r} must be an object, "
                         f"e.g. {{\"path\": \"/data/...\"}}")
    bad = set(entry) - SPEC_KEYS
    if bad:
        raise SystemExit(f"{source}: model {name!r} has unknown key(s) "
                         f"{sorted(bad)}; allowed: {sorted(SPEC_KEYS)}")
    cur = MODELS.setdefault(name, {})
    for k, v in entry.items():
        if k == "vars":
            cur.setdefault("vars", {}).update(v)
        elif k == "path":
            cur["path"] = _expand(v)
        else:
            cur[k] = v


def _set_obs(key, path):
    global ERA5_DIR, IMERG_DIR
    if key == "era5":
        ERA5_DIR = _expand(path)
    elif key == "imerg":
        IMERG_DIR = _expand(path)
    else:
        raise KeyError(key)


def apply_config(config_file=None, path_overrides=()):
    """Layer the JSON config and --path NAME=PATH overrides onto the
    built-in paths. Returns the config file used (or None)."""
    import json
    used = None
    cands = [config_file] if config_file else \
        [os.environ.get(CONFIG_ENV), CONFIG_DEFAULT]
    for c in cands:
        if not c:
            continue
        c = _expand(c)
        if not os.path.isfile(c):
            if config_file:
                raise SystemExit(f"config file not found: {c}")
            continue
        with open(c) as fh:
            try:
                conf = json.load(fh)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{c}: invalid JSON ({exc})")
        bad = set(conf) - {"obs", "models"}
        if bad:
            raise SystemExit(f"{c}: unknown top-level key(s) {sorted(bad)}; "
                             "use 'obs' and 'models'")
        for k, v in (conf.get("obs") or {}).items():
            try:
                _set_obs(k, v)
            except KeyError:
                raise SystemExit(f"{c}: obs key {k!r} must be era5 or imerg")
        for name, entry in (conf.get("models") or {}).items():
            _merge_model(name, entry, c)
        used = c
        break
    for item in path_overrides or ():
        if "=" not in item:
            raise SystemExit(f"--path {item!r}: use NAME=PATH")
        name, path = item.split("=", 1)
        if name.lower() in ("era5", "imerg"):
            _set_obs(name.lower(), path)
        else:
            _merge_model(name, {"path": path}, "--path")
    return used


def dump_config(path):
    """Write the paths/settings currently in effect as an editable config."""
    import json
    conf = {"obs": {"era5": ERA5_DIR, "imerg": IMERG_DIR},
            "models": {k: dict(v) for k, v in MODELS.items()}}
    with open(path, "w") as fh:
        json.dump(conf, fh, indent=2)
        fh.write("\n")
    print(f"wrote {path} - edit the paths, then run with --config {path} "
          f"(or name it {os.path.basename(CONFIG_DEFAULT)} next to the script)")


def list_models():
    """Print the obs/model paths in effect and whether each exists."""
    def row(name, path, note=""):
        ok = "ok " if path and os.path.exists(path) else "MISSING"
        print(f"  {name:<14} {ok}  {path}{note}")
    print("obs:")
    row("era5", ERA5_DIR)
    row("imerg", IMERG_DIR)
    print("models:")
    for name in MODELS:
        spec = model_spec(name)
        has = [k for k in ("heat", "precip") if spec["vars"].get(k)]
        row(name, spec["path"], f"   [{', '.join(has) or 'no vars'}]")

# --------------------------------------------------------------------------
# per-kind defaults (anything here can be overridden on the command line)
# --------------------------------------------------------------------------
KINDS = {
    "heat": dict(
        obs="era5", cmap="YlOrRd", extend="both", transparent_low=False,
        units="degC", cbar_label="2 m Tmax (degC)", ts_label="2 m Tmax (degC)",
        reduce="max",
        footprint="max", footprint_label="Max daily Tmax over window (degC)",
        track_cmap="cool", fc_color="orangered", threshold_pct=95,
        overlay_colors=OVERLAY_COLORS),
    "precip": dict(
        obs="imerg", cmap="YlGnBu", extend="max", transparent_low=True,
        units="mm/day", cbar_label="Precipitation (mm/day)",
        ts_label="Precip (mm/day)", reduce="mean",
        footprint="sum", footprint_label="Total precipitation over window (mm)",
        track_cmap="plasma", fc_color="dodgerblue", threshold_pct=97.5,
        # no blues: they clash with the dodgerblue forecast line
        overlay_colors=["tab:orange", "tab:green", "tab:purple", "tab:brown",
                        "tab:olive", "tab:pink"]),
}

# Ready-made cases mirroring the existing drivers. CLI flags override these.
# (Inits are the nearest AIFS-ENS stores actually on disk, see STORE NOTES.)
PRESETS = {
    "mali_heat": dict(
        kind="heat", name="mali_heat", domain=(-13.0, 5.0, 9.5, 25.5),
        model="AIFS-ENS", fc_label="AIFS", overlay=["Aurora"],
        overlay_tol=3,                        # mali_movie.py SNAP_TOL
        init="2024-03-29", end="2024-04-15",
        reduce="point", point=(14.44, -11.44), point_name="Kayes",
        threshold=44.0, vmin=28.0, vmax=48.0,
        event=("2024-04-01", "2024-04-04"),
        box=(-11.5, 11.5, 6.0, 4.0),          # WWA S. Mali study region
        cities=[("Kayes", 14.44, -11.44), ("Bamako", 12.64, -8.00)],
        title="Mali/Sahel heat, end of Ramadan 2024"),
    "nigeria_heat": dict(
        kind="heat", name="nigeria_heat", domain=(2.5, 15.0, 4.0, 14.0),
        model="AIFS-ENS", init="2025-05-29", end="2025-06-15",
        reduce="point", point=(12.00, 8.52), point_name="Kano",
        threshold=42.0, vmin=28.0, vmax=48.0,
        cities=[("Kano", 12.00, 8.52)],
        title="Nigeria heat, 2025"),
    "bangladesh_precip": dict(
        kind="precip", name="bangladesh_precip", domain=(89.5, 93.5, 21.5, 26.0),
        model="AIFS-ENS", init="2024-08-05", end="2024-08-26",
        reduce="mean", threshold=50.0, vmin=0.0, vmax=120.0,
        box=(90.9, 22.6, 1.4, 1.4),
        event=("2024-08-19", "2024-08-21"),
        cities=[("Feni", 23.02, 91.40), ("Cumilla", 23.46, 91.18)],
        title="Bangladesh eastern floods, Aug 2024"),
}


def model_spec(name=None, **overrides):
    """Registry entry merged over DEFAULT_SPEC, then non-None overrides."""
    if name is not None and name not in MODELS:
        raise SystemExit(f"unknown model {name!r}; registry has "
                         f"{', '.join(MODELS)} (or use --fc-path)")
    spec = {k: (dict(v) if isinstance(v, dict) else v)
            for k, v in DEFAULT_SPEC.items()}
    for k, v in (MODELS.get(name, {}) if name else {}).items():
        if k == "vars":
            spec["vars"].update(v)
        else:
            spec[k] = v
    for k, v in overrides.items():
        if v is None:
            continue
        if k == "vars":
            spec["vars"].update(v)
        else:
            spec[k] = v
    spec["name"] = name
    return spec


# ==========================================================================
# GRID HELPERS
# ==========================================================================

def _to_180(da):
    """0-360 -> -180..180, sorted ascending. No-op if already -180..180."""
    if float(da.lon.max()) > 180.0:
        da = da.assign_coords(lon=(((da.lon + 180) % 360) - 180)).sortby("lon")
    return da


def _grid_step(da):
    lat = np.asarray(da["lat"].values)
    return float(np.abs(np.diff(lat)).min()) if lat.size > 1 else 0.0


def _subset(da, domain, coarse_deg=0.5):
    """Standardise names/orientation and cut the domain box.

    Converts longitude to -180..180 and latitude to ascending BEFORE slicing,
    so any store convention and any box (incl. one crossing 0E) selects
    correctly. A grid coarser than `coarse_deg` gets a one-cell margin, so a
    small box on a ~3-deg model still holds a 2x2 (contourable) patch.
    Raises on an empty selection instead of plotting nothing.
    """
    ren = {k: v for k, v in (("latitude", "lat"), ("longitude", "lon"))
           if k in da.dims}
    if ren:
        da = da.rename(ren)
    da = _to_180(da)
    if da.lat.size > 1 and float(da.lat[0]) > float(da.lat[-1]):
        da = da.isel(lat=slice(None, None, -1))
    lon_min, lon_max, lat_min, lat_max = domain
    step = _grid_step(da)
    m = step if step > coarse_deg else 0.0
    da = da.sel(lat=slice(lat_min - m, lat_max + m),
                lon=slice(lon_min - m, lon_max + m))
    if da.sizes.get("lat", 0) == 0 or da.sizes.get("lon", 0) == 0:
        raise RuntimeError(f"empty selection for domain {domain} - check the "
                           "box is (lon_min, lon_max, lat_min, lat_max) in "
                           "-180..180")
    # maps plot .values as (lat, lon); IMERG and some stores are (lon, lat)
    return da.transpose(..., "lat", "lon")


def _pad(start, end, days=1):
    return (np.datetime64(pd.Timestamp(start) - pd.Timedelta(days=days)),
            np.datetime64(pd.Timestamp(end) + pd.Timedelta(days=days)))


# ==========================================================================
# OBSERVATIONS
# ==========================================================================

def _era5(var, start, end, domain):
    years = range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1)
    paths = [os.path.join(ERA5_DIR, f"era5_arco_6h_surface_{y}.zarr")
             for y in years]
    ds = xr.open_mfdataset(paths, engine="zarr", combine="by_coords",
                           chunks={})
    lo, hi = _pad(start, end)
    return _subset(ds[var], domain).sel(time=slice(lo, hi)).load()


def load_obs_heat(domain, start, end):
    """ERA5 daily max of 6-hourly 2 m temperature (degC)."""
    da = _era5("2m_temperature", start, end, domain) - 273.15
    daily = da.resample(time="1D").max().sel(time=slice(start, end))
    daily.attrs["units"] = "degC"
    return daily


def load_obs_precip_era5(domain, start, end):
    """ERA5 daily precipitation (mm/day). tp is the 6h accumulation ending at
    each stamp, so stamps shift back 6h before the daily sum."""
    da = _era5("total_precipitation", start, end, domain)
    da = da.assign_coords(time=da.time - pd.Timedelta("6h"))
    daily = (da.resample(time="1D").sum() * 1000.0).sel(time=slice(start, end))
    daily.attrs["units"] = "mm/day"
    return daily


def load_obs_precip_imerg(domain, start, end):
    """IMERG daily precipitation (mm/day) from the 12-hourly yearly files."""
    years = range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1)
    lo, hi = _pad(start, end)
    parts = []
    for y in years:
        da = xr.open_dataset(os.path.join(IMERG_DIR, f"{y}.nc"),
                             chunks={})["precipitation"]
        parts.append(_subset(da, domain).sel(time=slice(lo, hi)))
    da = (xr.concat(parts, "time") * IMERG_SCALE).load()
    if IMERG_END_LABELLED:
        da = da.assign_coords(time=da.time - pd.Timedelta("12h"))
    daily = da.resample(time="1D").sum().sel(time=slice(start, end))
    daily.attrs["units"] = "mm/day"
    return daily


def load_obs(kind, source, domain, start, end):
    if kind == "heat":
        return load_obs_heat(domain, start, end), "ERA5 (obs)"
    if source == "era5":
        return load_obs_precip_era5(domain, start, end), "ERA5 (obs)"
    return load_obs_precip_imerg(domain, start, end), "IMERG (obs)"


# ==========================================================================
# FORECASTS - any model
# ==========================================================================

_STORE_RE = re.compile(r"\.(zarr|zarr\.zip|nc|nc4)$")
_DATE_RE = re.compile(r"(\d{4})-?(\d{2})-?(\d{2})")
_INIT_COORDS = ("time", "init_time", "initialization_time", "reference_time",
                "forecast_reference_time")


def _is_store(path):
    return bool(_STORE_RE.search(os.path.basename(path.rstrip("/"))))


def _name_date(path):
    m = _DATE_RE.search(os.path.basename(path.rstrip("/")))
    if not m:
        return None
    try:
        return dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except ValueError:
        return None


def open_store(path):
    if path.endswith(".zip"):
        import zarr
        return xr.open_zarr(zarr.storage.ZipStore(path, mode="r"), chunks={})
    if path.endswith((".nc", ".nc4")):
        return xr.open_dataset(path, chunks={})
    return xr.open_zarr(path, chunks={})


def init_index(fc_path):
    """{init date: store path} for one model.

    fc_path may be a single store, or a directory holding per-init stores
    directly (init_20240325T00.zarr[.zip], ...) or one level down
    (NeuralGCM: climatology/2024/2024-03-25T00.zarr). Only *.zarr,
    *.zarr.zip and *.nc entries count, so *.complete / *.md5 markers are
    skipped; zarr stores are never descended into (slow on NFS).
    """
    if _is_store(fc_path):
        d = _name_date(fc_path)
        if d is None:
            ds = open_store(fc_path)
            for c in _INIT_COORDS:
                if c in ds.coords:
                    d = pd.Timestamp(np.asarray(ds[c].values).ravel()[0]).date()
                    break
        if d is None:
            raise SystemExit(f"cannot tell the init date of {fc_path}")
        return {d: fc_path}
    idx = {}
    for p in glob.glob(os.path.join(fc_path, "*")):
        if _is_store(p):
            cands = [p]
        elif os.path.isdir(p):
            cands = [q for q in glob.glob(os.path.join(p, "*")) if _is_store(q)]
        else:
            continue
        for q in cands:
            d = _name_date(q)
            if d is not None:
                idx[d] = q
    return idx


def pick_init(idx, target, tol):
    """Nearest init to `target` within tol days (earlier wins ties).
    Raises LookupError naming the nearest available inits."""
    target = pd.Timestamp(target).date()
    best = min(idx, key=lambda d: (abs((d - target).days), d > target),
               default=None)
    if best is None or abs((best - target).days) > tol:
        near = sorted(idx, key=lambda d: abs((d - target).days))[:4]
        raise LookupError(f"no init within {tol}d of {target}; nearest: "
                          + (", ".join(str(d) for d in sorted(near)) or "none"))
    return best, idx[best]


def _pick_var(ds, spec, kind, path):
    cands = spec["vars"].get(kind, [])
    for v in cands:
        if v in ds.data_vars:
            return v
    raise KeyError(f"no {kind} variable in {os.path.basename(path)} "
                   f"(tried {cands or 'nothing - model has none'}); store has "
                   f"{list(ds.data_vars)}. Pass --fc-var NAME.")


def _standardize(da):
    """lat/lon names, ensemble dim -> ENS_DIM, drop singleton level dims."""
    ren = {k: v for k, v in (("latitude", "lat"), ("longitude", "lon"))
           if k in da.dims}
    edim = _ensemble_dim(da)
    if edim and edim != ENS_DIM:
        ren[edim] = ENS_DIM
    if ren:
        da = da.rename(ren)
    for extra in ("surface", "level", "height", "heightAboveGround",
                  "isobaricInhPa", "step_type"):
        if extra in da.dims and da.sizes[extra] == 1:
            da = da.isel({extra: 0}, drop=True)
    return da


def _init_time(da, path):
    for c in _INIT_COORDS:
        if c in da.coords:
            v = np.asarray(da[c].values).ravel()
            if v.size and np.issubdtype(v.dtype, np.datetime64):
                return pd.Timestamp(v[0])
    d = _name_date(path)
    if d is None:
        raise RuntimeError(f"cannot find the init time in {path}")
    return pd.Timestamp(d)


def _lead_axis(da):
    """Lead dim: a timedelta-typed one if any, else the one remaining
    non-spatial / non-ensemble dim (e.g. NeuralGCM's integer days)."""
    cands = [d for d in da.dims if d not in ("lat", "lon", ENS_DIM)]
    td = [d for d in cands
          if np.issubdtype(np.asarray(da[d].values).dtype, np.timedelta64)]
    if td:
        return td[0]
    if len(cands) == 1:
        return cands[0]
    raise RuntimeError(f"cannot identify the lead axis among {cands}")


def open_forecast(path, spec, kind):
    """Open one init, return (da, init_t, var, lead_axis) with da on a
    'time' axis = step VALID time (heat) / accumulation END time (precip),
    lazily, still on the native grid and orientation."""
    ds = open_store(path)
    var = _pick_var(ds, spec, kind, path)
    da = _standardize(ds[var])
    init_t = _init_time(da, path)
    for c in _INIT_COORDS:                      # drop the length-1 init dim
        if c in da.dims:
            if da.sizes[c] != 1:
                raise RuntimeError(f"{c!r} has {da.sizes[c]} inits in one "
                                   f"store ({path}); expected 1")
            da = da.isel({c: 0}, drop=True)
        elif c in da.coords:
            da = da.drop_vars(c)
    axis = _lead_axis(da)
    vals = np.asarray(da[axis].values)
    if np.issubdtype(vals.dtype, np.timedelta64):
        off = vals.astype("timedelta64[ns]")
    else:                                       # integer days
        k = spec["inst_day_offset"] if kind == "heat" else spec["lead_day_offset"]
        off = ((vals.astype("int64") + k) * np.timedelta64(24, "h")) \
            .astype("timedelta64[ns]")
    da = da.assign_coords({axis: np.datetime64(init_t, "ns") + off}) \
           .rename({axis: "time"})
    return da, init_t, var, axis


def load_fc_heat(path, spec, domain, start, end):
    """Daily max of the instantaneous 2 m T steps (degC),
    (number?, time, lat, lon) on the model's native grid."""
    da, init_t, var, _ = open_forecast(path, spec, "heat")
    lo, hi = _pad(start, end)
    da = _subset(da, domain).sel(time=slice(lo, hi)).load()
    if da.sizes.get("time", 0) == 0:
        raise RuntimeError(f"{var}: no steps in {start}..{end} for init "
                           f"{init_t.date()}")
    if float(da.mean()) > 150.0:                # Kelvin
        da = da - 273.15
    n_day = pd.Series(1, index=pd.to_datetime(da.time.values).floor("D")) \
        .groupby(level=0).size().max()
    if n_day < 4:
        print(f"  WARNING {spec.get('name') or 'forecast'}: {n_day} {var} "
              "step(s) per day - daily max is a single snapshot, not "
              "comparable to the 6-hourly obs max")
    daily = da.resample(time="1D").max().sel(time=slice(start, end))
    daily.attrs["units"] = "degC"
    return daily, init_t


def load_fc_precip(path, spec, domain, start, end):
    """Daily precipitation (mm/day), (number?, time, lat, lon), native grid.

    Each step is an accumulation over (previous end, this end]; it is
    assigned to the calendar day of its MIDPOINT and summed. Days not covered
    by a full 24 h of steps are dropped. Right for 6-hourly and daily axes,
    and for AIFS daily tp (ends at init+N) it lands on day init+N-1 - the
    known off-by-one - without a model-specific shift.
    """
    da, init_t, var, _ = open_forecast(path, spec, "precip")
    ends = np.asarray(da["time"].values).astype("datetime64[ns]")
    prev = np.concatenate([[np.datetime64(init_t, "ns")], ends[:-1]])
    dur = ends - prev
    if spec["cumulative"]:                      # running total since init
        da = xr.concat([da.isel(time=[0]), da.diff("time")], "time")
    mid = ends - dur / 2
    s0 = np.datetime64(pd.Timestamp(start), "ns")
    s1 = np.datetime64(pd.Timestamp(end) + pd.Timedelta("1D"), "ns")
    keep = np.nonzero((mid >= s0) & (mid < s1))[0]
    if keep.size == 0:
        raise RuntimeError(f"{var}: no steps in {start}..{end} for init "
                           f"{init_t.date()}")
    da = _subset(da.isel(time=keep), domain).load()

    day = mid[keep].astype("datetime64[D]").astype("datetime64[ns]")
    cover = pd.Series(dur[keep] / np.timedelta64(1, "h"), index=day) \
        .groupby(level=0).sum()
    full = cover.index[cover >= 23.99].values
    daily = da.assign_coords(time=day).groupby("time").sum()
    daily = daily.sel(time=full)
    if daily.sizes["time"] == 0:
        raise RuntimeError(f"{var}: no complete days in {start}..{end}")

    units = spec["precip_units"]
    if units == "auto":
        units = "m" if float(daily.max()) < 1.0 else "mm"
        print(f"  {spec.get('name') or 'forecast'} {var}: precip units "
              f"GUESSED as {units!r} from magnitude - confirm with --peek "
              "and set precip_units / --fc-units")
    if units == "m":
        daily = daily * 1000.0
    daily.attrs["units"] = "mm/day"
    return daily, init_t


def load_fc(kind, path, spec, domain, start, end):
    f = load_fc_heat if kind == "heat" else load_fc_precip
    return f(path, spec, domain, start, end)


def peek(path, spec, kind, domain, n=12):
    """Print what the loader will detect for one init, then stop."""
    print(f"\n=== peek: {spec.get('name') or 'forecast'}  [{kind}] ===")
    print(f"  store : {path}")
    ds = open_store(path)
    print(f"  vars  : {list(ds.data_vars)}")
    da, init_t, var, axis = open_forecast(path, spec, kind)
    print(f"  using : {var!r}   init {init_t}   lead axis {axis!r} "
          f"(dtype {ds[var][axis].dtype if axis in ds[var].dims else '?'})")
    hrs = (pd.to_datetime(da.time.values) - init_t) / pd.Timedelta("1h")
    sp = float(np.median(np.diff(hrs))) if hrs.size > 1 else float("nan")
    print(f"  steps : {hrs.size}, spacing ~{sp:g} h, first ends/valid at "
          f"+{hrs[0]:g} h, last +{hrs[-1]:g} h")
    sub = _subset(da, domain)
    print(f"  grid  : {_grid_step(sub):g} deg, box {sub.sizes['lat']}x"
          f"{sub.sizes['lon']} cells")
    if ENS_DIM in sub.dims:
        print(f"  ens   : {sub.sizes[ENS_DIM]} members")
        sub = sub.mean(ENS_DIM)
    vals = np.asarray(sub.isel(time=slice(0, n)).mean(("lat", "lon"))
                      .load().values).ravel()
    for h, v in zip(hrs[:n], vals):
        print(f"    +{h:>6.0f} h   box-mean = {v:.6g}")
    if kind == "heat":
        print(f"  -> {'Kelvin (converted)' if np.nanmean(vals) > 150 else 'degC'}")
    else:
        mono = bool(np.all(np.diff(vals[np.isfinite(vals)]) >= 0))
        mx = float(np.nanmax(vals))
        print(f"  -> monotonic: {mono} (cumulative={mono}?)   "
              f"max {mx:.3g} (per step) -> "
              f"{'looks like metres' if mx < 0.2 else 'looks like mm'}")
        if not np.issubdtype(np.asarray(ds[var][axis].values).dtype,
                             np.timedelta64):
            print(f"  -> integer lead axis: step d ends at init + (d + "
                  f"{spec['lead_day_offset']}) d. If the movie is 1 day "
                  "early vs obs, raise lead_day_offset; late, lower it.")


# ==========================================================================
# REDUCTIONS / STATS
# ==========================================================================

def reduce_series(da, method, point=None):
    """(.., time, lat, lon) -> (.., time). Ensemble dim is kept."""
    if method == "point":
        return da.sel(lat=point[0], lon=point[1], method="nearest")
    if method == "mean":
        return da.weighted(np.cos(np.deg2rad(da.lat))).mean(("lat", "lon"))
    if method == "max":
        return da.max(("lat", "lon"))
    if method in ("p95", "p99"):
        return da.quantile(0.95 if method == "p95" else 0.99,
                           dim=("lat", "lon")).drop_vars("quantile")
    raise ValueError(method)


def forecast_line(fc_s, label, color):
    if ENS_DIM in fc_s.dims:
        lo = fc_s.quantile(0.10, ENS_DIM).drop_vars("quantile")
        hi = fc_s.quantile(0.90, ENS_DIM).drop_vars("quantile")
        return {"label": f"{label} (median)", "data": fc_s.median(ENS_DIM),
                "color": color, "zorder": 3, "band": (lo, hi),
                "band_label": f"{label} 10-90%"}
    return {"label": label, "data": fc_s, "color": color, "zorder": 3}


def exceedance_days(series, threshold):
    """(first, last) day the series is at/above threshold, or None."""
    over = series.where(series >= threshold, drop=True)
    if not over.sizes.get("time", 0):
        return None
    t = pd.to_datetime(over["time"].values)
    return str(t.min().date()), str(t.max().date())


def _largest_object(w):
    """Keep only the 8-connected exceedance object with the largest total
    weight (small flood fill; pc_env has no scipy and the grids are tiny)."""
    lab = np.zeros(w.shape, dtype=int)
    best, best_sum, n = 0, 0.0, 0
    ny, nx = w.shape
    for j0, i0 in zip(*np.nonzero(w > 0)):
        if lab[j0, i0]:
            continue
        n += 1
        lab[j0, i0] = n
        stack, tot = [(j0, i0)], 0.0
        while stack:
            j, i = stack.pop()
            tot += w[j, i]
            for dj in (-1, 0, 1):
                for di in (-1, 0, 1):
                    jj, ii = j + dj, i + di
                    if 0 <= jj < ny and 0 <= ii < nx and w[jj, ii] > 0 \
                            and not lab[jj, ii]:
                        lab[jj, ii] = n
                        stack.append((jj, ii))
        if tot > best_sum:
            best, best_sum = n, tot
    return np.where(lab == best, w, 0.0)


def centroid_track(field, threshold):
    """Daily centre of the main exceedance object. Each day, take the largest
    contiguous region above threshold and its centroid weighted by the amount
    above threshold (and cos-lat), so scattered cells elsewhere in the domain
    do not drag the track around. field (time, lat, lon) -> list of
    (time, lat, lon); days with no exceedance are skipped."""
    lat = field["lat"].values
    lon = field["lon"].values
    LON, LAT = np.meshgrid(lon, lat)
    coslat = np.cos(np.deg2rad(LAT))
    out = []
    for t in field["time"].values:
        x = np.asarray(field.sel(time=t).values)
        w = np.where(np.isfinite(x), np.clip(x - threshold, 0, None), 0) * coslat
        w = _largest_object(w)
        s = w.sum()
        if s > 0:
            out.append((t, (w * LAT).sum() / s, (w * LON).sum() / s))
    return out


# ==========================================================================
# EVENT TRACK / FOOTPRINT MAP
# ==========================================================================

def make_track_map(panels, outfile, *, domain, threshold, kind_cfg, units,
                   title="", cities=None, box=None, vmin=None, vmax=None,
                   dpi=170):
    """Static obs | forecast map of the event moving across the region.

    panels : {label: DataArray(time[, number], lat, lon)}; the ensemble is
             averaged for shading/contours, and each member's centroid track
             is drawn thinly behind the ens-mean track. Panels may be on
             different grids.
    """
    labels = list(panels)
    times = pd.to_datetime(panels[labels[0]]["time"].values)
    ndays = len(times)
    tcmap = plt.get_cmap(kind_cfg["track_cmap"], ndays)
    tnorm = mcolors.BoundaryNorm(np.arange(ndays + 1) - 0.5, ndays)
    day_index = {np.datetime64(t, "ns"): i for i, t in enumerate(times)}

    stat = kind_cfg["footprint"]
    foot = {k: getattr(v.mean(ENS_DIM) if ENS_DIM in v.dims else v, stat)("time")
            for k, v in panels.items()}
    if stat == "sum":
        flo = 0.0
        fhi = max(float(np.nanpercentile(f.values, 99)) for f in foot.values())
    else:
        flo = vmin if vmin is not None else \
            min(float(np.nanpercentile(f.values, 1)) for f in foot.values())
        fhi = vmax if vmax is not None else \
            max(float(np.nanmax(f.values)) for f in foot.values())
    flevels = MaxNLocator(nbins=12).tick_values(flo, fhi)

    lon_min, lon_max, lat_min, lat_max = domain
    scale = _basemap_scale()
    pc = ccrs.PlateCarree()
    min_sep = 0.06 * max(lon_max - lon_min, lat_max - lat_min)

    # explicit grid (not constrained_layout): cartopy gridline labels plus two
    # colorbars otherwise push the first panel off the canvas
    n = len(labels)
    aspect = min(max((lat_max - lat_min) / max(lon_max - lon_min, 1e-6)
                     * _map_aspect(lat_min, lat_max), 0.5), 1.4)
    fig = plt.figure(figsize=(6.0 * n + 1.5, 6.0 * aspect + 1.8))
    gs = fig.add_gridspec(2, n + 1, width_ratios=[1] * n + [0.04],
                          height_ratios=[1, 0.045], wspace=0.18, hspace=0.14,
                          left=0.06, right=0.90, top=0.86, bottom=0.09)
    axes = [fig.add_subplot(gs[0, k], projection=pc) for k in range(n)]

    for ax, label in zip(axes, labels):
        da = panels[label]
        mean = da.mean(ENS_DIM) if ENS_DIM in da.dims else da
        lon, lat = mean["lon"].values, mean["lat"].values
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=pc)
        ax.set_aspect(_map_aspect(lat_min, lat_max))

        shade = ax.contourf(lon, lat, foot[label].values, levels=flevels,
                            cmap=kind_cfg["cmap"], extend="both", alpha=0.55,
                            transform=pc)

        # daily threshold contour, coloured by date
        for t in mean["time"].values:
            x = mean.sel(time=t).values
            if np.nanmax(x) >= threshold:
                ax.contour(lon, lat, x, levels=[threshold], linewidths=1.3,
                           colors=[tcmap(day_index[np.datetime64(t, "ns")])],
                           transform=pc, zorder=3)

        # ensemble member centroids (small, faint, coloured by day), then the
        # ens-mean track (bold). Member points rather than lines: noisy
        # members (esp. precip) otherwise turn into unreadable spaghetti.
        if ENS_DIM in da.dims:
            pts = [p for m in da[ENS_DIM].values
                   for p in centroid_track(da.sel({ENS_DIM: m}), threshold)]
            if pts:
                ax.scatter([p[2] for p in pts], [p[1] for p in pts],
                           c=[day_index[np.datetime64(p[0], "ns")] for p in pts],
                           cmap=tcmap, norm=tnorm, s=9, alpha=0.45,
                           linewidths=0, transform=pc, zorder=4)
        tr = centroid_track(mean, threshold)
        if tr:
            ax.plot([p[2] for p in tr], [p[1] for p in tr], color="black",
                    lw=2.0, transform=pc, zorder=5)
            ax.scatter([p[2] for p in tr], [p[1] for p in tr],
                       c=[day_index[np.datetime64(p[0], "ns")] for p in tr],
                       cmap=tcmap, norm=tnorm, s=55, edgecolors="black",
                       linewidths=0.8, transform=pc, zorder=6)
            # date labels, skipping any that would sit on top of one already
            # placed (first and last day get priority)
            placed = []
            order = [0, len(tr) - 1] + list(range(1, len(tr) - 1))
            for j in dict.fromkeys(order):
                t, y, x = tr[j]
                if any(np.hypot(x - px, y - py) < min_sep for px, py in placed):
                    continue
                placed.append((x, y))
                ax.text(x, y, " " + pd.Timestamp(t).strftime("%m-%d"),
                        fontsize=9, transform=pc, zorder=7, path_effects=[
                            pe.withStroke(linewidth=2.5, foreground="white")])

        if scale is not None:
            ax.add_feature(cfeature.COASTLINE.with_scale(scale),
                           edgecolor="black", lw=0.7)
            ax.add_feature(cfeature.BORDERS.with_scale(scale),
                           edgecolor="black", lw=0.8)
        if box is not None:
            x0, y0, w, h = box
            ax.add_patch(plt.Rectangle((x0, y0), w, h, lw=1.5, edgecolor="grey",
                                       facecolor="none", transform=pc))
        for cname, clat, clon in cities or []:
            ax.plot(clon, clat, marker="^", ms=7, mfc="white", mec="black",
                    transform=pc, zorder=8)
            ax.text(clon + 0.15, clat - 0.15, cname, fontsize=9, va="top",
                    transform=pc, zorder=8, path_effects=[
                        pe.withStroke(linewidth=2.5, foreground="white")])
        gl = ax.gridlines(draw_labels=True, lw=0.3, alpha=0.4)
        gl.top_labels = gl.right_labels = False
        # plain text, not set_title: cartopy re-positions titles around the
        # gridline labels and pushes them off the figure
        ax.text(0.5, 1.02, label, transform=ax.transAxes, ha="center",
                va="bottom", fontsize=13)

    cb = fig.colorbar(shade, cax=fig.add_subplot(gs[1, :n]),
                      orientation="horizontal")
    cb.set_label(kind_cfg["footprint_label"])

    sm = plt.cm.ScalarMappable(cmap=tcmap, norm=tnorm)
    cd = fig.colorbar(sm, cax=fig.add_subplot(gs[0, n]), orientation="vertical")
    tick_step = max(1, ndays // 10)
    ticks = np.arange(0, ndays, tick_step)
    cd.set_ticks(ticks)
    cd.set_ticklabels([times[i].strftime("%m-%d") for i in ticks])
    cd.set_label(f"Valid day  (contour = {threshold:g} {units}; big dots = "
                 "centre of main exceedance region,\nsmall dots = ensemble "
                 "members)")

    span = f"{times[0]:%Y-%m-%d} to {times[-1]:%Y-%m-%d}"
    fig.suptitle(f"{title} - event footprint and track, {span}" if title
                 else f"Event footprint and track, {span}", fontsize=15)
    # no bbox_inches="tight": it drops the cartopy gridline labels and panel
    # titles here; the gridspec margins above leave room for them instead
    fig.savefig(outfile, dpi=dpi, facecolor="white")
    plt.close(fig)
    print(f"  wrote {outfile}")


# ==========================================================================
# DRIVER
# ==========================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--case", choices=sorted(PRESETS),
                   help="preset case; any other flag overrides it")
    p.add_argument("--kind", choices=sorted(KINDS))
    p.add_argument("--name", help="output filename prefix")
    p.add_argument("--title")
    p.add_argument("--domain", type=float, nargs=4,
                   metavar=("LON_MIN", "LON_MAX", "LAT_MIN", "LAT_MAX"))

    g = p.add_argument_group("forecast model")
    g.add_argument("--model",
                   help="model name from the registry or your config "
                        "(default AIFS-ENS unless --fc-path); see --list-models")
    g.add_argument("--fc-path", "--fc-dir", dest="fc_path",
                   help="forecast output: dir of per-init stores or a single "
                        ".zarr/.zarr.zip/.nc store (overrides the model's path)")
    g.add_argument("--fc-label", help="model name shown on plots/filenames")
    g.add_argument("--fc-var", help="variable name in the forecast store")
    g.add_argument("--fc-units", dest="precip_units",
                   choices=["m", "mm", "auto"], help="precip units in the store")
    g.add_argument("--fc-cumulative", dest="cumulative", action="store_true",
                   default=None, help="precip accumulates from init")
    g.add_argument("--lead-day-offset", type=int,
                   help="integer-day lead axes: precip step d ends at "
                        "init + (d + offset) days")
    g.add_argument("--overlay", nargs="*",
                   help="extra registry models drawn as time-series lines "
                        "(nearest init); --overlay with no names clears a "
                        "preset's overlays")
    g.add_argument("--peek", action="store_true",
                   help="print the detected forecast conventions and exit")

    p.add_argument("--init", nargs="+",
                   help="forecast init date(s), one movie each (each snapped "
                        "to the nearest store; duplicates after snapping are "
                        "rendered once)")
    p.add_argument("--lead", type=int,
                   help="instead of --init: init this many days before --peak")
    p.add_argument("--peak", help="event peak day for --lead "
                                  "(default: obs peak of the reduced series)")
    p.add_argument("--snap-tol", type=int, default=7,
                   help="max days between requested and available init")
    p.add_argument("--overlay-tol", type=int, default=3,
                   help="max days between the movie init and an --overlay "
                        "model's init (default 3, as in mali_movie.py)")
    p.add_argument("--start", help="time-series start (default: the init)")
    p.add_argument("--end", help="window end (default init + 15 d)")
    p.add_argument("--obs", choices=["imerg", "era5"],
                   help="precip obs source (heat always uses ERA5)")
    p.add_argument("--reduce", choices=["mean", "max", "p95", "p99", "point"])
    p.add_argument("--point", type=float, nargs=2, metavar=("LAT", "LON"))
    p.add_argument("--point-name", default=None)
    p.add_argument("--threshold", type=float,
                   help="event threshold (default: a high percentile of obs)")
    p.add_argument("--vmin", type=float)
    p.add_argument("--vmax", type=float)
    p.add_argument("--box", type=float, nargs=4,
                   metavar=("LON0", "LAT0", "WIDTH", "HEIGHT"))
    p.add_argument("--mask-country", help="Natural Earth NAME_LONG to clip maps to")
    p.add_argument("--outdir", default=".")
    p.add_argument("--fps", type=float, default=1)
    p.add_argument("--no-mp4", action="store_true")
    p.add_argument("--still", action="store_true",
                   help="render only the last movie frame (layout check)")
    p.add_argument("--show-threshold", action="store_true",
                   help="draw the threshold line + obs-onset marker on the "
                        "movie time series (off by default, as in mali_movie)")
    p.add_argument("--skip-movie", action="store_true")
    p.add_argument("--track", action="store_true",
                   help="also write the static event footprint/track map "
                        "(off by default, so output matches mali_movie.py)")
    p.add_argument("--skip-track", action="store_true",
                   help=argparse.SUPPRESS)          # kept for old commands
    p.add_argument("--track-window", nargs=2, metavar=("START", "END"),
                   help="days shown on the track map (default: event window "
                        "+/- 2 d, else the whole movie window)")
    g = p.add_argument_group("data paths")
    g.add_argument("--config", help="JSON file of obs/model paths "
                   f"(default ${CONFIG_ENV}, else {os.path.basename(CONFIG_DEFAULT)} "
                   "next to the script)")
    g.add_argument("--path", action="append", metavar="NAME=PATH",
                   help="override one path: a model name, era5 or imerg "
                        "(repeatable; also adds a new model)")
    g.add_argument("--dump-config", metavar="FILE",
                   help="write the paths in effect to FILE as a template, exit")
    g.add_argument("--list-models", action="store_true",
                   help="print the obs/model paths in effect and exit")
    args = p.parse_args()

    used = apply_config(args.config, args.path)
    if used:
        print(f"config: {used}")
    if args.dump_config:
        dump_config(args.dump_config)
        raise SystemExit(0)
    if args.list_models:
        list_models()
        raise SystemExit(0)

    cfg = dict(PRESETS.get(args.case, {}))
    for k, v in vars(args).items():
        if v is not None and k not in ("case", "config", "path",
                                        "dump_config", "list_models"):
            cfg[k] = v
    if args.fc_path and not args.model and not args.case:
        cfg.pop("model", None)         # custom output: generic conventions
    elif "model" not in cfg and not cfg.get("fc_path"):
        cfg["model"] = "AIFS-ENS"
    if args.lead is not None and args.init is None:
        cfg.pop("init", None)          # --lead overrides the preset's init
    for req in ("kind", "domain"):
        if req not in cfg:
            p.error(f"--{req} is required (or use --case)")
    if "init" not in cfg and "lead" not in cfg:
        p.error("give --init or --lead")
    if cfg.get("reduce") == "point" and "point" not in cfg:
        p.error("--reduce point needs --point LAT LON")
    if "point" in cfg and "reduce" not in cfg:
        cfg["reduce"] = "point"
    cfg.setdefault("name", f"{cfg['kind']}_event")
    cfg.setdefault("title", "")
    return cfg


def _rnd(x, step, up):
    return float((np.ceil if up else np.floor)(x / step) * step)


def _slug(s):
    return re.sub(r"[^A-Za-z0-9]+", "", s) or "fc"


def main():
    cfg = parse_args()
    inits = cfg.get("init")
    if inits is None:                       # --lead: one movie
        run(cfg, set())
        return
    if isinstance(inits, str):
        inits = [inits]
    done = set()
    for one in inits:
        print(f"\n=== movie for init {one} ===")
        try:
            run(dict(cfg, init=one), done)
        except SystemExit as exc:
            if len(inits) == 1:
                raise
            print(f"  init {one} failed: {exc}")


def run(cfg, done):
    """Render the outputs for one init. `done` holds snapped init dates
    already rendered in this call, so two requests that snap to the same
    store do not write the same movie twice."""
    kind = cfg["kind"]
    K = dict(KINDS[kind])
    domain = tuple(cfg["domain"])
    reduce = cfg.get("reduce", K["reduce"])
    point = tuple(cfg["point"]) if cfg.get("point") else None
    obs_src = cfg.get("obs", K["obs"])

    # --- which model -----------------------------------------------------
    spec = model_spec(
        cfg.get("model"), path=cfg.get("fc_path"),
        precip_units=cfg.get("precip_units"), cumulative=cfg.get("cumulative"),
        lead_day_offset=cfg.get("lead_day_offset"),
        vars={kind: [cfg["fc_var"]]} if cfg.get("fc_var") else None)
    if not spec["path"]:
        raise SystemExit("no forecast path: give --model or --fc-path")
    if not spec["vars"].get(kind):
        raise SystemExit(f"{spec['name']} has no {kind} variable in its "
                         "registry entry (Aurora stores no precipitation); "
                         "pass --fc-var if the store does have one")
    model_label = cfg.get("fc_label") or spec["name"] or "Forecast"

    idx = init_index(spec["path"])
    if not idx:
        raise SystemExit(f"no init stores found in {spec['path']}")

    # --- choose the init -------------------------------------------------
    if "init" in cfg:
        target = pd.Timestamp(cfg["init"])
    else:
        peak = cfg.get("peak")
        if peak is None:
            if "end" not in cfg:
                raise SystemExit("--lead without --peak needs --end (and "
                                 "optionally --start) to find the obs peak")
            s0 = cfg.get("start") or pd.Timestamp(cfg["end"]) - pd.Timedelta("30D")
            obs_probe, _ = load_obs(kind, obs_src, domain, s0, cfg["end"])
            peak = reduce_series(obs_probe, reduce, point).idxmax("time").values
            print(f"obs peak ({reduce}): {str(peak)[:10]}")
        target = pd.Timestamp(peak) - pd.Timedelta(days=cfg["lead"])
    try:
        init_d, fc_path = pick_init(idx, target, cfg["snap_tol"])
    except LookupError as exc:
        raise SystemExit(f"{model_label}: {exc}")
    init = pd.Timestamp(init_d)
    if init_d != target.date():
        print(f"requested init {target.date()} -> using nearest store {init_d}")
    if init_d in done:
        print(f"  init {init_d} already rendered - skipping")
        return
    done.add(init_d)
    print(f"forecast ({model_label}): {fc_path}")

    if cfg.get("peek"):
        peek(fc_path, spec, kind, domain)
        return

    start = pd.Timestamp(cfg.get("start") or init)
    end = pd.Timestamp(cfg.get("end") or init + pd.Timedelta("15D"))
    start_s, end_s = str(start.date()), str(end.date())
    movie_start = str(max(start, init).date())

    # --- load --------------------------------------------------------------
    print(f"loading obs ({obs_src if kind == 'precip' else 'era5'}) "
          f"{start_s}..{end_s} ...")
    obs, obs_label = load_obs(kind, obs_src, domain, start_s, end_s)
    print(f"  {dict(obs.sizes)}")

    print(f"loading forecast init {init_d} ...")
    fc, _ = load_fc(kind, fc_path, spec, domain, movie_start, end_s)
    print(f"  {dict(fc.sizes)}  (grid {_grid_step(fc):g} deg; obs "
          f"{_grid_step(obs):g} deg)")
    # each panel is drawn on its own grid, so no regridding is needed

    common = np.intersect1d(obs["time"].values, fc["time"].values)
    if common.size == 0:
        raise SystemExit("obs and forecast share no valid days - check dates")
    fc = fc.sel(time=common)
    obs_movie = obs.sel(time=common)
    fc.attrs["init"] = str(init_d)

    # --- threshold / colour scale --------------------------------------------
    obs_s = reduce_series(obs, reduce, point)
    threshold = cfg.get("threshold")
    if threshold is None:
        threshold = round(float(np.nanpercentile(obs_movie.values,
                                                 K["threshold_pct"])), 1)
        print(f"threshold (obs p{K['threshold_pct']}): {threshold} {K['units']}")
    both = np.concatenate([obs_movie.values.ravel(),
                           fc.mean(ENS_DIM).values.ravel()
                           if ENS_DIM in fc.dims else fc.values.ravel()])
    if kind == "heat":
        vmin = cfg.get("vmin", _rnd(np.nanpercentile(both, 2), 2, False))
        vmax = cfg.get("vmax", _rnd(np.nanmax(both), 2, True))
        ylim = (vmin, vmax)
    else:
        vmin = cfg.get("vmin", 0.0)
        vmax = cfg.get("vmax", _rnd(np.nanpercentile(both, 99.5), 10, True))
        ylim = (0, max(vmax, float(np.nanmax(obs_s.values)) * 1.05))

    event = cfg.get("event") or exceedance_days(obs_s, threshold)
    cities = list(cfg.get("cities", []))
    if point is not None and cfg.get("point_name") and \
            cfg["point_name"] not in [c[0] for c in cities]:
        cities.append((cfg["point_name"], point[0], point[1]))
    mask_geom = natural_earth_country(cfg["mask_country"]) \
        if cfg.get("mask_country") else None

    tag = f"{cfg['name']}_{_slug(model_label)}_init{init_d:%Y%m%d}"
    os.makedirs(cfg["outdir"], exist_ok=True)
    fc_label = f"{model_label} init {init_d}"
    where = (f"point @ {cfg.get('point_name') or f'{point[0]:.2f}N {point[1]:.2f}E'}"
             if reduce == "point" else f"{reduce} (domain)")

    # --- movie ---------------------------------------------------------------
    if not cfg.get("skip_movie"):
        lines = [
            # obs on the shared valid days only, as in mali_movie.py
            {"label": obs_label,
             "data": reduce_series(obs_movie, reduce, point), "color": "black",
             "lw": 2.0, "ms": 4, "zorder": 4},
            forecast_line(reduce_series(fc, reduce, point), fc_label,
                          K["fc_color"]),
        ]
        # overlay models: time-series lines only, nearest init to ours,
        # aligned onto the shared valid days (days before their init -> NaN)
        for j, name in enumerate(cfg.get("overlay") or []):
            ospec = model_spec(name)
            if not ospec["vars"].get(kind):
                print(f"  overlay {name}: no {kind} variable - skipping")
                continue
            try:
                od, opath = pick_init(init_index(ospec["path"]), init,
                                      cfg["overlay_tol"])
                ofc, _ = load_fc(kind, opath, ospec, domain, movie_start, end_s)
            except Exception as exc:
                print(f"  overlay {name} failed: {type(exc).__name__}: {exc}")
                continue
            os_ = reduce_series(ofc, reduce, point)
            if ENS_DIM in os_.dims:
                os_ = os_.median(ENS_DIM)
            os_ = os_.reindex(time=common)
            lab = f"{name} (init {od:%m-%d})" + \
                (" median" if ENS_DIM in ofc.dims else "")
            lines.append({"label": lab, "data": os_,
                          "color": K["overlay_colors"][
                              j % len(K["overlay_colors"])],
                          "lw": 1.6, "ms": 4, "zorder": 3})
            print(f"  overlay {name}: init {od} "
                  f"({abs((od - init_d).days)} d from {init_d}), "
                  f"grid {_grid_step(ofc):g} deg")

        print("rendering movie ...")
        show_thr = cfg.get("show_threshold", False)
        make_event_movie(
            {obs_label: obs_movie, fc_label: fc}, lines,
            os.path.join(cfg["outdir"], f"{tag}_movie.gif"),
            domain=domain, box=cfg.get("box"), mask_geom=mask_geom,
            cmap=K["cmap"], vmin=vmin, vmax=vmax, extend=K["extend"],
            cbar_label=K["cbar_label"],
            transparent_low=K["transparent_low"], alpha_gamma=0.5,
            threshold=threshold if show_thr else None,
            threshold_label=f"{threshold:g} {K['units']}",
            onset_from=obs_label if show_thr else None, event=event,
            map_feature_color="black", cities=cities or None,
            ylabel=f"{K['ts_label']}, {where}", ylim=ylim,
            title=cfg["title"],
            lead_from=fc_label,
            init_label=f"Initialized: {init_d}",
            fps=cfg["fps"], dpi=170, save_mp4=not cfg.get("no_mp4"),
            still_only=cfg.get("still", False))

    # --- event track map ----------------------------------------------------
    if cfg.get("track") and not cfg.get("skip_track"):
        print("rendering event track map ...")
        # default to the event window +/- 2 d: a whole 3-week window of daily
        # contours (esp. scattered convective rain) buries the event itself
        if cfg.get("track_window"):
            tw = cfg["track_window"]
        elif event is not None:
            tw = (pd.Timestamp(event[0]) - pd.Timedelta("2D"),
                  pd.Timestamp(event[1]) + pd.Timedelta("2D"))
        else:
            tw = (None, None)
        sel = dict(time=slice(*tw))
        if not obs_movie.sel(**sel).sizes["time"]:
            print(f"  track window {tw} outside movie window - using all days")
            sel = {}
        make_track_map(
            {obs_label: obs_movie.sel(**sel),
             f"{fc_label} (ens mean)" if ENS_DIM in fc.dims else fc_label:
                 fc.sel(**sel)},
            os.path.join(cfg["outdir"], f"{tag}_track.png"),
            domain=domain, threshold=threshold, kind_cfg=K, units=K["units"],
            title=cfg["title"], cities=cities, box=cfg.get("box"),
            vmin=vmin, vmax=vmax)


if __name__ == "__main__":
    main()