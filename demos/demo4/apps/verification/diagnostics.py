# Authors: The africlimate AI team
# SPDX-License-Identifier: MIT
"""Forecast verification diagnostics — shared sample foundation.

Builds the pooled (forecast probability, observed binary event) pairs that the
reliability, ROC, and cost-loss diagnostics all consume. This module is the only
place that knows the on-disk shapes: the calibrated GraphCast ``p_exceed``
forecasts, the ENACTS gridded observations (whose ``Lat``/``Lon`` are renamed to
the forecast's ``lat``/``lon``), and the DEM used for altitude stratification.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.stats import rankdata

from apps.verification import config

THRESHOLDS_MM_DAY = [10, 20, 40, 50, 60]

ALTITUDE_BANDS = {
    "all": None,
    "0-1000": (0.0, 1000.0),
    "1000-2000": (1000.0, 2000.0),
    "2000-3000": (2000.0, 3000.0),
    ">3000": (3000.0, float("inf")),
}

YEAR = 2022

_REPO_ROOT = Path(__file__).resolve().parent.parent
# Raw (uncalibrated) forecast tree — read by compute_deterministic for raw-skill metrics
# (the calibrated / IDR tree is _IDR_GRIDDED, below).
_RAW_GRIDDED = _REPO_ROOT / "data/raw/gridded"
_RAW_STATION = _REPO_ROOT / "data/raw/station/precipitation"   # raw forecast-at-station (precip-only)
_DEM_PATH = _REPO_ROOT / "data/constants-DEM_1_Arc-Minute.nc"

# Year-agnostic roots; (variable, year) select the per-variable / per-year subtree.
_IDR_GRIDDED = _REPO_ROOT / "data/idr/gridded"  # {variable}/{year}/{model}
_IDR_STATION = _REPO_ROOT / "data/idr/station/precipitation"  # station = precip-only
_ENACTS_DIR = _REPO_ROOT / "data/Enacts"        # one subdir per obs variable
_STATION_OBS_DIR = _REPO_ROOT / "data/station"  # gauge stations_{year}.nc
DEFAULT_YEAR = 2022
DEFAULT_VARIABLE = "precipitation"

# Per-variable facts the gridded consumer needs, in one place. Per variable:
#   obs_subdir/obs_prefix — ENACTS obs at {ENACTS}/{obs_subdir}/{obs_prefix}_{year}_ALL.nc
#   obs_var               — the obs variable name inside that file
#   to_rate               — obs is a daily flux to divide by 24 → mm/hr (precip), else as-is (°C)
#   raw_fc_subdir/raw_fc_var — the raw forecast under {RAW_GRIDDED}/{raw_fc_subdir}/{year}/{model}
# The calibrated (IDR) tree is always {IDR_GRIDDED}/{variable}/... with the median in `q50`.
# Add a row as a product onboards.
_VARIABLE = {
    "precipitation": {
        "obs_subdir": "precipitation", "obs_prefix": "rr_mrg", "obs_var": "precip",
        "to_rate": True,  "raw_fc_subdir": "precipitation",  "raw_fc_var": "precipitation",
    },
    "temperature_max": {
        "obs_subdir": "temperature_max", "obs_prefix": "tmax_mrg", "obs_var": "temp",
        "to_rate": False, "raw_fc_subdir": "temperature_2m", "raw_fc_var": "temperature_2m",
    },
}

# Model token (used in output paths and the UI) -> input folder under the year subtree.
MODELS = {"graphcast": "graphcast", "ifs": "ifs"}


def model_fc_root(model, year=DEFAULT_YEAR, variable=DEFAULT_VARIABLE, raw=False):
    """Gridded forecast dir for a (model, year, variable).

    raw=True → the raw (uncalibrated) forecast tree, keyed by the forecast subdir
    (precipitation, but temperature_2m for temperature_max — the raw variable name differs
    from the target). Otherwise the calibrated IDR output tree, keyed by the target variable.
    """
    if raw:
        return _RAW_GRIDDED / _VARIABLE[variable]["raw_fc_subdir"] / str(year) / MODELS[model]
    return _IDR_GRIDDED / variable / str(year) / MODELS[model]


def station_fc_root(model, year=DEFAULT_YEAR):
    """Station forecast dir for a (model, year) — under the station IDR tree."""
    return _IDR_STATION / str(year) / MODELS[model]


def station_raw_fc_root(model, year=DEFAULT_YEAR):
    """Raw (uncalibrated) forecast-at-station dir for a (model, year) — the calibration input."""
    return _RAW_STATION / str(year) / MODELS[model]


def gridded_obs_path(year=DEFAULT_YEAR, variable="precipitation"):
    """ENACTS gridded obs file for a (year, variable).

    e.g. precipitation → Enacts/precipitation/rr_mrg_{year}_ALL.nc;
    temperature_max → Enacts/temperature_max/tmax_mrg_{year}_ALL.nc.
    """
    spec = _VARIABLE[variable]
    return _ENACTS_DIR / spec["obs_subdir"] / f"{spec['obs_prefix']}_{year}_ALL.nc"


def ui_model_token(model, calibrated):
    """UI Model-facet token: calibrated (IDR) forecasts get a ``-idr`` suffix; raw stay plain.

    The data trees stay keyed by the real model; this is only the label the diagnostics
    path (and thus the dashboard's Model selector) shows, so ``graphcast`` (raw) and
    ``graphcast-idr`` (calibrated) appear side by side. The calibrated data still lives,
    self-documenting, under ``idr/`` — it is not renamed or re-published.
    """
    return f"{model}-idr" if calibrated else model


def parse_model_token(token):
    """Inverse of :func:`ui_model_token`: split a UI model token into (base_model, calibrated).

    ``'graphcast-idr' -> ('graphcast', True)``; ``'graphcast' -> ('graphcast', False)``.
    The CLI uses this so the token a user types (``graphcast`` vs ``graphcast-idr``) selects
    the source — raw forecast vs IDR-calibrated — and is also the token written out.
    """
    if token.endswith("-idr"):
        return token[:-4], True
    return token, False


_FC_ROOT = model_fc_root("graphcast")


@dataclass
class Samples:
    """Pooled (p, o) pairs for one lead, keyed by (threshold_mm_day, altitude_band).

    ``pairs[(thr, band)]`` is a tuple ``(p, o)`` of equal-length 1-D arrays: ``p``
    the forecast exceedance probabilities and ``o`` the binary observed events
    (1 if observed precip exceeded ``thr`` mm/day), pooled over time and space.
    ``thresholds`` and ``bands`` record the axes (and their order) this set covers.
    ``blocks`` (optional, same keys as ``pairs``) gives each pooled sample's day index,
    enabling the day-block bootstrap used for reliability confidence intervals.
    """
    lead_hours: int
    pairs: dict
    thresholds: list = None
    bands: list = None
    blocks: dict = None


def load_forecast(lead_hours, fc_root=_FC_ROOT):
    """Open the calibrated forecast file for one lead (``{lead_hours}h.nc``)."""
    return xr.open_dataset(Path(fc_root) / f"{lead_hours}h.nc",engine="scipy")


def load_obs(obs_path=None):
    """Open ENACTS observations, renaming ``Lat``/``Lon`` to the forecast's ``lat``/``lon``."""
    obs = xr.open_dataset(obs_path or gridded_obs_path())
    rename = {old: old.lower() for old in ("Lat", "Lon")
              if old in obs.dims or old in obs.coords}
    return obs.rename(rename)


def interp_dem(like, dem_path=_DEM_PATH):
    """Interpolate the DEM elevation onto the grid of ``like``, as a (lat, lon) DataArray."""
    dem = xr.open_dataset(dem_path)["z_bedrock"]
    return dem.interp(lat=like["lat"], lon=like["lon"], method="linear")


def altitude_masks(elevation):
    """Boolean (lat, lon) mask per altitude band, all land-only (elevation >= 0).

    Every band — including 'all' — excludes ocean/no-data cells, so 'all' equals the
    union of the numeric bands (a clean aggregate of the strata it summarises).
    """
    elev = np.asarray(elevation.values)
    land = np.isfinite(elev) & (elev >= 0)
    masks = {}
    for band, bounds in ALTITUDE_BANDS.items():
        if bounds is None:
            masks[band] = land
        else:
            lo, hi = bounds
            masks[band] = land & (elev >= lo) & (elev < hi)
    return masks


def pool_pairs(p_exceed, precip, masks, thresholds=THRESHOLDS_MM_DAY):
    """Pool (p, o) and day indices per (threshold, altitude band), dropping NaNs.

    ``p_exceed`` has dims (threshold, time, *space); ``precip`` has dims (time, *space)
    in mm/day, where *space* is one or more spatial dims — (lat, lon) for gridded
    targets, (station,) for station targets. All spatial dims are flattened to a
    single axis so the pooling is the same for both; the mask is flattened the same
    way. The event ``o`` is ``precip > threshold``; ``p`` is the forecast exceedance
    probability. Returns ``(pairs, blocks)`` where ``pairs[(thr, band)] = (p, o)`` and
    ``blocks[(thr, band)]`` is each sample's day index (the time coordinate position)
    — the unit resampled by the block bootstrap. One threshold slice is materialised
    at a time to keep the memory footprint bounded.
    """
    precip_v = np.asarray(precip.values)
    n_time = precip_v.shape[0]
    # C-order flatten reproduces the old nested (lat, lon) pooling exactly, so the
    # gridded result is byte-identical; for stations the single dim is a no-op.
    precip_v = precip_v.reshape(n_time, -1)            # (time, space)
    finite_obs = np.isfinite(precip_v)
    day = np.broadcast_to(np.arange(n_time)[:, None], precip_v.shape)
    pairs = {}
    blocks = {}
    for thr in thresholds:
        # Select by label, not position — trust the threshold coord the producer wrote.
        p_full = np.asarray(p_exceed.sel(threshold=thr).values).reshape(n_time, -1)
        o_full = precip_v > thr
        finite = finite_obs & np.isfinite(p_full)
        for band, mask in masks.items():
            sel = finite & np.asarray(mask).reshape(-1)[np.newaxis, :]
            pairs[(thr, band)] = (p_full[sel], o_full[sel].astype(np.int8))
            blocks[(thr, band)] = day[sel]
    return pairs, blocks


def _subset_gridded_to_region(fc, obs, region):
    """Crop ``fc``/``obs`` to an admin-1 region's bbox and NaN cells outside its polygon.

    Only admin-1 subregions of the deployment country (``VERIFICATION_COUNTRY``) subset;
    ``region`` None / the whole country / any non-subregion key returns the pair unchanged
    (whole-domain behaviour). The obs is masked (not the forecast) because pooling drops a
    cell whenever the obs is NaN, so out-of-polygon cells fall out on their own. Cropping to
    the bbox first keeps memory bounded — a single region is a small slice of the grid.
    """
    from verification.regions import _get_subregions, _subregion_keep

    if not region or region not in _get_subregions():
        return fc, obs
    lon = fc["lon"].values
    lat = fc["lat"].values
    keep = _subregion_keep(region, lon, lat)          # (nlat, nlon) bool
    lat_idx = np.where(keep.any(axis=1))[0]
    lon_idx = np.where(keep.any(axis=0))[0]
    if lat_idx.size == 0 or lon_idx.size == 0:
        return fc.isel(lat=slice(0, 0), lon=slice(0, 0)), obs.isel(lat=slice(0, 0), lon=slice(0, 0))
    lat_sl = slice(int(lat_idx[0]), int(lat_idx[-1]) + 1)
    lon_sl = slice(int(lon_idx[0]), int(lon_idx[-1]) + 1)
    fc = fc.isel(lat=lat_sl, lon=lon_sl)
    obs = obs.isel(lat=lat_sl, lon=lon_sl)
    keep_da = xr.DataArray(keep[lat_sl, lon_sl], dims=("lat", "lon"),
                           coords={"lat": fc["lat"], "lon": fc["lon"]})
    return fc, obs.where(keep_da)


def _subset_to_months(fc, obs, months):
    """Restrict ``fc``/``obs`` to rows whose ``time`` falls in ``months`` (1-based).

    ``months`` None / empty returns the pair unchanged (Yearly = the whole year). A season
    crossing the year boundary (e.g. Bega Oct–Jan → months 10,11,12,1) selects those calendar
    months of the single verification year. The two arrays share one ``time`` dim, so the same
    boolean mask applies to both.
    """
    if not months:
        return fc, obs
    keep = np.isin(fc["time"].dt.month.values, list(months))
    return fc.isel(time=keep), obs.isel(time=keep)


def build_samples(lead_hours, model="graphcast", year=DEFAULT_YEAR,
                  thresholds=THRESHOLDS_MM_DAY, bands=None,
                  fc_root=None, obs_path=None, dem_path=_DEM_PATH, region=None, months=None):
    """Build pooled (p, o) :class:`Samples` for one lead + model + year, restricted to the given axes.

    ``region`` (an admin-1 subregion of the deployment country) crops+masks the forecast and
    obs to that region before pooling; None / the whole country pools the full domain.
    ``months`` (1-based) restricts pooling to one season's calendar months; None = the whole year.
    """
    bands = bands if bands is not None else list(ALTITUDE_BANDS)
    fc = load_forecast(lead_hours, fc_root or model_fc_root(model, year))
    obs = load_obs(obs_path or gridded_obs_path(year))
    fc, obs = xr.align(fc, obs, join="inner")
    fc, obs = _subset_gridded_to_region(fc, obs, region)
    fc, obs = _subset_to_months(fc, obs, months)
    elevation = interp_dem(fc, dem_path)
    masks = {b: m for b, m in altitude_masks(elevation).items() if b in bands}
    pairs, blocks = pool_pairs(fc["p_exceed"], obs["precip"], masks, thresholds)
    return Samples(lead_hours=lead_hours, pairs=pairs, blocks=blocks,
                   thresholds=list(thresholds), bands=list(bands))


def load_obs_station(year, obs_dir=_STATION_OBS_DIR):
    """Open the EMA gauge obs for a year (``total_precipitation_rate``, mm/day)."""
    return xr.open_dataset(Path(obs_dir) / f"stations_{year}.nc")


def _subset_station_to_region(fc, obs_precip, region):
    """Keep only stations whose gauge coordinate falls inside an admin-1 region.

    The station analogue of :func:`_subset_gridded_to_region`: only admin-1 subregions
    filter; None / the whole country returns the pair unchanged. Uses the gauge coords
    (``station_lat``/``station_lon``, falling back to ``lat``/``lon`` on the station dim).
    """
    from verification.regions import _get_subregions, stations_in_region

    if not region or region not in _get_subregions():
        return fc, obs_precip
    lat_name = "station_lat" if "station_lat" in fc.coords else "lat"
    lon_name = "station_lon" if "station_lon" in fc.coords else "lon"
    keep = stations_in_region(fc[lon_name].values, fc[lat_name].values, region)
    return fc.isel(station=keep), obs_precip.isel(station=keep)


def build_samples_station(lead_hours, model="graphcast", year=DEFAULT_YEAR,
                          thresholds=THRESHOLDS_MM_DAY, bands=None,
                          fc_root=None, obs_dir=_STATION_OBS_DIR, region=None, months=None):
    """Build pooled (p, o) :class:`Samples` for one lead at stations.

    The station counterpart of :func:`build_samples`, on the (time, station) grid:
    the calibrated station forecast supplies ``p_exceed``; the EMA gauge supplies the
    event truth (gauge mm/day > threshold). The two are paired by the same
    ``(name, gauge-lat, gauge-lon)`` key + calendar date used in production
    (``align_stations``), and stations are stratified into altitude bands by their own
    ``elevation`` coord — no DEM interpolation, since each gauge carries its altitude.
    """
    from calibration.run_station import align_stations  # pure name+date pairing helper

    bands = bands if bands is not None else list(ALTITUDE_BANDS)
    fc = load_forecast(lead_hours, fc_root or station_fc_root(model, year))
    obs = load_obs_station(year, obs_dir)
    fc, obs_precip = align_stations(fc, obs["total_precipitation_rate"])
    fc, obs_precip = _subset_station_to_region(fc, obs_precip, region)
    fc, obs_precip = _subset_to_months(fc, obs_precip, months)
    masks = {b: m for b, m in altitude_masks(fc["elevation"]).items() if b in bands}
    pairs, blocks = pool_pairs(fc["p_exceed"], obs_precip, masks, thresholds)
    return Samples(lead_hours=lead_hours, pairs=pairs, blocks=blocks,
                   thresholds=list(thresholds), bands=list(bands))


def _axes(samples):
    """Resolve the (thresholds, bands) axes of a Samples, in canonical order."""
    keys = set(samples.pairs)
    thresholds = samples.thresholds or [t for t in THRESHOLDS_MM_DAY
                                        if any(k[0] == t for k in keys)]
    bands = samples.bands or [b for b in ALTITUDE_BANDS if any(k[1] == b for k in keys)]
    return thresholds, bands


ROC_PROBS = np.linspace(0.01, 1.0, 500)


def roc_counts(p, o, probs=ROC_PROBS):
    """Contingency counts (hits, false_alarms, misses, correct_negatives) at every cut-off.

    Declares an event where ``p >= p*``. Evaluates all cut-offs in one pass by sorting
    the event and non-event probabilities and counting with ``searchsorted`` — the cheap
    equivalent of the histogram/cumsum sweep, avoiding a 500-fold boolean scan. Returns a
    dict of int64 arrays, each aligned to ``probs``.
    """
    p = np.asarray(p)
    o = np.asarray(o).astype(bool)
    pe = np.sort(p[o])
    pn = np.sort(p[~o])
    ne, nn = pe.size, pn.size
    hits = ne - np.searchsorted(pe, probs, side="left")
    false_alarms = nn - np.searchsorted(pn, probs, side="left")
    return {
        "hits": hits.astype(np.int64),
        "false_alarms": false_alarms.astype(np.int64),
        "misses": (ne - hits).astype(np.int64),
        "correct_negatives": (nn - false_alarms).astype(np.int64),
    }


def compute_roc(samples, probs=ROC_PROBS, thresholds=None, bands=None):
    """Build the ROC contingency-table Dataset over (threshold, altitude, prob) for one lead."""
    axt, axb = _axes(samples)
    thresholds = thresholds if thresholds is not None else axt
    bands = bands if bands is not None else axb
    shape = (len(thresholds), len(bands), len(probs))
    fields = ("hits", "false_alarms", "misses", "correct_negatives")
    arrs = {k: np.zeros(shape, dtype=np.int64) for k in fields}
    for i, thr in enumerate(thresholds):
        for j, band in enumerate(bands):
            p, o = samples.pairs[(thr, band)]
            counts = roc_counts(p, o, probs)
            for k in fields:
                arrs[k][i, j, :] = counts[k]
    ds = xr.Dataset(
        {k: (("threshold", "altitude", "prob"), v) for k, v in arrs.items()},
        coords={"threshold": thresholds, "altitude": bands, "prob": probs},
    )
    ds["threshold"].attrs["units"] = "mm/day"
    ds["prob"].attrs["long_name"] = "probability cut-off p*"
    ds.attrs["lead_hours"] = samples.lead_hours
    return ds


def output_path(metric, lead_hours, outdir, year=DEFAULT_YEAR, subdir=None, model="graphcast",
                target="enacts", native_var=DEFAULT_VARIABLE, region=None, season=None):
    """The on-disk path for a diagnostic file, mirroring the engine's facet order.

    ``subdir`` inserts an extra segment before the file (e.g. ``annual_monthly_averages``
    for scalar metrics the dashboard discovers); curve metrics leave it None. ``model`` is
    the forecast token (the UI keys its Model facet off this segment). ``target`` is the
    observation product (``enacts`` for the gridded path, ``stations`` for gauges). ``native_var``
    is the variable segment the UI's ``kind_of`` reads (``precipitation`` → precip,
    ``temperature_max`` → Tmax), so variables never collide. ``region`` inserts an admin-1
    segment after the year (``.../{year}/{region}/{lead}h.nc``) for per-region threshold
    curves; None writes the region-less whole-domain path (unchanged legacy layout).
    ``season`` inserts a season-slug segment *after* the region one
    (``.../{year}/[{region}/]{season}/{lead}h.nc``) for per-season threshold curves; None /
    ``"yearly"`` writes the season-less path. The slug is a controlled vocabulary
    (see :mod:`verification.seasons`) so discovery can tell it apart from a region segment.
    """
    base = (Path(outdir) / config.domain() / model / native_var
            / target / metric / str(year))
    if subdir:
        base = base / subdir
    if region:
        base = base / region
    if season and season != "yearly":
        base = base / season
    return base / f"{lead_hours}h.nc"


def write_dataset(ds, metric, lead_hours, outdir, year=DEFAULT_YEAR, subdir=None, model="graphcast",
                  target="enacts", native_var=DEFAULT_VARIABLE, region=None, season=None):
    """Write a diagnostic Dataset to its :func:`output_path`, creating parent dirs."""
    path = output_path(metric, lead_hours, outdir, year, subdir, model=model, target=target,
                       native_var=native_var, region=region, season=season)
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)
    return path


def roc_rates(ds, threshold, band):
    """FPR and TPR arrays (ascending FPR, with (0,0)/(1,1) endpoints) for one ROC curve."""
    sub = ds.sel(threshold=threshold, altitude=band)
    h, m = sub["hits"].values, sub["misses"].values
    fa, cn = sub["false_alarms"].values, sub["correct_negatives"].values
    tpr = h / np.maximum(h + m, 1)
    fpr = fa / np.maximum(fa + cn, 1)
    order = np.argsort(fpr)
    fpr = np.concatenate([[0.0], fpr[order], [1.0]])
    tpr = np.concatenate([[0.0], tpr[order], [1.0]])
    return fpr, tpr


def plot_roc(ds, out_png, band="all"):
    """Save a ROC plot (one curve per threshold, with AUC in the legend) for one altitude band."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    trapezoid = getattr(np, "trapezoid", getattr(np, "trapz", None))
    fig, ax = plt.subplots(figsize=(6, 6))
    for thr in ds["threshold"].values:
        fpr, tpr = roc_rates(ds, thr, band)
        ax.plot(fpr, tpr, marker="", lw=2,
                label=f"{int(thr)} mm/day  (AUC={trapezoid(tpr, fpr):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="no skill")
    ax.set_xlabel("False alarm rate")
    ax.set_ylabel("Hit rate")
    ax.set_title(f"ROC — GraphCast vs ENACTS — {ds.attrs.get('lead_hours')}h — altitude: {band}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_aspect("equal")
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    return out_png


def _band_slug(band):
    """Filename-safe form of an altitude-band label."""
    return str(band).replace(">", "gt").replace(" ", "")


def plot_by_altitude(ds, out_dir, plot_fn, prefix):
    """Save one PNG per altitude band using ``plot_fn``; returns the written paths."""
    lead = ds.attrs.get("lead_hours")
    paths = []
    for band in (str(b) for b in ds["altitude"].values):
        png = Path(out_dir) / f"{prefix}_{lead}h_alt_{_band_slug(band)}.png"
        plot_fn(ds, png, band=band)
        paths.append(png)
    return paths


COST_LOSS_RATIOS = np.linspace(0.01, 0.99, 100)


def _value_envelope(hits, false_alarms, misses, correct_negatives, alphas):
    """Relative economic value V(alpha), maximised over cut-offs, from one curve's counts.

    For a user with cost-loss ratio ``alpha = C/L`` and event base rate ``s``, the value of
    acting on the forecast is compared against climatology and a perfect forecast:
    ``V = (E_clim - E_forecast) / (E_clim - E_perfect)``. The envelope takes the best V over
    all probability cut-offs. Returns ``(V_envelope, base_rate)``; V is zeros if no events.
    """
    n = hits[0] + misses[0] + false_alarms[0] + correct_negatives[0]
    if n == 0:                       # empty pool (e.g. a band absent from a small subregion)
        return np.zeros_like(alphas), 0.0
    s = (hits[0] + misses[0]) / n
    if s <= 0 or s >= 1:
        return np.zeros_like(alphas), float(s)
    hit_fa = (hits + false_alarms) / n
    miss = misses / n
    e_clim = np.minimum(alphas, s)
    e_perfect = alphas * s
    denom = e_clim - e_perfect
    e_forecast = alphas[:, None] * hit_fa[None, :] + miss[None, :]
    value = (e_clim[:, None] - e_forecast) / denom[:, None]
    return value.max(axis=1), float(s)


def compute_cost_loss(roc, alphas=COST_LOSS_RATIOS):
    """Build the cost-loss value-envelope Dataset from a ROC contingency Dataset (issue #4)."""
    thresholds = roc["threshold"].values
    bands = [str(b) for b in roc["altitude"].values]
    value = np.zeros((len(thresholds), len(bands), len(alphas)))
    base_rate = np.zeros((len(thresholds), len(bands)))
    for i, thr in enumerate(thresholds):
        for j, band in enumerate(roc["altitude"].values):
            sub = roc.sel(threshold=thr, altitude=band)
            value[i, j, :], base_rate[i, j] = _value_envelope(
                sub["hits"].values, sub["false_alarms"].values,
                sub["misses"].values, sub["correct_negatives"].values, alphas)
    ds = xr.Dataset(
        {"value": (("threshold", "altitude", "cost_loss_ratio"), value),
         "base_rate": (("threshold", "altitude"), base_rate),
         "max_value": (("threshold", "altitude"), value.max(axis=2))},
        coords={"threshold": thresholds, "altitude": bands, "cost_loss_ratio": alphas},
    )
    ds["threshold"].attrs["units"] = "mm/day"
    ds["cost_loss_ratio"].attrs["long_name"] = "cost-loss ratio alpha = C/L"
    ds.attrs["lead_hours"] = roc.attrs.get("lead_hours")
    return ds


def plot_cost_loss(ds, out_png, band="all"):
    """Save a cost-loss value curve (one curve per threshold) for one altitude band."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    alphas = ds["cost_loss_ratio"].values
    fig, ax = plt.subplots(figsize=(6, 5))
    for thr in ds["threshold"].values:
        v = ds["value"].sel(threshold=thr, altitude=band).values
        ax.plot(alphas, v, lw=2, label=f"{int(thr)} mm/day")
    ax.axhline(0, color="k", lw=1)
    ax.set_xlabel("Cost-loss ratio  α = C/L")
    ax.set_ylabel("Relative economic value V")
    ax.set_title(f"Cost-loss value — {ds.attrs.get('lead_hours')}h — altitude: {band}")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.1, 1.0)
    ax.legend(loc="upper right", fontsize=8)
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    return out_png


RELIABILITY_N_BINS = 10
RELIABILITY_N_BOOTSTRAPS = 50


def _block_bootstrap_ci(idx, o, block, n_bins, n_bootstraps, rng):
    """95% CI of each bin's observed frequency, resampling whole day-blocks with replacement.

    Bin edges are held fixed (from the full sample); each replicate redraws the set of days,
    so the interval reflects between-day (and within-day spatial) dependence rather than
    assuming i.i.d. cells. Counts are pre-aggregated per (day, bin) so a replicate is a cheap
    sum over the chosen days. Returns per-bin (lower, upper) arrays.
    """
    _, compact = np.unique(block, return_inverse=True)
    n_blocks = int(compact.max()) + 1
    tot = np.zeros((n_blocks, n_bins))
    pos = np.zeros((n_blocks, n_bins))
    np.add.at(tot, (compact, idx), 1.0)
    np.add.at(pos, (compact, idx), o.astype(float))
    boot = np.full((n_bootstraps, n_bins), np.nan)
    for r in range(n_bootstraps):
        chosen = rng.integers(0, n_blocks, size=n_blocks)
        bt = tot[chosen].sum(0)
        bp = pos[chosen].sum(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            boot[r] = np.where(bt > 0, bp / bt, np.nan)
    lo = np.full(n_bins, np.nan)
    hi = np.full(n_bins, np.nan)
    nonempty = tot.sum(0) > 0
    if nonempty.any():
        lo[nonempty], hi[nonempty] = np.nanpercentile(boot[:, nonempty], [2.5, 97.5], axis=0)
    return lo, hi


def _reliability_curve(p, o, n_bins, n_bootstraps, rng, block=None):
    """Per quantile bin: mean forecast prob, observed frequency, count, and bootstrap 95% CI.

    Bins hold roughly equal counts (quantile edges) — though under the point mass at p≈0 for
    rare thresholds the low bins can be empty and stay NaN. When ``block`` (a day index per
    sample) is given, the CI comes from a day-block bootstrap (:func:`_block_bootstrap_ci`),
    which respects spatial+temporal autocorrelation; otherwise it falls back to an i.i.d.
    binomial draw, which is only valid for genuinely independent samples.
    """
    # An empty pool (e.g. an altitude band with no cells inside a small subregion) has no
    # quantiles to bin — return all-NaN bins rather than letting np.quantile([]) raise.
    if np.asarray(p).size == 0:
        nan = np.full(n_bins, np.nan)
        return nan, nan.copy(), np.zeros(n_bins, dtype=np.int64), nan.copy(), nan.copy()
    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    idx = np.clip(np.searchsorted(edges[1:-1], p, side="right"), 0, n_bins - 1)
    fp = np.full(n_bins, np.nan)
    of = np.full(n_bins, np.nan)
    lo = np.full(n_bins, np.nan)
    hi = np.full(n_bins, np.nan)
    cnt = np.zeros(n_bins, dtype=np.int64)
    if block is not None:
        lo, hi = _block_bootstrap_ci(idx, o, block, n_bins, n_bootstraps, rng)
    for b in range(n_bins):
        sel = idx == b
        n = int(sel.sum())
        cnt[b] = n
        if n == 0:
            continue
        fp[b] = p[sel].mean()
        k = int(o[sel].sum())
        of[b] = k / n
        if block is None:
            boot = rng.binomial(n, k / n, size=n_bootstraps) / n
            lo[b], hi[b] = np.percentile(boot, [2.5, 97.5])
    return fp, of, cnt, lo, hi


def compute_reliability(samples, n_bins=RELIABILITY_N_BINS,
                        n_bootstraps=RELIABILITY_N_BOOTSTRAPS,
                        thresholds=None, bands=None, seed=0):
    """Build the reliability-diagram Dataset over (threshold, altitude, bin) for one lead (issue #2)."""
    axt, axb = _axes(samples)
    thresholds = thresholds if thresholds is not None else axt
    bands = bands if bands is not None else axb
    rng = np.random.default_rng(seed)
    shape = (len(thresholds), len(bands), n_bins)
    fp = np.full(shape, np.nan)
    of = np.full(shape, np.nan)
    lo = np.full(shape, np.nan)
    hi = np.full(shape, np.nan)
    cnt = np.zeros(shape, dtype=np.int64)
    for i, thr in enumerate(thresholds):
        for j, band in enumerate(bands):
            p, o = samples.pairs[(thr, band)]
            block = samples.blocks.get((thr, band)) if samples.blocks else None
            fp[i, j], of[i, j], cnt[i, j], lo[i, j], hi[i, j] = _reliability_curve(
                p, o, n_bins, n_bootstraps, rng, block=block)
    dims = ("threshold", "altitude", "bin")
    ds = xr.Dataset(
        {"forecast_probability": (dims, fp),
         "observed_frequency": (dims, of),
         "bin_count": (dims, cnt),
         "obs_freq_lower": (dims, lo),
         "obs_freq_upper": (dims, hi)},
        coords={"threshold": thresholds, "altitude": bands, "bin": np.arange(n_bins)},
    )
    ds["threshold"].attrs["units"] = "mm/day"
    ds.attrs["lead_hours"] = samples.lead_hours
    ds.attrs["n_bootstraps"] = n_bootstraps
    ds.attrs["binning"] = "quantile"
    return ds


def plot_reliability(ds, out_png, band="all"):
    """Save a reliability diagram (one curve per threshold, with bootstrap CI) for one band."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    band_ds = ds.sel(altitude=band)
    extent = np.concatenate([band_ds["forecast_probability"].values.ravel(),
                             band_ds["observed_frequency"].values.ravel(),
                             band_ds["obs_freq_upper"].values.ravel()])
    top = np.nanmax(extent)
    lim = 1.0 if not np.isfinite(top) else min(1.0, top * 1.08)

    fig, ax = plt.subplots(figsize=(6, 6))
    for thr in ds["threshold"].values:
        sub = ds.sel(threshold=thr, altitude=band)
        fp = sub["forecast_probability"].values
        of = sub["observed_frequency"].values
        lo = sub["obs_freq_lower"].values
        hi = sub["obs_freq_upper"].values
        ax.errorbar(fp, of, yerr=[of - lo, hi - of], marker="o", ms=4, lw=1.5,
                    capsize=2, label=f"{int(thr)} mm/day")
    ax.plot([0, lim], [0, lim], "k--", lw=1, label="perfect")
    ax.set_xlabel("Forecast probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title(f"Reliability — {ds.attrs.get('lead_hours')}h — altitude: {band}")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal")
    ax.legend(loc="lower right", fontsize=8)
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    return out_png


# ---------------------------------------------------------------------------
# Scalar-field metrics (CRPS, and the same shape as MAE / SEEPS) for the
# existing dashboard. These reduce a per-cell field to the four annual/monthly
# summaries the UI loader reads; they share nothing with the threshold family.
# ---------------------------------------------------------------------------

def annual_monthly_averages(da, time_dim="time", band_masks=None):
    """Reduce a (time, *space) field to the dashboard's summary variables.

    Mirrors the upstream weather-postprocessing transform: group by year and by month and
    average over time, then over the spatial dims (plain mean, skipna — no area weighting,
    to match the existing CRPS/MAE/SEEPS summaries). Returns a Dataset with annual_average,
    annual_spatial_average, monthly_average, and monthly_spatial_average.

    ``band_masks`` (label → boolean mask over the spatial dims; numeric altitude
    bands only — "all" stays the legacy unmasked scalars) appends the variables the
    dashboard's altitude selector reads:

    - ``annual_average_by_altitude(year, altitude)`` / ``monthly_average_by_altitude
      (month, altitude)`` — per-band spatial means, ``altitude`` coord = band labels;
    - ``altitude_band(*space)`` — int8 band index per cell/station (−1 = no band),
      index order matching the ``altitude`` coord, so the UI can mask maps and
      station markers without knowing the DEM.

    Without ``band_masks`` the file keeps its legacy shape (old data stays valid).
    """
    spatial = [d for d in da.dims if d != time_dim]
    annual_spatial = da.groupby(f"{time_dim}.year").mean(time_dim, skipna=True)
    monthly_spatial = da.groupby(f"{time_dim}.month").mean(time_dim, skipna=True)
    ds = xr.Dataset(
        {"annual_average": annual_spatial.mean(spatial, skipna=True),
         "annual_spatial_average": annual_spatial,
         "monthly_average": monthly_spatial.mean(spatial, skipna=True),
         "monthly_spatial_average": monthly_spatial},
    )
    if band_masks:
        labels = list(band_masks)
        coords = {d: da[d] for d in spatial if d in da.coords}
        masks_da = [xr.DataArray(np.asarray(m), dims=spatial, coords=coords)
                    for m in band_masks.values()]
        ds["annual_average_by_altitude"] = xr.concat(
            [annual_spatial.where(m).mean(spatial, skipna=True) for m in masks_da],
            dim="altitude").assign_coords(altitude=labels)
        ds["monthly_average_by_altitude"] = xr.concat(
            [monthly_spatial.where(m).mean(spatial, skipna=True) for m in masks_da],
            dim="altitude").assign_coords(altitude=labels)
        band_idx = np.full([da.sizes[d] for d in spatial], -1, dtype=np.int8)
        for i, m in enumerate(band_masks.values()):
            band_idx[np.asarray(m)] = i
        ds["altitude_band"] = xr.DataArray(
            band_idx, dims=spatial, coords=coords,
            attrs={"long_name": "altitude band index (-1 = no band)",
                   "band_labels": ",".join(labels)})
        ds["altitude"].attrs["units"] = "m"
    return ds


def band_masks_for(field, dem_path=_DEM_PATH):
    """Per-band boolean masks for a summary field's spatial layout, or None.

    Gridded (lat, lon) fields stratify by the DEM interpolated onto the field's
    grid; station fields use each gauge's own ``elevation`` coord (no DEM).
    Numeric bands only — "all" is the legacy unmasked scalars, not a band.
    Returns None (→ legacy-shaped summary file) when the field has no usable
    elevation source, e.g. the DEM file is absent on this deployment.
    """
    if "lat" in field.dims and "lon" in field.dims:
        if not Path(dem_path).exists():
            print(f"  note: DEM not found at {dem_path}; "
                  "writing summaries without altitude bands")
            return None
        elev = interp_dem(field, dem_path)
    elif "station" in field.dims and "elevation" in field.coords:
        elev = field["elevation"]
    else:
        return None
    masks = altitude_masks(elev)
    return {b: masks[b] for b, rng in ALTITUDE_BANDS.items() if rng is not None}


def compute_crps(lead_hours, model="graphcast", year=DEFAULT_YEAR,
                 variable=DEFAULT_VARIABLE, fc_root=None):
    """Aggregate a model's per-cell CRPS into the UI annual/monthly summaries.

    Works for gridded or station, precip or temperature — `crps` reduces over
    whatever spatial dim it has, in whatever units the calibration wrote. `fc_root`
    (if given) wins; otherwise the gridded IDR root for (model, year, variable).
    """
    fc = load_forecast(lead_hours, fc_root or model_fc_root(model, year, variable))
    ds = annual_monthly_averages(fc["crps"], band_masks=band_masks_for(fc["crps"]))
    units = fc["crps"].attrs.get("units", "")
    for v in ds.data_vars:
        if v != "altitude_band":          # int8 index layer, not a physical quantity
            ds[v].attrs["units"] = units
    ds.attrs["lead_hours"] = lead_hours
    return ds


def _safe_correlation(x, y, axis, corr_type='pearsonr'):
    """Compute correlation while silencing constant-input warnings.

    Returns NaN for slices where either input is constant or has fewer than two
    finite paired samples.
    """
    result = np.full(x.shape[1 - axis], np.nan, dtype=float)
    for i in range(x.shape[1 - axis]):
        mask = np.isfinite(x.take(i, axis=1 - axis)) & np.isfinite(y.take(i, axis=1 - axis))
        if mask.sum()>=2:
            x_vals = x.take(i, axis=1 - axis)[mask]
            y_vals = y.take(i, axis=1 - axis)[mask]
            if np.nanmin(x_vals)==np.nanmax(x_vals) or\
            np.nanmin(y_vals)==np.nanmax(y_vals):
                continue
            if corr_type=='spearmanr':
                x_vals = rankdata(x_vals,nan_policy='omit')
                y_vals = rankdata(y_vals,nan_policy='omit')
            result[i] = np.corrcoef(x_vals[None,:],y_vals[None,:])[0,1]
    return result

def corr_annual_monthly_averages(fc_raw, obs, corr_type='spearmanr', time_dim="time"):
    # For simplicity we replace the fcst init time with valid time
    # and drop prediction_timedelta, since we only care about the valid time for correlation

    fc = fc_raw.copy()
    fc[time_dim] = fc[time_dim] + fc["prediction_timedelta"]
    fc = fc.squeeze("prediction_timedelta")

    spatial_dims = [d for d in fc.dims if d != time_dim]

    fc_shape = fc.values.shape
    obs_shape = obs.values.shape
    if fc_shape != obs_shape:
        raise ValueError(f"Forecast and observation arrays must have the same shape. "
                         f"Got {fc_shape} and {obs_shape}.")
    # Compute annual and monthly averages along the time dimension
    annual_spatial_average = _safe_correlation(fc.values.reshape(fc_shape[0], -1), obs.values.reshape(obs_shape[0], -1), axis=0).reshape(fc_shape[1:])
    annual_average = _safe_correlation(fc.values.reshape(fc_shape[0], -1), obs.values.reshape(obs_shape[0], -1), axis=1)

    months = np.unique(fc[time_dim].dt.month.values)
    monthly_spatial_average = np.zeros((len(months), *fc_shape[1:]))
    monthly_average = np.zeros(len(months))
    for i, month in enumerate(months):
        fc_month = fc.sel({time_dim: fc[time_dim].dt.month == month})
        obs_month = obs.sel({time_dim: obs[time_dim].dt.month == month})
        monthly_spatial_average[i] = _safe_correlation(
            fc_month.values.reshape(fc_month.values.shape[0], -1),
            obs_month.values.reshape(obs_month.values.shape[0], -1),
            axis=0,corr_type=corr_type
        ).reshape(fc_shape[1:])
        monthly_average[i] = np.nanmean(_safe_correlation(
            fc_month.values.reshape(fc_month.values.shape[0], -1),
            obs_month.values.reshape(obs_month.values.shape[0], -1),
            axis=1,corr_type=corr_type,
        ), axis=0)

    return xr.Dataset(
        {"annual_average": (["time"], annual_average),
         "annual_spatial_average": (["latitude", "longitude"], annual_spatial_average),
         "monthly_average": (["month"], monthly_average),
         "monthly_spatial_average": (["month", "latitude", "longitude"], monthly_spatial_average)},
         coords={"time": np.unique(fc[time_dim].values),
                 "month": months,
                 "latitude": fc[spatial_dims[0]].values,
                 "longitude": fc[spatial_dims[1]].values},
    )

def _load_gridded_obs(variable, year):
    """ENACTS gridded obs for ``variable`` in the metric's standard unit — precipitation
    ÷24 → mm/hr, temperature left in °C (not a flux) — as a (time, lat, lon) DataArray."""
    spec = _VARIABLE[variable]
    obs = xr.open_dataset(gridded_obs_path(year, variable))
    rename = {o: o.lower() for o in ("Lat", "Lon") if o in obs.dims or o in obs.coords}
    da = obs.rename(rename)[spec["obs_var"]]
    return da / 24.0 if spec["to_rate"] else da


def compute_deterministic(lead_hours, model="graphcast", year=DEFAULT_YEAR,
                          variable=DEFAULT_VARIABLE, fc_root=None, metric='mae', source='raw'):
    """Deterministic skill (mae / bias / spearmanr / pearsonr) of a single-valued forecast
    against ENACTS, summarised into the UI annual/monthly file. Works for precipitation
    (mm/hr) and temperature_max (°C); the per-variable spec picks the obs/forecast vars.

    ``source='raw'``        scores the harmonized raw forecast (``raw_fc_var``). The valid
                            date is ``time + prediction_timedelta`` — the day the forecast's
                            daily window covers, the offset that maximises forecast/obs
                            correlation at every lead.
    ``source='calibrated'`` scores the IDR calibrated median (``q50``) on the q50's own
                            ``time`` axis (the calibration puts that on the valid date; no
                            verification-side shift is applied).
    """
    spec = _VARIABLE[variable]
    if source == 'raw':
        fc = load_forecast(lead_hours, fc_root or model_fc_root(model, year, variable, raw=True))
        fc_da = fc[spec["raw_fc_var"]]
        # Valid date = init + prediction_timedelta (the day the daily window covers).
        ptd = fc_da["prediction_timedelta"].values.reshape(-1)
        if ptd.size == 0:                  # e.g. day-9: window runs past the horizon
            raise FileNotFoundError(f"empty prediction_timedelta at {lead_hours}h")
        fc_dates = (fc_da["time"].values + ptd[0]).astype('datetime64[D]')
    elif source == 'calibrated':
        fc = load_forecast(lead_hours, fc_root or model_fc_root(model, year, variable))
        fc_da = fc["q50"]
        fc_dates = fc_da["time"].values.astype('datetime64[D]')
    else:
        raise ValueError(f"unknown source {source!r}; expected 'raw' or 'calibrated'")
    if "prediction_timedelta" in fc_da.dims:
        fc_da = fc_da.squeeze("prediction_timedelta", drop=True)

    obs = _load_gridded_obs(variable, year)
    # Snap obs lat/lon onto the forecast grid: the same ENACTS grid can differ in the last
    # float digit (precip bit-matches; temperature does not), which would misalign the
    # subtraction and silently NaN most cells.
    if (obs.sizes.get("lat"), obs.sizes.get("lon")) == (fc_da.sizes.get("lat"), fc_da.sizes.get("lon")):
        obs = obs.assign_coords(lat=fc_da["lat"], lon=fc_da["lon"])
    _, io, ifc = np.intersect1d(obs["time"].values.astype('datetime64[D]'),
                                fc_dates, return_indices=True)
    # Pair positionally, then put the forecast on the obs (valid) time axis so the
    # arithmetic aligns 1:1 and the monthly grouping uses the valid date.
    fc_da = fc_da.isel(time=ifc).assign_coords(time=obs["time"].values[io])
    obs_da = obs.isel(time=io)

    if metric in ('mae', 'bias'):
        field = np.abs(fc_da - obs_da) if metric == 'mae' else fc_da - obs_da
        ds = annual_monthly_averages(field, band_masks=band_masks_for(field))
    elif metric in ('spearmanr', 'pearsonr'):
        # The pair is already on a shared valid-date axis; corr_annual_monthly_averages
        # adds prediction_timedelta then drops it, so a zero shift leaves it in place.
        # (No altitude bands — the correlation reducer is not a spatial mean.)
        fc_ptd0 = fc_da.expand_dims(prediction_timedelta=[np.timedelta64(0, "ns")])
        ds = corr_annual_monthly_averages(fc_ptd0, obs_da, corr_type=metric)
    else:
        raise ValueError(f"unsupported deterministic metric {metric!r}")

    units = fc_da.attrs.get("units", "")
    for v in ds.data_vars:
        if v != "altitude_band":          # int8 index layer, not a physical quantity
            ds[v].attrs["units"] = units
    ds.attrs["lead_hours"] = lead_hours
    return ds


def compute_deterministic_station(lead_hours, model="graphcast", year=DEFAULT_YEAR,
                                  fc_root=None, obs_dir=_STATION_OBS_DIR, metric='mae', source='raw'):
    """Deterministic precip skill (mae / bias) at gauge stations — the station counterpart of
    :func:`compute_deterministic`, on the (time, station) grid.

    ``source='raw'``        scores the forecast-at-station; its valid date is
                            ``time + prediction_timedelta`` (``align_stations`` then floors
                            it to the day the gauge's window covers).
    ``source='calibrated'`` scores the IDR ``q50`` from the station tree (already valid-date).

    Forecast and gauge are paired by ``(name, gauge-lat, gauge-lon)`` + date via
    :func:`align_stations` and reduced over the ``station`` axis. Both are mm/hr (gauge ÷24).
    Correlation is not supported at stations (the gridded corr reducer assumes two spatial
    dims); use mae / bias.
    """
    from calibration.run_station import align_stations   # (name, lat, lon) + date pairing

    if source == 'raw':
        fc = load_forecast(lead_hours, fc_root or station_raw_fc_root(model, year))
        da = fc["precipitation"]
        ptd = da["prediction_timedelta"].values.reshape(-1)
        if ptd.size == 0:                  # e.g. day-9: window runs past the horizon
            raise FileNotFoundError(f"empty prediction_timedelta at {lead_hours}h")
        if "prediction_timedelta" in da.dims:
            da = da.squeeze("prediction_timedelta", drop=True)
        da = da.assign_coords(time=da["time"] + ptd[0])   # init -> valid; align_stations floors
    elif source == 'calibrated':
        fc = load_forecast(lead_hours, fc_root or station_fc_root(model, year))
        da = fc["q50"]
    else:
        raise ValueError(f"unknown source {source!r}; expected 'raw' or 'calibrated'")
    if metric not in ('mae', 'bias'):
        raise ValueError(f"station deterministic supports mae/bias, not {metric!r}")

    units = da.attrs.get("units", "")
    obs = load_obs_station(year, obs_dir)["total_precipitation_rate"] / 24.0   # mm/day -> mm/hr
    fc_al, obs_al = align_stations(da, obs)

    diff = fc_al - obs_al if metric == 'bias' else np.abs(fc_al - obs_al)
    # Band masks need a per-station elevation, taken from whichever side of the
    # aligned pair carries it: the calibrated forecast has its own `elevation`
    # coord, the raw forecast-at-station does not — there the gauge obs supplies
    # it (same station order as diff). Arithmetic can drop the coord, so don't
    # rely on diff carrying it.
    elev_src = fc_al if "elevation" in fc_al.coords else obs_al
    ds = annual_monthly_averages(diff, band_masks=band_masks_for(elev_src))
    for v in ds.data_vars:
        if v != "altitude_band":          # int8 index layer, not a physical quantity
            ds[v].attrs["units"] = units
    ds.attrs["lead_hours"] = lead_hours
    return ds


def plot_spatial(ds, out_png, metric="CRPS"):
    """Save a seasonal metric preview (domain-mean CRPS per month)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    monthly = np.atleast_1d(ds["monthly_average"].values.squeeze())
    months = ds["month"].values if "month" in ds.coords else np.arange(1, len(monthly) + 1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(months, monthly, marker="o", lw=2, color="#0a9938")
    ax.set_xlabel("Month")
    ax.set_ylabel(f"{metric} ({ds['monthly_average'].attrs.get('units', '')})")
    ax.set_title(f"{metric} seasonal cycle — {ds.attrs.get('lead_hours')}h")
    ax.set_xticks(range(1, 13))
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    return out_png
