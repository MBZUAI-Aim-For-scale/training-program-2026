"""Rainy-season onset: definitions, detection, and the observed + forecast series.

Definitions (from Narayana's note, matching the ROMP benchmarking package):

  ICPAC (Ethiopia; used across Africa)
      wet day >= 1 mm. Onset is the first day of a 3-day wet spell totalling
      more than 20 mm that is not followed by a dry spell of 7 or more days
      (each < 1 mm) within the next 21 days.
  Moron–Robertson (India)
      after 1 April, the first day of the first 5-day wet spell (each day >= 1 mm)
      whose total exceeds the local climatological 5-day wet-spell amount, not
      followed by a 10-day dry spell (< 5 mm in total) within the next 30 days.
      The local amounts are ROMP's MWset (4° grid over India).
  Moron–Robertson with MOK filter (India)
      the first such wet spell on or after 2 June (the 1901–2024 median monsoon
      onset over Kerala), with no dry-spell check. This is the version used to
      evaluate forecasts, since it does not need 30 days of rain after onset.

How a forecast gets an onset date: observed daily rain (CHIRPS) from the start of
the search window up to the day before the forecast starts, then the forecast's
daily rain. A wet spell whose dry-spell check runs past the end of the data is
reported as "possible onset" (not yet confirmed), not as onset.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

MWSET = Path(__file__).resolve().parent / "data" / "mwset4x4.nc4"

NONE, CONFIRMED, POSSIBLE = 0, 1, 2


@dataclass(frozen=True)
class Definition:
    key: str
    name: str
    home: str                    # where the definition was designed for
    wet_day_mm: float = 1.0
    spell_days: int = 3
    spell_mm: float | None = 20.0  # None -> local climatological amount (MWset)
    dry_kind: str | None = "days"  # "days": consecutive days < wet_day_mm; "total": sum < dry_mm; None: no check
    dry_days: int = 7
    dry_mm: float = 5.0          # only for dry_kind "total"
    dry_window: int = 21         # days after the wet spell searched for a dry spell
    search_from: tuple[int, int] | None = None  # (month, day) fixed by the definition, else the region's season start
    note: str = ""

    def describe(self) -> str:
        amount = (f"more than {self.spell_mm:g} mm" if self.spell_mm is not None
                  else "more than the local climatological amount")
        s = (f"First day of a {self.spell_days}-day wet spell (every day ≥ {self.wet_day_mm:g} mm) "
             f"totalling {amount}")
        if self.dry_kind == "days":
            s += (f", not followed by {self.dry_days} or more consecutive dry days "
                  f"(< {self.wet_day_mm:g} mm) within the next {self.dry_window} days")
        elif self.dry_kind == "total":
            s += (f", not followed by a {self.dry_days}-day dry spell (< {self.dry_mm:g} mm in total) "
                  f"within the next {self.dry_window} days")
        else:
            s += "; no dry-spell check"
        if self.search_from:
            s += f". Search starts {pd.Timestamp(2001, *self.search_from):%-d %B}"
        return s + "."

    @property
    def needs_after(self) -> int:
        """Days of rain needed after a wet spell starts to confirm it."""
        return self.spell_days + (self.dry_window if self.dry_kind else 0)


DEFINITIONS = {
    "icpac": Definition("icpac", "ICPAC", "Ethiopia; applied across Africa",
                        spell_days=3, spell_mm=20.0, dry_kind="days", dry_days=7, dry_window=21),
    "mr": Definition("mr", "Moron–Robertson", "India",
                     spell_days=5, spell_mm=None, dry_kind="total", dry_days=10, dry_mm=5.0,
                     dry_window=30, search_from=(4, 1)),
    "mr_mok": Definition("mr_mok", "Moron–Robertson with MOK filter", "India",
                         spell_days=5, spell_mm=None, dry_kind=None, search_from=(6, 2),
                         note="2 June is the median monsoon onset over Kerala, 1901–2024."),
}

# Start of the onset search window for each region's main rainy season(s), as
# (month, day). Two entries = two seasons; the latest one before the forecast
# start is used. To be checked against Narayana's country/phenomenon table.
SEASON_STARTS = {
    "Ethiopia": [(5, 1)], "Kenya": [(3, 1), (10, 1)], "Nigeria": [(3, 1)],
    "Rwanda": [(2, 15), (9, 1)], "Senegal": [(5, 1)], "Sierra Leone": [(4, 1)],
    "Bangladesh": [(4, 1)], "Indonesia": [(8, 1)], "Pakistan": [(6, 1)],
    "Philippines": [(4, 15)], "Chile": [(4, 1)], "Colombia": [(3, 1), (9, 1)],
    "Peru": [(9, 1)], "India": [(4, 1)],
}
MR_REGIONS = ("India", "Bangladesh")  # inside ROMP's MWset thresholds


def default_definition(region_name: str) -> str:
    return "mr_mok" if region_name in MR_REGIONS else "icpac"


def search_start(defn: Definition, region_name: str, init: pd.Timestamp) -> pd.Timestamp:
    """Latest season start on or before the forecast start (the definition's own date if it has one)."""
    starts = [defn.search_from] if defn.search_from else SEASON_STARTS.get(region_name, [(4, 1)])
    cands = [pd.Timestamp(y, m, d) for y in (init.year - 1, init.year) for m, d in starts]
    cands = [c for c in cands if c <= init.normalize()]
    return max(cands) if cands else pd.Timestamp(init.year, *starts[0])


def local_threshold(lat, lon) -> xr.DataArray:
    """ROMP's MWset 5-day wet-spell amounts, nearest 4° cell; NaN outside India."""
    with xr.open_dataset(MWSET) as ds:
        mw = ds.MWmean.transpose("lat", "lon").load()
    lat, lon = np.asarray(lat), np.asarray(lon)
    near = mw.sel(lat=lat, lon=lon, method="nearest")
    far = ((np.abs(near.lat.values - lat) > 2.01)[:, None]
           | (np.abs(near.lon.values - lon) > 2.01)[None, :])
    vals = np.where(far, np.nan, near.values)
    return xr.DataArray(vals, coords={"lat": lat, "lon": lon}, dims=("lat", "lon"), name="threshold")


def threshold_for(defn: Definition, rain: xr.DataArray) -> xr.DataArray:
    if defn.spell_mm is not None:
        return xr.full_like(rain.isel(time=0, drop=True), defn.spell_mm, dtype=float)
    return local_threshold(rain.lat, rain.lon)


def _windows(x: np.ndarray, n: int) -> np.ndarray:
    """Sum over each n-day window starting at each day: out[t] = x[t:t+n].sum(); NaN past the end."""
    T = x.shape[0]
    c = np.concatenate([np.zeros((1,) + x.shape[1:]), np.cumsum(x, axis=0)])
    out = np.full(x.shape, np.nan)
    if T >= n:
        out[:T - n + 1] = c[n:] - c[:-n]
    return out


def detect(rain: np.ndarray, thresh: np.ndarray, defn: Definition, start: int = 0):
    """First onset at or after day index `start` for every point.

    rain    (T, ...) daily mm; NaN = no data (never wet, never dry)
    thresh  (...) wet-spell amount; NaN = no onset possible here
    Returns (index (-1 = none), status NONE/CONFIRMED/POSSIBLE), both shaped (...).
    """
    T = rain.shape[0]
    shape = rain.shape[1:]
    idx = np.full(shape, -1)
    status = np.zeros(shape, dtype=np.int8)
    if T == 0:
        return idx, status
    have = ~np.isnan(rain)
    r = np.nan_to_num(rain)
    n = defn.spell_days
    wet_all = _windows((have & (r >= defn.wet_day_mm)).astype(float), n) == n
    spell_ok = wet_all & (_windows(r, n) > thresh)
    if defn.dry_kind:
        k = defn.dry_days
        if defn.dry_kind == "days":
            dry_start = _windows((have & (r < defn.wet_day_mm)).astype(float), k) == k
        else:
            dry_start = (_windows(have.astype(float), k) == k) & (_windows(r, k) < defn.dry_mm)
        # prefix count of dry-spell starts, to ask "any dry spell starting in [a, b]?"
        pc = np.concatenate([np.zeros((1,) + shape), np.cumsum(dry_start, axis=0)])
    undecided = np.ones(shape, dtype=bool)
    for t in range(start, T - n + 1):
        cand = undecided & spell_ok[t]
        if not cand.any():
            continue
        if not defn.dry_kind:
            idx[cand], status[cand] = t, CONFIRMED
            undecided &= ~cand
            continue
        a = t + n                          # first day after the wet spell
        b_full = a + defn.dry_window       # window [a, b_full)
        b = min(b_full, T)
        last_start = b - k                 # dry spell must fit inside the window
        dry = (pc[last_start + 1] - pc[a] > 0) if last_start >= a else np.zeros(shape, bool)
        complete = b_full <= T
        ok = cand & ~dry
        st = CONFIRMED if complete else POSSIBLE
        idx[ok], status[ok] = t, st
        undecided &= ~ok
        if not undecided.any():
            break
    return idx, status


def onset_dates(rain: xr.DataArray, defn: Definition, search_from: pd.Timestamp,
                thresh: xr.DataArray | None = None) -> xr.Dataset:
    """Onset date and status per point (and per member if rain has 'ensemble')."""
    thresh = threshold_for(defn, rain) if thresh is None else thresh
    times = pd.DatetimeIndex(rain.time.values)
    start = int(np.searchsorted(times, search_from))
    core = rain.transpose("time", ...)
    idx, status = detect(core.values.astype(float), np.broadcast_to(
        thresh.broadcast_like(core.isel(time=0, drop=True)).values, core.shape[1:]), defn, start)
    dates = np.where(idx >= 0, times.values[np.clip(idx, 0, None)], np.datetime64("NaT"))
    dims, coords = core.dims[1:], {d: core[d] for d in core.dims[1:]}
    return xr.Dataset({"onset": (dims, dates.astype("datetime64[ns]")),
                       "status": (dims, status)}, coords=coords)


# ── observed + forecast series ───────────────────────────────────────────────

def forecast_daily(ds: xr.Dataset, init: pd.Timestamp) -> xr.DataArray:
    """Forecast daily rain (mm/day) dated by forecast day: day 1 = the start date."""
    from .results import daily_precip
    d = daily_precip(ds)
    times = init.normalize() + pd.to_timedelta(d.day.values - 1, unit="D")
    return d.assign_coords(day=times).rename(day="time")


def to_grid(fc: xr.DataArray, like: xr.DataArray) -> xr.DataArray:
    """Model rain on the observation grid (linear; nearest at the edges), masked like the obs."""
    out = fc.interp(lat=like.lat, lon=like.lon)
    near = fc.interp(lat=like.lat, lon=like.lon, method="nearest",
                     kwargs={"fill_value": None})
    out = out.fillna(near)
    land = like.isel(time=0, drop=True).notnull() if "time" in like.dims else like.notnull()
    return out.where(land)


def joined(obs: xr.DataArray, fc: xr.DataArray, init: pd.Timestamp,
           search_from: pd.Timestamp) -> xr.DataArray:
    """Observed rain from search_from to the day before init, then the forecast.
    Days with no observation in between are NaN."""
    before = pd.date_range(search_from, init.normalize() - pd.Timedelta(days=1), freq="D")
    past = obs.reindex(time=before) if len(before) else obs.isel(time=slice(0, 0))
    if "ensemble" in fc.dims:
        past = past.expand_dims(ensemble=fc.ensemble.values)
    return xr.concat([past, fc.transpose(*past.dims)], "time")


def area_mean(da: xr.DataArray) -> xr.DataArray:
    w = np.cos(np.deg2rad(da.lat))
    return da.weighted(w).mean(["lat", "lon"], skipna=True)


def custom(defn: Definition, **changes) -> Definition:
    return replace(defn, **{k: v for k, v in changes.items() if v is not None})
