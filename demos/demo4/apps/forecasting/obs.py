"""Daily observed rainfall for onset: CHIRPS v2.0 at 0.25°, read from yearly files.

CHIRPS is satellite + rain gauge rainfall over land, 50°S–50°N, and is what ICPAC
and the ROMP benchmarking package use for onset. One file per year (~90 MB) is
downloaded once into {OUTPUT_DIR}/_obs/chirps/ (or found in $DEMO3_DATA/CHIRPS if
preloaded) and every region is cut from it. The current year's file is updated
by CHC about once a month, so it is re-downloaded when it is older than 7 days.
"""
from __future__ import annotations

import os
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .regions import Box
from .store import OUTPUT_DIR

URL = "https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_daily/netcdf/p25/chirps-v2.0.{year}.days_p25.nc"
CACHE = OUTPUT_DIR / "_obs" / "chirps"
PRELOADED = Path(os.environ.get("DEMO3_DATA", "/data")) / "CHIRPS"
LAT_LIMIT = 50.0
FIRST_YEAR = 1981
NAME = "CHIRPS"


def _file(year: int, refresh_days: float = 7) -> Path:
    name = f"chirps-v2.0.{year}.days_p25.nc"
    if (PRELOADED / name).exists():
        return PRELOADED / name
    path = CACHE / name
    stale = (year >= pd.Timestamp.utcnow().year and path.exists()
             and time.time() - path.stat().st_mtime > refresh_days * 86400)
    if not path.exists() or stale:
        CACHE.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        try:
            urllib.request.urlretrieve(URL.format(year=year), tmp)
            tmp.replace(path)
        except OSError:
            tmp.unlink(missing_ok=True)
            if not path.exists():
                raise
    return path


def covers(box: Box) -> bool:
    return box.lat_min < LAT_LIMIT and box.lat_max > -LAT_LIMIT


def last_day() -> pd.Timestamp | None:
    """Last day CHIRPS has data for (from the newest yearly file we can get)."""
    year = pd.Timestamp.utcnow().year
    for y in (year, year - 1):
        try:
            with xr.open_dataset(_file(y)) as ds:
                if "time" not in ds:
                    continue
                dates = pd.DatetimeIndex(ds.time.values).dropna()
                if not dates.empty:
                    return pd.Timestamp(dates.max()).normalize()
        except (OSError, ValueError):
            continue
    return None


def daily(box: Box, start, end) -> xr.DataArray:
    """Daily rainfall (mm/day), dims (time, lat, lon), for start..end inclusive.
    Days CHIRPS doesn't have yet are simply absent."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if pd.isna(start) or pd.isna(end):
        raise FileNotFoundError(f"{NAME} observation dates are unavailable.")
    start, end = start.normalize(), end.normalize()
    if start > end:
        raise ValueError("Observation start date must not be after the end date.")
    parts = []
    for y in range(start.year, end.year + 1):
        try:
            path = _file(y)
        except OSError:
            continue
        with xr.open_dataset(path) as ds:
            da = ds.precip.rename(latitude="lat", longitude="lon")
            da = da.sel(time=slice(max(start, pd.Timestamp(y, 1, 1)), min(end, pd.Timestamp(y, 12, 31))),
                        lat=slice(box.lat_min, box.lat_max), lon=slice(box.lon_min, box.lon_max))
            if all(da.sizes[dim] > 0 for dim in ("time", "lat", "lon")):
                parts.append(da.load())
    if not parts:
        raise FileNotFoundError(f"no {NAME} data for {start:%Y-%m-%d}..{end:%Y-%m-%d}")
    out = xr.concat(parts, "time") if len(parts) > 1 else parts[0]
    out = out.where(out >= 0)  # missing values (sea) are negative in some files
    out.name, out.attrs = "precip", {"units": "mm/day", "source": NAME}
    return out.astype(np.float32)
