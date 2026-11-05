"""Observations: daily rainfall from CHIRPS v3.0 (Precipitation and Onset) and daily maximum
temperature from CHIRTS (Heat), both from UCSB's Climate Hazards Center.

Rainfall: CHIRPS v3.0 (IMERG-disaggregated, "sat") at 0.25°.

CHIRPS is satellite + rain gauge rainfall over land, 60°S–60°N. The "sat" version
spreads each pentad's total over its days using NASA IMERG, so it starts in 1998.
Two sources, same data on the same grid:

  * Programme countries: yearly country boxes in the training bucket
    (gs://aim4scale_training_2026/CHIRPSv3_Country_Boxes_025deg, ~3 MB a year),
    downloaded once into {OUTPUT_DIR}/_obs/chirps_v3/.
  * Anywhere else (India, custom boxes), or if the bucket can't be read: CHC's
    global 0.25° daily GeoTIFFs (~0.9 MB a day), downloaded into the same folder.

The current year is updated by CHC about once a month, so current-year files are
re-fetched when they are older than 7 days. $DEMO3_DATA/CHIRPSv3 is checked first
for preloaded country files.

Temperature: CHIRTS daily Tmax (CHIRTS-ERA5) at 0.05°, land only, from yearly country boxes
in the training bucket (gs://aim4scale_training_2026/CHIRTS_Tmax_Country_Boxes, 3-100 MB a
year), downloaded once into {OUTPUT_DIR}/_obs/chirts/ ($DEMO3_DATA/CHIRTS checked first).
Programme countries only: anywhere else the page falls back to ERA5.
"""
from __future__ import annotations

import os
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .regions import Box
from .store import OUTPUT_DIR

BUCKET = os.environ.get("DEMO3_CHIRPS_BUCKET",
                        "gs://aim4scale_training_2026/CHIRPSv3_Country_Boxes_025deg").rstrip("/")
TIF_URL = "https://data.chc.ucsb.edu/products/CHIRPS/v3.0/daily/final/sat/p25/{y}/"
CACHE = OUTPUT_DIR / "_obs" / "chirps_v3"
PRELOADED = Path(os.environ.get("DEMO3_DATA", "/data")) / "CHIRPSv3"
COUNTRIES = ("Bangladesh", "Bolivia", "Chile", "Colombia", "Ethiopia", "Indonesia", "Kenya", "Nigeria",
             "Pakistan", "Peru", "Philippines", "Rwanda", "Senegal", "Sierra Leone")
LAT_LIMIT = 60.0
FIRST_YEAR = 1998
NAME = "CHIRPS v3"

TMAX_BUCKET = os.environ.get("DEMO3_CHIRTS_BUCKET",
                             "gs://aim4scale_training_2026/CHIRTS_Tmax_Country_Boxes").rstrip("/")
TMAX_CACHE = OUTPUT_DIR / "_obs" / "chirts"
TMAX_PRELOADED = Path(os.environ.get("DEMO3_DATA", "/data")) / "CHIRTS"
TMAX_FIRST_YEAR = 1981
TMAX_NAME = "CHIRTS"


def _stale(path: Path, year: int, refresh_days: float = 7) -> bool:
    return (year >= pd.Timestamp.utcnow().year and path.exists()
            and time.time() - path.stat().st_mtime > refresh_days * 86400)


# ── country boxes in the bucket ──────────────────────────────────────────────

def _bucket_file(bucket: str, cache: Path, preloaded: Path, c: str, name: str, year: int) -> Path:
    if (preloaded / c / name).exists():
        return preloaded / c / name
    path = cache / c / name
    if not path.exists() or _stale(path, year):
        import fsspec
        fs, root = fsspec.core.url_to_fs(bucket)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        try:
            fs.get(f"{root}/{c}/{name}", str(tmp))
            tmp.replace(path)
        except Exception as e:
            tmp.unlink(missing_ok=True)
            if not path.exists():
                raise OSError(f"couldn't get {name} from the bucket: {e}") from e
    return path


def _country_file(country: str, year: int) -> Path:
    c = country.replace(" ", "_")
    return _bucket_file(BUCKET, CACHE, PRELOADED, c, f"CHIRPSv3_{c}_{year}_box025.nc", year)


def _from_bucket(country: str, box: Box, start, end) -> xr.DataArray:
    parts = []
    for y in range(start.year, end.year + 1):
        try:
            path = _country_file(country, y)
        except OSError:
            if y < pd.Timestamp.utcnow().year:
                raise
            continue  # current year not uploaded yet
        with xr.open_dataset(path) as ds:
            da = ds.precip.assign_coords(lon=(ds.lon + 180) % 360 - 180).sortby("lat").sortby("lon")
            parts.append(da.sel(time=slice(start, end), lat=slice(box.lat_min, box.lat_max),
                                lon=slice(box.lon_min, box.lon_max)).load())
    return xr.concat(parts, "time") if parts else None


# ── CHC global daily GeoTIFFs ────────────────────────────────────────────────

_listing: dict = {}


def _tif_days(year: int) -> list[pd.Timestamp]:
    """Days CHC has a 0.25° file for in this year (directory listing, cached for an hour)."""
    hit = _listing.get(year)
    if hit and time.time() - hit[0] < 3600:
        return hit[1]
    try:
        with urllib.request.urlopen(TIF_URL.format(y=year), timeout=30) as r:
            html = r.read().decode()
        days = sorted({pd.Timestamp(*map(int, m))
                       for m in re.findall(r"chirps-v3\.0\.sat\.(\d{4})\.(\d\d)\.(\d\d)\.tif", html)})
    except OSError:
        days = []
    _listing[year] = (time.time(), days)
    return days


def _tif(day: pd.Timestamp) -> Path:
    name = f"chirps-v3.0.sat.{day:%Y.%m.%d}.tif"
    path = CACHE / "p25" / f"{day:%Y}" / name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        try:
            urllib.request.urlretrieve(TIF_URL.format(y=day.year) + name, tmp)
            tmp.replace(path)
        except OSError:
            tmp.unlink(missing_ok=True)
            raise
    return path


def _read_tif(path: Path, box: Box) -> xr.DataArray:
    """One day from a global CHIRPS GeoTIFF (north-up, 0.25°, 60°S–60°N), cut to the box."""
    from PIL import Image
    with Image.open(path) as im:
        a = np.array(im, dtype=np.float32)
        dx, dy = im.tag_v2[33550][:2]
        x0, y0 = im.tag_v2[33922][3:5]
    lat = y0 - dy * (np.arange(a.shape[0]) + 0.5)
    lon = x0 + dx * (np.arange(a.shape[1]) + 0.5)
    da = xr.DataArray(a, coords={"lat": lat, "lon": lon}, dims=("lat", "lon")).sortby("lat")
    return da.sel(lat=slice(box.lat_min, box.lat_max), lon=slice(box.lon_min, box.lon_max))


def _from_tifs(box: Box, start, end) -> xr.DataArray:
    have = {d for y in range(start.year, end.year + 1) for d in _tif_days(y)}
    days = [d for d in pd.date_range(start, end, freq="D") if d in have]
    if not days:
        return None
    with ThreadPoolExecutor(8) as pool:
        paths = list(pool.map(_tif, days))
    return xr.concat([_read_tif(p, box) for p in paths], pd.Index(days, name="time"))


# ── public ───────────────────────────────────────────────────────────────────

def covers(box: Box) -> bool:
    return box.lat_min < LAT_LIMIT and box.lat_max > -LAT_LIMIT


def last_day(region: str | None = None) -> pd.Timestamp | None:
    """Last day CHIRPS v3 has data for (from the bucket file for a programme country,
    else CHC's daily files)."""
    year = pd.Timestamp.utcnow().year
    for y in (year, year - 1):
        if region in COUNTRIES:
            try:
                with xr.open_dataset(_country_file(region, y)) as ds:
                    return pd.Timestamp(ds.time.values[-1])
            except (OSError, ValueError):
                pass
        days = _tif_days(y)
        if days:
            return days[-1]
    return None


def daily(box: Box, start, end, region: str | None = None) -> xr.DataArray:
    """Daily rainfall (mm/day), dims (time, lat, lon), for start..end inclusive.
    Days CHIRPS doesn't have yet are simply absent. A programme country's data come
    from its bucket box, which is cut close to the border (the plot's margin is empty)."""
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    out = None
    if region in COUNTRIES:
        try:
            out = _from_bucket(region, box, start, end)
        except OSError:
            out = None  # no bucket access: fall back to CHC
    if out is None:
        out = _from_tifs(box, start, end)
    if out is None or not out.sizes["time"]:
        raise FileNotFoundError(f"no {NAME} data for {start:%Y-%m-%d}..{end:%Y-%m-%d}")
    out = out.where(out >= 0)  # sea and missing are -9999
    out.name, out.attrs = "precip", {"units": "mm/day", "source": NAME}
    return out.astype(np.float32)


# ── CHIRTS daily maximum temperature (Heat) ─────────────────────────────────

def tmax_covers(region: str | None) -> bool:
    return region in COUNTRIES


def _tmax_file(country: str, year: int) -> Path:
    c = country.replace(" ", "_")
    return _bucket_file(TMAX_BUCKET, TMAX_CACHE, TMAX_PRELOADED, c, f"CHIRTS_Tmax_{c}_{year}_box.nc", year)


def _tmax_year(country: str, year: int, box: Box | None = None, start=None, end=None) -> xr.DataArray:
    with xr.open_dataset(_tmax_file(country, year)) as ds:
        da = ds.temp.assign_coords(lon=(ds.lon + 180) % 360 - 180).sortby("lat").sortby("lon")
        if box is not None:
            da = da.sel(lat=slice(box.lat_min, box.lat_max), lon=slice(box.lon_min, box.lon_max))
        if start is not None:
            da = da.sel(time=slice(start, end))
        return da.load()


_tmax_last: dict = {}


def tmax_last_day(region: str) -> pd.Timestamp | None:
    """Last day the CHIRTS file for a programme country has data for (the newest file is
    padded to the end of the month with empty days). Remembered for an hour."""
    hit = _tmax_last.get(region)
    if hit and time.time() - hit[0] < 3600:
        return hit[1]
    day = _tmax_last_day(region)
    _tmax_last[region] = (time.time(), day)
    return day


def _tmax_last_day(region: str) -> pd.Timestamp | None:
    year = pd.Timestamp.utcnow().year
    for y in (year, year - 1):
        try:
            with xr.open_dataset(_tmax_file(region, y)) as ds:
                t = ds.temp
                some = t.isel(lat=slice(None, None, 10), lon=slice(None, None, 10)).notnull().any(["lat", "lon"])
                days = t.time.values[some.values]
                if len(days):
                    return pd.Timestamp(days[-1])
        except (OSError, ValueError):
            pass
    return None


def tmax_daily(box: Box, start, end, region: str) -> xr.DataArray:
    """Daily maximum 2 m temperature (°C), dims (time, lat, lon), for start..end inclusive,
    land only. Days CHIRTS doesn't have yet are absent."""
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    parts = []
    for y in range(start.year, end.year + 1):
        try:
            parts.append(_tmax_year(region, y, box, start, end))
        except OSError:
            if y < pd.Timestamp.utcnow().year:
                raise
    out = xr.concat(parts, "time").dropna("time", how="all") if parts else None
    if out is None or not out.sizes["time"]:
        raise FileNotFoundError(f"no {TMAX_NAME} data for {start:%Y-%m-%d}..{end:%Y-%m-%d}")
    out.name, out.attrs = "tmax", {"units": "degC", "source": TMAX_NAME}
    return out.astype(np.float32)
