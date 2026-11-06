"""Observations: daily rainfall from CHIRPS v3.0 (Precipitation and Onset) and daily maximum
temperature from CHIRTS (Heat), both from UCSB's Climate Hazards Center.

Rainfall: CHIRPS v3.0 (IMERG-disaggregated, "sat"). CHIRPS is satellite + rain gauge
rainfall over land, 60°S–60°N.

  * Programme countries: yearly country files in the training bucket, on the models' 0.25°
    grid (cells centred on whole and quarter degrees, as GraphCast's and the bucket's
    forecasts are): gs://aim4scale_training_2026/CHIRPSv3_Country_0p25deg, from 2000.
  * Anywhere else (India, custom boxes), or if neither this Spark nor the bucket has them:
    CHC's global 0.25° daily GeoTIFFs (~0.9 MB a day), on CHIRPS's own grid (cells half a
    cell off the models'), downloaded into {OUTPUT_DIR}/_obs/chirps_v3/p25/.

Temperature: CHIRTS daily Tmax (CHIRTS-ERA5), land only, from yearly country files on the
same 0.25° grid (gs://aim4scale_training_2026/CHIRTS_Country_0p25deg/Tmax, from 2000, in K:
returned in °C). Programme countries only: anywhere else the page falls back to ERA5.

Country files are read from this Spark's country data first ($DEMO_COUNTRY_DATA, default
/country_data: docker/run.sh mounts what Demo 1's country-data button downloaded, the bucket's
own paths), and from the bucket otherwise (downloaded once into {OUTPUT_DIR}/_obs/*_0p25/).
The current year's file is fetched from the bucket (re-fetched when older than 7 days, as
CHC adds a month at a time), and the country data's copy is used if the bucket can't be read.

Whatever the source, one series is all from one source (never some years on one grid and
some on another), and every model is put on its grid (onset.to_grid).
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

BUCKET = os.environ.get("DEMO3_OBS_BUCKET", "gs://aim4scale_training_2026").rstrip("/")
# This Spark's country data: the bucket's own paths (docker/run.sh mounts it read-only)
COUNTRY_DATA = Path(os.environ.get("DEMO_COUNTRY_DATA", "/country_data"))
TIF_URL = "https://data.chc.ucsb.edu/products/CHIRPS/v3.0/daily/final/sat/p25/{y}/"
CHIRPS_DIR = "CHIRPSv3_Country_0p25deg"   # (the models' 0.25° grid)
# New folders: files of the same names on CHIRPS's own grid may be in the old ones
CACHE = OUTPUT_DIR / "_obs" / "chirps_v3_0p25"
TIF_CACHE = OUTPUT_DIR / "_obs" / "chirps_v3" / "p25"
COUNTRIES = ("Bangladesh", "Bolivia", "Chile", "Colombia", "Ethiopia", "Indonesia", "Kenya", "Nigeria",
             "Pakistan", "Peru", "Philippines", "Rwanda", "Senegal", "Sierra Leone")
LAT_LIMIT = 60.0
FIRST_YEAR = 2000  # (the 0.25° country files' first year)
NAME = "CHIRPS v3"

TMAX_DIR = "CHIRTS_Country_0p25deg/Tmax"
TMAX_CACHE = OUTPUT_DIR / "_obs" / "chirts_0p25"
TMAX_FIRST_YEAR = 2000
TMAX_NAME = "CHIRTS"


def _stale(path: Path, year: int, refresh_days: float = 7) -> bool:
    return (year >= pd.Timestamp.utcnow().year and path.exists()
            and time.time() - path.stat().st_mtime > refresh_days * 86400)


# ── country boxes in the bucket ──────────────────────────────────────────────

def _bucket_file(folder: str, cache: Path, c: str, name: str, year: int) -> Path:
    """A country file: this Spark's copy (past years), else the bucket's (downloaded once into
    `cache`; the current year's re-fetched when stale), else this Spark's copy after all."""
    local = COUNTRY_DATA / folder / c / name
    current = year >= pd.Timestamp.utcnow().year
    if local.is_file() and not current:
        return local
    path = cache / c / name
    if not path.exists() or _stale(path, year):
        import fsspec
        fs, root = fsspec.core.url_to_fs(BUCKET)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        try:
            fs.get(f"{root}/{folder}/{c}/{name}", str(tmp))
            tmp.replace(path)
        except Exception as e:
            tmp.unlink(missing_ok=True)
            if not path.exists():
                if local.is_file():
                    return local
                raise OSError(f"couldn't get {name} from the bucket: {e}") from e
    return path


def _country_file(country: str, year: int) -> Path:
    c = country.replace(" ", "_")
    return _bucket_file(CHIRPS_DIR, CACHE, c, f"CHIRPSv3_{c}_{year}_box025.nc", year)


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
    path = TIF_CACHE / f"{day:%Y}" / name
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
                    day = _newest_day(ds)
                if day is not None:
                    return day
            except (OSError, ValueError):
                pass
        days = _tif_days(y)
        if days:
            return days[-1]
    return None


def _dates(start, end) -> tuple[pd.Timestamp, pd.Timestamp]:
    """start..end as days; missing dates mean there's nothing to look up (before any file)."""
    start = pd.Timestamp(start) if start is not None else pd.NaT
    end = pd.Timestamp(end) if end is not None else pd.NaT
    if pd.isna(start) or pd.isna(end):
        raise FileNotFoundError(f"{NAME} observation dates are unavailable.")
    start, end = start.normalize(), end.normalize()
    if start > end:
        raise ValueError("Observation start date must not be after the end date.")
    return start, end


def _newest_day(ds: xr.Dataset) -> pd.Timestamp | None:
    """The file's last valid day (None for an empty or invalid calendar)."""
    if "time" not in ds:
        return None
    dates = pd.DatetimeIndex(ds.time.values).dropna()
    return pd.Timestamp(dates.max()).normalize() if not dates.empty else None


def daily(box: Box, start, end, region: str | None = None) -> xr.DataArray:
    """Daily rainfall (mm/day), dims (time, lat, lon), for start..end inclusive.
    Days CHIRPS doesn't have yet are simply absent. A programme country's data come
    from its bucket box, which is cut close to the border (the plot's margin is empty)."""
    start, end = _dates(start, end)
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
    return _bucket_file(TMAX_DIR, TMAX_CACHE, c, f"CHIRTS-ERA5_Tmax_{c}_{year}_box025.nc", year)


def _tmax_var(ds: xr.Dataset) -> xr.DataArray:
    """The file's daily Tmax in °C (the 0.25° files have it as `tmax`, in K)."""
    t = ds["tmax"] if "tmax" in ds else ds["temp"]
    if str(t.attrs.get("units", "")).strip().upper() in ("K", "KELVIN", "DEGK"):
        t = t - 273.15
    return t


def _tmax_year(country: str, year: int, box: Box | None = None, start=None, end=None) -> xr.DataArray:
    with xr.open_dataset(_tmax_file(country, year)) as ds:
        da = _tmax_var(ds)
        da = da.assign_coords(lon=(da.lon + 180) % 360 - 180).sortby("lat").sortby("lon")
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
                t = _tmax_var(ds)
                some = t.notnull().any(["lat", "lon"])  # (a 0.25° country box is small enough to scan whole)
                days = t.time.values[some.values]
                if len(days):
                    return pd.Timestamp(days[-1])
        except (OSError, ValueError):
            pass
    return None


def tmax_daily(box: Box, start, end, region: str) -> xr.DataArray:
    """Daily maximum 2 m temperature (°C), dims (time, lat, lon), for start..end inclusive,
    land only. Days CHIRTS doesn't have yet are absent."""
    start, end = _dates(start, end)
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
