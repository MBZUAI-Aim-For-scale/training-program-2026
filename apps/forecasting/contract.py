"""Output format shared by every model runner and every visualization.

One NetCDF file per run, global field (regions are cropped at plot time):

    dims      (ensemble, lead_time, lat, lon)
    ensemble  0..N-1 (size 1 for deterministic models)
    lead_time integer hours since init, 6-hourly, starting at 0
    lat       ascending, regular grid
    lon       -180..180, ascending
    tp        total precipitation accumulated over the preceding 6 h [mm]
              (NaN at lead 0)
    t2m       2 m temperature [K]
    z500      geopotential at 500 hPa [m2 s-2]

Coordinates also carry `valid_time` (datetime64 along lead_time) and a scalar
`init_time`. Global attributes: model, model_name, init_source, members,
synthetic (0/1), created.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import xarray as xr

VARIABLES = {
    "tp": ("Total precipitation (6 h accumulation)", "mm"),
    "t2m": ("2 m temperature", "K"),
    "z500": ("Geopotential at 500 hPa", "m2 s-2"),
}


def standardize(ds: xr.Dataset) -> xr.Dataset:
    """Put a runner's output into the shared layout (names and units must already match)."""
    if "lon" in ds.coords and float(ds.lon.max()) > 180:
        ds = ds.assign_coords(lon=((ds.lon + 180) % 360) - 180)
    ds = ds.sortby("lat").sortby("lon")
    if "ensemble" not in ds.dims:
        ds = ds.expand_dims(ensemble=[0])
    return ds.transpose("ensemble", "lead_time", "lat", "lon")


def finalize(ds: xr.Dataset, *, model_key: str, model_name: str, init: pd.Timestamp,
             init_source: str, synthetic: bool = False) -> xr.Dataset:
    ds = standardize(ds)
    if not set(VARIABLES) & set(ds.data_vars):
        raise ValueError("runner output contains none of the expected variables")
    # A model that doesn't forecast a variable (e.g. no precipitation) gets NaNs.
    missing = sorted(set(VARIABLES) - set(ds.data_vars))
    template = ds[next(v for v in VARIABLES if v in ds.data_vars)]
    for v in missing:
        ds[v] = xr.full_like(template, np.nan)
    ds = ds[list(VARIABLES)]
    # Drop coordinates outside the format (e.g. source-specific time variables).
    ds = ds.drop_vars([c for c in ds.coords if c not in ("ensemble", "lead_time", "lat", "lon")])
    for v, (long_name, units) in VARIABLES.items():
        ds[v].attrs.update(long_name=long_name, units=units)
    ds["lead_time"] = ds.lead_time.astype("int32")
    # Plain "hours" (not "hours since …") so xarray never decodes it as a date.
    ds.lead_time.attrs.update(units="hours", long_name="hours after init_time")
    ds = ds.assign_coords(
        valid_time=("lead_time", (init + pd.to_timedelta(ds.lead_time.values, unit="h")).values),
        init_time=np.datetime64(init.to_datetime64()),
    )
    ds.attrs.update(
        model=model_key, model_name=model_name, init_source=init_source,
        members=int(ds.sizes["ensemble"]), synthetic=int(synthetic),
        missing_variables=",".join(missing),
        created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    return ds


def write(ds: xr.Dataset, path) -> None:
    """Write atomically: a failed write never leaves a half-written run behind."""
    from pathlib import Path
    path = Path(path)
    tmp = path.with_name(path.name + ".partial")
    enc = {v: {"zlib": True, "complevel": 3, "dtype": "float32"} for v in VARIABLES}
    try:
        ds.to_netcdf(tmp, encoding=enc)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)
