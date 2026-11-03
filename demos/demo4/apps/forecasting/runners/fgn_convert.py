"""Build FGN (WeatherNext 2) starting conditions from ECMWF open data.

Google publishes FGN-ready inputs for one date only (2024-10-07 00Z). This
script makes the same file for any date ECMWF open data covers (from
2024-03-01): the IFS analyses (step 0) at init - 6 h and init, on the 13
pressure levels FGN uses, sampled onto FGN Mini's 1 deg grid and written in
Google's layout (dims batch/time/level/lat/lon, timedelta `time` starting at
init - 6 h, `datetime` coordinate). Frames after the two inputs are NaN; FGN
only reads their shape.

What open data lacks, and what stands in for it:
  sea_surface_temperature  IFS skin temperature over the ocean, floored at
                           seawater freezing under sea ice (NaN on land, as in
                           Google's files); vs Google's 2024-10-07 file:
                           correlation 0.997, bias -0.08 K
  geopotential_at_surface, land_sea_mask
                           fixed fields, copied from Google's sample file
  total_precipitation_6hr, cyclone_*
                           FGN outputs only; written as NaN placeholders

Runs in the e2s018 environment (it has ecmwf-opendata and pygrib):
    python runners/fgn_convert.py --init 2026-05-15T00 --steps 40 \\
        --static <Google sample file> --out fgn_inputs.nc
"""
from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
G = 9.80665
SEAWATER_FREEZING = 271.46  # K, the value ERA5 / IFS use for SST under sea ice
PRESSURE = {"t": "temperature", "gh": "geopotential", "u": "u_component_of_wind",
            "v": "v_component_of_wind", "w": "vertical_velocity", "q": "specific_humidity"}
SURFACE = {"2t": "2m_temperature", "msl": "mean_sea_level_pressure",
           "10u": "10m_u_component_of_wind", "10v": "10m_v_component_of_wind",
           "skt": "sea_surface_temperature"}
STATIC = ("geopotential_at_surface", "land_sea_mask")


def _retrieve(client, when: pd.Timestamp, params, levels, target: Path) -> None:
    """Analysis (step 0) at `when`. The 06/18Z runs were stream "scda" until 2026,
    then moved to "oper"; try both."""
    streams = ["oper"] if when.hour in (0, 12) else ["oper", "scda"]
    last = None
    for stream in streams:
        req = dict(date=when.strftime("%Y%m%d"), time=when.hour, stream=stream, type="fc",
                   step=0, param=list(params))
        if levels:
            req["levelist"] = levels
        try:
            client.retrieve(target=str(target), **req)
            return
        except Exception as e:  # missing index or entries on this stream
            last = e
    raise RuntimeError(f"ECMWF open data has no {sorted(params)} analysis for {when:%Y-%m-%d %HZ}"
                       f" ({last})")


# "point" takes the 1 deg grid points; on 2024-10-07 it reproduces Google's own file
# (correlation 1.000 for every field but SST). "mean" averages each 1 deg cell.
REGRID = "point"


def _to_grid(values: np.ndarray, lats: np.ndarray, lons: np.ndarray,
             lat_out: np.ndarray, lon_out: np.ndarray) -> np.ndarray:
    """0.25 deg open-data field -> FGN's 1 deg grid (lat -90..90, lon 0..359), by
    averaging the 0.25 deg points within each 1 deg cell (cell centred on the
    1 deg point; edge points get half weight), or by taking the points."""
    da = xr.DataArray(values, dims=("lat", "lon"), coords={"lat": lats, "lon": lons % 360})
    da = da.sortby("lat").sortby("lon")
    if REGRID == "point":
        return da.sel(lat=lat_out, lon=lon_out).values.astype("float32")
    # wrap so cells at lon 0 see points at 359.5..359.75
    da = xr.concat([da.isel(lon=slice(-2, None)).assign_coords(lon=da.lon[-2:] - 360), da,
                    da.isel(lon=slice(0, 2)).assign_coords(lon=da.lon[:2] + 360)], "lon")
    w = xr.DataArray([0.5, 1, 1, 1, 0.5], dims="window")
    lon_mean = da.rolling(lon=5, center=True).construct("window").dot(w) / w.sum()
    lat_mean = lon_mean.rolling(lat=5, center=True, min_periods=3).construct("window")
    lat_w = w.where(lat_mean.notnull())
    out = (lat_mean.fillna(0) * lat_w).sum("window") / lat_w.sum("window")
    return out.sel(lat=lat_out, lon=lon_out).values.astype("float32")


def _read(path: Path, lat_out, lon_out, names: dict) -> dict:
    import pygrib
    out, grid = {}, None
    with pygrib.open(str(path)) as grbs:
        for g in grbs:
            if grid is None:
                lats, lons = g.latlons()
                grid = (lats[:, 0], lons[0, :])
            name = names[g.shortName]
            field = _to_grid(g.values.filled(np.nan) if np.ma.isMaskedArray(g.values) else g.values,
                             *grid, lat_out, lon_out)
            if g.typeOfLevel == "isobaricInhPa":
                out.setdefault(name, {})[int(g.level)] = field
            else:
                out[name] = field
    return out


def build(init: pd.Timestamp, steps: int, static_file: Path, source: str = "aws") -> xr.Dataset:
    from ecmwf.opendata import Client
    client = Client(source=source)
    static = xr.open_dataset(static_file)
    lat_out, lon_out = static.lat.values, static.lon.values
    ocean = static.land_sea_mask.values < 0.5

    frames = {}
    with tempfile.TemporaryDirectory() as tmp:
        for when in (init - pd.Timedelta(hours=6), init):
            f_pl, f_sfc = Path(tmp) / "pl.grib2", Path(tmp) / "sfc.grib2"
            _retrieve(client, when, PRESSURE, LEVELS, f_pl)
            _retrieve(client, when, SURFACE, None, f_sfc)
            fields = _read(f_pl, lat_out, lon_out, PRESSURE)
            fields.update(_read(f_sfc, lat_out, lon_out, SURFACE))
            frames[when] = fields
            print(f"  {when:%Y-%m-%d %HZ}: {len(fields)} variables", flush=True)

    n_time = steps + 2
    times = pd.to_timedelta(np.arange(n_time) * 6, unit="h")
    shape2, shape3 = (1, n_time, len(lat_out), len(lon_out)), (1, n_time, len(LEVELS), len(lat_out), len(lon_out))
    data = {}
    for name in PRESSURE.values():
        arr = np.full(shape3, np.nan, "float32")
        for i, when in enumerate(frames):
            arr[0, i] = np.stack([frames[when][name][lev] for lev in LEVELS])
        if name == "geopotential":
            arr *= G  # open data gives geopotential height (m)
        data[name] = (("batch", "time", "level", "lat", "lon"), arr)
    for name in SURFACE.values():
        arr = np.full(shape2, np.nan, "float32")
        for i, when in enumerate(frames):
            arr[0, i] = frames[when][name]
        if name == "sea_surface_temperature":
            # skin temperature over sea ice is the ice surface; sea water stays near freezing
            arr = np.maximum(arr, SEAWATER_FREEZING)
            arr[:, :, ~ocean] = np.nan
        data[name] = (("batch", "time", "lat", "lon"), arr)
    # outputs-only variables: placeholders with the right shape
    for name, v in static.data_vars.items():
        if name not in data and name not in STATIC and v.dims == ("batch", "time", "lat", "lon"):
            data[name] = (v.dims, np.full(shape2, np.nan, "float32"))
    ds = xr.Dataset(data, coords={"time": times, "level": np.array(LEVELS, "int32"),
                                  "lat": lat_out, "lon": lon_out})
    for name in STATIC:
        ds[name] = static[name].load()
    ds = ds.assign_coords(datetime=(("batch", "time"),
                                    (init - pd.Timedelta(hours=6) + times).values[None, :]))
    ds.attrs.update(source="ECMWF open data (IFS analyses, step 0)", init=f"{init:%Y-%m-%dT%H}")
    return ds


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True, help="YYYY-MM-DDTHH (UTC; 00/06/12/18)")
    ap.add_argument("--steps", type=int, required=True, help="number of 6 h forecast steps")
    ap.add_argument("--static", type=Path, required=True, help="Google's FGN sample file (1 deg)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--source", default="aws", help="ECMWF mirror: aws | azure | ecmwf")
    ap.add_argument("--regrid", choices=["mean", "point"], default=REGRID)
    a = ap.parse_args()
    globals()["REGRID"] = a.regrid
    t0 = time.time()
    ds = build(pd.Timestamp(a.init), a.steps, a.static, a.source)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.out.with_suffix(".tmp.nc")
    ds.to_netcdf(tmp)
    tmp.replace(a.out)
    print(f"wrote {a.out} in {time.time() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
