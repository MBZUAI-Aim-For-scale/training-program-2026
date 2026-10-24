"""Run Panchali's forecast_event_movie.py against Demo 3 outputs on the Spark.

forecast_event_movie.py is kept unmodified (drop in new versions as they come).
This wrapper patches two of its functions, then calls its main():

  open_store  Demo 3 output files store lead_time as integer hours; her loader
              would read an integer lead axis as days, so convert it to a real
              timedelta on open.
  _era5       her obs loader reads yearly 6-hourly zarr files on the DSI
              cluster. If that folder isn't there (the Spark), fetch the box
              from Google's ARCO ERA5 instead, rebuild the same 6-hourly layout
              (instantaneous 2 m T; 6 h precipitation totals ending at each
              stamp), and cache it under DEMO3_OBS_CACHE.

All of her command-line flags work unchanged. One extra flag:
    --result FILE   write {"gif": ..., "png": ..., "error": ...} when finished
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import forecast_event_movie as fem  # noqa: E402

ARCO = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
OBS_CACHE = Path(os.environ.get(
    "DEMO3_OBS_CACHE", Path(__file__).resolve().parents[2] / "outputs" / "_obs_cache"))

_orig_open_store = fem.open_store
_orig_era5 = fem._era5


def open_store(path):
    ds = _orig_open_store(path)
    lt = ds.get("lead_time")
    if lt is not None and np.issubdtype(lt.dtype, np.integer) and lt.attrs.get("units") == "hours":
        ds = ds.drop_vars([c for c in ("valid_time",) if c in ds.coords])
        ds = ds.assign_coords(lead_time=pd.to_timedelta(lt.values, unit="h").values)
    return ds


def _era5(var, start, end, domain):
    if os.path.isdir(fem.ERA5_DIR):
        return _orig_era5(var, start, end, domain)
    lo, hi = fem._pad(start, end)
    key = f"{var}|{domain}|{lo}|{hi}"
    cached = OBS_CACHE / f"era5_{var}_{hashlib.md5(key.encode()).hexdigest()[:12]}.nc"
    if cached.exists():
        return xr.open_dataarray(cached).load()
    print(f"  fetching ERA5 {var} {str(lo)[:10]}..{str(hi)[:10]} from ARCO "
          f"(first time only; cached in {OBS_CACHE})", flush=True)
    ds = xr.open_zarr(ARCO, chunks={}, storage_options=dict(token="anon"))
    # the time axis is padded out to 2050 with empty chunks; the attrs say where data stops
    last = pd.Timestamp(ds.attrs.get("valid_time_stop_era5t")
                        or ds.attrs.get("valid_time_stop") or ds.time.values[-1])
    if pd.Timestamp(lo) > last:
        raise SystemExit(f"ERA5 on ARCO currently ends {last:%Y-%m-%d}; no observations "
                         f"for {str(lo)[:10]} onwards yet")
    da = fem._subset(ds[var], domain)
    if var == "total_precipitation":
        # hourly totals -> 6 h totals ending at each 00/06/12/18 stamp
        da = _load_in_parts(da.sel(time=slice(np.datetime64(lo) - np.timedelta64(5, "h"), hi)))
        da = da.rolling(time=6, min_periods=6).sum()
    else:
        # instantaneous field: keep the 6-hourly stamps, like the forecasts
        da = _load_in_parts(da.sel(time=slice(lo, hi)))
    da = da.sel(time=da.time.dt.hour % 6 == 0).sel(time=slice(lo, hi)).dropna("time", how="all")
    OBS_CACHE.mkdir(parents=True, exist_ok=True)
    da.to_netcdf(cached)
    return da


def _load_in_parts(da, parts=10):
    """Load a lazy ARCO selection in ~`parts` pieces, printing "ERA5 fetch k/N" after each
    so the app can show a progress bar (each piece still downloads its hours in parallel)."""
    n = da.sizes["time"]
    step = max(24, -(-n // parts))
    pieces = range(0, n, step)
    out = []
    for k, i in enumerate(pieces, 1):
        out.append(da.isel(time=slice(i, i + step)).load())
        print(f"  ERA5 fetch {k}/{len(pieces)}", flush=True)
    return xr.concat(out, "time") if len(out) > 1 else out[0]


fem.open_store = open_store
fem._era5 = _era5


def main() -> int:
    argv = sys.argv[1:]
    result = None
    if "--result" in argv:
        i = argv.index("--result")
        result = Path(argv[i + 1])
        del argv[i:i + 2]
    outdir = Path(argv[argv.index("--outdir") + 1]) if "--outdir" in argv else Path(".")
    sys.argv = [fem.__file__] + argv
    out = {"gif": None, "png": None, "error": None}
    try:
        fem.main()
    except SystemExit as e:  # her script exits with a message on bad inputs
        if e.code not in (0, None):
            out["error"] = str(e.code)
    except Exception as e:
        traceback.print_exc()
        out["error"] = f"{type(e).__name__}: {e}"
    gifs = sorted(glob.glob(str(outdir / "*_movie.gif")), key=os.path.getmtime)
    pngs = sorted(glob.glob(str(outdir / "*_track.png")), key=os.path.getmtime)
    out["gif"] = gifs[-1] if gifs else None
    out["png"] = pngs[-1] if pngs else None
    if result:
        result.write_text(json.dumps(out))
    print(json.dumps(out))
    return 0 if not out["error"] else 1


if __name__ == "__main__":
    sys.exit(main())
