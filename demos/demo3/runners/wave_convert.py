"""Convert the ECMWF wave input states (DSI: /net/monsoon/ECMWF_WAVE_DATASET) to NetCDF.

Each source file, input_state_YYYYMMDDT00.pkl, is a pickled dict
{"date": datetime, "fields": {name: float64 array (2, 542080)}} holding the
11 ocean-wave fields AIFS v2 needs (swh, mwp, mwd, cdww, wmb, h1012 ... h2530)
at 18Z the day before and 00Z, on AIFS's N320 grid; NaN over land.

Output, wave_YYYYMMDDT00.nc: the same fields as float32 (zlib), plus cos_mwd and
sin_mwd (computed here, on the native grid, so directions don't get averaged
across 0/360°), dims (time: 2, values: 542080). NetCDF instead of pickle so the
files can be shared safely (loading a pickle can run code).

    python runners/wave_convert.py input_state_20250701T00.pkl [...] --out DIR
"""
from __future__ import annotations

import argparse
import pickle
from datetime import timedelta
from pathlib import Path

import numpy as np
import xarray as xr

STATIC = ("wmb",)  # model bathymetry: an invariant AIFS takes from its own package


def convert(src: Path, out_dir: Path) -> Path:
    with open(src, "rb") as f:
        state = pickle.load(f)
    date, fields = state["date"], state["fields"]
    times = np.array([date - timedelta(hours=6), date], dtype="datetime64[ns]")
    data = {k: (("time", "values"), np.asarray(v, dtype=np.float32))
            for k, v in fields.items() if k not in STATIC}
    rad = np.deg2rad(np.asarray(fields["mwd"], dtype=np.float64))
    data["cos_mwd"] = (("time", "values"), np.cos(rad).astype(np.float32))
    data["sin_mwd"] = (("time", "values"), np.sin(rad).astype(np.float32))
    ds = xr.Dataset(data, coords={"time": times})
    ds.attrs.update(grid="N320 (AIFS v2 native, 542080 points)", source=src.name,
                    note="18Z the day before and 00Z; NaN over land")
    out = out_dir / f"wave_{date:%Y%m%dT%H}.nc"
    tmp = out.with_suffix(".part")
    ds.to_netcdf(tmp, encoding={k: {"zlib": True, "complevel": 4, "shuffle": True} for k in ds.data_vars})
    tmp.replace(out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=Path("."))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    for src in a.inputs:
        out = convert(src, a.out)
        print(f"{src.name} -> {out.name} ({out.stat().st_size / 1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
