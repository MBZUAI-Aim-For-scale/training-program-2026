"""Earth2Studio data source for AIFS v2 before ECMWF open data has its inputs:
ocean-wave fields from the ECMWF wave dataset, everything else from ARCO ERA5.

AIFS v2 needs 11 wave inputs (swh, mwp, cos/sin_mwd, cdww, h1012 ... h2530) that
ERA5 on ARCO doesn't carry, plus snow cover (snowc), which ARCO lacks but which
follows from ERA5 snow depth and density (ECMWF's formula). The wave fields come
from runners/wave_convert.py's files (00Z starts, 2000-2025), found in WAVE_DIR
or downloaded from WAVE_BUCKET, and are put on the 0.25° grid with the same
N320 -> 0.25° matrix Earth2Studio uses for AIFS v2's own outputs.
"""
from __future__ import annotations

import glob
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

WAVE_VARS = ("swh", "mwp", "cos_mwd", "sin_mwd", "cdww",
             "h1012", "h1214", "h1417", "h1721", "h2125", "h2530")
WAVE_DIR = Path(os.environ.get("DEMO3_WAVE_DIR", Path.home() / ".cache" / "wave"))
WAVE_BUCKET = os.environ.get("DEMO3_WAVE_BUCKET", "gs://aim4scale_training_2026/wave/n320")
FIRST, LAST = datetime(2000, 1, 1), datetime(2025, 12, 31)
LAT = np.linspace(90, -90, 721)
LON = np.arange(0, 360, 0.25)


def available(init) -> bool:
    """Wave inputs exist for 00Z starts from 1 Jan 2000 to 31 Dec 2025."""
    return init.hour == 0 and FIRST <= init.replace(tzinfo=None) <= LAST


def source_for(init, ifs_mirror: str = "aws"):
    """Wave + ERA5 for 00Z starts in 2000-2025, else ECMWF open data (from 13 May 2026)."""
    if available(init):
        return WaveERA5()
    from earth2studio.data import IFS
    # ECMWF open-data GRIB downloads can be slow; the default 600 s timeout is too short.
    return IFS(source=ifs_mirror, async_timeout=3600)


def _wave_file(day: datetime) -> Path:
    name = f"wave_{day:%Y%m%dT00}.nc"
    path = WAVE_DIR / name
    if not path.exists():
        import fsspec
        WAVE_DIR.mkdir(parents=True, exist_ok=True)
        fs, root = fsspec.core.url_to_fs(WAVE_BUCKET)
        fs.get(f"{root}/{name}", str(path.with_suffix(".part")))
        path.with_suffix(".part").replace(path)
    return path


def _inverse_matrix():
    """Earth2Studio's AIFS v2 N320 -> 0.25° matrix (downloaded with the model)."""
    import scipy.sparse as sp
    hits = glob.glob(str(Path.home() / ".cache" / "earth2studio" /
                         "aifs-single-2.0_inverse_interpolation_matrix" / "*.npz"))
    if not hits:
        raise FileNotFoundError("AIFS v2 interpolation matrix not in the cache; load AIFS2 first")
    m = np.load(hits[0])
    return sp.csr_matrix((m["data"], m["indices"], m["indptr"]), shape=tuple(m["shape"]))


class WaveERA5:
    """time, variable -> DataArray (time, variable, lat, lon) on ARCO's 0.25° grid."""

    def __init__(self):
        from _e2s import arco
        from earth2studio.lexicon import ARCOLexicon
        ARCOLexicon.VOCAB.setdefault("rsn", "snow_density::")  # for snow cover
        self.arco = arco()
        self._matrix = None

    def _wave(self, times, names) -> np.ndarray:
        if self._matrix is None:
            self._matrix = _inverse_matrix()
        out = np.empty((len(times), len(names), LAT.size, LON.size), dtype=np.float32)
        for i, t in enumerate(times):
            t = np.datetime64(t, "ns")
            day = (t + np.timedelta64(6, "h")).astype("datetime64[D]")  # 18Z belongs to the next day's file
            with xr.open_dataset(_wave_file(pd.Timestamp(day))) as ds:
                sel = ds.sel(time=t)
                for j, n in enumerate(names):
                    grid = (self._matrix @ sel[n].values.astype(np.float64)).reshape(LAT.size, LON.size)
                    out[i, j] = np.roll(grid, LON.size // 2, axis=1)  # matrix gives lon -180..180; ARCO 0..360
        return out

    def __call__(self, time, variable) -> xr.DataArray:
        times = [np.datetime64(t, "ns") for t in np.atleast_1d(time)]
        variable = list(np.atleast_1d(variable))
        wave = [v for v in variable if v in WAVE_VARS]
        rest = [v for v in variable if v not in WAVE_VARS and v != "snowc"]
        parts = []
        if rest or "snowc" in variable:
            extra = ["sd", "rsn"] if "snowc" in variable else []
            era = self.arco([pd.Timestamp(t).to_pydatetime() for t in times],
                            rest + [e for e in extra if e not in rest])
            if "snowc" in variable:
                sd, rsn = era.sel(variable="sd"), era.sel(variable="rsn")
                # ECMWF snow cover: 100 * min(1, snow depth [m of snow] / 0.1 m), depth = 1000 * sd / rsn
                cover = 100 * np.minimum(1.0, (1000.0 * sd / rsn.where(rsn > 0)) / 0.1)
                snowc = cover.fillna(0).expand_dims(variable=["snowc"], axis=1)
                era = xr.concat([era.sel(variable=rest), snowc], "variable") if rest else snowc
            parts.append(era.assign_coords(lat=LAT, lon=LON))
        if wave:
            arr = self._wave(times, wave)
            parts.append(xr.DataArray(arr, dims=("time", "variable", "lat", "lon"),
                                      coords={"time": np.array(times), "variable": wave, "lat": LAT, "lon": LON}))
        da = xr.concat(parts, "variable") if len(parts) > 1 else parts[0]
        return da.sel(variable=variable)
