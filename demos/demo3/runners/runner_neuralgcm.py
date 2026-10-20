"""NeuralGCM (Google) stochastic precipitation model, 2.8 deg, initialized from ARCO ERA5.

Only the v1_precip checkpoints forecast precipitation, and they exist only at
2.8 deg (64 x 128 Gaussian grid), so output stays on that native grid instead
of being interpolated to 0.25 deg. NeuralGCM predicts pressure-level fields
but not 2 m temperature, so t2m is left missing. Members use different random
keys; sea-surface temperature and sea ice are held at their initial values.
"""
from __future__ import annotations

import pickle
import time
from pathlib import Path

import numpy as np
import xarray as xr

from _e2s import peak_gpu_gb

CHECKPOINT = "gs://neuralgcm/models/v1_precip/stochastic_precip_2_8_deg.pkl"
CACHE = Path.home() / ".cache" / "neuralgcm"
ARCO = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"


def run(init, lead_hours, members, report):
    import gcsfs
    import jax
    import neuralgcm
    from dinosaur import horizontal_interpolation, spherical_harmonic, xarray_utils

    t0 = time.time()
    report("Loading NeuralGCM checkpoint…", 0.02)
    local = CACHE / Path(CHECKPOINT).name  # keep a copy so later runs work offline
    if not local.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        gcsfs.GCSFileSystem(token="anon").get(CHECKPOINT, str(local) + ".part")
        Path(str(local) + ".part").replace(local)
    with open(local, "rb") as f:
        model = neuralgcm.PressureLevelModel.from_checkpoint(pickle.load(f))
    load_s = round(time.time() - t0, 1)
    report(f"Model loaded in {load_s:.0f} s; fetching ERA5 initial conditions…", 0.1, load_s=load_s)

    t1 = time.time()
    era5 = xr.open_zarr(ARCO, chunks=None, storage_options=dict(token="anon"))
    era5 = era5[model.input_variables + model.forcing_variables].sel(time=[init.to_datetime64()]).compute()
    era5_grid = spherical_harmonic.Grid(
        latitude_nodes=era5.sizes["latitude"], longitude_nodes=era5.sizes["longitude"],
        latitude_spacing=xarray_utils.infer_latitude_spacing(era5.latitude),
        longitude_offset=xarray_utils.infer_longitude_offset(era5.longitude))
    regridder = horizontal_interpolation.ConservativeRegridder(era5_grid, model.data_coords.horizontal,
                                                              skipna=True)
    era5 = xarray_utils.fill_nan_with_nearest(xarray_utils.regrid(era5, regridder))
    inputs = model.inputs_from_xarray(era5.isel(time=0))
    forcings = model.forcings_from_xarray(era5.isel(time=0))
    persisted = model.forcings_from_xarray(era5.head(time=1))
    fetch_s = round(time.time() - t1, 1)
    report(f"Initial conditions ready in {fetch_s:.0f} s", 0.15, fetch_s=fetch_s)
    t1 = time.time()

    steps = lead_hours // 6 + 1
    out = []
    for m in range(members):
        report(f"Running member {m + 1}/{members}…", 0.1 + 0.8 * m / members)
        state = model.encode(inputs, forcings, jax.random.key(42 + m))
        _, pred = model.unroll(state, persisted, steps=steps, timedelta=np.timedelta64(6, "h"),
                               start_with_input=True)
        ds = model.data_to_xarray(pred, times=np.arange(steps) * 6)
        out.append(ds)
    run_s = round(time.time() - t1, 1)
    report(f"Forecast finished in {run_s:.0f} s", 0.95, run_s=run_s, peak_gpu_gb=peak_gpu_gb())

    ds = xr.concat(out, dim="ensemble").rename(time="lead_time", latitude="lat", longitude="lon")
    res = xr.Dataset({"z500": ds.geopotential.sel(level=500, drop=True)})
    precip = next((v for v in ds.data_vars if "precip" in v), None)
    if precip:
        # Despite its name, precipitation_cumulative_mean is the running total since
        # init in metres (checked: 2.6 mm global mean over day 1 vs ~2.7 mm/day observed).
        tp = ds[precip].squeeze([d for d in ("surface", "level") if d in ds[precip].dims], drop=True)
        tp = tp.diff("lead_time").reindex(lead_time=tp.lead_time) * 1000.0
        res["tp"] = tp.clip(min=0).where(tp.lead_time > 0)
    res = res.assign_coords(ensemble=np.arange(members), lead_time=res.lead_time.values.astype(int))
    return res
