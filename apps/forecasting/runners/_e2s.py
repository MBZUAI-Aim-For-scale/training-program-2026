"""Shared helper for models run through NVIDIA Earth2Studio."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import xarray as xr

# Earth2Studio precipitation variables, in preference order; all are in metres.
PRECIP_VARS = ("tp06", "tp1h")


def peak_gpu_gb() -> float | None:
    """Peak GPU memory of this process in GB, counting both PyTorch and JAX
    (GraphCast and NeuralGCM run on JAX, which torch's counter can't see)."""
    peaks = []
    try:
        import torch
        if torch.cuda.is_available():
            peaks.append(torch.cuda.max_memory_allocated())
    except ImportError:
        pass
    try:
        import jax
        peaks += [(d.memory_stats() or {}).get("peak_bytes_in_use", 0)
                  for d in jax.local_devices() if d.platform == "gpu"]
    except ImportError:
        pass
    return round(max(peaks) / 1e9, 1) if any(peaks) else None


ARCO_ATTRS_URL = ("https://storage.googleapis.com/gcp-public-data-arco-era5/ar/"
                  "full_37-1h-0p25deg-chunk-1.zarr-v3/.zattrs")


def arco():
    """Earth2Studio's ARCO ERA5 source, told the end date from Google's live metadata.

    Earth2Studio reads ERA5's end date from the store's metadata, but through its
    download cache, which keeps files for a year; a stale copy made it refuse dates
    after 30 Apr 2026 although ERA5 ran to 30 Jun. A subclass attribute takes
    precedence over the value Earth2Studio writes onto ARCO itself.
    """
    import json
    import urllib.request
    from datetime import datetime
    from earth2studio.data import ARCO
    try:
        with urllib.request.urlopen(ARCO_ATTRS_URL, timeout=10) as r:
            stop = datetime.strptime(json.load(r)["valid_time_stop"], "%Y-%m-%d")
    except Exception:
        return ARCO()

    class ARCOLive(ARCO):
        ARCO_TIME_STOP = stop

    return ARCOLive()


def run_e2s(model_cls, data, init: pd.Timestamp, lead_hours: int, members: int, report,
            *, ensemble: bool = False, step_hours: int = 6, seed: int = 0,
            precip_fix=None, device: str = "cuda") -> xr.Dataset:
    """Run `model_cls` from `data` and return t2m [K], z500 [m2 s-2] and tp [mm / 6 h]
    on 6-hourly leads. `step_hours` is the lead-time spacing the model yields (Aurora 1.5
    yields hourly steps). `precip_fix` maps the model's raw precipitation [m] to
    corrected values [m] before aggregation."""
    import torch
    import earth2studio.run as run
    from earth2studio.io import XarrayBackend

    t0 = time.time()
    report("Loading model weights…", 0.02)
    model = model_cls.load_model(model_cls.load_default_package()).to(device)
    load_s = round(time.time() - t0, 1)
    report(f"Model loaded in {load_s:.0f} s", 0.05, load_s=load_s)

    # Fetch the initial conditions on their own first, so the download is timed
    # separately from the model; the data sources cache it, so the run reuses it.
    from earth2studio.data import fetch_data
    ic = model.input_coords()
    t_f = time.time()
    report("Fetching initial conditions…", 0.06)
    fetch_data(data, time=np.array([init.to_datetime64()]), variable=np.array(ic["variable"]),
               lead_time=np.array(ic["lead_time"]), device="cpu")
    fetch_s = round(time.time() - t_f, 1)
    report(f"Initial conditions ready in {fetch_s:.0f} s; running…", 0.1, fetch_s=fetch_s)

    available = set(model.output_coords(model.input_coords())["variable"].tolist())
    precip = next((v for v in PRECIP_VARS if v in available), None)
    wanted = [v for v in ("t2m", "z500") if v in available] + ([precip] if precip else [])
    out_coords = {"variable": np.array(wanted)}
    nsteps = lead_hours // step_hours
    io = XarrayBackend()
    torch.manual_seed(seed)
    torch.cuda.reset_peak_memory_stats()
    t1 = time.time()
    if ensemble:
        from earth2studio.perturbation import Zero
        run.ensemble([init.to_pydatetime()], nsteps, members, model, data, io, Zero(),
                     batch_size=1, output_coords=out_coords)
    else:
        run.deterministic([init.to_pydatetime()], nsteps, model, data, io, output_coords=out_coords)
    run_s = round(time.time() - t1, 1)
    peak_gb = peak_gpu_gb()
    report(f"Forecast finished in {run_s:.0f} s", 0.95, run_s=run_s, peak_gpu_gb=peak_gb)

    ds = io.root.isel(time=0, drop=True)
    hours = (ds.lead_time.values / np.timedelta64(1, "h")).astype(int)
    ds = ds.assign_coords(lead_time=hours)
    if "ensemble" not in ds.dims:
        ds = ds.expand_dims(ensemble=[0])
    ds = ds.assign_coords(ensemble=np.arange(ds.sizes["ensemble"]))

    if precip and precip_fix is not None:
        ds[precip] = precip_fix(ds[precip])
    six = hours[hours % 6 == 0]
    out = xr.Dataset({v: ds[v].sel(lead_time=six) for v in ("t2m", "z500") if v in ds})
    if precip == "tp06":
        tp = ds.tp06.sel(lead_time=six) * 1000.0
    elif precip == "tp1h":
        # Sum hourly totals over each 6 h window ending at lead h: (h-6, h].
        block = xr.DataArray(((hours + 5) // 6) * 6, dims="lead_time", coords={"lead_time": hours},
                             name="group")
        tp = (ds.tp1h.where(ds.lead_time > 0).groupby(block).sum(min_count=6)
              .rename(group="lead_time").reindex(lead_time=six) * 1000.0)
    if precip:
        # Some models emit tiny negative rainfall; clip, and leave lead 0 undefined.
        out["tp"] = tp.clip(min=0).where(tp.lead_time > 0)
    return out
