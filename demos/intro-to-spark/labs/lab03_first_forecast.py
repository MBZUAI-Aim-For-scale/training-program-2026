# %% [markdown]
# # Lab 03 — Your first AI weather forecast on Spark
# **Goal:** run a global AI forecast end to end with Earth2Studio and look at the UAE.
#
# Requires: `pip install earth2studio[sfno,data]` (already in the `spark-intro` image, see Lab 00) and internet for initial
# conditions + model weights (first run only). Expected time: 5–15 min.
#
# Pipeline: **data source (GFS analysis) -> prognostic model (SFNO) -> Zarr output -> plot**

# %%
import time
from datetime import datetime, timedelta, timezone
import numpy as np
import torch
import xarray as xr

from earth2studio.data import GFS
from earth2studio.io import ZarrBackend
from earth2studio.models.px import SFNO
import earth2studio.run as run

# Start from yesterday 00 UTC (GFS archives are available shortly after each cycle).
start = (datetime.now(timezone.utc) - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
start = start.replace(tzinfo=None)
NSTEPS = 20  # 6-hourly steps -> 5 days. Raise to 40 for 10 days.
print("init:", start, "| steps:", NSTEPS)

# %% [markdown]
# ## Load model
# The first call downloads weights into `EARTH2STUDIO_CACHE` (default `~/.cache/earth2studio`).

# %%
t0 = time.perf_counter()
model = SFNO.load_model(SFNO.load_default_package())
print(f"model loaded in {time.perf_counter()-t0:.0f}s, params: {sum(p.numel() for p in model.parameters())/1e6:.0f} M")

# %% [markdown]
# ## Run forecast

# %%
# Best practice: save only the variables you need. All 73 SFNO channels for 21 steps is ~6 GB;
# these five are ~0.4 GB and write much faster.
VARS = np.array(["t2m", "u10m", "v10m", "msl", "tcwv"])
out_path = "outputs/lab03_forecast.zarr"
io = ZarrBackend(file_name=out_path, backend_kwargs={"overwrite": True})
torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()
t0 = time.perf_counter()
run.deterministic([start], NSTEPS, model, GFS(), io, output_coords={"variable": VARS})
torch.cuda.synchronize()
wall = time.perf_counter() - t0
peak = torch.cuda.max_memory_allocated() / 2**30
print(f"forecast wall time: {wall:.0f}s (includes data download) | peak GPU alloc: {peak:.1f} GiB")

# %% [markdown]
# ## Look at the UAE
# Box: lat 22–26.5 N, lon 51–56.5 E. Earth2Studio longitudes are 0–360, which already covers 51–56.5.

# %%
ds = xr.open_zarr(out_path)
print(list(ds.data_vars)[:8], "...")
t2m = ds["t2m"].isel(time=0) - 273.15          # (lead_time, lat, lon)
box = dict(lat=slice(26.5, 22), lon=slice(51, 56.5))  # lat is descending in ERA5/GFS grids
t2m_uae = t2m.sel(**box)

import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
for a, step in zip(ax, [0, NSTEPS // 2, NSTEPS]):
    im = t2m_uae.isel(lead_time=step).plot(ax=a, vmin=15, vmax=45, cmap="turbo", add_colorbar=False)
    a.set_title(f"+{6*step} h")
fig.colorbar(im, ax=ax, label="2 m temperature (°C)")
plt.savefig("outputs/lab03_uae_t2m.png", dpi=120, bbox_inches="tight"); plt.show()

# %% [markdown]
# ## Exercise
# 1. Swap `GFS()` for `ARCO()` (ERA5 from Google) for a past date. Does the forecast change?
# 2. Change `NSTEPS`. How does wall time and peak memory scale?
# 3. `print(model.output_coords(model.input_coords())["variable"])` lists all 73 variables SFNO predicts.
#    Which would you need for irrigation, spraying, or frost advisories? Add them to `VARS` and rerun.
#    Note SFNO has no precipitation output: rainfall needs a model or diagnostic that provides `tp`.
#
# Troubleshooting:
# - *no kernel image / sm_121*: your PyTorch lacks Blackwell kernels; use the `spark-intro` image (Lab 00).
# - *download errors*: check internet and proxy; model weights come from NGC/HF.
# - `lat` slice empty: grids are sometimes ascending; use `slice(22, 26.5)` instead.
