# %% [markdown]
# # Lab 02 — Your first AI weather forecast for Africa
# **Goal:** run a global AI forecast end to end with Earth2Studio, then read it for Africa: where will it be
# hottest, windiest and most humid over the next 5 days?
#
# Pipeline: **data source (GFS analysis) -> prognostic model (SFNO) -> Zarr output -> maps and a city table**
#
# Requires the `spark-intro` container (Lab 00) and internet for initial conditions and weights (first run only).
# Expected time: 5–10 min. Measured on Spark: forecast ~45 s, 15 GiB GPU.

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
NSTEPS = 20  # 6-hourly steps -> 5 days
print("init:", start, "| steps:", NSTEPS)

# %% [markdown]
# ## Load the model and run
# The first call downloads weights into `EARTH2STUDIO_CACHE`. Save only the variables you need: all 73 SFNO
# channels for 21 steps is ~6 GB, these five are ~0.4 GB.

# %%
t0 = time.perf_counter()
model = SFNO.load_model(SFNO.load_default_package())
print(f"model loaded in {time.perf_counter()-t0:.0f}s, {sum(p.numel() for p in model.parameters())/1e6:.0f} M params")

VARS = np.array(["t2m", "u10m", "v10m", "msl", "tcwv"])
out_path = "outputs/lab02_forecast.zarr"
io = ZarrBackend(file_name=out_path, backend_kwargs={"overwrite": True})
torch.cuda.reset_peak_memory_stats()
t0 = time.perf_counter()
run.deterministic([start], NSTEPS, model, GFS(), io, output_coords={"variable": VARS})
torch.cuda.synchronize()
print(f"forecast wall time: {time.perf_counter()-t0:.0f}s (incl. GFS download) | "
      f"peak GPU: {torch.cuda.max_memory_allocated()/2**30:.1f} GiB")

# %% [markdown]
# ## Cut out Africa
# Model grids run 0–360° east, so West Africa (west of 0°) sits at the far end of the array. Converting to
# -180–180° puts the whole continent in one piece. This is the most common gotcha with African domains.

# %%
def to_africa(ds):
    ds = ds.assign_coords(lon=((ds.lon + 180) % 360) - 180).sortby("lon")
    return ds.sel(lat=slice(38, -35), lon=slice(-18, 52))    # latitude is descending

ds = to_africa(xr.open_zarr(out_path).isel(time=0)).isel(lead_time=slice(1, None)).load()
t2m = ds["t2m"] - 273.15
wind = np.hypot(ds["u10m"], ds["v10m"])

CITIES = {"Cairo": (30.04, 31.24), "Dakar": (14.72, -17.47), "Niamey": (13.51, 2.11), "Lagos": (6.52, 3.38),
          "Khartoum": (15.50, 32.56), "Addis Ababa": (9.03, 38.74), "Kinshasa": (-4.44, 15.27),
          "Nairobi": (-1.29, 36.82), "Lusaka": (-15.39, 28.32), "Johannesburg": (-26.20, 28.05)}

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt

panels = [(t2m.max("lead_time"), "Hottest temperature in 5 days (°C)", "turbo", 10, 45),
          (wind.max("lead_time"), "Strongest 10 m wind in 5 days (m/s)", "viridis", 0, 15),
          (ds["tcwv"].mean("lead_time"), "Mean water vapour (kg/m²): where rain can form", "YlGnBu", 0, 60)]
fig, axs = plt.subplots(1, 3, figsize=(18, 6.5), subplot_kw={"projection": ccrs.PlateCarree()})
for a, (da, title, cmap, vmin, vmax) in zip(axs, panels):
    im = a.pcolormesh(da.lon, da.lat, da, cmap=cmap, vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
    a.coastlines(linewidth=0.7); a.add_feature(cfeature.BORDERS, linewidth=0.3)
    for name, (la, lo) in CITIES.items():
        a.plot(lo, la, "ko", ms=2.5, transform=ccrs.PlateCarree())
    a.set_title(title); fig.colorbar(im, ax=a, shrink=0.7, orientation="horizontal", pad=0.04)
fig.suptitle(f"SFNO forecast from GFS, init {start:%Y-%m-%d %H} UTC")
plt.savefig("outputs/lab02_africa.png", dpi=110, bbox_inches="tight"); plt.show()

# %% [markdown]
# ## City table

# %%
import pandas as pd
rows = []
for name, (la, lo) in CITIES.items():
    p = dict(lat=la, lon=lo, method="nearest")
    rows.append({"city": name, "max °C": float(t2m.sel(**p).max()), "min °C": float(t2m.sel(**p).min()),
                 "max wind m/s": float(wind.sel(**p).max())})
pd.DataFrame(rows).set_index("city").round(1)

# %% [markdown]
# ## Exercise
# 1. Where is the heat this week: Sahel, Sudan, or southern Africa? Does that match the season?
# 2. The model outputs every 6 h (00, 06, 12, 18 UTC). In Lagos that is 01, 07, 13, 19 local time; in Nairobi
#    03, 09, 15, 21. Which city's "max" is most likely to miss the afternoon peak?
# 3. `print(model.output_coords(model.input_coords())["variable"])` lists all 73 SFNO variables. SFNO has
#    **no rainfall** output; Lab 04 uses AIFS-ENS, which has.
# 4. Swap `GFS()` for `ARCO()` (ERA5) with a past date you remember (a heatwave, a dust storm). Lab 03 scores it.
#
# Troubleshooting: *no kernel image / sm_121* -> wrong PyTorch, use the `spark-intro` container (Lab 00). Empty map -> latitude
# slice in the wrong order (these grids run north to south).
