# %% [markdown]
# # Lab 04 — Africa rainfall ensemble
# **Goal:** forecast 5 days of rainfall for the whole of Africa with an AI ensemble, and turn it into
# probabilities a farmer or water manager can use.
#
# Model: **AIFS-ENS** (ECMWF). Unlike SFNO (Lab 02) it predicts precipitation directly (`tp06`, 6-hour
# accumulated). It is trained as an ensemble, so each member draws its own noise and the spread between members
# tells you how sure the forecast is.
#
# **Run once, use many times:** this lab also saves temperature, dewpoint, wind and sunshine for Africa. Labs 05
# (advisories), 06 (crop stress) and 08 (solar) read this file instead of running the model again.
#
# Requires the `spark-intro` container (Lab 00) and internet (IFS initial conditions + ~1 GB of weights on first run).
# Expected time: ~12 min for 8 members. Peak memory: ~36 GiB GPU, ~60 GiB system, so run it alone on a shared box.

# %%
import time
from datetime import datetime, timedelta, timezone
import numpy as np
import torch
import xarray as xr

from earth2studio.data import IFS
from earth2studio.io import ZarrBackend
from earth2studio.models.px import AIFSENS
from earth2studio.perturbation import Zero
import earth2studio.run as run

# Start from yesterday 00 UTC (IFS open data is published a few hours after each cycle).
start = (datetime.now(timezone.utc) - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
start = start.replace(tzinfo=None)
NSTEPS = 20     # 6-hourly steps -> 5 days
NENSEMBLE = 8   # members; each adds ~1 min. More members give smoother probabilities.
print("init:", start, "| steps:", NSTEPS, "| members:", NENSEMBLE)

# Africa: 35 S – 38 N, 18 W – 52 E. Model longitudes run 0–360, so West Africa is 342–360 and the rest 0–52;
# Earth2Studio accepts that wrapped list directly.
LAT = np.arange(38, -35.01, -0.25)
LON = np.concatenate([np.arange(342, 360, 0.25), np.arange(0, 52.01, 0.25)])
VARS = ["tp06", "t2m", "d2m", "u10m", "v10m", "ssrd06", "tcc"]
CITIES = {"Nairobi": (-1.29, 36.82), "Addis Ababa": (9.03, 38.74), "Kampala": (0.35, 32.58),
          "Dar es Salaam": (-6.79, 39.21), "Mogadishu": (2.05, 45.32), "Kano": (12.00, 8.52),
          "Accra": (5.60, -0.19), "Lusaka": (-15.39, 28.32)}

# %% [markdown]
# ## Load the model and run the ensemble

# %%
t0 = time.perf_counter()
model = AIFSENS.load_model(AIFSENS.load_default_package())
print(f"model loaded in {time.perf_counter()-t0:.0f}s")

out_path = "outputs/lab04_africa_ensemble.zarr"
io = ZarrBackend(file_name=out_path, backend_kwargs={"overwrite": True})
output_coords = {"variable": np.array(VARS), "lat": LAT, "lon": LON}

torch.cuda.reset_peak_memory_stats()
t0 = time.perf_counter()
# IFS open data has several mirrors; the AWS one throttles heavy requests ("503 Slow Down"), so use Google's.
# If this hangs, try source="ecmwf" or "azure".
ifs = IFS(source="google")
run.ensemble([start], NSTEPS, NENSEMBLE, model, ifs, io, Zero(), batch_size=1, output_coords=output_coords)
torch.cuda.synchronize()
print(f"ensemble wall time: {time.perf_counter()-t0:.0f}s | peak GPU alloc: "
      f"{torch.cuda.max_memory_allocated()/2**30:.1f} GiB")

# %% [markdown]
# ## 5-day rainfall: ensemble mean and probability of a wet spell
# `tp06` is in metres of water per 6 h. Lead time 0 is the initial state, so it is left out of totals.
# For maps we convert longitude to -180–180 so the continent is in one piece.

# %%
ds = xr.open_zarr(out_path).isel(time=0)
ds = ds.assign_coords(lon=((ds.lon + 180) % 360) - 180).sortby("lon")
rain = (ds["tp06"].isel(lead_time=slice(1, None)).clip(min=0) * 1000).load()   # mm per 6 h
total = rain.sum("lead_time")                                                  # (ensemble, lat, lon)

WET_MM = 25  # 5-day total that counts as a "wet spell" for planting decisions (placeholder)
prob = (total > WET_MM).mean("ensemble") * 100

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 2, figsize=(14, 7), subplot_kw={"projection": ccrs.PlateCarree()})
panels = [(total.mean("ensemble"), "Ensemble-mean 5-day rainfall (mm)", "YlGnBu", 0, 100),
          (prob, f"Chance of > {WET_MM} mm in 5 days (%)", "Purples", 0, 100)]
for a, (da, title, cmap, vmin, vmax) in zip(ax, panels):
    im = a.pcolormesh(da.lon, da.lat, da, cmap=cmap, vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
    a.coastlines(linewidth=0.7); a.add_feature(cfeature.BORDERS, linewidth=0.3)
    a.add_feature(cfeature.LAKES, facecolor="none", edgecolor="k", linewidth=0.3)
    for name, (la, lo) in CITIES.items():
        a.plot(lo, la, "ko", ms=2.5, transform=ccrs.PlateCarree())
        a.text(lo + 0.5, la + 0.5, name, fontsize=7, transform=ccrs.PlateCarree())
    a.set_title(title)
    fig.colorbar(im, ax=a, shrink=0.6)
fig.suptitle(f"AIFS-ENS, {NENSEMBLE} members, init {start:%Y-%m-%d %H} UTC")
plt.savefig("outputs/lab04_africa_rain.png", dpi=110, bbox_inches="tight"); plt.show()

# %% [markdown]
# ## Daily rainfall at cities: how much do the members agree?

# %%
daily = rain.coarsen(lead_time=4, boundary="trim").sum()   # 4 x 6 h = 1 day
days = np.arange(1, daily.sizes["lead_time"] + 1)
fig, axs = plt.subplots(2, len(CITIES) // 2, figsize=(15, 5.5), sharey=True, sharex=True)
for a, (name, (la, lo)) in zip(axs.flat, CITIES.items()):
    pt = daily.sel(lat=la, lon=lo, method="nearest")
    for m in range(pt.sizes["ensemble"]):
        a.plot(days, pt.isel(ensemble=m), "-", c="tab:blue", alpha=0.3)
    a.plot(days, pt.mean("ensemble"), "o-", c="k", label="mean")
    a.set_title(name)
for a in axs[1]:
    a.set_xlabel("day")
for a in axs[:, 0]:
    a.set_ylabel("rain (mm/day)")
axs[0, 0].legend()
plt.tight_layout(); plt.savefig("outputs/lab04_city_rain.png", dpi=120, bbox_inches="tight"); plt.show()

# %% [markdown]
# ## Exercise
# 1. Raise `NENSEMBLE` to 16. How do wall time and peak memory scale? Do the probabilities change?
# 2. Change `WET_MM` to a threshold that matters for a crop you know (e.g. planting after the first 20 mm of the
#    rains). Where is the chance above 50 %?
# 3. Add `"cp06"` (convective precipitation) to `VARS`. Over which areas is most rain convective?
# 4. Is this week wetter or drier than normal? Lab 10 builds a 40-year rainfall climate to answer that.
#
# Caveats: 0.25 deg (~28 km) cannot resolve individual storms, and AI models still under-predict heavy rain
# extremes. Treat probabilities as guidance for planning, not warnings.
