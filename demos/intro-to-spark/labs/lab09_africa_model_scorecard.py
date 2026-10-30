# %% [markdown]
# # Lab 09 — Which AI model should Africa trust? A scorecard
# **Goal:** run several AI weather models from the same start dates, score them all against ERA5 over Africa,
# and build the kind of scorecard a met service uses to pick a model. One date is an anecdote (Lab 03); four
# dates across the seasons start to be evidence.
#
# Models (all run from ERA5 initial conditions): **SFNO** (NVIDIA), **AIFS** (ECMWF), **Aurora** (Microsoft). Baseline: **persistence** (the start state, unchanged).
#
# Requires internet (ERA5, weights on first run). Expected time: ~25 min. One model at a time on the GPU.

# %%
from datetime import datetime, timedelta
import time
import numpy as np
import pandas as pd
import torch
import xarray as xr

from earth2studio.data import ARCO
from earth2studio.io import ZarrBackend
from earth2studio.models.px import AIFS, SFNO, Aurora
import earth2studio.run as run

INITS = [datetime(2023, 1, 5), datetime(2023, 4, 5), datetime(2023, 7, 5), datetime(2023, 10, 5)]
NSTEPS = 20                                     # 5 days
VARS = ["t2m", "msl", "u10m", "v10m", "z500"]
LAT = np.arange(37.5, -35.01, -0.25)            # Aurora's grid stops at 89.75 S; all three share these points
LON = np.concatenate([np.arange(342, 360, 0.25), np.arange(0, 52.01, 0.25)])
MODELS = {"SFNO": SFNO, "AIFS": AIFS, "Aurora": Aurora}

# %% [markdown]
# ## 1. Run every model from every date
# Only Africa and five variables are saved, so all runs together are a few hundred MB.

# %%
arco = ARCO()
runtime = {}
for name, M in MODELS.items():
    t0 = time.perf_counter()
    model = M.load_model(M.load_default_package())
    for init in INITS:
        io = ZarrBackend(file_name=f"outputs/lab09_{name}_{init:%Y%m%d}.zarr", backend_kwargs={"overwrite": True})
        run.deterministic([init], NSTEPS, model, arco, io, verbose=False,
                          output_coords={"variable": np.array(VARS), "lat": LAT, "lon": LON})
    runtime[name] = (time.perf_counter() - t0) / len(INITS)
    print(f"{name}: {runtime[name]:.0f}s per 5-day forecast (incl. ERA5 download)")
    del model; torch.cuda.empty_cache()

# %% [markdown]
# ## 2. Score against ERA5
# Area-weighted RMSE over the Africa box at every lead time. Wind is scored as speed.

# %%
def prep(da):
    da = da.copy()
    ws = np.hypot(da.sel(variable="u10m"), da.sel(variable="v10m")).expand_dims(variable=["ws10m"])
    return xr.concat([da.sel(variable=["t2m", "msl", "z500"]), ws], "variable")

w = xr.DataArray(np.cos(np.deg2rad(LAT)), coords={"lat": LAT}, dims="lat")
rmse = lambda a, b: np.sqrt((((a - b) ** 2) * w).sum(("lat", "lon")) / (w.sum() * len(LON)))

scores = []
for init in INITS:
    valid = [init + timedelta(hours=6 * i) for i in range(NSTEPS + 1)]
    truth = prep(arco(valid, VARS).sel(lat=LAT, lon=LON)).rename(time="valid")
    truth = truth.assign_coords(valid=np.arange(NSTEPS + 1) * 6)
    persist = truth.isel(valid=0, drop=True)
    scores.append(rmse(persist, truth).to_dataframe("rmse").reset_index().assign(model="persistence", init=init))
    for name in MODELS:
        fc = xr.open_zarr(f"outputs/lab09_{name}_{init:%Y%m%d}.zarr").isel(time=0)
        fc = prep(fc.to_array("variable")).rename(lead_time="valid")
        fc = fc.assign_coords(valid=np.arange(NSTEPS + 1) * 6)
        scores.append(rmse(fc, truth).to_dataframe("rmse").reset_index().assign(model=name, init=init))
scores = pd.concat(scores).rename(columns={"valid": "lead_h"})
mean = scores.groupby(["variable", "model", "lead_h"]).rmse.mean().unstack("model")

# %% [markdown]
# ## 3. The scorecard
# Skill = how much of persistence's error the model removes: 1 - RMSE(model) / RMSE(persistence). 0 = no better
# than "nothing changes", 1 = perfect. Green is good.

# %%
import matplotlib.pyplot as plt
LEADS = [24, 72, 120]
UNITS = {"t2m": "K", "msl": "Pa", "ws10m": "m/s", "z500": "m²/s²"}
skill = pd.DataFrame({(v, f"day {h//24}"): 1 - mean.loc[(v, h), list(MODELS)] / mean.loc[(v, h), "persistence"]
                      for v in UNITS for h in LEADS}).T
fig, ax = plt.subplots(1, 2, figsize=(15, 5.5), gridspec_kw={"width_ratios": [1, 1.6]})
im = ax[0].imshow(skill.values, cmap="RdYlGn", vmin=-0.5, vmax=1, aspect="auto")
ax[0].set_xticks(range(len(MODELS)), MODELS.keys()); ax[0].set_yticks(range(len(skill)), [f"{v} {d}" for v, d in skill.index])
for (i, j), s in np.ndenumerate(skill.values):
    ax[0].text(j, i, f"{s:.2f}", ha="center", va="center", fontsize=8)
ax[0].set_title(f"Skill vs persistence over Africa ({len(INITS)} dates)"); fig.colorbar(im, ax=ax[0], shrink=0.8)
for name, c in zip(["persistence", *MODELS], ["gray", "#76B900", "tab:blue", "tab:purple"]):
    ax[1].plot(mean.loc["t2m"].index, mean.loc["t2m"][name], "-o", ms=3, c=c, label=name)
ax[1].set_xlabel("lead time (h)"); ax[1].set_ylabel("t2m RMSE over Africa (K)"); ax[1].legend(); ax[1].grid(alpha=.3)
ax[1].set_title("2 m temperature error")
plt.tight_layout(); plt.savefig("outputs/lab09_scorecard.png", dpi=110, bbox_inches="tight"); plt.show()

table = mean.xs(120, level="lead_h").T.round(2)
table["runtime_s"] = pd.Series(runtime).round(0)
print("Day-5 RMSE by variable (units: " + ", ".join(f"{k} {v}" for k, v in UNITS.items()) + ")")
table

# %% [markdown]
# ## What to notice
# - Temperature has a strong daily cycle, so "persistence" of a 00 UTC state is poor by 12 UTC: every model
#   beats it easily. Pressure and the 500 hPa height are the fairer tests of the dynamics.
# - In our run (4 dates in 2023), **AIFS** had the lowest day-5 error on all four variables, and its lead grows
#   with time (500 hPa height: skill 0.80 at day 5 vs ~0.5 for SFNO and Aurora). SFNO and Aurora are close.
# - Cost matters too: SFNO ran a 5-day forecast in ~40 s, AIFS ~70 s, Aurora ~4 min (peaks 15 / 35 / 24 GiB).
# - Four dates are still few. A real evaluation uses a year of start dates and scores the **variables you
#   care about** (rain, heat) at **your** stations, not a continental average.
#
# ## Exercise
# 1. Restrict the score to your country (change `LAT`/`LON` before scoring). Does the ranking change?
# 2. Add dates in your rainy season. Does any model degrade more than the others?
# 3. Score against stations instead of ERA5 using Lab 03's GHCN code. Does the ranking survive?
# 4. FourCastNet (`FCN`) needs relative humidity (`r500`, `r850`), which the ARCO copy of ERA5 lacks. How would
#    you derive it from `q` and `t`? (Earth2Studio has a `DerivedRH` diagnostic.)
