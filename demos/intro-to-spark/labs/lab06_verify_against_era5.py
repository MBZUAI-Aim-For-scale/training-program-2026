# %% [markdown]
# # Lab 06 — Is the forecast any good? Verify against ERA5
# **Goal:** run SFNO from a past date, compare it with what actually happened (ERA5 reanalysis), and
# beat a naive baseline. A small version of Day 3's "Putting Your Forecasts to the Test" and "Ground Truth Challenge".
#
# Requires internet (ARCO ERA5 on Google Cloud). Expected time: 15–20 min.
#
# Caveat: ERA5 is itself a model product. Station or national gridded data (e.g. CHIRPS for rainfall) is
# the "ground truth" each country should use; the method below is the same.

# %%
from datetime import datetime, timedelta
import numpy as np
import xarray as xr

from earth2studio.data import ARCO
from earth2studio.io import ZarrBackend
from earth2studio.models.px import SFNO
import earth2studio.run as run

INIT = datetime(2023, 7, 1)  # a past date ERA5 covers; try a heatwave or dust event you remember
NSTEPS = 20                  # 5 days
VARS = np.array(["t2m", "u10m", "v10m"])
BOX = dict(lat=slice(26.5, 22), lon=slice(51, 56.5))  # UAE

model = SFNO.load_model(SFNO.load_default_package())
arco = ARCO()
out_path = "outputs/lab06_forecast.zarr"
io = ZarrBackend(file_name=out_path, backend_kwargs={"overwrite": True})
run.deterministic([INIT], NSTEPS, model, arco, io, output_coords={"variable": VARS})
fc = xr.open_zarr(out_path)["t2m"].isel(time=0).sel(**BOX).load() - 273.15   # (lead_time, lat, lon)

# %% [markdown]
# ## Truth and a baseline
# Baseline = **"yesterday again"**: for each valid time, use the ERA5 value at the same hour on the day
# before the forecast started. It knows the diurnal cycle but nothing about the coming days.
# A model that cannot beat this is not adding value.
#
# Note: SFNO's error at lead 0 is exactly zero because it starts from the ERA5 state we score it against.
# That is not skill; read the curve from +6 h onward.

# %%
valid = [INIT + timedelta(hours=6 * i) for i in range(NSTEPS + 1)]
before = [INIT - timedelta(hours=24) + timedelta(hours=6 * (i % 4)) for i in range(NSTEPS + 1)]

def era5_t2m(times):
    da = arco(times, ["t2m"]).sel(variable="t2m")
    # nearest-match to the forecast grid so shapes line up exactly
    # (.values: fc.lat/lon carry a scalar `time` coord that would clash with ERA5's time dimension)
    return da.sel(lat=fc.lat.values, lon=fc.lon.values, method="nearest").values - 273.15

truth = era5_t2m(valid)
unique_before = sorted(set(before))  # only 4 distinct times; fetch once, then repeat
baseline = era5_t2m(unique_before)[[unique_before.index(t) for t in before]]
model_fc = fc.values

w = np.cos(np.deg2rad(fc.lat.values))[None, :, None]  # area weights
def rmse(a, b):
    return np.sqrt((((a - b) ** 2) * w).sum((1, 2)) / (w.sum() * a.shape[2]))
def bias(a, b):
    return ((a - b) * w).sum((1, 2)) / (w.sum() * a.shape[2])

hours = np.arange(NSTEPS + 1) * 6
print(f"{'lead h':>6} {'SFNO rmse':>10} {'base rmse':>10} {'SFNO bias':>10}")
for h, r1, r2, b in zip(hours, rmse(model_fc, truth), rmse(baseline, truth), bias(model_fc, truth)):
    print(f"{h:>6} {r1:>10.2f} {r2:>10.2f} {b:>10.2f}")

# %%
import matplotlib.pyplot as plt
plt.figure(figsize=(7, 3.5))
plt.plot(hours, rmse(model_fc, truth), "o-", label="SFNO", c="#76B900")
plt.plot(hours, rmse(baseline, truth), "s--", label="yesterday again", c="gray")
plt.xlabel("lead time (h)"); plt.ylabel("t2m RMSE over UAE (°C)"); plt.legend(); plt.grid(alpha=.3)
plt.title(f"Init {INIT:%Y-%m-%d %HZ}")
plt.tight_layout(); plt.savefig("outputs/lab06_rmse.png", dpi=120); plt.show()

# %% [markdown]
# ## Threshold skill: does it catch the events farmers care about?
# RMSE can look fine while alerts are wrong. Count hits / misses / false alarms for t2m > threshold
# over all grid points and lead times.

# %%
THRESH = 42.0
obs, fcst = truth > THRESH, model_fc > THRESH
hits, misses = (obs & fcst).sum(), (obs & ~fcst).sum()
false_alarms = (~obs & fcst).sum()
pod = hits / max(hits + misses, 1)
far = false_alarms / max(hits + false_alarms, 1)
print(f"t2m > {THRESH}°C  hits={hits} misses={misses} false_alarms={false_alarms}")
print(f"probability of detection = {pod:.2f}, false alarm ratio = {far:.2f}")

# %% [markdown]
# ## Exercise
# 1. Try a winter date and a summer date. Where does the model beat the baseline, and from which lead time?
# 2. Repeat for wind speed (`u10m`, `v10m`) with the spraying threshold from Lab 05.
# 3. One case is an anecdote. How many init dates would you need before trusting a score? What does that cost
#    in Spark time (time one run, multiply)?
# 4. Is there a bias? If SFNO is consistently 1.5 °C too cool, how would that change your heat alert threshold?
