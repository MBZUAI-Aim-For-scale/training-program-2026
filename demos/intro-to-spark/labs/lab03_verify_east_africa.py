# %% [markdown]
# # Lab 03 — Is the forecast any good? Check it against ERA5 and weather stations
# **Goal:** rerun SFNO from a past date over East Africa, then compare it with two kinds of truth:
# **ERA5** (a gridded reanalysis) and **real weather stations** (NOAA GHCN-Daily). Beat a naive baseline,
# and see why station data matters in a region with mountains.
#
# Requires internet (ARCO ERA5 on Google Cloud, GHCN on AWS). Expected time: ~5 min.

# %%
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import xarray as xr

from earth2studio.data import ARCO, GHCNDaily
from earth2studio.io import ZarrBackend
from earth2studio.models.px import SFNO
import earth2studio.run as run

INIT = datetime(2024, 2, 15)   # hot, dry season before the long rains; try your own dates
NSTEPS = 20                    # 5 days
BOX = dict(lat=slice(15, -12), lon=slice(28, 52))   # East Africa (latitude descending)

model = SFNO.load_model(SFNO.load_default_package())
arco = ARCO()
out_path = "outputs/lab03_forecast.zarr"
io = ZarrBackend(file_name=out_path, backend_kwargs={"overwrite": True})
run.deterministic([INIT], NSTEPS, model, arco, io, output_coords={"variable": np.array(["t2m"])})
fc_full = xr.open_zarr(out_path)["t2m"].isel(time=0).load() - 273.15     # (lead_time, lat, lon), global
fc = fc_full.sel(**BOX)

# %% [markdown]
# ## 1. Against ERA5, with a baseline
# Baseline = **"yesterday again"**: for each valid time, the ERA5 value at the same hour on the day before the
# forecast started. A model that cannot beat this is not adding value.

# %%
valid = [INIT + timedelta(hours=6 * i) for i in range(NSTEPS + 1)]
before = [INIT - timedelta(hours=24) + timedelta(hours=6 * (i % 4)) for i in range(NSTEPS + 1)]

def era5_t2m(times):
    da = arco(times, ["t2m"]).sel(variable="t2m")
    # (.values: fc.lat/lon carry a scalar `time` coord that would clash with ERA5's time dimension)
    return da.sel(lat=fc.lat.values, lon=fc.lon.values, method="nearest").values - 273.15

truth = era5_t2m(valid)
unique_before = sorted(set(before))
baseline = era5_t2m(unique_before)[[unique_before.index(t) for t in before]]

w = np.cos(np.deg2rad(fc.lat.values))[None, :, None]   # area weights
rmse = lambda a, b: np.sqrt((((a - b) ** 2) * w).sum((1, 2)) / (w.sum() * a.shape[2]))
hours = np.arange(NSTEPS + 1) * 6
r_model, r_base = rmse(fc.values, truth), rmse(baseline, truth)
print(f"SFNO beats 'yesterday again' at {(r_model[1:] < r_base[1:]).sum()} of {NSTEPS} lead times")

import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
ax[0].plot(hours, r_model, "o-", c="#76B900", label="SFNO"); ax[0].plot(hours, r_base, "s--", c="gray", label="yesterday again")
ax[0].set_xlabel("lead time (h)"); ax[0].set_ylabel("t2m RMSE vs ERA5 (°C)"); ax[0].legend(); ax[0].grid(alpha=.3)
ax[0].set_title(f"East Africa, init {INIT:%Y-%m-%d}")
plt.close(fig)  # shown and saved after the station panel is added below (Jupyter closes figures at cell end)

# %% [markdown]
# ## 2. Against weather stations
# GHCN-Daily has decades of daily max/min temperature and rainfall from national met services. We pick every
# station in the box that reports daily maximum temperature recently, and compare it with the forecast's daily
# max. (The model outputs 00/06/12/18 UTC = 03/09/15/21 East Africa Time, so 15:00 is close to the true peak.)

# %%
meta, inv = GHCNDaily.get_station_metadata(), GHCNDaily.get_station_inventory()
recent = inv[(inv.ELEMENT == "TMAX") & (inv.LASTYEAR >= INIT.year) & (inv.FIRSTYEAR <= INIT.year)].ID
st = meta[meta.ID.isin(recent) & meta.LAT.between(-12, 15) & meta.LON.between(28, 52)].set_index("ID")
print(len(st), "stations in the box")

days = [INIT + timedelta(days=k) for k in range(1, 5)]       # 4 whole UTC days inside the 5-day forecast
obs = GHCNDaily(stations=list(st.index), verbose=False)(days, ["t2m_max"])
obs = obs.assign(tmax=obs.observation - 273.15, day=pd.to_datetime(obs.time).dt.normalize())

fc_daily = fc_full.isel(lead_time=slice(4, 20)).coarsen(lead_time=4).max()       # day k = leads 4k..4k+3
fc_daily = fc_daily.assign_coords(lead_time=pd.DatetimeIndex(days)).rename(lead_time="day")
era5_daily = xr.DataArray(era5_t2m([d + timedelta(hours=6 * h) for d in days for h in range(4)]).reshape(4, 4, *fc.shape[1:]).max(1),
                          coords={"day": pd.DatetimeIndex(days), "lat": fc.lat, "lon": fc.lon})

rows = []
for _, o in obs.iterrows():
    s = st.loc[o.station]
    p = dict(lat=s.LAT, lon=s.LON % 360, day=o.day, method="nearest")
    rows.append({"station": s.NAME.strip().title(), "elev_m": s.ELEV, "obs": o.tmax,
                 "SFNO": float(fc_daily.sel(**p)), "ERA5": float(era5_daily.sel(**p))})
cmp = pd.DataFrame(rows)
per_station = cmp.groupby("station").mean().sort_values("elev_m")
per_station["SFNO-obs"] = per_station.SFNO - per_station.obs
per_station["ERA5-obs"] = per_station.ERA5 - per_station.obs
print(per_station.round(1).to_string())
print(f"\nmean abs error over {len(cmp)} station-days: SFNO {abs(cmp.SFNO-cmp.obs).mean():.1f} °C, "
      f"ERA5 {abs(cmp.ERA5-cmp.obs).mean():.1f} °C")

ax[1].scatter(per_station.elev_m, per_station["SFNO-obs"], c="#76B900", label="SFNO")
ax[1].scatter(per_station.elev_m, per_station["ERA5-obs"], c="gray", marker="s", label="ERA5")
ax[1].axhline(0, c="k", lw=0.8); ax[1].set_xlabel("station elevation (m)"); ax[1].set_ylabel("daily max error (°C)")
ax[1].set_title("Error at stations vs elevation"); ax[1].legend(); ax[1].grid(alpha=.3)
from IPython.display import display
fig.tight_layout(); fig.savefig("outputs/lab03_verification.png", dpi=120); display(fig)

# %% [markdown]
# ## What to notice
# - Against ERA5 the model looks very good: ERA5 and SFNO see the same smooth 25 km world.
# - Against stations both are worse, and the errors depend on the station: a 25 km grid box averages
#   highlands, valleys, sea and land, so a station on a coast, an island or a hill can be several degrees off.
#   Four samples a day also miss part of the afternoon peak, so the daily max tends to be too low.
#   ERA5 is a model product, not ground truth. Lab 11 learns a correction for each station from its own history.
# - Station records have gaps: not every station reports every day. Plan for that in any operational check.
#
# ## Exercise
# 1. Try a date in the long rains (April) and in the cool season (July). When does the model beat the baseline?
# 2. Which stations have the largest error? Look them up: mountain, coast, lake shore, city?
# 3. Swap `t2m_max` for `tp` (daily rain, in metres) and compare with ERA5 rain. Why is rain much harder to verify
#    at a station than temperature?
# 4. One date is an anecdote. How many would you need before trusting a score, and how long would that take on
#    the Spark? (Lab 09 runs a multi-date, multi-model scorecard.)
