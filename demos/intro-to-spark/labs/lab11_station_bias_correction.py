# %% [markdown]
# # Lab 11 — Make gridded data fit your stations (bias correction)
# **Goal:** gridded weather data (reanalysis, AI forecasts) describe the average of a big grid box. A farm, an
# airport or a town on a hillside is not average. With a few years of a station's own records you can learn a
# correction, and this is where a national met service's station archive becomes valuable. On the Spark, those
# records never leave the building.
#
# We use ERA5 at 1.5° (~165 km, like a coarse global model) and daily maximum temperature from ~30 stations in
# Kenya, Ethiopia, Tanzania and Uganda (NOAA GHCN-Daily). Train on 2010–2019, test on 2020–2022.
#
# Requires internet the first time (~1 GB of ERA5 read, cached as a small file). Expected time: ~5 min.

# %%
import os
import time
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime

from earth2studio.data import GHCNDaily

YEARS = (2010, 2022)
meta, inv = GHCNDaily.get_station_metadata(), GHCNDaily.get_station_inventory()
ok = inv[(inv.ELEMENT == "TMAX") & (inv.FIRSTYEAR <= YEARS[0]) & (inv.LASTYEAR >= YEARS[1])].ID
st = meta[meta.ID.isin(ok) & meta.ID.str[:2].isin(["KE", "ET", "TZ", "UG"])].set_index("ID")
print(len(st), "stations")

CACHE = "outputs/lab11_station_tmax.parquet"
if not os.path.exists(CACHE):
    days = pd.date_range(f"{YEARS[0]}-01-01", f"{YEARS[1]}-12-31", freq="D").to_pydatetime().tolist()
    obs = GHCNDaily(stations=list(st.index), verbose=False)(days, ["t2m_max"])
    obs.to_parquet(CACHE)
obs = pd.read_parquet(CACHE)
obs = obs.assign(day=pd.to_datetime(obs.time).dt.normalize(), tmax=obs.observation - 273.15)[["station", "day", "tmax"]]
print(f"{len(obs)} station-days")

# %% [markdown]
# ## ERA5 at the nearest grid point
# Daily max = max of the 00/06/12/18 UTC values (03/09/15/21 East Africa Time), as a forecast model would give.

# %%
ECACHE = "outputs/lab11_era5_tmax_1p5deg.nc"
if not os.path.exists(ECACHE):
    t0 = time.perf_counter()
    src = xr.open_zarr("gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr",
                       storage_options={"token": "anon"})["2m_temperature"]
    src = src.sel(time=slice(f"{YEARS[0]}-01-01", f"{YEARS[1]}-12-31"), latitude=slice(-15, 18), longitude=slice(27, 50))
    (src.resample(time="1D").max() - 273.15).rename("tmax").to_netcdf(ECACHE)
    print(f"ERA5 cached in {time.perf_counter()-t0:.0f}s")
era5 = xr.open_dataarray(ECACHE)

grid = era5.sel(latitude=xr.DataArray(st.LAT.values, dims="station"), longitude=xr.DataArray(st.LON.values, dims="station"),
                method="nearest")
gdf = grid.assign_coords(station=st.index.values).to_dataframe("era5").reset_index()[["station", "time", "era5"]]
df = obs.merge(gdf.rename(columns={"time": "day"}), on=["station", "day"]).dropna()
df = df.join(st[["LAT", "LON", "ELEV", "NAME"]], on="station")
doy = df.day.dt.dayofyear
df["sin"], df["cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
train, test = df[df.day.dt.year <= 2019], df[df.day.dt.year >= 2020]
print(f"train {len(train)} / test {len(test)} station-days")

# %% [markdown]
# ## Three ways to get a station's temperature
# 1. **Raw**: just read the grid box.
# 2. **Per-station linear correction** (classic "model output statistics"): for each station,
#    `station = a + b·grid + c·sin(season) + d·cos(season)`, fitted on its own history.
# 3. **One machine-learning model for all stations** (gradient boosting), with the station's latitude,
#    longitude and elevation as inputs. Tested on stations it **never saw**, it tells us whether we can correct
#    places that have no history.

# %%
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression

pred = test[["station", "day", "tmax", "era5", "NAME", "ELEV"]].copy()
pred["raw"] = test.era5

pred["linear"] = np.nan
for sid, g in train.groupby("station"):
    m = LinearRegression().fit(g[["era5", "sin", "cos"]], g.tmax)
    idx = test.station == sid
    if idx.any():
        pred.loc[idx, "linear"] = m.predict(test.loc[idx, ["era5", "sin", "cos"]])

X = ["era5", "sin", "cos", "LAT", "LON", "ELEV"]
ids = st.index.values
rng = np.random.default_rng(0); rng.shuffle(ids)
folds = np.array_split(ids, 5)                    # leave-stations-out: each station is predicted by a model
pred["ml_unseen"] = np.nan                         # trained without it
t0 = time.perf_counter()
for f in folds:
    m = HistGradientBoostingRegressor(max_iter=300).fit(train.loc[~train.station.isin(f), X], train.loc[~train.station.isin(f), "tmax"])
    idx = test.station.isin(f)
    pred.loc[idx, "ml_unseen"] = m.predict(test.loc[idx, X])
print(f"5 gradient-boosting fits in {time.perf_counter()-t0:.0f}s")

mae = lambda c: (pred[c] - pred.tmax).abs().mean()
print(f"\nMean absolute error, 2020–2022 ({len(pred)} station-days):")
for c, label in [("raw", "raw grid box"), ("linear", "per-station linear"), ("ml_unseen", "one ML model, unseen stations")]:
    print(f"  {label:32s} {mae(c):.2f} °C")

# %% [markdown]
# ## Station by station

# %%
per = pred.groupby("NAME").apply(lambda g: pd.Series({
    "elev_m": g.ELEV.iloc[0], "raw": (g.raw - g.tmax).abs().mean(), "linear": (g.linear - g.tmax).abs().mean(),
    "ml_unseen": (g.ml_unseen - g.tmax).abs().mean(), "raw_bias": (g.raw - g.tmax).mean()}), include_groups=False)
per = per.sort_values("elev_m")
per.index = per.index.str.strip().str.title()

import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
ax[0].scatter(per.elev_m, per.raw_bias, c="gray")
for name, r in per.iterrows():
    ax[0].annotate(name, (r.elev_m, r.raw_bias), fontsize=6)
ax[0].axhline(0, c="k", lw=.8); ax[0].set_xlabel("station elevation (m)"); ax[0].set_ylabel("grid minus station, daily max (°C)")
ax[0].set_title("Raw 1.5° grid: high stations are colder than their grid box")
x = np.arange(len(per))
for k, (c, col) in enumerate([("raw", "gray"), ("linear", "#76B900"), ("ml_unseen", "tab:blue")]):
    ax[1].bar(x + (k - 1) * 0.27, per[c], 0.27, color=col, label=c)
ax[1].set_xticks(x, per.index, rotation=90, fontsize=7); ax[1].set_ylabel("mean abs error (°C)"); ax[1].legend()
ax[1].set_title("Error per station, 2020–2022")
plt.tight_layout(); plt.savefig("outputs/lab11_bias_correction.png", dpi=110, bbox_inches="tight"); plt.show()
per.round(2)

# %% [markdown]
# ## What to notice
# - Most of the raw error is a fixed offset that depends on height and setting (lake shore, coast, city), so
#   even a simple per-station correction removes most of it.
# - The single ML model barely helps at stations it never saw, and is worse at several (on our run: 2.2 °C raw,
#   0.8 °C per-station, 1.9 °C unseen). ~20 stations are too few to learn how lake shores, rift valleys and
#   escarpments change temperature. **A station's own history is worth far more than a clever model**, which
#   is the case for sharing more national station data, at least inside the country.
# - In operations you would train on the **forecast model's own past forecasts** (a "hindcast" set, e.g. a year
#   of Lab 09-style runs) rather than ERA5, so the correction also learns the model's errors at each lead time.
#
# ## Exercise
# 1. Add `t2m_min` (night temperature: frost on Kenyan tea and Ethiopian highland crops). Is it harder?
# 2. Train with only 2 years of history. How much skill do you lose?
# 3. Swap in rainfall (`tp`). Why does a simple correction work so much worse for rain?
# 4. Your met service has hundreds more stations than GHCN. What would it take to run this weekly on the Spark?
