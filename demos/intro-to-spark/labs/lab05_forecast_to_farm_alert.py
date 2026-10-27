# %% [markdown]
# # Lab 05 — From forecast to farm advisory
# **Goal:** turn the Lab 03 forecast into impact-based information: heat stress and spraying-window flags
# for a point of interest. This is the "forecast -> thresholds -> message" link of the training workflow.
#
# Requires `outputs/lab03_forecast.zarr` from Lab 03. Thresholds are **placeholders**: set your own.

# %%
import numpy as np
import pandas as pd
import xarray as xr

ds = xr.open_zarr("outputs/lab03_forecast.zarr")
init = pd.Timestamp(ds["time"].values[0])
lead = pd.to_timedelta(ds["lead_time"].values)

# Farmers plan in local days, not UTC. Change for your country.
TZ = "Asia/Dubai"
valid_local = (init + lead).tz_localize("UTC").tz_convert(TZ)

# Point of interest: Al Ain agricultural area (approx.). Replace with a farm/region of your choice.
LAT, LON = 24.2, 55.7
pt = ds.sel(lat=LAT, lon=LON, method="nearest").isel(time=0)

df = pd.DataFrame({
    "valid_local": valid_local,
    "t2m_C": pt["t2m"].values - 273.15,
    "wind10_ms": np.hypot(pt["u10m"].values, pt["v10m"].values),
})
df.head()

# %% [markdown]
# ## Thresholds (edit these)
# Caveat: the model outputs every 6 h (04, 10, 16, 22 local time in the UAE), so the "daily max" below is the
# max of four samples and can miss the true afternoon peak. Ask: does that bias the heat alert up or down?

# %%
HEAT_C = 35.0       # placeholder chosen so the alert fires in November (Al Ain highs ~35 °C); summer stress is ~42+
SPRAY_MAX_WIND = 4  # m/s, safe spraying limit, placeholder

df["heat_stress"] = df["t2m_C"] > HEAT_C
df["spray_ok"] = df["wind10_ms"] < SPRAY_MAX_WIND
df["date"] = df["valid_local"].dt.date

daily = df.groupby("date").agg(
    tmax_C=("t2m_C", "max"),
    wind_max_ms=("wind10_ms", "max"),
    heat_stress=("heat_stress", "any"),
    spray_hours=("spray_ok", "sum"),
    samples=("t2m_C", "size"),
)
daily["spray_hours"] *= 6  # each step = 6 h
daily = daily[daily["samples"] == 4]  # drop partial first/last days
daily.round(1)

# %% [markdown]
# ## Plot

# %%
import matplotlib.pyplot as plt
fig, ax = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
ax[0].plot(df["valid_local"], df["t2m_C"], "o-", c="tab:red"); ax[0].axhline(HEAT_C, ls="--", c="k")
ax[0].set_ylabel("2 m temp (°C)")
ax[1].plot(df["valid_local"], df["wind10_ms"], "o-", c="tab:blue"); ax[1].axhline(SPRAY_MAX_WIND, ls="--", c="k")
ax[1].set_ylabel("10 m wind (m/s)"); ax[1].set_xlabel(f"local time ({TZ})")
fig.suptitle(f"AI forecast at ({LAT}N, {LON}E) with advisory thresholds")
plt.tight_layout(); plt.savefig("outputs/lab05_point_forecast.png", dpi=120); plt.show()

# %% [markdown]
# ## Draft a message
# The Ag track covers framing and channels. Here we generate a plain-language draft from the table.

# %%
def advisory(day, row):
    parts = []
    if row.heat_stress:
        parts.append(f"high heat expected (up to {row.tmax_C:.0f}°C): irrigate early morning, shade livestock")
    if row.spray_hours >= 6:
        parts.append(f"calm enough to spray for about {int(row.spray_hours)} h")
    else:
        parts.append("too windy for spraying; wait")
    return f"{pd.Timestamp(day):%a %d %b}: " + "; ".join(parts) + "."

for day, row in daily.iterrows():
    print(advisory(day, row))

# %% [markdown]
# ## Exercise
# 1. Change the point to another region and thresholds to another crop. How do alerts change?
# 2. A single deterministic forecast has no uncertainty. How would an ensemble change the message
#    ("70% chance of > 42 °C")? What would that cost on Spark (Lab 04)?
# 3. Before trusting any alert, check the model's skill: Lab 06.
