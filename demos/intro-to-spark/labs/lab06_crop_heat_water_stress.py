# %% [markdown]
# # Lab 06 — Crop and livestock stress across Africa
# **Goal:** turn the Lab 04 ensemble into the stress indicators agronomists actually use, with a probability
# for each: **heat days** (damage to flowering maize), **vapour-pressure deficit** (how hard the air pulls water
# out of plants), **dry spells**, and the **temperature-humidity index** for cattle.
#
# Requires `outputs/lab04_africa_ensemble.zarr` from Lab 04 (no GPU, no download). Expected time: 2 min.
# All thresholds are **placeholders** from the general literature: set them with your crop and livestock experts.

# %%
import numpy as np
import pandas as pd
import xarray as xr

ds = xr.open_zarr("outputs/lab04_africa_ensemble.zarr").isel(time=0)
ds = ds.assign_coords(lon=((ds.lon + 180) % 360) - 180).sortby("lon")   # wrapped 342..52 -> -18..52
init = pd.Timestamp(ds.time.values)
ds = ds.isel(lead_time=slice(1, None)).load()          # drop the initial state
NENS, NDAYS = ds.sizes["ensemble"], ds.sizes["lead_time"] // 4
t = ds["t2m"] - 273.15                                  # °C, every 6 h
td = ds["d2m"] - 273.15
print(f"init {init:%Y-%m-%d %H} UTC | {NENS} members | {NDAYS} days")

HEAT_C = 35        # maize pollen dies above ~35 °C during flowering (placeholder)
VPD_HIGH = 3.0     # kPa: most crops close their stomata and stop growing (placeholder)
DRY_MM = 5         # 5-day rain below this = dry spell (placeholder)
THI_DANGER = 79    # cattle: "danger" band in the common THI scale (placeholder)

# %% [markdown]
# ## The indicators
# Vapour pressure from temperature (Tetens formula): `e(T) = 0.6108 exp(17.27 T / (T + 237.3))` kPa.
# VPD = e(air temperature) - e(dewpoint). Relative humidity = e(dewpoint) / e(air temperature).
# (Earth2Studio also ships `DerivedVPD` and `DerivedRHDewpoint` diagnostics that do the same on the GPU.)

# %%
e = lambda T: 0.6108 * np.exp(17.27 * T / (T + 237.3))
vpd = e(t) - e(td)                                      # kPa
rh = (100 * e(td) / e(t)).clip(max=100)
thi = 0.8 * t + rh / 100 * (t - 14.4) + 46.4           # temperature-humidity index (NRC 1971 form)

daily = lambda da, how: getattr(da.coarsen(lead_time=4, boundary="trim"), how)()
tmax, vpdmax, thimax = daily(t, "max"), daily(vpd, "max"), daily(thi, "max")
rain5 = (ds["tp06"].clip(min=0) * 1000).sum("lead_time")   # mm in 5 days

indicators = {
    f"Chance of 2+ days above {HEAT_C} °C (%)": ((tmax > HEAT_C).sum("lead_time") >= 2).mean("ensemble") * 100,
    "Peak vapour-pressure deficit (kPa, ensemble mean)": vpdmax.max("lead_time").mean("ensemble"),
    f"Chance of a dry 5 days, < {DRY_MM} mm (%)": (rain5 < DRY_MM).mean("ensemble") * 100,
    f"Chance of 2+ days of cattle heat danger, THI > {THI_DANGER} (%)":
        ((thimax > THI_DANGER).sum("lead_time") >= 2).mean("ensemble") * 100,
}

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt

styles = [("Reds", 0, 100), ("YlOrBr", 0, 6), ("Oranges", 0, 100), ("RdPu", 0, 100)]
fig, axs = plt.subplots(2, 2, figsize=(14, 14), subplot_kw={"projection": ccrs.PlateCarree()})
for a, (title, da), (cmap, vmin, vmax) in zip(axs.flat, indicators.items(), styles):
    im = a.pcolormesh(da.lon, da.lat, da, cmap=cmap, vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
    a.add_feature(cfeature.OCEAN, facecolor="white", zorder=2)   # land only: these are farm indicators
    a.coastlines(linewidth=0.6, zorder=3); a.add_feature(cfeature.BORDERS, linewidth=0.3, zorder=3)
    a.set_title(title, fontsize=10); fig.colorbar(im, ax=a, shrink=0.75)
fig.suptitle(f"Crop and livestock stress, next {NDAYS} days (AIFS-ENS, {NENS} members, init {init:%Y-%m-%d})")
plt.tight_layout(); plt.savefig("outputs/lab06_stress_maps.png", dpi=100, bbox_inches="tight"); plt.show()

# %% [markdown]
# ## A table per farming region
# Averages over a few major agricultural zones. A ministry or a cooperative would use its own districts
# (e.g. from a shapefile) in exactly the same way.

# %%
ZONES = {   # (lat north, lat south, lon west, lon east)
    "Sahel (Senegal–Chad)":           (16, 11, -17, 20),
    "Nigeria middle belt":            (11, 7, 3, 12),
    "Ethiopian highlands":            (14, 6, 36, 41),
    "Kenya & Uganda highlands":       (2, -2, 31, 38),
    "Tanzania southern highlands":    (-7, -10, 32, 37),
    "Zambia–Malawi maize belt":       (-12, -16, 25, 35),
    "South Africa Highveld":          (-25, -28, 26, 31),
}
rows = {}
for name, (n, s, w_, e_) in ZONES.items():
    w = np.cos(np.deg2rad(ds.lat.sel(lat=slice(n, s))))
    rows[name] = {k.split(" (")[0]: float(v.sel(lat=slice(n, s), lon=slice(w_, e_)).weighted(w).mean())
                  for k, v in indicators.items()}
table = pd.DataFrame(rows).T.round(1)
table.columns = ["% 2+ hot days", "peak VPD kPa", "% dry 5 days", "% cattle heat danger"]
table

# %% [markdown]
# ## How sure are we? Look at one place, member by member
# A probability of 50 % with 8 members means 4 members say yes. Seeing them makes that concrete.

# %%
PLACE, (LA, LO) = "Kano, Nigeria", (12.0, 8.5)
fig, ax = plt.subplots(1, 2, figsize=(12, 3.5))
days = np.arange(1, NDAYS + 1)
for m in range(NENS):
    ax[0].plot(days, tmax.sel(lat=LA, lon=LO, method="nearest").isel(ensemble=m), c="tab:red", alpha=0.35)
    ax[1].plot(days, vpdmax.sel(lat=LA, lon=LO, method="nearest").isel(ensemble=m), c="tab:brown", alpha=0.35)
ax[0].axhline(HEAT_C, ls="--", c="k"); ax[0].set_ylabel("daily max °C")
ax[1].axhline(VPD_HIGH, ls="--", c="k"); ax[1].set_ylabel("daily max VPD (kPa)")
for a in ax:
    a.set_xlabel("day"); a.set_title(PLACE)
plt.tight_layout(); plt.savefig("outputs/lab06_point.png", dpi=120); plt.show()

# %% [markdown]
# ## Exercise
# 1. Which zone is under the most stress this week? Is that the crop's sensitive stage (flowering) right now?
# 2. The model gives 4 samples a day (00/06/12/18 UTC). In West Africa the hottest sample is ~13:00, in the
#    Horn ~15:00. Do the "hot days" undercount the true afternoon maximum? Check against Lab 03's station data.
# 3. Combine indicators into one alert: "hot AND dry". How would you choose between a strict rule (all members
#    agree) and a loose one (2 of 8)? Who pays for a false alarm, and who pays for a miss?
# 4. Replace the zones with your country's districts. (Hint: `regionmask` or `rioxarray` can clip to a shapefile.)
