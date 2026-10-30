# %% [markdown]
# # Lab 08 — Sunshine forecast for solar mini-grids and solar irrigation pumps
# **Goal:** forecast how much solar energy reaches the ground across Africa in the next 5 days, and turn it
# into numbers an operator uses: **kWh per installed kW of panels** for a mini-grid, and **cubic metres of
# water** a solar pump can lift for irrigation.
#
# Two independent forecasts, and we compare them:
# 1. **AIFS-ENS** surface solar radiation (`ssrd06`), already in the Lab 04 output: 8 members, so we get a range.
# 2. **SFNO + Earth-2's `SolarRadiationAFNO6H`**: a deterministic SFNO forecast fed into an NVIDIA diagnostic model
#    trained to predict solar radiation from the weather state. This shows how Earth-2 chains a forecast model
#    with a specialised "what does this weather mean for X" model.
#
# Requires `outputs/lab04_africa_ensemble.zarr` (Lab 04) and internet for GFS and weights on first run.
# Expected time: ~8 min.

# %%
import time
import numpy as np
import pandas as pd
import torch
import xarray as xr

def to_africa(ds):
    return ds.assign_coords(lon=((ds.lon + 180) % 360) - 180).sortby("lon")

ens = to_africa(xr.open_zarr("outputs/lab04_africa_ensemble.zarr").isel(time=0))
init = pd.Timestamp(ens.time.values).to_pydatetime()
# ssrd06 is J/m² accumulated over the 6 h ending at each lead time; lead 0 has no past 6 h.
ghi_ens = (ens["ssrd06"].isel(lead_time=slice(1, None)) / 3.6e6).load()          # kWh/m² per 6 h
ghi_ens_daily = ghi_ens.coarsen(lead_time=4, boundary="trim").sum()             # kWh/m² per day
NDAYS = ghi_ens_daily.sizes["lead_time"]
print(f"init {init:%Y-%m-%d %H} UTC, {ens.sizes['ensemble']} members, {NDAYS} days")

# %% [markdown]
# ## Forecast 2: SFNO -> SolarRadiationAFNO6H
# `run.diagnostic` runs the forecast model and, at every step, passes its output to the diagnostic model.

# %%
from earth2studio.data import GFS
from earth2studio.io import ZarrBackend
from earth2studio.models.dx import SolarRadiationAFNO6H
from earth2studio.models.px import SFNO
import earth2studio.run as run

LAT = np.arange(38, -35.01, -0.25)
LON = np.concatenate([np.arange(342, 360, 0.25), np.arange(0, 52.01, 0.25)])
t0 = time.perf_counter()
px = SFNO.load_model(SFNO.load_default_package())   # SFNO outputs every input the solar model needs
dx = SolarRadiationAFNO6H.load_model(SolarRadiationAFNO6H.load_default_package())
io = ZarrBackend(file_name="outputs/lab08_solar_afno.zarr", backend_kwargs={"overwrite": True})
torch.cuda.reset_peak_memory_stats()
run.diagnostic([init], NDAYS * 4, px, dx, GFS(), io, output_coords={"lat": LAT, "lon": LON})
print(f"SFNO + solar diagnostic: {time.perf_counter()-t0:.0f}s incl. load, "
      f"peak GPU {torch.cuda.max_memory_allocated()/2**30:.1f} GiB")
del px, dx; torch.cuda.empty_cache()

afno = to_africa(xr.open_zarr("outputs/lab08_solar_afno.zarr").isel(time=0))
ghi_afno_daily = (afno["ssrd"].isel(lead_time=slice(1, None)) / 3.6e6).load().coarsen(lead_time=4, boundary="trim").sum()

# %% [markdown]
# ## Map: average daily sunshine, and how uncertain it is
# Clouds make solar hard to forecast: where the ensemble members disagree, a mini-grid operator should keep
# more diesel or battery in reserve.

# %%
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt

mean_ens = ghi_ens_daily.mean(("ensemble", "lead_time"))
spread = ghi_ens_daily.std("ensemble").mean("lead_time")
mean_afno = ghi_afno_daily.mean("lead_time")
panels = [(mean_ens, "AIFS-ENS mean (kWh/m²/day)", "inferno", 2, 8),
          (mean_afno, "SFNO + SolarRadiationAFNO (kWh/m²/day)", "inferno", 2, 8),
          (spread, "AIFS-ENS spread between members (kWh/m²/day)", "Blues", 0, 1.5)]
fig, axs = plt.subplots(1, 3, figsize=(18, 6.5), subplot_kw={"projection": ccrs.PlateCarree()})
for a, (da, title, cmap, vmin, vmax) in zip(axs, panels):
    im = a.pcolormesh(da.lon, da.lat, da, cmap=cmap, vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
    a.coastlines(linewidth=0.6); a.add_feature(cfeature.BORDERS, linewidth=0.3)
    a.set_title(title, fontsize=10); fig.colorbar(im, ax=a, shrink=0.7, orientation="horizontal", pad=0.04)
fig.suptitle(f"Surface solar radiation, next {NDAYS} days from {init:%Y-%m-%d}")
plt.savefig("outputs/lab08_solar_maps.png", dpi=110, bbox_inches="tight"); plt.show()
diff = (mean_afno - mean_ens)
print(f"SFNO+AFNO minus AIFS-ENS over Africa: mean {float(diff.mean()):+.2f}, "
      f"90% of points within ±{float(abs(diff).quantile(0.9)):.2f} kWh/m²/day")

# %% [markdown]
# ## From sunshine to energy and water
# - **PV yield** per installed kW of panels ≈ daily irradiation (kWh/m²) × performance ratio. Panels are rated at
#   1 kW/m², so 6 kWh/m² of sun gives ~6 × 0.75 = 4.5 kWh per kWp.
# - **Solar pump**: energy E lifts water to height H: volume = E × pump efficiency / (ρ g H).

# %%
PR = 0.75          # performance ratio: heat, dust, wiring, inverter losses (placeholder)
MINIGRID_KWP = 50  # a village mini-grid
PUMP_KWP, PUMP_EFF, HEAD_M = 2.0, 0.45, 30   # a small borehole pump (placeholders)

SITES = {"Garissa, Kenya (55 MW plant)": (-0.45, 39.65), "Lodwar, Kenya (mini-grids)": (3.12, 35.60),
         "Kano, Nigeria": (12.00, 8.52), "Ouarzazate, Morocco (Noor)": (30.92, -6.90),
         "Benban, Egypt": (24.45, 32.73), "De Aar, South Africa": (-30.65, 24.00),
         "Mzuzu, Malawi": (-11.46, 34.02)}
rows, dates = [], [(init + pd.Timedelta(days=d)).strftime("%a %d") for d in range(NDAYS)]
for name, (la, lo) in SITES.items():
    p = dict(lat=la, lon=lo, method="nearest")
    e = ghi_ens_daily.sel(**p)                                    # (ensemble, day)
    kwh = e * PR * MINIGRID_KWP
    water = e * PR * PUMP_KWP * PUMP_EFF * 3.6e6 / (1000 * 9.81 * HEAD_M)
    rows.append({"site": name,
                 "sun kWh/m²/day": float(e.mean()), "AFNO kWh/m²/day": float(ghi_afno_daily.sel(**p).mean()),
                 f"{MINIGRID_KWP} kWp: kWh/day": float(kwh.mean()),
                 "worst day (low member)": f"{dates[int(kwh.min('ensemble').argmin())]}: {float(kwh.min()):.0f}",
                 "pump m³/day": float(water.mean())})
pd.DataFrame(rows).set_index("site").round(1)

# %% [markdown]
# ## Day by day at one site, with the ensemble range

# %%
SITE = "Lodwar, Kenya (mini-grids)"
la, lo = SITES[SITE]
e = ghi_ens_daily.sel(lat=la, lon=lo, method="nearest") * PR * MINIGRID_KWP
fig, ax = plt.subplots(figsize=(8, 3.5))
ax.fill_between(dates, e.min("ensemble"), e.max("ensemble"), alpha=0.3, color="orange", label="AIFS-ENS range")
ax.plot(dates, e.mean("ensemble"), "o-", c="darkorange", label="AIFS-ENS mean")
ax.plot(dates, ghi_afno_daily.sel(lat=la, lon=lo, method="nearest") * PR * MINIGRID_KWP, "s--", c="k",
        label="SFNO + SolarRadiationAFNO")
ax.set_ylabel(f"kWh/day from {MINIGRID_KWP} kWp"); ax.set_title(SITE); ax.legend(fontsize=8); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig("outputs/lab08_site.png", dpi=120); plt.show()

# %% [markdown]
# ## Exercise
# 1. A clinic on the mini-grid needs 120 kWh every day. On which days, and in how many members, does the
#    forecast fall short? How much battery would cover the worst member?
# 2. Where do the two forecasts disagree most? Is it cloudy (rainy-season) regions or deserts?
# 3. Dust (harmattan) cuts sunshine across the Sahel from December to March. Neither model sees aerosol directly:
#    how would you check how much that matters? (Hint: Earth2Studio has a `PlanetaryComputerSentinel3AOD` source.)
# 4. Wind for the same mini-grids: Lab 04 saved `u10m`/`v10m`; AIFS-ENS also has 100 m wind (`u100m`, `v100m`).
