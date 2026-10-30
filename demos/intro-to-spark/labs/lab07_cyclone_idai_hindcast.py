# %% [markdown]
# # Lab 07 — Cyclone Idai: would AI have seen it coming?
# **Goal:** rerun AI weather models from 10 March 2019, five days before Tropical Cyclone Idai hit Beira,
# Mozambique (the night of 14–15 March), and compare their storm tracks with the official best track.
# Idai killed over 1,000 people in Mozambique, Zimbabwe and Malawi and flooded ~3,000 km² of farmland.
#
# - Initial conditions: ERA5 (ARCO), exactly as an operational centre would have had them (approximately).
# - Truth: **IBTrACS**, the global archive of cyclone best tracks, via Earth2Studio.
# - Tracker: a simple "follow the lowest pressure" tracker you can read in 15 lines.
#
# Requires internet (ERA5, IBTrACS, model weights on first run). Expected time: ~10 min.

# %%
from datetime import datetime, timedelta
import time
import numpy as np
import pandas as pd
import torch
import xarray as xr

from earth2studio.data import ARCO, IBTrACS
from earth2studio.io import ZarrBackend
from earth2studio.models.px import AIFS, SFNO, Aurora
import earth2studio.run as run

INIT = datetime(2019, 3, 10)
NSTEPS = 24                       # 6 days, past landfall
BEIRA = (-19.84, 34.84)
LAT = np.arange(-8, -32.01, -0.25)           # Mozambique Channel and the coast
LON = np.arange(28, 56.01, 0.25)
valid = [INIT + timedelta(hours=6 * i) for i in range(NSTEPS + 1)]

# %% [markdown]
# ## 1. The truth: Idai's best track

# %%
ib = IBTrACS(region="SI", verbose=False)(valid, ["mslp", "tcwnd"])
ib = ib[ib.storm_name == "IDAI"]
best = (ib.pivot_table(index="time", columns="variable", values="observation", aggfunc="mean")
          .join(ib.groupby("time")[["lat", "lon"]].mean()))
best.index = pd.to_datetime(best.index).round("h")
if best["mslp"].median() > 2000:     # Pa -> hPa
    best["mslp"] /= 100
best.round(1)

# %% [markdown]
# ## 2. Rerun three AI models from 10 March: SFNO (NVIDIA), AIFS (ECMWF), Aurora (Microsoft)
# Each model only needs ERA5 for the start time; everything after that is forecast. We keep sea-level pressure
# and 10 m wind in a box over the Mozambique Channel.

# %%
arco = ARCO()
tracks_in = {}
for name, M in [("SFNO", SFNO), ("AIFS", AIFS), ("Aurora", Aurora)]:
    t0 = time.perf_counter()
    model = M.load_model(M.load_default_package())
    path = f"outputs/lab07_{name.lower()}.zarr"
    io = ZarrBackend(file_name=path, backend_kwargs={"overwrite": True})
    torch.cuda.reset_peak_memory_stats()
    run.deterministic([INIT], NSTEPS, model, arco, io,
                      output_coords={"variable": np.array(["msl", "u10m", "v10m"]), "lat": LAT, "lon": LON})
    tracks_in[name] = xr.open_zarr(path).isel(time=0).load()
    print(f"{name}: {time.perf_counter()-t0:.0f}s, peak GPU {torch.cuda.max_memory_allocated()/2**30:.1f} GiB")
    del model; torch.cuda.empty_cache()

# %% [markdown]
# ## 3. Track the storm
# Start at the best-track position at 10 March 00 UTC. At each 6-hour step look within 3° (~330 km) of the last
# position for the lowest sea-level pressure: that is the new centre. Record its pressure and the strongest
# 10 m wind within 2° of it.

# %%
def track(ds, lat0, lon0, search=3.0, radius=2.0):
    out, la, lo = [], lat0, lon0
    wind = np.hypot(ds.u10m, ds.v10m)
    for i in range(ds.sizes["lead_time"]):
        box = ds.msl.isel(lead_time=i).sel(lat=slice(la + search, la - search), lon=slice(lo - search, lo + search))
        j = np.unravel_index(int(box.argmin()), box.shape)
        la, lo = float(box.lat[j[0]]), float(box.lon[j[1]])
        w = wind.isel(lead_time=i).sel(lat=slice(la + radius, la - radius), lon=slice(lo - radius, lo + radius))
        out.append({"time": valid[i], "lat": la, "lon": lo, "mslp": float(box.min()) / 100, "wind": float(w.max())})
    return pd.DataFrame(out).set_index("time")

def km(a_lat, a_lon, b_lat, b_lon):   # great-circle distance
    a_lat, a_lon, b_lat, b_lon = map(np.deg2rad, (a_lat, a_lon, b_lat, b_lon))
    h = np.sin((b_lat - a_lat) / 2) ** 2 + np.cos(a_lat) * np.cos(b_lat) * np.sin((b_lon - a_lon) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(h))

lat0, lon0 = best.loc[pd.Timestamp(INIT), ["lat", "lon"]]
tracks = {name: track(ds, lat0, lon0) for name, ds in tracks_in.items()}
errors = pd.DataFrame({name: km(tr.lat, tr.lon, best.lat.reindex(tr.index), best.lon.reindex(tr.index))
                       for name, tr in tracks.items()})
errors.index = (errors.index - pd.Timestamp(INIT)) / pd.Timedelta(hours=1)
print("track error (km) by lead time (h):")
print(errors.iloc[::4].round(0).to_string())

LANDFALL = pd.Timestamp("2019-03-15 00:00")
print(f"\nAt {LANDFALL:%d %b %H UTC} (landfall, 5 days ahead): distance from Beira")
print(f"  best track: {km(*best.loc[LANDFALL, ['lat','lon']], *BEIRA):.0f} km")
for name, tr in tracks.items():
    print(f"  {name:12s}: {km(*tr.loc[LANDFALL, ['lat','lon']], *BEIRA):.0f} km, centre {tr.loc[LANDFALL,'mslp']:.0f} hPa")

# %% [markdown]
# ## 4. Plot

# %%
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt

colors = {"SFNO": "#76B900", "AIFS": "tab:blue", "Aurora": "tab:purple"}
fig = plt.figure(figsize=(16, 6))
a = fig.add_subplot(1, 3, 1, projection=ccrs.PlateCarree())
a.set_extent([30, 50, -28, -12]); a.coastlines(); a.add_feature(cfeature.BORDERS, linewidth=0.4)
a.add_feature(cfeature.LAND, facecolor="#f2efe6")
a.plot(best.lon, best.lat, "k-o", ms=3, label="best track (IBTrACS)", transform=ccrs.PlateCarree())
for name, tr in tracks.items():
    a.plot(tr.lon, tr.lat, "-o", ms=2, c=colors[name], label=name, transform=ccrs.PlateCarree())
a.plot(BEIRA[1], BEIRA[0], "r*", ms=14, transform=ccrs.PlateCarree()); a.text(BEIRA[1] - 3.2, BEIRA[0], "Beira", c="r")
a.legend(loc="lower right", fontsize=8); a.set_title(f"Tracks from {INIT:%d %b %Y}")

b = fig.add_subplot(1, 3, 2)
b.plot(best.index, best.mslp, "k-o", ms=3, label="best track")
for name, tr in tracks.items():
    b.plot(tr.index, tr.mslp, "-", c=colors[name], label=name)
b.axvline(LANDFALL, c="r", ls="--"); b.set_ylabel("central pressure (hPa)"); b.legend(fontsize=8)
b.tick_params(axis="x", rotation=45); b.set_title("Intensity")

c = fig.add_subplot(1, 3, 3)
for name in errors:
    c.plot(errors.index, errors[name], "-o", ms=3, c=colors[name], label=name)
c.set_xlabel("lead time (h)"); c.set_ylabel("track error (km)"); c.grid(alpha=.3); c.legend(fontsize=8)
c.set_title("Position error")
plt.tight_layout(); plt.savefig("outputs/lab07_idai.png", dpi=110, bbox_inches="tight"); plt.show()

# %% [markdown]
# ## What to notice
# - **Track** is the useful part. In our run all three models, started 5 days before landfall, brought Idai
#   ashore in central Mozambique, but ~160–180 km south of Beira; errors stayed under ~100 km for 2–3 days
#   (Aurora best early on). "Central Mozambique coast, Sofala province" 5 days ahead is already actionable:
#   move people, harvest early, pre-position aid.
# - **Intensity** is badly underestimated: 970–990 hPa against a real 940–960. ERA5 itself smooths a cyclone's core at 25 km, and models trained on
#   it inherit that. Never read an AI model's central pressure as the storm's real strength.
# - Idai's worst impact was **flooding** from days of rain over the Buzi and Pungwe basins. SFNO and Aurora don't
#   output rain; AIFS does (`tp06`). Add it to the saved variables and map the rain over Sofala province.
#
# ## Exercise
# 1. Start one day later (11 March). How much does the track error at landfall shrink?
# 2. Try Cyclone Freddy's second landfall (init 2023-03-06, landfall near Quelimane on 11 March). Freddy lasted
#    over five weeks, a record.
# 3. How many hours of warning does each model give that the centre will pass within 100 km of Beira?
# 4. Would an ensemble help? What does that cost on the Spark (Lab 01)?
