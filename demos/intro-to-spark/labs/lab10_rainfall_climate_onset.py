# %% [markdown]
# # Lab 10 — 42 years of African rainfall: normal, onset of the rains, and this week's anomaly
# **Goal:** use the Spark as a data-processing box, not just a model runner. From 42 years of ERA5 rainfall
# (1981–2022) we compute:
# 1. the **normal** (climatology) for every day of the year;
# 2. the **onset of the rains** in every year at every grid point, with an agronomic definition, and whether it
#    is getting later; on the GPU, testing **24 onset definitions at once**;
# 3. whether **this week's forecast** (Lab 04) is wetter or drier than normal.
#
# Data: WeatherBench 2's copy of ERA5 at 1.5° (public, Google Cloud). The Africa subset is downloaded once
# (~3–5 GB read, a few minutes) and cached as a 150 MB file. Expected time: ~10 min first run, 2 min after.
# Lab 04's output is needed for the last section only.

# %%
import os
import time
import numpy as np
import pandas as pd
import xarray as xr

CACHE = "outputs/lab10_era5_daily_rain_africa.nc"
if not os.path.exists(CACHE):
    t0 = time.perf_counter()
    src = xr.open_zarr("gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr",
                       storage_options={"token": "anon"})["total_precipitation_6hr"]
    lon = src.longitude.values
    src = src.isel(longitude=np.where((lon >= 342) | (lon <= 52))[0]).sel(latitude=slice(-35, 38),
                                                                            time=slice("1981-01-01", "2022-12-31"))
    daily = (src.resample(time="1D").sum() * 1000).astype("float32")      # m per 6 h -> mm per day
    daily = daily.rename(latitude="lat", longitude="lon")
    daily = daily.assign_coords(lon=((daily.lon + 180) % 360) - 180).sortby("lon").sortby("lat", ascending=False)
    daily.to_dataset(name="tp").to_netcdf(CACHE)
    print(f"downloaded and cached in {time.perf_counter()-t0:.0f}s")
tp = xr.open_dataset(CACHE)["tp"].transpose("time", "lat", "lon").load()   # (source order is lon, lat)
tp = tp.sel(time=~((tp.time.dt.month == 2) & (tp.time.dt.day == 29)))   # 365-day years keep the arithmetic simple
print(tp.sizes, f"{tp.nbytes/2**20:.0f} MB in memory")

# %% [markdown]
# ## 1. The normal: mean annual rainfall and the seasonal cycle

# %%
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt

annual = tp.groupby("time.year").sum().mean("year")
doy_clim = tp.groupby("time.dayofyear").mean()                    # (365, lat, lon), mm/day
doy_clim = doy_clim.rolling(dayofyear=31, center=True, min_periods=1).mean()   # smooth daily noise

PLACES = {"Niamey": (13.5, 2.1), "Kano": (12.0, 8.5), "Addis Ababa": (9.0, 38.7), "Nairobi": (-1.3, 36.8),
          "Kinshasa": (-4.4, 15.3), "Lusaka": (-15.4, 28.3)}
fig = plt.figure(figsize=(16, 6))
a = fig.add_subplot(1, 2, 1, projection=ccrs.PlateCarree())
im = a.pcolormesh(annual.lon, annual.lat, annual, cmap="YlGnBu", vmin=0, vmax=2500, transform=ccrs.PlateCarree())
a.coastlines(); a.add_feature(cfeature.BORDERS, linewidth=0.3); fig.colorbar(im, ax=a, shrink=0.8, label="mm/year")
for n, (la, lo) in PLACES.items():
    a.plot(lo, la, "ro", ms=3, transform=ccrs.PlateCarree()); a.text(lo + 0.8, la, n, fontsize=8, transform=ccrs.PlateCarree())
a.set_title("Mean annual rainfall, ERA5 1981–2022")
b = fig.add_subplot(1, 2, 2)
for n, (la, lo) in PLACES.items():
    b.plot(doy_clim.dayofyear, doy_clim.sel(lat=la, lon=lo, method="nearest"), label=n)
b.set_xticks([1, 60, 121, 182, 244, 305], ["Jan", "Mar", "May", "Jul", "Sep", "Nov"])
b.set_ylabel("mm/day (31-day smoothed)"); b.legend(); b.grid(alpha=.3); b.set_title("One rainy season or two?")
plt.tight_layout(); plt.savefig("outputs/lab10_climate.png", dpi=110, bbox_inches="tight"); plt.show()

# %% [markdown]
# ## 2. Onset of the rains, every year, every grid point
# A common agronomic definition (variants of Stern et al. 1981, Sivakumar 1988): the onset is the first day,
# after a search start date, when **3-day rain reaches 20 mm** and **no dry spell of 7+ days (< 1 mm/day)
# follows within 30 days** (otherwise the seedlings die: a "false start").
#
# We compute it with array operations only (running sums), so the same code runs on the CPU with NumPy or on
# the GPU with CuPy. Two search windows cover most of Africa: from **1 March** (West African monsoon, East
# Africa long rains) and from **1 September** (East Africa short rains, southern Africa summer rains).

# %%
import cupy as cp

def onset(xp, R, min3=20.0, dry_len=7, look=30, dry_mm=1.0):
    """R: (year, day_in_window, lat, lon) rain. Returns onset day index (NaN if none)."""
    c = xp.cumsum(R, axis=1); c = xp.concatenate([xp.zeros_like(c[:, :1]), c], axis=1)
    wet3 = (c[:, 3:] - c[:, :-3]) >= min3                                   # 3-day total from day d
    dry = (R < dry_mm).astype(xp.int32)
    cd = xp.cumsum(dry, axis=1); cd = xp.concatenate([xp.zeros_like(cd[:, :1]), cd], axis=1)
    spell = (cd[:, dry_len:] - cd[:, :-dry_len]) == dry_len                 # a dry spell starts on day d
    cs = xp.cumsum(spell, axis=1); cs = xp.concatenate([xp.zeros_like(cs[:, :1]), cs], axis=1)
    n = R.shape[1] - look - 3
    false_start = (cs[:, 1 + look - dry_len + 1: 1 + look - dry_len + 1 + n] - cs[:, 1:1 + n]) > 0
    ok = wet3[:, :n] & ~false_start
    first = xp.argmax(ok, axis=1).astype(xp.float32)
    return xp.where(ok.any(axis=1), first, xp.nan)

def windows(start_month, length=200):
    """Stack one window per year starting on day 1 of start_month: (year, day, lat, lon)."""
    years = range(1981, 2022)                       # 2022 windows would run past the end of the data
    starts = [np.where((tp.time.dt.year == y) & (tp.time.dt.month == start_month) & (tp.time.dt.day == 1))[0][0]
              for y in years]
    return np.stack([tp.values[s:s + length] for s in starts]), list(years)

results = {}
for name, month in [("from 1 March", 3), ("from 1 September", 9)]:
    R, years = windows(month)
    t0 = time.perf_counter(); on_cpu = onset(np, R); t_cpu = time.perf_counter() - t0
    Rg = cp.asarray(R); onset(cp, Rg); cp.cuda.Device().synchronize()            # warm-up (kernel compile)
    t0 = time.perf_counter(); on_gpu = onset(cp, Rg); cp.cuda.Device().synchronize(); t_gpu = time.perf_counter() - t0
    assert np.allclose(on_cpu, cp.asnumpy(on_gpu), equal_nan=True)
    results[name] = xr.DataArray(on_cpu, dims=("year", "lat", "lon"), coords={"year": years, "lat": tp.lat, "lon": tp.lon})
    print(f"onset {name}: {R.shape} -> CPU {t_cpu*1e3:.0f} ms, GPU {t_gpu*1e3:.1f} ms")

# %% [markdown]
# ### Is the onset getting later?
# Median onset date, and the trend in days per decade (linear fit over 1981–2021). Grey = no reliable onset in
# most years (deserts, or rain all year).

# %%
fig, axs = plt.subplots(2, 2, figsize=(14, 13), subplot_kw={"projection": ccrs.PlateCarree()})
for row, (name, on) in zip(axs, results.items()):
    valid = on.notnull().mean("year") > 0.7
    med = on.median("year").where(valid)
    trend = on.polyfit("year", 1, skipna=True).polyfit_coefficients.sel(degree=1).where(valid) * 10
    start = pd.Timestamp(f"2001-{3 if 'March' in name else 9:02d}-01")
    ticks = [0, 30, 61, 91, 122, 152]
    for a, (da, title, cmap, vmin, vmax) in zip(row, [(med, f"Median onset, days after {start:%d %B}", "viridis", 0, 150),
                                                      (trend, f"Onset trend {name} (days/decade, + = later)", "RdBu_r", -10, 10)]):
        a.set_facecolor("#ddd")
        im = a.pcolormesh(da.lon, da.lat, da, cmap=cmap, vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
        a.add_feature(cfeature.OCEAN, facecolor="white", zorder=2)       # farmers' question: land only
        a.coastlines(zorder=3); a.add_feature(cfeature.BORDERS, linewidth=0.3); a.set_title(title, fontsize=10)
        cb = fig.colorbar(im, ax=a, shrink=0.7)
        if cmap == "viridis":
            cb.set_ticks(ticks); cb.set_ticklabels([(start + pd.Timedelta(days=t)).strftime("%d %b") for t in ticks])
plt.tight_layout(); plt.savefig("outputs/lab10_onset.png", dpi=100, bbox_inches="tight"); plt.show()

# %% [markdown]
# ### Agronomists argue about the definition. Test 24 of them on the GPU.
# How much does the onset map change if the 3-day threshold is 15–40 mm and the dry spell 5–10 days? On the
# CPU this is a coffee break at 0.25° resolution; here it is seconds even at 1.5°.

# %%
R, years = windows(3)
Rg = cp.asarray(R)
grid = [(m, d) for m in (15, 20, 25, 30, 35, 40) for d in (5, 7, 8, 10)]
t0 = time.perf_counter()
spread = cp.stack([cp.nanmedian(onset(cp, Rg, min3=m, dry_len=d), axis=0) for m, d in grid])
cp.cuda.Device().synchronize()
print(f"{len(grid)} definitions x {len(years)} years x {R.shape[2]*R.shape[3]} points in {time.perf_counter()-t0:.1f}s on GPU")
sens = xr.DataArray(cp.asnumpy(spread.max(0) - spread.min(0)), dims=("lat", "lon"), coords={"lat": tp.lat, "lon": tp.lon})
for n, (la, lo) in PLACES.items():
    print(f"  {n:12s}: median onset shifts by up to {float(sens.sel(lat=la, lon=lo, method='nearest')):.0f} days across definitions")

# %% [markdown]
# ## 3. This week vs normal
# Compare Lab 04's 5-day forecast total with the normal for the same 5 calendar days. (Lab 04 is at 0.25°, the
# climate at 1.5°: we average the forecast onto the coarse grid first.)

# %%
if os.path.exists("outputs/lab04_africa_ensemble.zarr"):
    fc = xr.open_zarr("outputs/lab04_africa_ensemble.zarr").isel(time=0)
    fc = fc.assign_coords(lon=((fc.lon + 180) % 360) - 180).sortby("lon")
    init = pd.Timestamp(fc.time.values)
    total = (fc["tp06"].isel(lead_time=slice(1, None)).clip(min=0) * 1000).sum("lead_time").mean("ensemble").load()
    total = total.coarsen(lat=6, lon=6, boundary="trim").mean().sortby("lat").interp(lat=tp.lat, lon=tp.lon)
    days = [(init + pd.Timedelta(days=d)).dayofyear for d in range(5)]
    normal = doy_clim.sel(dayofyear=[min(d, 365) for d in days]).sum("dayofyear")
    pct = (100 * total / normal).where(normal > 5)              # skip places where 5 mm is already "a lot"
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": ccrs.PlateCarree()})
    im = ax.pcolormesh(pct.lon, pct.lat, pct, cmap="BrBG", vmin=0, vmax=200, transform=ccrs.PlateCarree())
    ax.coastlines(); ax.add_feature(cfeature.BORDERS, linewidth=0.3); fig.colorbar(im, ax=ax, shrink=0.7, label="% of normal")
    ax.set_title(f"Forecast rain {init:%d %b}–{init + pd.Timedelta(days=4):%d %b %Y} as % of the 1981–2022 normal")
    plt.savefig("outputs/lab10_anomaly.png", dpi=110, bbox_inches="tight"); plt.show()
else:
    print("Run Lab 04 first for this section.")

# %% [markdown]
# ## Exercise
# 1. Pick your district. In which years did the rains start late, and does that match what farmers remember
#    (and crop yields)?
# 2. The trend map is noisy. Which areas show a consistent signal, and how would you test whether it is real
#    (hint: count years, check neighbours)?
# 3. ERA5 rainfall is a model product. Repeat for a station from Lab 03's GHCN data, or for CHIRPS (satellite +
#    station, 0.05°): which onset dates differ most?
# 4. At 0.25° the arrays are 36x larger. Estimate the CPU and GPU time from the numbers above.
