# %% [markdown]
# # Lab 12 — The data behind an AI forecast
# **Goal:** the model is the easy part. Before an AI weather model can run every day for your country you need
# the right **input data**, on time, in the right units and grid, plus **storage** for what comes out and
# **observations** to check it. This lab measures each of these on the Spark, so you can plan a service.
#
# 1. What each model needs as input (and which data sources can supply it)
# 2. How fresh the input data is right now
# 3. How much you download per forecast, and how the cache grows
# 4. Units and grid conventions that silently break things
# 5. Storing forecasts: format and chunking decide how fast people can use them
# 6. Observations: how many weather stations Africa shares, compared with Europe
#
# Requires internet (a few hundred MB). Uses the Lab 04 output for section 5 if it exists. Expected time: ~5 min.
# Memory: models are loaded on the CPU one at a time (~10 GiB), no GPU needed.

# %%
import os
import shutil
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import xarray as xr

def du(path):
    """Size of a file or folder in bytes."""
    if os.path.isfile(path):
        return os.path.getsize(path)
    return sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(path) for f in fs)

GB, MB = 2**30, 2**20
os.makedirs("outputs", exist_ok=True)

# %% [markdown]
# ## 1. What does each model need as input?
# Every Earth2Studio model publishes its `input_coords()`: the variables, the grid and how many past time steps
# it needs. We load three models (CPU only, just to read this) and compare. A "variable" is one 2-D field, so
# temperature at 13 pressure levels counts as 13.

# %%
import torch
from earth2studio.models.px import AIFSENS, SFNO, Aurora

rows, inputs = [], {}
for name, cls in [("SFNO", SFNO), ("AIFS-ENS", AIFSENS), ("Aurora", Aurora)]:
    t0 = time.perf_counter()
    m = cls.load_model(cls.load_default_package())
    ic, oc = m.input_coords(), m.output_coords(m.input_coords())
    v = [str(x) for x in ic["variable"]]
    inputs[name] = v
    levels = sorted({int(x[1:]) for x in v if x[1:].isdigit() and x[0] in "zqtuvwr"})
    shape = [len(ic[k]) for k in ("lead_time", "variable", "lat", "lon")]
    rows.append({"model": name,
                 "input fields": len(v),
                 "pressure levels": len(levels),
                 "grid": f"{len(ic['lat'])} x {len(ic['lon'])}" if "lat" in ic else "other",
                 "past steps needed": len(ic["lead_time"]),
                 "one input state (MB, fp32)": np.prod(shape) * 4 / MB,
                 "step (h)": int(oc["lead_time"][-1] / np.timedelta64(1, "h")),
                 "weights in memory (GB, as loaded)": sum(p.numel() * p.element_size() for p in m.parameters()) / GB})
    del m
    print(f"{name}: read in {time.perf_counter()-t0:.0f}s")
pd.DataFrame(rows).set_index("model").round(2)

# %% [markdown]
# Not every model needs the same surface fields. AIFS-ENS wants soil and skin temperature, dewpoint and total
# column water; SFNO wants total column water vapour and 100 m wind; Aurora only the basic four. Fixed fields
# (terrain height, land-sea mask) come inside the model packages, so they are not listed.

# %%
allv = sorted(set().union(*inputs.values()))
surface = [x for x in allv if not (x[1:].isdigit() and x[0] in "zqtuvwr")]
print("Surface / special fields each model needs (✓):")
pd.DataFrame({n: ["✓" if x in inputs[n] else "" for x in surface] for n in inputs}, index=surface)

# %% [markdown]
# ### Can a data source supply them?
# Each data source has a **lexicon**: the list of Earth2Studio variable names it can deliver. Checking it before you
# run saves you from a failure 10 minutes in (building these labs hit exactly this: FourCastNet needs relative humidity at
# 500 hPa, which the ARCO ERA5 copy does not have).

# %%
from earth2studio.lexicon import ARCO_ERA5Lexicon, GFSLexicon, IFSLexicon, WB2Lexicon

sources = {"GFS (NOAA, live)": GFSLexicon, "IFS (ECMWF open data, live)": IFSLexicon,
           "ARCO ERA5 (reanalysis, 1940-)": ARCO_ERA5Lexicon, "WB2 ERA5 (reanalysis, 1959-2023)": WB2Lexicon}
cov = pd.DataFrame({s: {n: (lambda miss: "all" if not miss else f"missing {len(miss)}: {', '.join(miss[:4])}")(
                            [x for x in v if x not in lex.VOCAB]) for n, v in inputs.items()}
                    for s, lex in sources.items()})
cov

# %% [markdown]
# "Missing" does not always mean impossible: Earth2Studio fills some fields (e.g. fixed fields like the land-sea mask
# come with the model package), and a few can be derived (relative humidity from specific humidity). But it means
# you **must test that source with that model before relying on it**.
#
# ## 2. How fresh is the data right now?
# A forecast can only start from an analysis that has been published. We ask NOAA's GFS bucket which of the last
# 8 cycles (00, 06, 12, 18 UTC) are there yet.

# %%
from earth2studio.data import GFS

now = datetime.now(timezone.utc).replace(tzinfo=None)
last = now.replace(minute=0, second=0, microsecond=0) - timedelta(hours=now.hour % 6)
cycles = [last - timedelta(hours=6 * k) for k in range(8)]
avail = {c: GFS.available(c) for c in cycles}
newest = max(c for c, a in avail.items() if a)
for c, a in avail.items():
    print(f"{c:%Y-%m-%d %H} UTC  {'available' if a else 'not yet':10s}  ({(now - c).total_seconds()/3600:.1f} h ago)")
print(f"\nnow {now:%H:%M} UTC; newest GFS cycle {newest:%Y-%m-%d %H} UTC is {(now-newest).total_seconds()/3600:.1f} h old")

# %% [markdown]
# GFS files appear on the bucket a few hours after the cycle time, and the full set takes longer still. ECMWF's open
# IFS data is published on a similar schedule. Plan the daily run around this: e.g. for a forecast ready by
# 07:00 local time in Nairobi (04:00 UTC), you can start from the **18 UTC cycle of the day before**.
#
# ## 3. How much do you download per forecast?
# We send this lab's downloads to a fresh cache folder so we can measure them, then fetch SFNO's inputs from GFS.
# Earth2Studio reads only the fields it needs out of each GRIB file (HTTP range requests), not the whole file.

# %%
from earth2studio.data import fetch_data
from earth2studio.data.utils import datasource_cache_root

main_cache = datasource_cache_root()
lab_cache = os.path.abspath("outputs/lab12_cache")
shutil.rmtree(lab_cache, ignore_errors=True)
os.environ["EARTH2STUDIO_CACHE"] = lab_cache

# A cycle's folder appears before all its files do, so in a pipeline start from the cycle before the newest.
ic_time = newest - timedelta(hours=6)
t0 = time.perf_counter()
x, coords = fetch_data(GFS(verbose=False), np.array([ic_time], dtype="datetime64[ns]"), inputs["SFNO"], device="cpu")
dt = time.perf_counter() - t0
os.environ["EARTH2STUDIO_CACHE"] = main_cache

down = du(lab_cache)
print(f"SFNO inputs from GFS {ic_time:%Y-%m-%d %H} UTC: {len(inputs['SFNO'])} fields in {dt:.0f}s")
print(f"  downloaded (GRIB2, compressed): {down/MB:.0f} MB")
print(f"  in memory (fp32 on the 721x1440 grid): {x.numel()*4/MB:.0f} MB")
print(f"  for comparison, the whole GFS 0.25° analysis file is ~500 MB")

# %% [markdown]
# ### Your cache, and how it grows
# Earth2Studio keeps every download and every model package in one cache folder. On a shared Spark it lives on the
# mounted data disk, so teams don't download the same thing twice. Here is what this Spark has accumulated so far:

# %%
items = [(d, du(os.path.join(main_cache, d))) for d in os.listdir(main_cache)]
cache = pd.DataFrame(items, columns=["folder", "GB"]).set_index("folder").GB.div(GB).sort_values(ascending=False)
print(f"cache at {main_cache}: {cache.sum():.1f} GB in {len(cache)} folders")
cache.head(12).round(2)

# %% [markdown]
# Model weights are a one-off cost (a few GB each). The input data is the part that **grows**: on a box that has run
# the other labs, `arco_era5` (every hindcast in Labs 03, 07 and 09, read at full resolution) is usually the
# largest folder. Budget for it, set `cache=False` on the data source for one-off runs, or clean it regularly.
#
# ## 4. Units and grid conventions that silently break things
# The fetched array is a good example. Nothing here raises an error if you get it wrong, you just get wrong numbers.

# %%
da = xr.DataArray(x[0, 0].numpy(), dims=["variable", "lat", "lon"],
                  coords={"variable": coords["variable"], "lat": coords["lat"], "lon": coords["lon"]})
print("latitude runs", float(da.lat[0]), "->", float(da.lat[-1]), "(north to south)")
print("longitude runs", float(da.lon[0]), "->", float(da.lon[-1]), "(0-360, not -180-180)")

wrong = da.sel(lat=slice(-35, 38), lon=slice(-18, 52))
print(f"\nnaive Africa crop sel(lat=slice(-35, 38), lon=slice(-18, 52)): shape {dict(wrong.sizes)}  <- empty!")
africa = da.assign_coords(lon=((da.lon + 180) % 360) - 180).sortby("lon").sel(lat=slice(38, -35), lon=slice(-18, 52))
print(f"correct: shift lon to -180-180, slice lat north->south: shape {dict(africa.sizes)}")

t2m = africa.sel(variable="t2m")
print(f"\nt2m over Africa: {float(t2m.min()):.0f} to {float(t2m.max()):.0f}  <- Kelvin, subtract 273.15")
print(f"msl: {float(africa.sel(variable='msl').mean()):.0f}  <- Pascals, divide by 100 for hPa")

# %% [markdown]
# | Pitfall | What it looks like | What to do |
# |---|---|---|
# | Longitude 0–360 | West Africa (18° W = 342° E) vanishes from a crop | `((lon + 180) % 360) - 180` then `sortby("lon")` |
# | Latitude north→south | `slice(-35, 38)` returns nothing | Slice `slice(38, -35)`, or `sortby("lat")` |
# | Kelvin, Pascals, metres | 300 "degrees"; rain of 0.004 | K − 273.15; Pa / 100; m × 1000 = mm |
# | Accumulations (`tp06`, `ssrd06`) | Totals off by the first step | Lead 0 has no past 6 h: skip it; values are per 6 h, not per hour |
# | Radiation in J/m² | Sunshine of 2 000 000 | J/m² / 3.6e6 = kWh/m² |
# | Time zones | "Daily max" in UTC days misses the afternoon | Model times are UTC; East Africa is UTC+3, West Africa UTC+0/+1 |
# | Grid-box averages | Station 6 °C off (Lab 11) | Bias-correct to stations before giving point values |
#
# ## 5. Storing forecasts: format and chunking
# How you store forecasts decides how fast people can read them. A map viewer reads **one time, all places**; a
# farm advisory reads **one place, all times**. Zarr stores data in chunks; reading needs whole chunks.
# We store Lab 04's 5-day rainfall ensemble for Africa three ways and time both kinds of read.

# %%
src = "outputs/lab04_africa_ensemble.zarr"
if not os.path.exists(src):
    print("Lab 04 output not found, skipping section 5")
else:
    rain = xr.open_zarr(src)["tp06"].isel(time=0).load()     # (ensemble, lead_time, lat, lon)
    # fresh array without the source file's storage settings, so each layout below starts from the same thing
    rain = xr.DataArray(rain.values, dims=rain.dims, coords={k: rain[k].values for k in rain.dims}, name="tp06")
    print(f"tp06 ensemble: {dict(rain.sizes)}, {rain.nbytes/MB:.0f} MB in memory")
    layouts = {
        "NetCDF, no compression": ("outputs/lab12_rain.nc", None),
        "Zarr, chunked as maps (1 lead time per chunk)": ("outputs/lab12_rain_maps.zarr",
                                                         {"ensemble": -1, "lead_time": 1, "lat": -1, "lon": -1}),
        "Zarr, chunked as time series (32x32 tiles)": ("outputs/lab12_rain_series.zarr",
                                                      {"ensemble": -1, "lead_time": -1, "lat": 32, "lon": 32}),
    }
    res = []
    for label, (path, chunks) in layouts.items():
        shutil.rmtree(path, ignore_errors=True)
        if chunks is None:
            rain.to_netcdf(path)
        else:
            rain.chunk(chunks).to_dataset().to_zarr(path, mode="w")

        def read(sel):
            ds = xr.open_dataset(path) if chunks is None else xr.open_zarr(path)
            t0 = time.perf_counter(); ds["tp06"].isel(**sel).load(); ds.close()
            return (time.perf_counter() - t0) * 1000
        point = dict(lat=rain.sizes["lat"] // 2, lon=rain.sizes["lon"] // 2)
        res.append({"layout": label, "size on disk (MB)": du(path) / MB,
                    "read 1 place, all times (ms)": np.median([read(point) for _ in range(5)]),
                    "read 1 time, all places (ms)": np.median([read(dict(lead_time=10)) for _ in range(5)])})
    print(pd.DataFrame(res).set_index("layout").round(1).to_string())

# %% [markdown]
# On our Spark, the Zarr chunking that matched the read was **10–30x faster** than the one that didn't (about 3 ms
# vs 30 ms for a time series; 5–7 ms vs 95 ms for a map). Uncompressed NetCDF was fastest of all here, because this
# 53 MB file sits in memory after the first read, but it took 3x the disk space. For archives of many GB, or reads
# over a network, compressed and chunked Zarr wins. **Rule: chunk for your most common read.**
#
# A year of storage for one daily run like Lab 04 (7 fields, 8 members, 5 days, Africa at 0.25°):

# %%
if os.path.exists(src):
    per_run = du(src)
    print(f"one run: {per_run/MB:.0f} MB on disk -> one year of daily runs: {per_run*365/GB:.0f} GB")
    print(f"same run for the whole globe would be ~{per_run*(721*1440)/(rain.sizes['lat']*rain.sizes['lon'])/GB:.1f} GB "
          f"-> {per_run*(721*1440)/(rain.sizes['lat']*rain.sizes['lon'])*365/2**40:.1f} TB per year")

# %% [markdown]
# Save only your region and the variables you use (the `output_coords` argument in every Earth2Studio run does this).
#
# ## 6. Observations: how many stations does Africa share?
# You need observations to verify a forecast (Lab 03) and to correct it (Lab 11). NOAA's GHCN-Daily is the
# largest free archive. How many stations reported daily maximum temperature recently, per area?

# %%
from earth2studio.data import GHCNDaily
meta, inv = GHCNDaily.get_station_metadata(), GHCNDaily.get_station_inventory()
recent = inv[(inv.ELEMENT == "TMAX") & (inv.LASTYEAR >= now.year - 1)].ID
st = meta[meta.ID.isin(recent)]

# Country outlines (Natural Earth, via cartopy) give each country's land area and which stations fall inside it.
import pyproj
import shapely
from cartopy.io import shapereader
geod = pyproj.Geod(ellps="WGS84")
countries = []
for r in shapereader.Reader(shapereader.natural_earth("110m", "cultural", "admin_0_countries")).records():
    a = r.attributes
    if a["CONTINENT"] in ("Africa", "Europe") and a["ADMIN"] != "Russia":   # Russia is mostly in Asia
        n = int(shapely.contains_xy(r.geometry, st.LON.values, st.LAT.values).sum())
        countries.append({"country": a["ADMIN"], "continent": a["CONTINENT"], "stations": n,
                          "area (M km²)": abs(geod.geometry_area_perimeter(r.geometry)[0]) / 1e12})
cdf = pd.DataFrame(countries)
cdf["per M km²"] = cdf.stations / cdf["area (M km²)"]
tot = cdf.groupby("continent")[["stations", "area (M km²)"]].sum()
tot["per M km²"] = tot.stations / tot["area (M km²)"]
print(f"Stations reporting daily max temperature in {now.year-1}-{now.year} (Europe without Russia):")
print(tot.round(1), "\n")
print("African countries with the most and the fewest stations per area (over 100 000 km²):")
big = cdf[(cdf.continent == "Africa") & (cdf["area (M km²)"] > 0.1)].sort_values("per M km²", ascending=False)
print(pd.concat([big.head(5), big.tail(5)]).set_index("country")[["stations", "area (M km²)", "per M km²"]].round(1))

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(7, 7.5), subplot_kw={"projection": ccrs.PlateCarree()})
ax.set_extent([-20, 55, -37, 40]); ax.coastlines(linewidth=0.6); ax.add_feature(cfeature.BORDERS, linewidth=0.3)
ax.plot(st.LON, st.LAT, "o", ms=1.5, c="tab:red", transform=ccrs.PlateCarree())
ax.set_title(f"GHCN-Daily stations with TMAX in {now.year-1}-{now.year}")
plt.savefig("outputs/lab12_stations.png", dpi=110, bbox_inches="tight"); plt.show()

# %% [markdown]
# Most national met services hold many more stations than they share internationally. On a Spark the data stays in
# the building, so the archive can be used for verification and bias correction without being published.
#
# ## Clean up

# %%
for p in [lab_cache, "outputs/lab12_rain.nc", "outputs/lab12_rain_maps.zarr", "outputs/lab12_rain_series.zarr"]:
    if os.path.isdir(p):
        shutil.rmtree(p)
    elif os.path.exists(p):
        os.remove(p)
print("removed this lab's temporary files")

# %% [markdown]
# ## What to take away
# - **Inputs:** each model needs a specific list of fields; check the source's lexicon against `input_coords()`
#   before building a pipeline.
# - **Timeliness:** live analyses arrive hours after the cycle time. Pick the cycle your deadline allows.
# - **Volume:** inputs are small (tens to hundreds of MB per run); weights are a one-off few GB; **outputs and
#   reanalysis caches are what grow**. Crop to your region and variables.
# - **Conventions:** longitude, latitude order, units and accumulations cause silent errors. Check them once and
#   wrap them in a helper.
# - **Observations** are the scarce resource in Africa, and the one only your institution can add.
#
# See the one-page checklist: [`../docs/06-data-guideline-one-pager.md`](../docs/06-data-guideline-one-pager.md).
#
# ## Exercise
# 1. Repeat section 3 for AIFS-ENS from `IFS(source="google")`. Which model needs more data per run?
# 2. Change the chunk size in section 5 to 8x8 and 128x128. Where is the sweet spot for each kind of read?
# 3. Count stations reporting **rainfall** (`PRCP`) instead of TMAX. Which countries have the most?
# 4. Your deadline is 06:00 local time. Using section 2 and Lab 04's runtime, which cycle can you use?
