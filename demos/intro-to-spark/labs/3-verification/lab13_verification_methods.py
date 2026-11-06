# %% [markdown]
# # Lab 13 — How good is the forecast? Verification methods
# **Goal:** "RMSE went down" is not enough to decide whether to trust a forecast. A farmer cares whether it said rain
# when it rained; a flood manager cares whether a 70 % chance means 70 %; a map viewer cares whether the storm was
# in roughly the right place. Each question has its own score. This lab applies the standard ones to one AI
# ensemble forecast over Africa, mostly with Earth2Studio's built-in `earth2studio.statistics`.
#
# **Case:** AIFS-ENS, 8 members, started **22 April 2024**, during the long rains that brought severe flooding to
# Kenya and Tanzania that season. Truth is ERA5 reanalysis; at the end we also check against rain gauges.
#
# | Question | Method | Section |
# |---|---|---|
# | How far off is a value, on average? Is it too warm or too cold? | Bias, MAE, RMSE, skill vs baselines, anomaly correlation | 3 |
# | Did it rain when the forecast said it would? | Contingency table: POD, FAR, CSI, frequency bias, ETS | 4 |
# | Was the rain in roughly the right place? | Fractions Skill Score (FSS) at growing scales | 5 |
# | Is the ensemble's uncertainty honest? | CRPS, spread vs error, rank histogram | 6 |
# | Can I trust "70 % chance of rain"? | Brier score, reliability diagram, ROC | 6 |
# | Does it match what a gauge measured? | Point verification at stations | 7 |
#
# Requires internet (ERA5 on Google Cloud; AIFS-ENS weights cached from Lab 04). The forecast runs once (~9 min,
# ~36 GiB GPU, ~60 GiB system, so run it alone on a shared box) and is saved; re-runs take ~1 min.

# %%
import os
import time
from collections import OrderedDict
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch
import xarray as xr

INIT = datetime(2024, 4, 22)
NSTEPS, NENSEMBLE = 20, 8      # 5 days, 6-hourly
LAT = np.arange(38, -35.01, -0.25)
LON = np.concatenate([np.arange(342, 360, 0.25), np.arange(0, 52.01, 0.25)])
OUT = "../outputs/lab13_hindcast.zarr"
USE_CACHED = True              # reuse OUT if it exists; set False to run the model again

def to_africa(da):
    return da.assign_coords(lon=((da.lon + 180) % 360) - 180).sortby("lon")

# %% [markdown]
# ## 1. Run the hindcast (once)
# A **hindcast** is a forecast for a past date, so we can compare it with what happened. We start AIFS-ENS from
# ERA5 (ARCO) rather than IFS: ERA5 has every field the model needs (Lab 12, section 1) for any date since 1940.

# %%
from earth2studio.data import ARCO

if USE_CACHED and os.path.exists(OUT):
    print(f"using cached hindcast {OUT}")
else:
    import earth2studio.run as run
    from earth2studio.io import ZarrBackend
    from earth2studio.models.px import AIFSENS
    from earth2studio.perturbation import Zero

    model = AIFSENS.load_model(AIFSENS.load_default_package())
    io = ZarrBackend(file_name=OUT, backend_kwargs={"overwrite": True})
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    run.ensemble([INIT], NSTEPS, NENSEMBLE, model, ARCO(), io, Zero(), batch_size=1,
                 output_coords={"variable": np.array(["t2m", "tp06"]), "lat": LAT, "lon": LON})
    print(f"hindcast: {time.perf_counter()-t0:.0f}s, peak GPU {torch.cuda.max_memory_allocated()/2**30:.1f} GiB")
    del model; torch.cuda.empty_cache()

fc = to_africa(xr.open_zarr(OUT).isel(time=0)).load()
leads = fc.lead_time.values
hours = (leads / np.timedelta64(1, "h")).astype(int)
print(f"forecast: {dict(fc.t2m.sizes)}")

# %% [markdown]
# ## 2. The truth: ERA5 at the same times
# ERA5 is itself a model blended with observations, so "error" here means "difference from ERA5". Over data-poor
# regions, ERA5 rain in particular can be wrong; section 7 checks it against gauges.
#
# We also fetch two **baselines** every forecast must beat:
# - **Persistence**: the same hour yesterday (it keeps the day-night cycle right).
# - **Climatology**: the 1990–2019 average for that day and hour (WeatherBench 2).

# %%
from earth2studio.data import WB2Climatology

t0 = time.perf_counter()
valid = [INIT + timedelta(hours=int(h)) for h in hours]
arco = ARCO(verbose=False)
obs = to_africa(arco(valid, ["t2m", "tp06"]).sel(lat=LAT, lon=LON)).load()
past = to_africa(arco([INIT - timedelta(hours=h) for h in (24, 18, 12, 6)], ["t2m"]).sel(lat=LAT, lon=LON)).load()
clim = to_africa(WB2Climatology("1990-2019_6h_1440x721.zarr", verbose=False)(valid, ["t2m"])
                 .sel(lat=LAT, lon=LON)).load()
print(f"ERA5 + climatology fetched in {time.perf_counter()-t0:.0f}s")

def by_lead(da):
    """Turn a (time, ...) array at the valid times into a (lead_time, ...) array."""
    return da.rename(time="lead_time").assign_coords(lead_time=leads)

t2m_f = fc.t2m - 273.15                                            # (ensemble, lead_time, lat, lon), °C
t2m_o = by_lead(obs.sel(variable="t2m", drop=True)) - 273.15
t2m_clim = by_lead(clim.sel(variable="t2m", drop=True)) - 273.15
# persistence: the value at the same hour of the day before the start
t2m_pers = xr.concat([past.sel(variable="t2m", drop=True).isel(time=(h % 24) // 6) for h in hours], dim="time")
t2m_pers = by_lead(t2m_pers.assign_coords(time=valid)) - 273.15

# Daily rain (mm), days 1-5: sum of four 6-h totals. Lead 0 has no past 6 h, so it is skipped.
def daily(da):
    d = (da.isel(lead_time=slice(1, None)).clip(min=0) * 1000).coarsen(lead_time=4, boundary="trim").sum()
    return d.rename(lead_time="day").assign_coords(day=np.arange(1, d.sizes["lead_time"] + 1))
rain_f = daily(fc.tp06)                                            # (ensemble, day, lat, lon)
rain_o = daily(by_lead(obs.sel(variable="tp06", drop=True)))       # (day, lat, lon)

# Land mask: most users care about land, and rain over the ocean is easy to get "right" by saying no rain.
import shapely
from cartopy.io import shapereader
land_geom = shapely.union_all(list(shapereader.Reader(shapereader.natural_earth("110m", "physical", "land")).geometries()))
lon2d, lat2d = np.meshgrid(fc.lon.values, fc.lat.values)
land = xr.DataArray(shapely.contains_xy(land_geom, lon2d, lat2d), dims=("lat", "lon"),
                    coords={"lat": fc.lat, "lon": fc.lon})
print(f"{int(land.sum())} land points of {land.size}")

# %%
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt

EA = dict(lat=slice(15, -12), lon=slice(28, 52))     # East Africa
panels = [(rain_f.sum("day").mean("ensemble").sel(**EA), "Forecast: ensemble-mean 5-day rain (mm)", "YlGnBu", 0, 200),
          (rain_o.sum("day").sel(**EA), "ERA5: 5-day rain (mm)", "YlGnBu", 0, 200),
          ((rain_f.sum("day") > 50).mean("ensemble").sel(**EA) * 100, "Forecast: chance of > 50 mm (%)", "Purples", 0, 100)]
fig, axs = plt.subplots(1, 3, figsize=(16, 5.5), subplot_kw={"projection": ccrs.PlateCarree()})
for a, (da, title, cmap, vmin, vmax) in zip(axs, panels):
    im = a.pcolormesh(da.lon, da.lat, da, cmap=cmap, vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
    a.coastlines(linewidth=0.6); a.add_feature(cfeature.BORDERS, linewidth=0.3)
    a.add_feature(cfeature.LAKES, facecolor="none", edgecolor="k", linewidth=0.3)
    a.plot(36.82, -1.29, "r*", ms=8, transform=ccrs.PlateCarree())       # Nairobi
    a.set_title(title, fontsize=10); fig.colorbar(im, ax=a, shrink=0.75)
fig.suptitle(f"AIFS-ENS from {INIT:%d %b %Y}, days 1-5 (red star: Nairobi)")
plt.savefig("../outputs/lab13_case.png", dpi=110, bbox_inches="tight"); plt.show()

# %% [markdown]
# ## 3. Continuous scores: temperature
# - **Bias** = mean(forecast − truth): is it too warm or too cold overall?
# - **MAE** = mean |forecast − truth|: the typical error, in °C.
# - **RMSE** = √mean((forecast − truth)²): like MAE but punishes big misses more.
# - **Skill score** = 1 − RMSE / RMSE(baseline): 0 means no better than the baseline, 1 is perfect.
# - **ACC** (anomaly correlation): correlation of the forecast's and the truth's departures from climatology.
#   It ignores the easy part (it is hot in the Sahara) and asks if the forecast got the unusual part right.
#   Above ~0.6 is usually called useful.
#
# Grid points shrink towards the poles, so means are weighted by cos(latitude). Earth2Studio's `rmse` and `mae`
# take the weights and work on torch tensors with a coordinate dictionary, like every Earth2Studio model.

# %%
from earth2studio.statistics import crps, fss, mae, rmse, spread_skill_ratio

def cs(da):
    return OrderedDict((d, da[d].values) for d in da.dims)

def score(metric, f, o):
    """Run an earth2studio.statistics metric on two xarray arrays and return xarray."""
    out, oc = metric(torch.as_tensor(f.values, dtype=torch.float32), cs(f),
                     torch.as_tensor(o.values, dtype=torch.float32), cs(o))
    return xr.DataArray(out.numpy(), dims=list(oc), coords=oc)

W = torch.as_tensor(np.cos(np.deg2rad(fc.lat.values))[:, None] * land.values, dtype=torch.float32)   # land only
RMSE, MAE = rmse(["lat", "lon"], weights=W), mae(["lat", "lon"], weights=W)

candidates = {"one member": t2m_f.isel(ensemble=0, drop=True), "ensemble mean": t2m_f.mean("ensemble"),
              "persistence (yesterday)": t2m_pers, "climatology": t2m_clim}
rm = {k: score(RMSE, v, t2m_o) for k, v in candidates.items()}
wl = xr.DataArray(W.numpy(), dims=("lat", "lon"), coords={"lat": fc.lat, "lon": fc.lon})
bias = {k: ((v - t2m_o) * wl).sum(("lat", "lon")) / wl.sum() for k, v in candidates.items()}

def acc(f):
    af, ao = f - t2m_clim, t2m_o - t2m_clim
    return (wl * af * ao).sum(("lat", "lon")) / np.sqrt((wl * af**2).sum(("lat", "lon")) * (wl * ao**2).sum(("lat", "lon")))

day = lambda d: {"day": d} | {k: float(v.sel(lead_time=np.timedelta64(24 * d, "h"))) for k, v in rm.items()}
tab = pd.DataFrame([day(d) for d in (1, 3, 5)]).set_index("day")
print("t2m RMSE over African land (°C):"); print(tab.round(2).to_string())
print(f"\nensemble-mean skill vs persistence, day 5: {1 - tab.loc[5, 'ensemble mean']/tab.loc[5, 'persistence (yesterday)']:.2f}"
      f" | vs climatology: {1 - tab.loc[5, 'ensemble mean']/tab.loc[5, 'climatology']:.2f}")
print(f"MAE of the ensemble mean, day 5: {float(score(MAE, candidates['ensemble mean'], t2m_o).isel(lead_time=-1)):.2f} °C")

fig, ax = plt.subplots(1, 3, figsize=(17, 4.2))
for k, v in rm.items():
    ax[0].plot(hours, v, "o-" if "ens" in k else "-", ms=3, label=k)
    ax[1].plot(hours, bias[k], "-", label=k)
ax[0].set_title("RMSE (°C), African land"); ax[1].set_title("Bias (°C): forecast minus ERA5"); ax[1].axhline(0, c="k", lw=.7)
ax[2].plot(hours, acc(candidates["ensemble mean"]), "o-", ms=3, label="ensemble mean")
ax[2].plot(hours, acc(candidates["one member"]), "-", label="one member")
ax[2].plot(hours, acc(t2m_pers), "-", label="persistence")
ax[2].axhline(0.6, c="gray", ls="--", lw=.8); ax[2].set_title("Anomaly correlation (ACC)"); ax[2].set_ylim(0, 1)
for a in ax:
    a.set_xlabel("lead time (h)"); a.grid(alpha=.3); a.legend(fontsize=7)
plt.tight_layout(); plt.savefig("../outputs/lab13_continuous.png", dpi=110); plt.show()

# %% [markdown]
# **On our run:** the ensemble mean's RMSE was 1.0–1.4 °C over days 1–5, against 2.4–3.2 °C for persistence and
# ~2.7 °C for climatology (skill 0.4–0.5 at day 5). A single member was ~30 % worse than the mean: averaging
# cancels the members' random errors. Lead 0 has zero error because the forecast starts from ERA5 itself.
#
# Where is it biased? Mean error over days 2-5 for the ensemble mean:

# %%
bmap = (candidates["ensemble mean"] - t2m_o).sel(lead_time=slice(np.timedelta64(24, "h"), None)).mean("lead_time")
fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"projection": ccrs.PlateCarree()})
im = ax.pcolormesh(bmap.lon, bmap.lat, bmap.where(land), cmap="RdBu_r", vmin=-4, vmax=4, transform=ccrs.PlateCarree())
ax.coastlines(linewidth=0.6); ax.add_feature(cfeature.BORDERS, linewidth=0.3)
fig.colorbar(im, ax=ax, shrink=0.6, label="°C (red: forecast too warm)")
ax.set_title("t2m bias of the ensemble mean, days 2-5")
plt.savefig("../outputs/lab13_bias_map.png", dpi=110, bbox_inches="tight"); plt.show()

# %% [markdown]
# ## 4. Categorical scores: did it rain when it said it would?
# Pick a threshold (here mm/day) and count four outcomes at every land point and day:
#
# | | Rain observed | No rain observed |
# |---|---|---|
# | **Rain forecast** | hit (a) | false alarm (b) |
# | **No rain forecast** | miss (c) | correct no (d) |
#
# - **POD** (probability of detection) = a/(a+c): share of rain events caught. 1 is best.
# - **FAR** (false alarm ratio) = b/(a+b): share of rain forecasts that were wrong. 0 is best.
# - **CSI** (critical success index) = a/(a+b+c): ignores the easy "correct no". 1 is best.
# - **Frequency bias** = (a+b)/(a+c): does it forecast rain as often as it happens? 1 is unbiased.
# - **ETS** (equitable threat score): CSI minus the hits you'd get by chance. 0 = no skill.

# %%
def contingency(f, o, thr, mask=land):
    F, O = (f >= thr).where(mask).values, (o >= thr).where(mask).values
    ok = ~np.isnan(F) & ~np.isnan(O)
    F, O = F[ok].astype(bool), O[ok].astype(bool)
    a, b, c, d = (F & O).sum(), (F & ~O).sum(), (~F & O).sum(), (~F & ~O).sum()
    r = (a + b) * (a + c) / (a + b + c + d)                  # hits expected by chance
    return {"hits": a, "false alarms": b, "misses": c, "POD": a / (a + c), "FAR": b / max(a + b, 1),
            "CSI": a / (a + b + c), "freq. bias": (a + b) / (a + c), "ETS": (a - r) / (a + b + c - r)}

rows = []
for thr in (1, 10, 25):
    for name, f in [("one member", rain_f.isel(ensemble=0, drop=True)), ("ensemble mean", rain_f.mean("ensemble"))]:
        rows.append({"threshold (mm/day)": thr, "forecast": name} | contingency(f, rain_o, thr))
cat = pd.DataFrame(rows).set_index(["threshold (mm/day)", "forecast"])
print(f"Daily rain over African land, days 1-5, vs ERA5:"); print(cat.round(2).to_string())

csi_day = pd.DataFrame({name: [contingency(f.sel(day=d), rain_o.sel(day=d), 10)["CSI"] for d in rain_o.day.values]
                        for name, f in [("one member", rain_f.isel(ensemble=0, drop=True)),
                                        ("ensemble mean", rain_f.mean("ensemble"))]}, index=rain_o.day.values)
print("\nCSI for > 10 mm/day, by forecast day:"); print(csi_day.round(2).to_string())

# %% [markdown]
# Look at the **frequency bias** of the ensemble mean: averaging members smears rain out, so it forecasts light
# rain too often (1.27 at 1 mm on our run) and heavy rain far too rarely (0.10 at 25 mm: it almost never
# forecasts it). A single member keeps realistic rain amounts (0.68 at 25 mm) but places them badly (CSI 0.07). A mean is a good
# best-guess **amount** but a poor **event** forecast. For events, use the members' probabilities (section 6).
#
# ## 5. Was the rain in roughly the right place? Fractions Skill Score
# A forecast that puts a storm 30 km off is punished twice by point scores: a miss where it rained and a false
# alarm where it didn't (the "double penalty"). The **FSS** compares the *fraction* of rainy points within a window
# around each point, for growing window sizes. FSS = 1 is perfect; above **0.5 + f/2** (f = fraction of points that
# had rain) is usually called useful. The scale where it crosses that line tells you how precise the forecast is.

# %%
WINDOWS = [3, 9, 21, 41]              # grid points; × 0.25° × ~28 km → ~80, 250, 600, 1150 km
FSS = fss(["lat", "lon"], window_sizes=WINDOWS, thresholds=[10.0], spatial_dimensions=("lat", "lon"))
FSS_ENS = fss(["lat", "lon"], window_sizes=WINDOWS, thresholds=[10.0], spatial_dimensions=("lat", "lon"),
              ensemble_dimension="ensemble")
f_det = score(FSS, rain_f.isel(ensemble=0, drop=True), rain_o).isel(threshold=0)
f_ens = score(FSS_ENS, rain_f, rain_o).isel(threshold=0)
frac = float((rain_o >= 10).mean())
km = np.array(WINDOWS) * 0.25 * 111

fig, ax = plt.subplots(figsize=(7, 4))
for d, c in [(1, "tab:green"), (3, "tab:orange"), (5, "tab:red")]:
    ax.plot(km, f_det.sel(day=d), "o--", c=c, label=f"day {d}, one member")
    ax.plot(km, f_ens.sel(day=d), "o-", c=c, label=f"day {d}, ensemble")
ax.axhline(0.5 + frac / 2, c="gray", ls=":", label="useful (0.5 + f/2)")
ax.set_xlabel("window size (km)"); ax.set_ylabel("FSS, rain > 10 mm/day"); ax.set_ylim(0, 1); ax.grid(alpha=.3)
ax.legend(fontsize=7); ax.set_title("Fractions Skill Score over the Africa box")
plt.tight_layout(); plt.savefig("../outputs/lab13_fss.png", dpi=110); plt.show()
print(f"fraction of points with > 10 mm/day: {frac:.3f}")
print(f_ens.to_pandas().round(2).rename(columns=dict(zip(WINDOWS, [f"{k:.0f} km" for k in km]))).to_string())

# %% [markdown]
# **On our run** FSS for > 10 mm/day stayed above the useful line (0.53) at every scale and day, even ~80 km. The
# scores do not fall steadily with lead time (day 4 beat day 2): one start date is noisy.
#
# ## 6. Probabilistic scores: is the ensemble honest?
# - **CRPS** (continuous ranked probability score): the probabilistic version of MAE, in the same units (°C, mm).
#   For a single forecast it *equals* MAE, so CRPS of the ensemble vs MAE of one member is a fair comparison.
# - **Spread vs error**: if the ensemble is honest, its spread (std of members) matches the error of its mean.
#   Ratio < 1 = over-confident (too little spread), > 1 = under-confident.
# - **Rank histogram**: where does the truth fall among the sorted members? Flat = honest; U-shape = truth often
#   outside all members (too little spread); a slope = bias.

# %%
CRPS = crps("ensemble", ["lat", "lon"], weights=W)
SSR = spread_skill_ratio("ensemble", ["lat", "lon"], reduction_weights=W)
crps_t = score(CRPS, t2m_f, t2m_o)
mae_t = score(MAE, t2m_f.isel(ensemble=0, drop=True), t2m_o)
ssr_t = score(SSR, t2m_f, t2m_o)
crps_r = score(CRPS, rain_f, rain_o)
mae_r = score(MAE, rain_f.isel(ensemble=0, drop=True), rain_o)
ssr_r = score(SSR, rain_f, rain_o)
print("t2m, day 5:  CRPS of ensemble {:.2f} °C vs MAE of one member {:.2f} °C; spread/error {:.2f}".format(
      float(crps_t.isel(lead_time=-1)), float(mae_t.isel(lead_time=-1)), float(ssr_t.isel(lead_time=-1))))
print("rain, by day: CRPS (mm)", crps_r.values.round(2), "| one-member MAE", mae_r.values.round(2),
      "| spread/error", ssr_r.values.round(2))

# rank of the truth among the 8 members, land points, all lead times from day 1
f_l = t2m_f.sel(lead_time=slice(np.timedelta64(24, "h"), None)).where(land)
o_l = t2m_o.sel(lead_time=slice(np.timedelta64(24, "h"), None)).where(land)
ranks = (f_l < o_l).sum("ensemble").where(o_l.notnull()).values.ravel()
ranks = ranks[~np.isnan(ranks)].astype(int)
hist = np.bincount(ranks, minlength=NENSEMBLE + 1) / len(ranks)

# %% [markdown]
# ### Brier score, reliability and ROC for "more than 10 mm tomorrow"
# The forecast probability is the share of members above 10 mm/day (0, 1/8, ..., 8/8).
# - **Brier score** = mean (probability − outcome)², 0 is perfect. **Brier skill score** compares it with always
#   forecasting the observed frequency (climatology): above 0 is skill.
# - **Reliability diagram**: of all the times the forecast said 50 %, did it rain about half the time? Points on the
#   diagonal = reliable.
# - **ROC curve**: hit rate vs false-alarm rate as you lower the probability needed to issue a warning. Area
#   (AUC) 0.5 = no skill, 1 = perfect. It measures whether the forecast can *discriminate* rainy from dry days.

# %%
THR = 10.0
p = (rain_f >= THR).mean("ensemble").where(land).values.ravel()
o = (rain_o >= THR).where(land).values.ravel()
ok = ~np.isnan(p) & ~np.isnan(o)
p, o = p[ok], o[ok].astype(bool)
bs = np.mean((p - o) ** 2)
base = o.mean()
bss = 1 - bs / (base * (1 - base))
levels = np.arange(NENSEMBLE + 1) / NENSEMBLE
rel = [(o[np.isclose(p, l)].mean() if np.isclose(p, l).sum() > 50 else np.nan) for l in levels]
cnt = [np.isclose(p, l).sum() for l in levels]
hit = [((p >= l) & o).sum() / o.sum() for l in levels[1:]]
fa = [((p >= l) & ~o).sum() / (~o).sum() for l in levels[1:]]
roc_x, roc_y = np.r_[0, fa[::-1], 1], np.r_[0, hit[::-1], 1]
auc = np.trapezoid(roc_y, roc_x)
print(f"rain > {THR:.0f} mm/day, days 1-5, African land: observed frequency {base:.3f}, Brier {bs:.3f}, "
      f"Brier skill score {bss:.2f}, ROC area {auc:.2f}")

fig, ax = plt.subplots(1, 4, figsize=(19, 4.2))
ax[0].plot(hours, crps_t, "o-", ms=3, label="CRPS, ensemble"); ax[0].plot(hours, mae_t, "-", label="MAE, one member")
ax[0].set_title("t2m: CRPS vs MAE (°C)"); ax[0].set_xlabel("lead time (h)")
ax[1].plot(hours, ssr_t, "o-", ms=3, label="t2m"); ax[1].plot(rain_o.day * 24, ssr_r, "s-", label="daily rain")
ax[1].axhline(1, c="k", lw=.8); ax[1].set_title("Spread / error (1 = honest)"); ax[1].set_xlabel("lead time (h)")
ax[2].bar(np.arange(NENSEMBLE + 1), hist, color="tab:blue"); ax[2].axhline(1 / (NENSEMBLE + 1), c="k", ls="--")
ax[2].set_title("t2m rank histogram (flat = honest)"); ax[2].set_xlabel("members below the truth")
ax[3].plot([0, 1], [0, 1], "k--", lw=.8); ax[3].plot(levels, rel, "o-", label="reliability")
ax[3].plot(roc_x, roc_y, "s-", c="tab:red", label=f"ROC (area {auc:.2f})")
ax[3].set_xlabel("forecast probability / false-alarm rate"); ax[3].set_ylabel("observed frequency / hit rate")
ax[3].set_title(f"Rain > {THR:.0f} mm/day: reliability and ROC")
for a in ax:
    a.grid(alpha=.3)
    if a.get_legend_handles_labels()[0]:
        a.legend(fontsize=7)
plt.tight_layout(); plt.savefig("../outputs/lab13_probabilistic.png", dpi=110); plt.show()
pd.DataFrame({"forecast probability": levels, "cases": cnt, "observed frequency": rel}).round(3)

# %% [markdown]
# **On our run:**
# - CRPS of the ensemble was 0.8 °C at day 5, against 1.4 °C MAE for one member: the ensemble is worth running.
# - Spread/error was 0.8–1.0 for temperature (nearly honest) but 0.5–0.8 for rain: **over-confident about rain**,
#   the members agree more than they should.
# - The rank histogram is flat except the last bar (truth warmer than all 8 members ~15 % of the time instead of
#   11 %): a slight cold bias.
# - For rain > 10 mm/day the reliability curve sits close to the diagonal, Brier skill 0.39 and ROC area 0.93.
#
# ## 7. Against rain gauges
# ERA5 is not the truth either. Here we take every GHCN-Daily rain gauge in East Africa that reported during the
# forecast, and compare the gauge, ERA5 at the nearest grid point, and the forecast.
#
# GHCN daily rain is usually the 24 h ending at the morning reading (06 UTC in East Africa), so we sum the forecast
# and ERA5 over 06 UTC to 06 UTC. Stations do not all follow this, which adds error that is not the model's fault.

# %%
from earth2studio.data import GHCNDaily

meta, inv = GHCNDaily.get_station_metadata(), GHCNDaily.get_station_inventory()
ids = inv[(inv.ELEMENT == "PRCP") & (inv.FIRSTYEAR <= INIT.year) & (inv.LASTYEAR >= INIT.year)].ID
st = meta[meta.ID.isin(ids) & meta.LAT.between(-12, 15) & meta.LON.between(28, 52)].set_index("ID")
days = [INIT + timedelta(days=k) for k in range(1, 5)]           # gauge days ending 06 UTC on days 1-4
g = GHCNDaily(stations=list(st.index), verbose=False)(days, ["tp"])
g = g.assign(day=pd.to_datetime(g.time).dt.normalize(), gauge=g.observation * 1000)[["station", "day", "gauge"]]
g = g.drop_duplicates(["station", "day"])
print(f"{g.station.nunique()} gauges reporting, {len(g)} station-days")

def window06(da6):
    """Sum 6-h rain into 06-06 UTC days ending on each date in `days` (da6 has lead_time in hours)."""
    out = []
    for dd in days:
        k = (dd - INIT).days
        sel = [np.timedelta64(h, "h") for h in (24 * k - 12, 24 * k - 6, 24 * k, 24 * k + 6)]
        out.append((da6.sel(lead_time=sel).clip(min=0) * 1000).sum("lead_time"))
    return xr.concat(out, dim=pd.Index(pd.to_datetime(days), name="day"))

f06 = window06(fc.tp06)                                           # (day, ensemble, lat, lon)
o06 = window06(by_lead(obs.sel(variable="tp06", drop=True)))
pts = dict(lat=xr.DataArray(st.LAT.values, dims="station"), lon=xr.DataArray(st.LON.values, dims="station"),
           method="nearest")
fp, op = f06.sel(**pts).assign_coords(station=st.index.values), o06.sel(**pts).assign_coords(station=st.index.values)
m = g.merge(op.to_dataframe("era5").reset_index()[["station", "day", "era5"]], on=["station", "day"])
m = m.merge(fp.mean("ensemble").to_dataframe("ens_mean").reset_index()[["station", "day", "ens_mean"]], on=["station", "day"])
m = m.merge((fp >= THR).mean("ensemble").to_dataframe("p10").reset_index()[["station", "day", "p10"]], on=["station", "day"])
m = m.dropna()

rows = []
for name, col in [("ERA5", "era5"), ("forecast ensemble mean", "ens_mean")]:
    F, O = m[col] >= THR, m.gauge >= THR
    a, b, c = (F & O).sum(), (F & ~O).sum(), (~F & O).sum()
    rows.append({"vs gauges": name, "MAE (mm)": (m[col] - m.gauge).abs().mean(), "hits": a, "false alarms": b,
                 "misses": c, "POD": a / max(a + c, 1), "CSI": a / max(a + b + c, 1)})
print(f"{len(m)} gauge-days, {int((m.gauge >= THR).sum())} with > {THR:.0f} mm")
print(pd.DataFrame(rows).set_index("vs gauges").round(2).to_string())
print(f"Brier score of P(>{THR:.0f} mm) at gauges: {np.mean((m.p10 - (m.gauge >= THR))**2):.3f}")

fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(m.gauge, m.era5, s=12, label="ERA5"); ax.scatter(m.gauge, m.ens_mean, s=12, marker="x", label="forecast mean")
mx = max(m.gauge.max(), m.era5.max(), m.ens_mean.max()) + 5
ax.plot([0, mx], [0, mx], "k--", lw=.8); ax.set_xlabel("gauge (mm/day)"); ax.set_ylabel("gridded (mm/day)")
ax.set_title("East Africa gauges, 23-26 Apr 2024"); ax.legend(); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig("../outputs/lab13_gauges.png", dpi=110); plt.show()

# %% [markdown]
# **On our run:** 16 gauges, 38 gauge-days, 13 of them over 10 mm. ERA5 differed from the gauges by 17 mm/day on
# average, no closer than the forecast (16 mm/day). 38 cases are far too few to rank the two; the lesson is that
# over East Africa a reanalysis and a gauge can disagree as much as a forecast does, so a forecast cannot be
# judged more precisely than its "truth".
#
# ## What to take away
# - **Always compare with a baseline.** A number like "RMSE 1.5 °C" means nothing until you know persistence and
#   climatology score 2.5 °C.
# - **Pick the score for the decision.** Amounts: MAE/RMSE. Events (rain yes/no): CSI, POD, FAR, frequency bias.
#   Location: FSS. Probabilities: Brier, reliability, ROC. Ensemble honesty: CRPS, spread/error, rank histogram.
# - **Averaging smooths rain**: the ensemble mean wins RMSE but over-forecasts drizzle and under-forecasts heavy rain.
# - **Reanalysis is not truth** where there are few stations; check against gauges, and mind the reporting window.
# - **One case is an anecdote.** Scores from one start date are noisy and the points are not independent. A real
#   verification uses a season or more of start dates (Lab 09 does four) before deciding anything.
#
# ## Exercise
# 1. Change the rain threshold `THR` to 25 and 50 mm/day. Which scores fall apart first, and why?
# 2. Recompute section 4 over East Africa only (`EA`). Is the forecast better or worse there than for all of Africa?
# 3. Run a second start date (e.g. 2024-04-29, change `INIT` and `OUT`) and combine both. How much do the scores move?
# 4. A water authority issues a flood warning when P(> 50 mm in 5 days) ≥ 50 %. Using section 6's method, would that
#    rule have caught the rain around Nairobi? How many false alarms would it have raised?
