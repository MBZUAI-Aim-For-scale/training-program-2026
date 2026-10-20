"""Models offered in Demo 3.

One entry per model on the 2026 list. `env` names the Python environment the
model's runner executes in (models whose dependencies conflict get separate
environments inside the container). A model can run live only if its
environment exists at {ENV_ROOT}/{env}; otherwise only its saved runs load.
"""
from __future__ import annotations

import functools
import json
import os
import urllib.request
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

ARCO_ERA5 = "ARCO ERA5 (Google)"
IFS_OPEN = "IFS analysis (ECMWF Open Data)"
# ECMWF open data on AWS starts 2024-03-01 and is current to within a few days.
IFS_MIN, IFS_MAX = date(2024, 3, 1), date.today() - timedelta(days=2)
# ECMWF open data has every input AIFS v2 needs (incl. 10 hPa levels) only from this date.
AIFS2_MIN = date(2026, 5, 13)
# Before that, AIFS v2 starts from the ECMWF wave dataset (ocean-wave inputs, 00Z, 2000-2025;
# copied to the shared bucket) plus ARCO ERA5 for everything else.
WAVE_MIN, WAVE_MAX = date(2000, 1, 1), date(2025, 12, 31)
WAVE_BUCKET = os.environ.get("DEMO3_WAVE_BUCKET", "gs://aim4scale_training_2026/wave/n320")
WAVE_DIR = Path(os.environ.get("DEMO3_WAVE_DIR", Path.home() / ".cache" / "wave"))


@functools.lru_cache(maxsize=1)
def _wave_days(_bucket_minute: int) -> frozenset:
    """Days whose wave inputs are on this Spark or in the bucket (listing refreshed every 10 min)."""
    days = {p.name[5:13] for p in WAVE_DIR.glob("wave_*T00.nc")}
    try:
        import fsspec
        fs, root = fsspec.core.url_to_fs(WAVE_BUCKET)
        days |= {p.rsplit("/", 1)[-1][5:13] for p in fs.ls(root, detail=False) if p.endswith("T00.nc")}
    except Exception:
        pass
    return frozenset(days)


def wave_ready(day: date) -> bool:
    import time as _time
    return f"{day:%Y%m%d}" in _wave_days(int(_time.time() // 600))
# Google's ARCO ERA5 copy: Earth2Studio serves it up to `valid_time_stop` (final ERA5, about
# 3 months behind); the preliminary ERA5T part runs to `valid_time_stop_era5t` (about a week
# behind) and is what the event movie's observations can use. Both move forward over time.
ARCO_ATTRS_URL = ("https://storage.googleapis.com/gcp-public-data-arco-era5/ar/"
                  "full_37-1h-0p25deg-chunk-1.zarr-v3/.zattrs")
ERA5_FALLBACK = {"valid_time_stop": "2026-06-30", "valid_time_stop_era5t": "2026-09-30"}


@functools.lru_cache(maxsize=1)
def era5_dates() -> tuple[date, date]:
    """(last day Earth2Studio serves, last day of preliminary ERA5T) from Google's metadata."""
    try:
        with urllib.request.urlopen(ARCO_ATTRS_URL, timeout=5) as r:
            attrs = json.load(r)
    except Exception:
        attrs = ERA5_FALLBACK
    attrs = {**ERA5_FALLBACK, **attrs}
    return (date.fromisoformat(attrs["valid_time_stop"]),
            date.fromisoformat(attrs["valid_time_stop_era5t"]))


# Forecast variables every run aims to provide, with the names shown to participants.
VAR_LABELS = {"tp": "rainfall", "t2m": "2 m temperature", "z500": "500 hPa geopotential (z500)"}

# Event thresholds used across the demos (from Docko Sow); participants can type others
HEAT_THRESHOLDS = [37, 38, 39, 40, 41, 42, 43, 44, 45]                       # °C, daily max
RAIN_THRESHOLDS = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]         # mm/day
RUNNERS_DIR = Path(__file__).resolve().parent.parent / "runners"
# Model environments inside the container: {ENV_ROOT}/{env}/bin/python
ENV_ROOT = Path(os.environ.get("DEMO3_ENV_ROOT", "/opt/envs"))


@dataclass(frozen=True)
class Model:
    key: str
    name: str
    org: str
    kind: str                     # "deterministic" | "ensemble"
    init_source: str
    env: str                      # runner environment name
    max_lead_days: int
    max_members: int = 1
    default_members: int = 1
    slow: bool = False            # suggest loading a saved run in class
    init_min: date = date(2020, 1, 1)
    init_max: date | None = None  # None: the last day ERA5 is available (see era5_dates)
    notes: str = ""
    missing: tuple[str, ...] = ()  # variables (VAR_LABELS keys) the model does not forecast
    grid_deg: float = 0.25          # output grid spacing
    weights_gb: float = 0.0         # model weights on disk (downloaded once)
    output_mb_10d: float = 0.0      # saved NetCDF for 10 days at the default members
    inputs: str = ""                # what the starting conditions are and how they're prepared

    @property
    def live(self) -> bool:
        """True if this container can run the model: its environment is installed and it
        has a runner (FGN, for now, only has Google's saved sample case)."""
        return ((ENV_ROOT / self.env / "bin" / "python").exists()
                and (RUNNERS_DIR / f"runner_{self.key}.py").exists())

    def date_range(self) -> tuple[date, date]:
        """First and last start date this model can use."""
        if self.init_max is not None:
            return self.init_min, self.init_max
        # the last ERA5 day is served up to 00Z; stop the day before so 12Z works too
        return self.init_min, era5_dates()[0] - timedelta(days=1)

    def date_reason(self) -> str:
        lo, hi = self.date_range()
        if self.init_source == ARCO_ERA5:
            return (f"{self.name} starts from ERA5 (Google's copy), which is available up to "
                    f"{hi:%-d %b %Y} (it runs about 3 months behind real time).")
        if self.key.startswith("aifs2"):
            return ("AIFS v2 needs ocean-wave inputs: for 00Z starts from 2000 to 2025 they come from "
                    "the ECMWF wave dataset (the rest from ERA5); from 13 May 2026, everything comes "
                    "from ECMWF's open data (current to 2 days ago). 1 Jan – 12 May 2026 isn't covered.")
        return (f"{self.name} starts from ECMWF open data, available from {lo:%-d %b %Y} "
                "to 2 days ago.")


MODELS: list[Model] = [
    Model("aifs2_single", "AIFS Single v2.0", "ECMWF", "deterministic", IFS_OPEN, "e2s018",
          max_lead_days=15, init_min=WAVE_MIN, init_max=IFS_MAX,
          notes="From 13 May 2026 it starts from ECMWF's IFS analyses (open data). For 00Z starts "
                "in 2000–2025 it starts from the ECMWF wave dataset plus ERA5.",
          weights_gb=1.0, output_mb_10d=262,
          inputs="ECMWF's IFS analysis at the start time and 6 h before: surface, soil and 14 pressure-level fields, interpolated from the 0.25° grid onto the model's own (N320) grid."),
    Model("graphcast", "GraphCast", "Google DeepMind", "deterministic", ARCO_ERA5, "graphcast",
          max_lead_days=10,
          weights_gb=0.14, output_mb_10d=269,
          inputs='ERA5 at the start time and 6 h before: 6 atmospheric variables on 13 pressure levels plus surface fields, on the 0.25° grid.'),
    Model("aurora15", "Aurora 1.5", "Microsoft", "deterministic", ARCO_ERA5, "e2s018",
          max_lead_days=10, slow=True,
          weights_gb=4.9, output_mb_10d=269,
          inputs='ERA5 at the start time and 6 h before: 5 atmospheric variables on 13 pressure levels plus 4 surface fields, on the 0.25° grid. The model steps one hour at a time.'),
    Model("aifs2_ens", "AIFS ENS v2", "ECMWF", "ensemble", IFS_OPEN, "e2s018ens",
          max_lead_days=15, max_members=3, default_members=2, slow=True,
          init_min=WAVE_MIN, init_max=IFS_MAX,
          notes="From 13 May 2026 it starts from ECMWF's IFS analyses (open data). For 00Z starts "
                "in 2000–2025 it starts from the ECMWF wave dataset plus ERA5.",
          weights_gb=2.4, output_mb_10d=795,
          inputs="ECMWF's IFS analysis at the start time and 6 h before: surface, soil and 14 pressure-level fields, interpolated onto the model's own (N320) grid. Each member adds its own random noise."),
    Model("atlas_crps", "Atlas CRPS", "NVIDIA", "ensemble", ARCO_ERA5, "e2s018",
          max_lead_days=10, max_members=3, default_members=2,
          weights_gb=12.5, output_mb_10d=900,
          inputs='ERA5 at the start time: surface and pressure-level fields on the 0.25° grid. Each member draws fresh random noise.'),
    Model("neuralgcm", "NeuralGCM", "Google", "ensemble", ARCO_ERA5, "neuralgcm",
          max_lead_days=10, max_members=3, default_members=2, missing=("t2m",), grid_deg=2.8,
          notes="2.8° (~300 km) stochastic precipitation version — the only NeuralGCM checkpoint "
                "that forecasts rain. No 2 m temperature output.",
          weights_gb=0.045, output_mb_10d=4.5,
          inputs="ERA5 at the start time, regridded from 0.25° to the model's 2.8° grid; sea-surface temperature and sea ice are held at their starting values."),
    Model("fgn", "FGN Mini (WeatherNext 2, 1°)", "Google DeepMind", "ensemble",
          "IFS analysis (ECMWF Open Data)", "fgn",
          max_lead_days=10, max_members=3, default_members=3, grid_deg=1.0,
          init_min=IFS_MIN, init_max=IFS_MAX,
          notes="The full 0.25° FGN does not fit in a Spark's memory, so this is Google's 1° Mini "
                "version (cyclone-tuned weights, the only Mini Google publishes). Starting conditions "
                "are built from ECMWF open data; on 7 Oct 2024 it uses Google's own sample file.",
          weights_gb=0.23, output_mb_10d=75,
          inputs="ECMWF's IFS analysis at the start time and 6 h before: 6 variables on 13 pressure levels plus surface fields, sampled onto a 1° grid and written in Google's format by our converter (Google's own sample file on 7 Oct 2024)."),
]

BY_KEY = {m.key: m for m in MODELS}


def missing_in_run(ds) -> set[str]:
    """Variables a saved run does not provide: listed as missing, absent, or all NaN."""
    out = set(filter(None, ds.attrs.get("missing_variables", "").split(",")))
    for v in VAR_LABELS:
        if v not in ds.data_vars or not bool(ds[v].notnull().any()):
            out.add(v)
    return out

# Variable each use case needs
USE_CASE_NEEDS = {"Heat": "t2m", "Precipitation": "tp", "Onset": "tp"}
# Models whose rainfall is known to be off on the Spark (checked against observations)
RAIN_CAVEATS = {"aurora15": "Aurora's rainfall comes out 2–3 times too dry in our runs (still being "
                            "investigated), so it will rarely reach rainfall or onset thresholds."}
CHIRPS_LAT = 50.0  # observed rainfall for onset covers 50°S–50°N


def check(m: Model, init, lead_days: int, members: int, box, region_name: str,
          use_case: str) -> list[tuple[str, str]]:
    """Problems with a request, as (level, message): "error" means it cannot run,
    "warning" means it runs but something will be missing or limited."""
    out = []
    lo, hi = m.date_range()
    if not lo <= init.date() <= hi:
        out.append(("error", f"{m.name} can't start on {init:%-d %b %Y}: start dates run from "
                             f"{lo:%-d %b %Y} to {hi:%-d %b %Y}. {m.date_reason()}"))
    if init.hour not in (0, 12):
        out.append(("error", "Forecasts start at 00Z or 12Z."))
    if m.key.startswith("aifs2") and lo <= init.date() < AIFS2_MIN:
        if init.date() > WAVE_MAX:
            out.append(("error", f"{m.name} can't start between 1 Jan and 12 May 2026: the ECMWF wave "
                                 "dataset ends in 2025 and ECMWF's open data has every input only "
                                 "from 13 May 2026."))
        elif init.hour != 0:
            out.append(("error", f"Before 13 May 2026, {m.name} starts from the ECMWF wave dataset, "
                                 "which has 00Z starts only. Choose 00Z."))
        elif not wave_ready(init.date()):
            out.append(("error", f"The ocean-wave inputs {m.name} needs for {init:%-d %b %Y} aren't in "
                                 "the shared bucket yet (they are being copied from DSI)."))
    if lead_days > m.max_lead_days:
        out.append(("error", f"{m.name} forecasts at most {m.max_lead_days} days ahead."))
    if m.kind == "deterministic" and members != 1:
        out.append(("error", f"{m.name} is deterministic: it gives one forecast, not an ensemble."))
    elif members > m.max_members:
        out.append(("error", f"{m.name} runs at most {m.max_members} members on the Spark."))
    need = USE_CASE_NEEDS.get(use_case)
    if need in m.missing:
        out.append(("warning", f"{m.name} does not forecast {VAR_LABELS[need]}, so there is no "
                               f"{use_case.lower()} result for this model. Choose another model, or "
                               "switch to a use case it supports."))
    if need == "tp" and m.key in RAIN_CAVEATS:
        out.append(("warning", RAIN_CAVEATS[m.key]))
    if use_case == "Onset" and box is not None and box.is_valid():
        if box.lat_min >= CHIRPS_LAT or box.lat_max <= -CHIRPS_LAT:
            out.append(("error", f"Onset uses observed rainfall from CHIRPS, which covers 50°S–50°N "
                                 f"only; {region_name} is outside it."))
    if box is not None and box.is_valid():
        n = min((box.lat_max - box.lat_min), (box.lon_max - box.lon_min)) / m.grid_deg
        km = round(m.grid_deg * 111, -1)
        if n < 2:
            out.append(("error", f"{region_name} is too small for {m.name}'s {m.grid_deg:g}° "
                                 f"(~{km:.0f} km) grid: the maps would hold fewer than 2 grid "
                                 "points across. Choose a larger region."))
        elif n < 4:
            out.append(("warning", f"{m.name}'s grid is {m.grid_deg:g}° (~{km:.0f} km), so "
                                   f"{region_name} is only about {int(n) + 1} grid points across; "
                                   "maps will look very coarse."))
    return out


TIMINGS_PATH = Path(__file__).resolve().parent.parent / "timings.json"


def load_timings() -> dict:
    try:
        return json.loads(TIMINGS_PATH.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def timing_label(model: Model, timings: dict | None = None) -> str:
    """Bracketed runtime shown next to a model, e.g. '(~4 min for 10 days, 3 members)'."""
    t = (timings if timings is not None else load_timings()).get(model.key)
    if not t:
        return "(not timed yet)"
    total = t.get("load_s", 0) + t.get("run_s", 0)
    if total < 90:
        dur = f"~{round(total)} s"
    elif total < 5400:
        dur = f"~{round(total / 60)} min"
    else:
        dur = f"~{total / 3600:.1f} h"
    detail = f"{t.get('lead_days', '?')} days"
    if model.kind == "ensemble":
        detail += f", {t.get('members', '?')} members"
    return f"({dur} for {detail})"
