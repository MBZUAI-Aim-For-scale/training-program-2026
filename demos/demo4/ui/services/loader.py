# Authors: The africlimate AI team
# SPDX-License-Identifier: MIT
"""Engine-backed summary loader for the dashboard.

Reads the annual+monthly summary files through the engine's
:func:`ui.services.engine.open_summary` (the single home for the
result-path scheme) and adapts them to the ``LeadTimeData`` shape the charts
consume. Replaces the UI's old hand-rolled ModelLoader + path builder.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
import xarray as xr
from ui.services.engine import open_summary


@dataclass
class LeadTimeData:
    """All precomputed statistics for a single model × lead-time slice."""
    annual_average:          float                # scalar — y-value for the time-series chart
    annual_spatial_average:  xr.DataArray         # (lat, lon)        — annual spatial map
    monthly_average:         np.ndarray           # (12,)             — monthly time-series
    monthly_spatial_average: xr.DataArray | None  # (month, lat, lon) — may be freed after precaching


def _to_lead_time_data(ds: xr.Dataset) -> LeadTimeData:
    """Adapt one summary dataset to LeadTimeData (same extraction as before)."""
    mon_spatial = ds["monthly_spatial_average"]
    for dim in ("year", "lead_time"):
        if dim in mon_spatial.dims and mon_spatial.sizes[dim] == 1:
            mon_spatial = mon_spatial.squeeze(dim)
    return LeadTimeData(
        annual_average          = float(ds["annual_average"].values.flat[0]),
        annual_spatial_average  = ds["annual_spatial_average"].squeeze(),
        # atleast_1d so a single-month file stays a length-1 vector.
        monthly_average         = np.atleast_1d(ds["monthly_average"].values.squeeze()),
        monthly_spatial_average = mon_spatial,
    )


def load_month_spatial(
    forecast: str, native_var: str, target: str, metric: str, year: str,
    lead: int, month: int, *, root: Union[Path, str, None] = None,
) -> "xr.DataArray | None":
    """One month's spatial field from a summary file, reloaded on demand.

    The gridded ``monthly_spatial_average`` is freed from memory after precaching, so the
    spatial map's monthly view re-reads just the requested month's slice here rather than
    holding every month in RAM. Returns a (lat/lon) DataArray, or None if absent.
    """
    try:
        with open_summary(forecast, native_var, target, metric, year, lead, root=root) as ds:
            da = ds["monthly_spatial_average"]
            for dim in ("year", "lead_time"):
                if dim in da.dims and da.sizes[dim] == 1:
                    da = da.squeeze(dim)
            return da.sel(month=month).load()
    except (FileNotFoundError, KeyError):
        return None


def load_season_spatial(
    forecast: str, native_var: str, target: str, metric: str, year: str,
    lead: int, months: "tuple[int, ...] | list[int]",
    *, root: Union[Path, str, None] = None,
) -> "xr.DataArray | None":
    """Season-mean spatial field, reloaded on demand and averaged in one open.

    Like :func:`load_month_spatial`, but averages the ``monthly_spatial_average``
    slices for every requested month that the file actually has (their
    intersection) into a single (lat/lon or station) DataArray — the season mean.
    Opening the summary once and reducing here is cheaper than one open per month.
    Returns None if the file / variable is missing or none of ``months`` are present.
    """
    try:
        with open_summary(forecast, native_var, target, metric, year, lead, root=root) as ds:
            da = ds["monthly_spatial_average"]
            for dim in ("year", "lead_time"):
                if dim in da.dims and da.sizes[dim] == 1:
                    da = da.squeeze(dim)
            month_dim = next((d for d in ("month", "time") if d in da.dims), None)
            if month_dim is None:
                return None
            present = [m for m in da[month_dim].values if int(m) in set(months)]
            if not present:
                return None
            return da.sel({month_dim: present}).mean(month_dim, skipna=True).load()
    except (FileNotFoundError, KeyError):
        return None


def load_daily_series(
    forecast: str, native_var: str, target: str, metric: str, year: str,
    lead: int, *, root: Union[Path, str, None] = None,
) -> "tuple[np.ndarray, np.ndarray] | None":
    """The daily metric series (``annual_average`` over time) for one run + lead, on demand.

    For correlation metrics ``annual_average`` is a per-day series (spatial-pattern
    correlation), not a scalar — the Lead-Time tab's correlation view reads it here rather
    than from the collapsed ``LeadTimeData.annual_average``. Returns ``(times, values)`` or
    None if absent / not a time series.
    """
    try:
        with open_summary(forecast, native_var, target, metric, year, lead, root=root) as ds:
            da = ds["annual_average"]
            for dim in ("year", "lead_time"):
                if dim in da.dims and da.sizes[dim] == 1:
                    da = da.squeeze(dim)
            if "time" not in da.dims:
                return None
            da = da.load()
            return da["time"].values, np.asarray(da.values, dtype=float)
    except (FileNotFoundError, KeyError):
        return None


def load_run(
    forecast:   str,
    native_var: str,
    target:     str,
    metric:     str,
    year:       str,
    leads:      list[int],
    *,
    root:       Union[Path, str, None] = None,
) -> dict[int, LeadTimeData]:
    """Return ``{lead: LeadTimeData}`` for every requested lead present on disk.

    Missing leads are skipped (a run rarely covers every canonical lead);
    callers get whatever exists. ``root`` is the results root the UI reads
    from (``DATA_DIR/africa``).
    """
    out: dict[int, LeadTimeData] = {}
    for lt in leads:
        try:
            with open_summary(forecast, native_var, target, metric, year, lt,
                              root=root) as ds:
                out[lt] = _to_lead_time_data(ds.load())
        except FileNotFoundError:
            continue
        except Exception as exc:  # noqa: BLE001 — log + skip, never crash the page
            print(f"[loader] {forecast}/{target}/{metric}/{lt}h: {exc}")
    return out
