# Authors: The africlimate AI team
# SPDX-License-Identifier: MIT
from __future__ import annotations

import numpy as np
import xarray as xr
from ui.utils.regions import mask_region

# Per-LeadTimeData regional result caches keyed by (id(ltd), region).
# id() is stable because LeadTimeData objects are held in _DATA_CACHE.
_MEAN_CACHE:    dict[tuple, float]         = {}
_MONTHLY_CACHE: dict[tuple, np.ndarray]    = {}
_GRID_CACHE:    dict[tuple, xr.DataArray]  = {}
_CLIM_GRID_CACHE: dict[tuple, xr.DataArray] = {}  # keyed by (id(clim), region)


def is_region_warm(ltd: "LeadTimeData", region: str) -> bool:  # type: ignore[name-defined]
    """Return True if the scalar/monthly caches are populated for this (ltd, region).

    Spatial grids are computed on demand by the map tab and are not required
    for time-series or seasonal charts, so _GRID_CACHE is not checked here.
    """
    key = (id(ltd), region)
    return key in _MEAN_CACHE and key in _MONTHLY_CACHE


def ensure_region_warm(
    all_data: "dict[str, dict[int, LeadTimeData]]",  # type: ignore[name-defined]
    region: str,
    on_progress: "Callable[[int, int], None] | None" = None,  # type: ignore[name-defined]
) -> None:
    """Populate scalar and monthly caches for every LeadTimeData in all_data.

    Spatial grids (annual_spatial_grid) are NOT warmed here — they are large
    (2 MB each × all lead times × all regions = hundreds of GB) and are built
    on demand by the spatial map tab only for the lead time being viewed.

    on_progress(done, total) is called after each cache operation so callers
    can drive a progress bar with fine-grained percentage updates.
    """
    items = [
        ltd
        for lt_map in all_data.values()
        for ltd in lt_map.values()
        if ltd is not None
    ]
    total = len(items) * 2  # mean + monthly per LeadTimeData
    done = 0
    for ltd in items:
        regional_annual_mean(ltd, region)
        done += 1
        if on_progress:
            on_progress(done, total)
        regional_monthly_means(ltd, region)
        done += 1
        if on_progress:
            on_progress(done, total)


def subset_region(da: xr.DataArray, region: str) -> xr.DataArray:
    """Mask and crop a DataArray to the given region using exact country borders (cached)."""
    key = (id(da), region)
    if key not in _CLIM_GRID_CACHE:
        _CLIM_GRID_CACHE[key] = mask_region(da, region)
    return _CLIM_GRID_CACHE[key]


def annual_spatial_grid(ltd: "LeadTimeData", region: str) -> xr.DataArray:  # type: ignore[name-defined]
    """Return the masked annual spatial grid for a LeadTimeData object (cached).

    Caches the result of mask_region(ltd.annual_spatial_average.squeeze(), region)
    so spatial map renders don't recompute the da.where(keep).isel() chain on every
    frame — the masked numpy array is built once and reused.
    """
    key = (id(ltd), region)
    if key not in _GRID_CACHE:
        _GRID_CACHE[key] = mask_region(ltd.annual_spatial_average.squeeze(), region)
    return _GRID_CACHE[key]


def regional_annual_mean(ltd: "LeadTimeData", region: str) -> float:  # type: ignore[name-defined]
    """Spatial mean of annual_spatial_average within the region (cached)."""
    key = (id(ltd), region)
    if key in _MEAN_CACHE:
        return _MEAN_CACHE[key]

    if region == "Africa":
        result = float(ltd.annual_average)
    else:
        try:
            grid = mask_region(ltd.annual_spatial_average, region)
            val  = float(grid.mean().values)
            result = val if not np.isnan(val) else float(ltd.annual_average)
        except Exception:
            result = float(ltd.annual_average)

    _MEAN_CACHE[key] = result
    return result


def regional_monthly_means(ltd: "LeadTimeData", region: str) -> np.ndarray:  # type: ignore[name-defined]
    """(12,) array of regional spatial means indexed Jan→Dec.

    Values are placed at their TRUE calendar-month positions using the
    NetCDF's `month` coordinate. Months absent from the data are NaN —
    not silently filled with the next available month, which previously
    made April's value appear under January for 2024 IFS+IDR (months 4-12).
    """
    key = (id(ltd), region)
    if key in _MONTHLY_CACHE:
        return _MONTHLY_CACHE[key]

    da = ltd.monthly_spatial_average

    # Read the real month numbers from the NetCDF — these are the
    # calendar months each value actually represents (e.g. [4, 5, ..., 12]
    # for an April-onwards file).
    months: list[int] | None = None
    if da is not None:
        try:
            month_dim = next(d for d in ("month", "time") if d in da.dims)
            months = [int(m) for m in da[month_dim].values]
        except (StopIteration, Exception):
            months = None

    if region == "Africa":
        arr = np.asarray(ltd.monthly_average)
    elif da is None:
        # monthly_spatial_average was freed after precaching; all regional
        # monthly caches should have been populated before freeing.  If we
        # reach here it means this (ltd, region) pair was never precached
        # (e.g. a new region added after startup) — return NaN so charts
        # degrade gracefully rather than silently showing Africa values.
        result = np.full(12, np.nan)
        _MONTHLY_CACHE[key] = result
        return result
    else:
        try:
            da_sub  = mask_region(da, region)
            sp_dims = [d for d in da_sub.dims if d not in ("month", "time")]
            vals    = da_sub.mean(dim=sp_dims).values
            arr     = (vals.astype(float)
                       if not np.all(np.isnan(vals))
                       else np.asarray(ltd.monthly_average))
        except Exception:
            arr = np.asarray(ltd.monthly_average)

    # A single-month file can arrive as a 0-d scalar; normalise to 1-d so
    # the month-length checks below never index an empty shape.
    arr = np.atleast_1d(arr)
    result = np.full(12, np.nan)
    if months is not None and len(months) == arr.shape[0]:
        # Honour the file's month labels — value at index 0 of `arr` may
        # correspond to any calendar month (e.g. April for a year that
        # only has data from April onwards).
        for m, v in zip(months, arr):
            if 1 <= m <= 12:
                result[m - 1] = v
    else:
        # Fallback when month coords are unavailable (e.g. cache is read
        # back after `monthly_spatial_average` was freed and we never saw
        # the coordinate). Assume the array is already Jan-indexed.
        n = min(arr.shape[0], 12)
        result[:n] = arr[:n]

    _MONTHLY_CACHE[key] = result
    return result


def regional_season_mean(
    ltd: "LeadTimeData",  # type: ignore[name-defined]
    region: str,
    months: "tuple[int, ...] | list[int] | None",
) -> float:
    """Regional season mean for a LeadTimeData/ClimData object.

    ``months`` is a tuple/list of 1-based calendar months (e.g. (6, 7, 8) for
    Kiremt); ``None`` (or empty) means Yearly, delegating to
    :func:`regional_annual_mean`. Otherwise the value is the unweighted nanmean
    of the cached regional monthly means over those months — a season with no
    data (all months NaN) returns NaN so charts degrade gracefully.
    """
    if not months:
        return regional_annual_mean(ltd, region)
    monthly = regional_monthly_means(ltd, region)
    vals = monthly[[m - 1 for m in months]]
    return float(np.nanmean(vals)) if not np.all(np.isnan(vals)) else float("nan")


def prewarm_regional_cache(
    data_cache: dict,
    regions:    list[str],
) -> None:
    """Pre-compute regional scalar and monthly means for every (LeadTimeData, region) pair.

    Spatial grids are intentionally excluded: storing a full Africa-sized masked
    DataArray for every (ltd, region) combination would require hundreds of GB.
    They are computed on demand by the spatial map tab and cached lazily.
    """
    total = 0
    for region in regions:                  # Africa first → blocs → individual countries
        for lt_map in data_cache.values():
            if not isinstance(lt_map, dict):
                continue
            for ltd in lt_map.values():
                if ltd is None:
                    continue
                regional_annual_mean(ltd, region)
                regional_monthly_means(ltd, region)
                total += 1
    print(f"[prewarm_regional_cache] {total} (region × ltd) scalar/monthly entries cached")
