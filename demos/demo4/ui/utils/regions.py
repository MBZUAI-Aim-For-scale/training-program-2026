# Authors: The africlimate AI team
# SPDX-License-Identifier: MIT
"""Display-time region masking for the dashboard.

The region→countries table and the pure masking geometry now live one layer down in
:mod:`verification.regions` (so the diagnostics pipeline can subset the IDR to a region
before pooling, without importing the UI). This module re-exports that public surface
unchanged and adds the **display-time** concerns the dashboard needs: a per-process mask
cache, the ``mask_region`` field wrapper (crop-to-bbox + NaN outside), station masking,
and mask prewarming.
"""
from __future__ import annotations

import numpy as np
import xarray as xr

from apps.verification.regions import (
    REGION_COUNTRIES,
    _country_keep,
    _get_subregions,
    _subregion_keep,
    available_regions,
    default_region,
    region_boundary,
    regions_for_domain,
    stations_in_region,
)

__all__ = [
    "REGION_COUNTRIES", "available_regions", "default_region", "regions_for_domain",
    "stations_in_region", "region_boundary", "mask_region", "prewarm_masks",
]

# Cache: (region, grid identity) → (keep array, lat slice, lon slice).
_MASK_CACHE: dict[tuple, tuple] = {}


def mask_region(da: xr.DataArray, region: str) -> xr.DataArray:
    """Return da with pixels outside the region set to NaN, cropped to its bbox.

    ``region`` is one of: 'Africa' (or any None-member key) → returned as-is; a
    REGION_COUNTRIES key → masked with the union of its Natural Earth country polygons
    (SADEC, GHAF, … combine all members); or an admin-1 subregion of the deployment
    country → masked with its shapefile polygon.
    """
    is_sub = region in _get_subregions()
    if not is_sub and REGION_COUNTRIES.get(region) is None:
        return da  # full domain (Africa) or unknown key — no masking needed

    if "station" in da.dims:
        return _mask_stations(da, region)

    # Detect spatial coordinate names
    lat_name = "latitude" if "latitude" in da.coords else "lat"
    lon_name = "longitude" if "longitude" in da.coords else "lon"

    lat = da[lat_name].values
    lon = da[lon_name].values

    # Cache key: region + grid identity (shape + first/last value)
    grid_key = (lat.shape[0], float(lat[0]), float(lat[-1]),
                lon.shape[0], float(lon[0]), float(lon[-1]))
    cache_key = (region, grid_key)

    if cache_key not in _MASK_CACHE:
        keep_raw = (_subregion_keep(region, lon, lat) if is_sub
                    else _country_keep(region, lon, lat, lat_name, lon_name))
        if keep_raw is None or not keep_raw.any():
            _MASK_CACHE[cache_key] = (None, None, None)
        else:
            lat_idx = np.where(keep_raw.any(axis=1))[0]
            lon_idx = np.where(keep_raw.any(axis=0))[0]
            buf = 2
            lat_sl = slice(max(0, lat_idx[0] - buf), lat_idx[-1] + buf + 1)
            lon_sl = slice(max(0, lon_idx[0] - buf), lon_idx[-1] + buf + 1)
            # Store the raw numpy array — rebuilding with each da's exact coords
            # at query time prevents xarray from broadcasting into a 4D outer product
            # when coordinate float values differ slightly across DataArrays.
            _MASK_CACHE[cache_key] = (keep_raw, lat_sl, lon_sl)

    keep_raw, lat_sl, lon_sl = _MASK_CACHE[cache_key]
    if keep_raw is None:
        return da

    # Always rebuild with current da's coordinates to guarantee alignment
    keep = xr.DataArray(
        keep_raw,
        dims=[lat_name, lon_name],
        coords={lat_name: da[lat_name], lon_name: da[lon_name]},
    )
    masked = da.where(keep)
    if lat_sl is not None and lon_sl is not None:
        masked = masked.isel({lat_name: lat_sl, lon_name: lon_sl})
    return masked


def _mask_stations(da: xr.DataArray, region: str) -> xr.DataArray:
    """Station analogue of grid masking: keep only stations inside the region.

    A station field has no lat/lon grid to NaN-out, so masking means dropping the
    stations whose gauge coordinate falls outside the region. Coord names may be
    ``station_lat``/``station_lon`` (raw/deterministic files) or plain ``lat``/``lon``.
    """
    lat_name = "station_lat" if "station_lat" in da.coords else "lat"
    lon_name = "station_lon" if "station_lon" in da.coords else "lon"
    keep = stations_in_region(da[lon_name].values, da[lat_name].values, region)
    return da.isel(station=keep)


def prewarm_masks(sample_da: xr.DataArray) -> None:
    """Pre-compute and cache masks for every known region.

    Call once at app startup with any full-domain DataArray so subsequent
    mask_region() calls are instant. A station field (no lat/lon grid) has nothing
    to grid-mask, so it is skipped rather than raising for every region.
    """
    if "station" in sample_da.dims:
        return
    subregions = _get_subregions()
    for region in available_regions():
        if region not in subregions and REGION_COUNTRIES.get(region) is None:
            continue  # Africa / full domain — no mask needed (subregions aren't keys)
        try:
            mask_region(sample_da, region)
        except Exception as exc:
            print(f"[prewarm_masks] {region}: {exc}")
