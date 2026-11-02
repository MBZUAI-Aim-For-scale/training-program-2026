"""Bounds for rectilinear geographic grids, including full boundary cells."""

import numpy as np


def geographic_extent(lons, lats, values=None, padding=0.08, region_bounds=None):
    """Return west/east/south/north around finite cells with symmetric margins.

    All-missing fields use the coordinate domain. Coordinates may be descending
    or irregular; cell edges use the adjacent spacing at each outer center.
    Optional region_bounds (west, east, south, north) frame the requested region
    instead, with fixed geographic padding of at least 0.25 degrees per side.
    """
    lon = np.asarray(lons, dtype=float)
    lat = np.asarray(lats, dtype=float)
    if lon.ndim != 1 or lat.ndim != 1:
        raise ValueError("Map extent requires one-dimensional lon/lat coordinates")
    if region_bounds is not None:
        west, east, south, north = map(float, region_bounds)
        if not np.isfinite([west, east, south, north]).all() or west >= east or south >= north:
            raise ValueError("Region bounds must be finite west/east/south/north limits")

        def region_margin(low, high):
            margin = max((high - low) * padding, 0.25)
            return low - margin, high + margin

        west, east = region_margin(west, east)
        south, north = region_margin(south, north)
        return west, east, max(-90, south), min(90, north)
    valid_lon, valid_lat = np.isfinite(lon), np.isfinite(lat)
    if values is not None:
        valid = np.isfinite(np.asarray(values, dtype=float))
        if valid.shape != (lat.size, lon.size):
            raise ValueError("Map values must have shape (lat, lon)")
        valid &= valid_lat[:, None] & valid_lon[None, :]
        if valid.any():
            valid_lon &= valid.any(axis=0)
            valid_lat &= valid.any(axis=1)

    def bounds(coords, selected):
        full = np.unique(coords[np.isfinite(coords)])
        chosen = np.sort(coords[selected])
        if not chosen.size:
            raise ValueError("Map coordinates contain no finite values")
        low, high = chosen[0], chosen[-1]
        i, j = np.searchsorted(full, [low, high])
        left = full[i] - full[i - 1] if i else (full[1] - full[0] if full.size > 1 else 1.0)
        right = full[j + 1] - full[j] if j < full.size - 1 else (full[-1] - full[-2] if full.size > 1 else 1.0)
        low, high = low - left / 2, high + right / 2
        margin = (high - low) * padding
        return low - margin, high + margin

    west, east = bounds(lon, valid_lon)
    south, north = bounds(lat, valid_lat)
    return west, east, max(-90, south), min(90, north)
