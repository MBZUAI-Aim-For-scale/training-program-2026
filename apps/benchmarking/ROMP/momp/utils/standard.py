"""
momp/utils/standard.py
Standardization and dimension formatting utilities for NetCDF datasets.
"""

import numpy as np
import pandas as pd
import xarray as xr

# Standard dimension order: time-like dims first, then spatial
_STANDARD_DIM_ORDER = ["init_time", "time", "member", "step", "lat", "lon"]


def align_spatial_grid(data, reference, *, label='data'):
    """Reorder identical grids; require explicit regridding for different cells."""
    if np.ndim(data) == 0:
        return data.item() if hasattr(data, 'item') else data
    for axis in ('lat', 'lon'):
        for name, value in ((label, data), ('reference', reference)):
            if axis not in value.dims or axis not in value.coords or value[axis].dims != (axis,):
                raise ValueError(f"{name} grid requires a one-dimensional {axis} coordinate")
            coordinates = np.asarray(value[axis].values)
            if (not np.issubdtype(coordinates.dtype, np.number) or
                    not np.isfinite(coordinates).all() or
                    len(np.unique(coordinates)) != len(coordinates)):
                raise ValueError(f"{name} grid has invalid or duplicate {axis} coordinates")
        if not np.array_equal(np.sort(data[axis].values), np.sort(reference[axis].values)):
            raise ValueError(
                f"{label} grid {axis} coordinates do not match the reference; "
                "regrid the inputs explicitly before onset evaluation")
    return data.sel(lat=reference.lat, lon=reference.lon)


def dim_order_fmt(ds: xr.Dataset) -> xr.Dataset:
    """
    Transpose all variables in ds so their dimensions follow the standard order:
        (init_time, time, member, step, lat, lon)

    Only the dimensions actually present in each variable are used — missing ones
    are skipped. For example, a variable with dims (lat, lon, init_time, step)
    becomes (init_time, step, lat, lon).

    Any dimension not listed in _STANDARD_DIM_ORDER is appended at the end in
    their original relative order, so unknown dims are never dropped.
    """
    new_vars = {}
    for var in ds.data_vars:
        da = ds[var]
        current_dims = list(da.dims)

        # Build target order: known dims in standard order, then any unknown dims
        ordered = [d for d in _STANDARD_DIM_ORDER if d in current_dims]
        remainder = [d for d in current_dims if d not in _STANDARD_DIM_ORDER]
        target_dims = ordered + remainder

        if target_dims != current_dims:
            da = da.transpose(*target_dims)

        new_vars[var] = da

    return ds.assign(new_vars)


def _promote_coords(ds: xr.Dataset) -> xr.Dataset:
    """Promotes candidate coordinate variables from data_vars into active coords."""
    coord_candidates = [
        "lat", "latitude", "lon", "longitude", "time", "date",
        "init_time", "step", "day", "prediction_timedelta",
        "member", "number", "sample", "adm3_name", "woreda", "district"
    ]
    for dv in list(ds.data_vars):
        if dv.lower() in coord_candidates or (ds[dv].ndim == 1 and ds[dv].dims == (dv,)):
            ds = ds.set_coords(dv)
    return ds


def dim_fmt(ds: xr.Dataset) -> xr.Dataset:
    """Standardize dimension names for observation datasets."""
    ds = _promote_coords(ds)
    coord_list = list(ds.coords.keys())

    rename = {}
    for target in ("lat", "lon"):
        if target not in coord_list:
            aliases = [name for name in coord_list if target in name.lower()]
            if aliases:
                rename[aliases[0]] = target
    if rename:
        ds = ds.rename(rename)

    coord_list = list(ds.coords.keys())

    if set(ds.dims) == {"lat", "lon"} and len(ds.dims) == 2:
        return ds

    if "time" not in coord_list:
        keywords = ["time", "date"]
        time_coords = [
            variable
            for variable in coord_list
            if any(keyword in variable.lower() for keyword in keywords)
        ]
        if time_coords:
            ds = ds.rename({time_coords[0]: "time"})

    return dim_order_fmt(ds)


def dim_fmt_model(ds: xr.Dataset, preserve_admin=False) -> xr.Dataset:
    """Standardize dimension names for deterministic reforecast model data."""
    ds = _promote_coords(ds)
    coord_list = list(ds.coords.keys())

    rename = {}
    for target in ("lat", "lon"):
        if target not in coord_list:
            aliases = [name for name in coord_list if target in name.lower()]
            if aliases:
                rename[aliases[0]] = target
    if rename:
        ds = ds.rename(rename)
    if "lon" not in ds.coords and "lat" not in ds.coords and not preserve_admin:
        # Support regional/tabular datasets indexed by administrative area (e.g., adm3_name)
        spatial_candidates = ["adm3_name", "adm2_name", "adm1_name", "woreda", "district", "station", "location"]
        for cand in spatial_candidates:
            if cand in ds.coords or cand in ds.dims or cand in ds.data_vars:
                if cand in ds.data_vars:
                    ds = ds.set_coords(cand)
                ds = ds.rename({cand: "lat"})
                ds = ds.expand_dims(lon=[0])
                break


    coord_list = list(ds.coords.keys())

    if "init_time" not in coord_list:
        time_coords = [variable for variable in coord_list if "time" in variable.lower()]
        if time_coords:
            ds = ds.rename({time_coords[0]: "init_time"})

    coord_list = list(ds.coords.keys())

    if "step" not in coord_list:
        keywords = ["day", "step", "prediction_timedelta", "lead"]
        step_coords = [
            variable
            for variable in coord_list
            if any(keyword in variable.lower() for keyword in keywords)
        ]
        if step_coords:
            ds = ds.rename({step_coords[0]: "step"})

    if "step" in ds.coords:
        step = ds["step"]
        if np.issubdtype(step.dtype, np.timedelta64):
            days = np.asarray(step.values / np.timedelta64(1, "D"), dtype=float)
        else:
            units = str(step.attrs.get("units", "days")).strip().lower()
            factors = {"day": 1, "days": 1, "d": 1, "hour": 24, "hours": 24, "h": 24,
                       "minute": 1440, "minutes": 1440, "second": 86400, "seconds": 86400}
            if units not in factors:
                raise ValueError(f"Unsupported forecast lead units {units!r}; daily amounts are required.")
            days = np.asarray(step.values, dtype=float) / factors[units]
        if (days.ndim != 1 or not np.isfinite(days).all() or
                not np.allclose(days, np.rint(days), rtol=0, atol=1e-8)):
            raise ValueError("Forecast leads must identify whole daily rainfall amounts; subdaily data require aggregation.")
        days = np.rint(days).astype(int)
        if days.size and (days[0] not in (0, 1) or not np.all(np.diff(days) == 1)):
            raise ValueError("Forecast leads must be consecutive daily steps beginning at day 0 or 1.")
        ds = ds.assign_coords(step=xr.DataArray(days, dims=step.dims, attrs={"units": "days"}))

    return dim_order_fmt(ds)


def dim_fmt_model_ensemble(ds: xr.Dataset, preserve_admin=False) -> xr.Dataset:
    """Standardize dimension names for probabilistic reforecast model data."""

    ds = dim_fmt_model(ds, preserve_admin=preserve_admin)

    coord_list = list(ds.coords.keys())

    if "member" not in coord_list:
        keywords = ["number", "sample", "member"]
        ensemble_coords = [
            variable
            for variable in coord_list
            if any(keyword in variable.lower() for keyword in keywords)
        ]
        if ensemble_coords:
            ds = ds.rename({ensemble_coords[0]: "member"})

    return dim_order_fmt(ds)
