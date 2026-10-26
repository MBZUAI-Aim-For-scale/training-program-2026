"""
momp/utils/standard.py
Standardization and dimension formatting utilities for NetCDF datasets.
"""

import pandas as pd
import xarray as xr

# Standard dimension order: time-like dims first, then spatial
_STANDARD_DIM_ORDER = ["init_time", "time", "member", "step", "lat", "lon"]


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
        if dv.lower() in coord_candidates or any(c in dv.lower() for c in ["lat", "lon", "time", "step", "member"]):
            ds = ds.set_coords(dv)
    return ds


def dim_fmt(ds: xr.Dataset) -> xr.Dataset:
    """Standardize dimension names for observation datasets."""
    ds = _promote_coords(ds)
    coord_list = list(ds.coords.keys())

    if "lon" not in coord_list:
        lat_coords = [variable for variable in coord_list if "lat" in variable.lower()]
        lon_coords = [variable for variable in coord_list if "lon" in variable.lower()]

        if lat_coords and lon_coords:
            ds = ds.rename({lat_coords[0]: "lat", lon_coords[0]: "lon"})

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


def dim_fmt_model(ds: xr.Dataset) -> xr.Dataset:
    """Standardize dimension names for deterministic reforecast model data."""
    ds = _promote_coords(ds)
    coord_list = list(ds.coords.keys())

    if "lon" not in coord_list:
        lat_coords = [variable for variable in coord_list if "lat" in variable.lower()]
        lon_coords = [variable for variable in coord_list if "lon" in variable.lower()]

        if lat_coords and lon_coords:
            ds = ds.rename({lat_coords[0]: "lat", lon_coords[0]: "lon"})
        else:
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

    # Convert TimedeltaIndex to integer days
    if "step" in ds.indexes and isinstance(ds.indexes["step"], pd.TimedeltaIndex):
        ds = ds.assign_coords(step=ds.step.dt.days)

    return dim_order_fmt(ds)


def dim_fmt_model_ensemble(ds: xr.Dataset) -> xr.Dataset:
    """Standardize dimension names for probabilistic reforecast model data."""

    ds = dim_fmt_model(ds)

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
