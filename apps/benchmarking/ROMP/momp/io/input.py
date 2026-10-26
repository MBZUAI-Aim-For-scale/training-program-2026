"""
momp/io/input.py
Data loading utilities for observation and model forecast datasets.
"""

import os
import sys
from datetime import datetime
from pathlib import Path
import pandas as pd
import xarray as xr

# Import config safely across main and worker processes
try:
    import config
except ImportError:
    try:
        from ai_weather import config
    except ImportError:
        _curr = Path(__file__).resolve()
        for _parent in _curr.parents:
            if (_parent / "ai_weather" / "config.py").exists() or (_parent / "config.py").exists():
                sys.path.insert(0, str(_parent / "ai_weather"))
                sys.path.insert(0, str(_parent))
                break
        import config

from momp.utils.standard import dim_fmt, dim_fmt_model, dim_fmt_model_ensemble
from momp.utils.region import region_select
from momp.utils import storage
from momp.utils.practical import restore_args
import momp.params as params


def load_thresh_file(*, thresh_file=None, thresh_var=None, wet_threshold=None, region=None, **kwargs):
    kwargs_clean = kwargs.copy()
    kwargs_clean.pop("region", None)
    kwargs_clean.pop("thresh_file", None)
    kwargs_clean.pop("thresh_var", None)
    kwargs_clean.pop("wet_threshold", None)

    if thresh_file is None:
        thresh_file = getattr(params, "thresh_file", None)
    if thresh_var is None:
        thresh_var = getattr(params, "thresh_var", None)
    if wet_threshold is None:
        wet_threshold = getattr(params, "wet_threshold", 20)
    if region is None:
        region = getattr(params, "region", "Ethiopia")

    if thresh_file:
        thresh_ds = storage.open_dataset(thresh_file)
        thresh_da = thresh_ds[thresh_var]
        thresh_da = region_select(thresh_da, region=region, **kwargs_clean)
    else:
        thresh_da = wet_threshold

    return thresh_da


def get_initialization_dates(year=None, *, date_filter_year=None, init_days=None, start_date=None, end_date=None, init_type="weekly", **kwargs):
    kwargs_clean = kwargs.copy()
    kw_year = kwargs_clean.pop("year", None)
    year = year if year is not None else kw_year

    if init_type == "monthly":
        return get_initialization_dates_monthly(year, date_filter_year=date_filter_year, init_days=init_days, start_date=start_date, end_date=end_date, **kwargs_clean)

    if date_filter_year is None:
        date_filter_year = getattr(params, "date_filter_year", 2024)
    if init_days is None:
        init_days = getattr(params, "init_days", (0, 3))
    if start_date is None:
        start_date = getattr(params, "start_date", (2003, 5, 1))
    if end_date is None:
        end_date = getattr(params, "end_date", (2023, 7, 31))

    start_MMDD = start_date[1:]
    end_MMDD = end_date[1:]

    start_date_obj = datetime(date_filter_year, *start_MMDD)
    end_date_obj = datetime(date_filter_year, *end_MMDD)
    date_range = pd.date_range(start_date_obj, end_date_obj, freq='D')

    filtered_dates = date_range[date_range.weekday.isin(init_days)]
    filtered_dates_yr = pd.to_datetime(filtered_dates.strftime(f'{year}-%m-%d'))

    return filtered_dates_yr


def get_initialization_dates_monthly(year=None, *, date_filter_year=None, init_days=None, start_date=None, end_date=None, **kwargs):
    kwargs_clean = kwargs.copy()
    kw_year = kwargs_clean.pop("year", None)
    year = year if year is not None else kw_year

    if start_date is None:
        start_date = getattr(params, "start_date", (2003, 5, 1))
    if end_date is None:
        end_date = getattr(params, "end_date", (2023, 7, 31))
    if init_days is None:
        init_days = getattr(params, "init_days", (1, 15))

    start_month = start_date[1]
    end_month   = end_date[1]
    
    filtered_dates_yr = pd.DatetimeIndex([
        pd.Timestamp(year, month, day)
        for month in range(start_month, end_month + 1)
        for day in init_days
    ])

    return filtered_dates_yr


def load_imd_rainfall(year=None, *, obs_dir=None, obs_file_pattern=None, obs_var=None, obs_unit_cvt=None, region=None, 
                      grid_point=False, lat_select=None, lon_select=None, **kwargs):
    kwargs_clean = kwargs.copy()
    kw_year = kwargs_clean.pop("year", None)
    year = year if year is not None else kw_year

    if obs_dir is None:
        obs_dir = getattr(params, "obs_dir", None)
    if obs_file_pattern is None:
        obs_file_pattern = getattr(params, "obs_file_pattern", ("{}.nc",))
    if obs_var is None:
        obs_var = getattr(params, "obs_var", "precip")
    if region is None:
        region = getattr(params, "region", "Ethiopia")

    kwargs_clean.pop("region", None)

    if obs_dir and config.is_uri(obs_dir):
        obs_dir = f"gs://{config._strip(obs_dir)}"

    region_clean = str(region).replace(" ", "_")

    file_patterns = []
    if isinstance(obs_file_pattern, (list, tuple)):
        file_patterns.extend(p.format(year) for p in obs_file_pattern if "{}" in p)
    elif isinstance(obs_file_pattern, str) and "{}" in obs_file_pattern:
        file_patterns.append(obs_file_pattern.format(year))

    fallbacks = [
        f"CHIRPS_{region_clean}_{year}.nc",
        f"CHIRPS_{region}_{year}.nc",
        f"{year}.nc"
    ]
    for fb in fallbacks:
        if fb not in file_patterns:
            file_patterns.append(fb)

    obs_file = None
    for pattern in file_patterns:
        test_path = storage.join(obs_dir, pattern)
        if storage.exists(test_path):
            obs_file = test_path
            break

    if obs_file is None:
        available_files = storage.list_nc(obs_dir)
        raise FileNotFoundError(
            f"No observation file found for region '{region}' year {year} in {obs_dir}. "
            f"Tried patterns: {file_patterns}. Available files: {available_files[:5] if available_files else '[]'}"
        )

    ds = storage.open_dataset(obs_file)
    ds = dim_fmt(ds)
    ds = region_select(ds, region=region, **kwargs_clean)

    rainfall = ds[obs_var]
    if obs_unit_cvt:
        rainfall *= obs_unit_cvt

    if grid_point:
        rainfall = rainfall.sel(lat=[lat_select], lon=[lon_select])

    return rainfall


def get_forecast_deterministic_twice_weekly(year=None, *, model_dir=None, model_var=None, date_filter_year=None, init_days=None, 
                                            start_date=None, end_date=None, unit_cvt=None, file_pattern=None, region=None, **kwargs):
    kwargs_clean = kwargs.copy()
    kw_year = kwargs_clean.pop("year", None)
    year = year if year is not None else kw_year

    if not model_dir:
        model_dir = kwargs_clean.get("model_dir") or kwargs_clean.get("model_dir_list") or getattr(params, "model_dir_list", None)
    if isinstance(model_dir, (list, tuple)) and len(model_dir) > 0:
        model_dir = model_dir[0]

    if not file_pattern:
        file_pattern = kwargs_clean.get("file_pattern") or kwargs_clean.get("file_pattern_list") or getattr(params, "file_pattern_list", ("{}.nc",))
    if isinstance(file_pattern, (list, tuple)) and len(file_pattern) > 0:
        file_pattern = file_pattern[0]

    if not model_var:
        model_var = kwargs_clean.get("model_var") or kwargs_clean.get("model_var_list") or getattr(params, "model_var_list", "tp")
    if isinstance(model_var, (list, tuple)) and len(model_var) > 0:
        model_var = model_var[0]

    if region is None:
        region = kwargs_clean.get("region") or getattr(params, "region", "Ethiopia")

    kwargs_clean = restore_args(get_forecast_deterministic_twice_weekly, kwargs_clean, locals())
    kwargs_clean.pop("year", None)
    kwargs_clean.pop("region", None)

    if model_dir and config.is_uri(model_dir):
        model_dir = f"gs://{config._strip(model_dir)}"

    fname = file_pattern.format(year) if "{}" in str(file_pattern) else f"{year}.nc"
    file_path = storage.join(model_dir, fname)

    if not storage.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    filtered_dates_yr = get_initialization_dates(year, **kwargs_clean)

    ds = storage.open_dataset(file_path)
    ds = dim_fmt_model(ds)

    available_init_times = pd.to_datetime(ds.init_time.values)
    matching_times = available_init_times[available_init_times.isin(filtered_dates_yr)]
    if len(matching_times) == 0:
        raise ValueError(f"No matching initialization times found for year {year} in {file_path}")
    ds = ds.sel(init_time=matching_times)

    ds = region_select(ds, region=region, **kwargs_clean)

    # Detect or fallback 'step' dimension name
    if 'step' not in ds.dims and 'step' not in ds.coords:
        unrecognized_dims = [d for d in ds.dims if d not in ('init_time', 'lat', 'lon', 'member')]
        if unrecognized_dims:
            ds = ds.rename({unrecognized_dims[0]: 'step'})

    if 'step' in ds.dims or 'step' in ds.coords:
        if ds['step'][0].values == 0:
            ds = ds.sel(step=slice(1, None))
    else:
        raise KeyError(f"'step' dimension not found in dataset. Active dimensions: {list(ds.dims)}")

    p_model = ds[model_var]

    if unit_cvt:
        p_model *= unit_cvt

    ds.close()
    return p_model


def get_forecast_probabilistic_twice_weekly(year=None, *, model_dir=None, model_var=None, date_filter_year=None, init_days=None,
                                            start_date=None, end_date=None, unit_cvt=None, file_pattern=None,
                                            members=None, region=None, **kwargs):
    kwargs_clean = kwargs.copy()
    kw_year = kwargs_clean.pop("year", None)
    year = year if year is not None else kw_year

    if not model_dir:
        model_dir = kwargs_clean.get("model_dir") or kwargs_clean.get("model_dir_list") or getattr(params, "model_dir_list", None)
    if isinstance(model_dir, (list, tuple)) and len(model_dir) > 0:
        model_dir = model_dir[0]

    if not file_pattern:
        file_pattern = kwargs_clean.get("file_pattern") or kwargs_clean.get("file_pattern_list") or getattr(params, "file_pattern_list", ("{}.nc",))
    if isinstance(file_pattern, (list, tuple)) and len(file_pattern) > 0:
        file_pattern = file_pattern[0]

    if not model_var:
        model_var = kwargs_clean.get("model_var") or kwargs_clean.get("model_var_list") or getattr(params, "model_var_list", "tp")
    if isinstance(model_var, (list, tuple)) and len(model_var) > 0:
        model_var = model_var[0]

    if region is None:
        region = kwargs_clean.get("region") or getattr(params, "region", "Ethiopia")

    kwargs_clean = restore_args(get_forecast_probabilistic_twice_weekly, kwargs_clean, locals())
    kwargs_clean.pop("year", None)
    kwargs_clean.pop("region", None)

    if model_dir and config.is_uri(model_dir):
        model_dir = f"gs://{config._strip(model_dir)}"

    fname = file_pattern.format(year) if "{}" in str(file_pattern) else f"{year}.nc"
    file_path = storage.join(model_dir, fname)

    if not storage.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    filtered_dates_yr = get_initialization_dates(year, **kwargs_clean)

    ds = storage.open_dataset(file_path)
    ds.load()
    ds = dim_fmt_model_ensemble(ds)

    available_init_times = pd.to_datetime(ds.init_time.values)
    matching_times = available_init_times[available_init_times.isin(filtered_dates_yr)]

    if len(matching_times) == 0:
        raise ValueError(f"No matching initialization times found for year {year} in {file_path}")

    ds = ds.sel(init_time=matching_times)

    if 'step' not in ds.dims and 'step' not in ds.coords:
        unrecognized_dims = [d for d in ds.dims if d not in ('init_time', 'lat', 'lon', 'member')]
        if unrecognized_dims:
            ds = ds.rename({unrecognized_dims[0]: 'step'})

    if 'step' in ds.dims or 'step' in ds.coords:
        if ds['step'][0].values == 0:
            ds = ds.sel(step=slice(1, None))
    else:
        raise KeyError(f"'step' dimension not found in dataset. Active dimensions: {list(ds.dims)}")

    if isinstance(members, str) and members.strip().lower() in ("all", ""):
        members = None

    if members:
        ds = ds.sel(member=list(members))

    ds = region_select(ds, region=region, **kwargs_clean)

    p_model = ds[model_var]

    if unit_cvt:
        p_model *= unit_cvt

    ds.close()
    return p_model
