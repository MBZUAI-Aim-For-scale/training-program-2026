"""
momp/io/input.py
Data loading utilities for observation and model forecast datasets.
"""

import os
import re
import sys
import tempfile
from datetime import datetime
from functools import lru_cache
from pathlib import Path
import numpy as np
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

from ai_weather.precipitation import precipitation_to_mm
from momp.utils.standard import dim_fmt, dim_fmt_model, dim_fmt_model_ensemble
from momp.utils.region import region_select
from momp.utils import storage
from momp.utils.practical import restore_args
import momp.params as params


class MissingEnsembleDimensionError(ValueError):
    """Raised when a probabilistic forecast file cannot provide members."""


class NoMatchingInitializationTimesError(ValueError):
    """Raised when a forecast file has no initialization dates for a year."""


def _resolve_data_variable(ds, requested_var, source):
    variables = list(ds.data_vars)
    if requested_var in variables:
        return requested_var

    normalized = {str(name).strip().lower(): name for name in variables}
    requested_key = str(requested_var or "").strip().lower()
    aliases = [requested_key, "rainfall", "precip", "precipitation", "pr", "tp", "total_precipitation"]
    for alias in aliases:
        if alias in normalized:
            return normalized[alias]

    if len(variables) == 1:
        return variables[0]

    raise KeyError(
        f"Could not resolve rainfall variable {requested_var!r} in {source}. "
        f"Available data variables: {variables}"
    )


def _normalize_admin_name(value):
    return re.sub(r"[^a-z0-9]", "", str(value).strip().lower())


def _download_admin_boundaries(region, configured_dir=None):
    """Stage the configured region's administrative boundaries from the demo bucket."""
    if storage.is_uri(configured_dir):
        remote_dir = str(configured_dir).rstrip("/")
    else:
        bucket = os.getenv("DEMO_BUCKET", "").strip().removeprefix("gs://").strip("/")
        if not bucket:
            return None
        remote_dir = f"gs://{bucket}/demo4/DATA/{region}/shapefile"
    remote_files = storage.glob(f"{remote_dir}/**")
    boundary_files = [
        path for path in remote_files
        if path.lower().endswith((".shp", ".geojson", ".json", ".shx", ".dbf", ".prj", ".cpg"))
    ]
    if not boundary_files:
        return None

    local_dir = Path(tempfile.gettempdir()) / "momp_admin_boundaries" / re.sub(r"[^a-zA-Z0-9_-]", "_", region)
    local_dir.mkdir(parents=True, exist_ok=True)
    fs = storage.gcs_fs()
    for remote_path in boundary_files:
        relative = remote_path[len(remote_dir):].lstrip("/")
        target = local_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            fs.get(storage._strip(remote_path), str(target))
    return local_dir


@lru_cache(maxsize=8)
def _build_admin_grid_index(admin_names, latitudes, longitudes, shapefile_dir, region="Ethiopia"):
    import geopandas as gpd

    source = Path(shapefile_dir).expanduser()
    if source.is_file():
        boundary_files = [source]
    elif source.is_dir():
        boundary_files = [
            path
            for pattern in ("*.geojson", "*.json", "*.shp")
            for path in source.rglob(pattern)
        ]
    else:
        boundary_files = []

    if not boundary_files:
        downloaded_dir = _download_admin_boundaries(region, shapefile_dir)
        if downloaded_dir is not None:
            boundary_files = [
                path
                for pattern in ("*.geojson", "*.json", "*.shp")
                for path in downloaded_dir.rglob(pattern)
            ]

    boundary_files.sort(
        key=lambda path: (
            not any(token in path.name.lower() for token in ("adm3", "admin3", "woreda")),
            path.name.lower(),
        )
    )
    if not boundary_files:
        raise FileNotFoundError(f"No administrative boundary file found in {source}")

    boundaries = gpd.read_file(boundary_files[0])
    name_field = next(
        (field for field in ("ADM3_EN", "ADM3_NAME", "NAME_3", "adm3_name", "name") if field in boundaries),
        None,
    )
    if name_field is None:
        raise ValueError(f"No ADM3 name field found in {boundary_files[0]}")
    if boundaries.crs is not None and str(boundaries.crs).upper() not in {"EPSG:4326", "OGC:CRS84"}:
        boundaries = boundaries.to_crs("EPSG:4326")

    admin_indices = {
        _normalize_admin_name(name): index
        for index, name in enumerate(admin_names)
    }
    boundaries = boundaries.copy()
    boundaries["_admin_index"] = boundaries[name_field].map(
        lambda name: admin_indices.get(_normalize_admin_name(name), -1)
    )
    boundaries = boundaries[boundaries["_admin_index"] >= 0]
    if boundaries.empty:
        raise ValueError("No forecast ADM3 names match the configured administrative boundaries")

    lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)
    points = gpd.GeoDataFrame(
        {"_grid_index": np.arange(lon_grid.size)},
        geometry=gpd.points_from_xy(lon_grid.ravel(), lat_grid.ravel()),
        crs="EPSG:4326",
    )
    try:
        joined = gpd.sjoin(
            points,
            boundaries[["_admin_index", "geometry"]],
            how="left",
            predicate="intersects",
        )
    except ImportError as exc:
        raise RuntimeError("ADM3 forecast mapping requires a GeoPandas spatial index") from exc

    joined = joined.dropna(subset=["_admin_index"]).drop_duplicates("_grid_index")
    grid_index = np.full(lon_grid.size, -1, dtype=np.int32)
    grid_index[joined["_grid_index"].to_numpy(dtype=np.int64)] = joined["_admin_index"].to_numpy(dtype=np.int32)
    if not np.any(grid_index >= 0):
        raise ValueError("No observation grid cells overlap matched ADM3 forecast areas")
    return grid_index.reshape(lon_grid.shape)


def admin_forecast_grid_index(admin_names, grid, shpfile_dir=None, region="Ethiopia"):
    if shpfile_dir is None:
        shpfile_dir = getattr(params, "shpfile_dir", None)
    if not shpfile_dir:
        raise ValueError("ADM3 forecasts require a configured shpfile_dir")
    return _build_admin_grid_index(
        tuple(str(name) for name in admin_names),
        tuple(float(value) for value in grid.lat.values),
        tuple(float(value) for value in grid.lon.values),
        str(shpfile_dir),
        str(region),
    )


def available_forecast_years(years, *, model_dir=None, file_pattern=None):
    if years is None:
        return ()
    if isinstance(model_dir, (list, tuple)):
        model_dir = model_dir[0] if model_dir else None
    if isinstance(file_pattern, (list, tuple)):
        file_pattern = file_pattern[0] if file_pattern else None
    file_pattern = str(file_pattern or "{}.nc")

    available = []
    for year in years:
        filename = file_pattern.format(year) if "{}" in file_pattern else f"{year}.nc"
        forecast_path = storage.join(model_dir, filename)
        if storage.exists(forecast_path):
            available.append(year)
        else:
            print(f"Skipping forecast year {year}: file not found at {forecast_path}")
    return tuple(available)


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
    """Select seasonal dates, preserving weekly reference-calendar anchoring."""
    kwargs_clean = kwargs.copy()
    kw_year = kwargs_clean.pop("year", None)
    year = year if year is not None else kw_year
    init_type = str(init_type).strip().lower()
    if init_type not in {"weekly", "daily", "monthly"}:
        raise ValueError("Init Type must be weekly, daily or monthly.")

    if init_type == "monthly":
        return get_initialization_dates_monthly(year, date_filter_year=date_filter_year, init_days=init_days, start_date=start_date, end_date=end_date, **kwargs_clean)

    if date_filter_year is None:
        date_filter_year = getattr(params, "date_filter_year", 2024)
    year = int(year if year is not None else date_filter_year)
    if start_date is None:
        start_date = getattr(params, "start_date", (2003, 5, 1))
    if end_date is None:
        end_date = getattr(params, "end_date", (2023, 7, 31))

    # Validate month/day using a leap year; absent dates are omitted naturally.
    datetime(2000, *start_date[1:])
    datetime(2000, *end_date[1:])
    start_md = start_date[1] * 100 + start_date[2]
    end_md = end_date[1] * 100 + end_date[2]
    calendar_year = year if init_type == "daily" else int(date_filter_year)
    dates = pd.date_range(f"{calendar_year}-01-01", f"{calendar_year}-12-31", freq="D")
    month_days = dates.month * 100 + dates.day
    in_season = ((month_days >= start_md) & (month_days <= end_md) if start_md <= end_md
                 else (month_days >= start_md) | (month_days <= end_md))
    dates = dates[in_season]

    if init_type == "weekly":
        if init_days is None:
            init_days = (getattr(params, "init_days", (0, 3))
                         if getattr(params, "init_type", "weekly") == "weekly" else (0, 3))
        try:
            days = tuple(init_days)
        except TypeError:
            raise ValueError("Initialization weekdays must be a nonempty sequence of integers from 0 to 6.") from None
        if not days or any(isinstance(day, bool) or not isinstance(day, (int, np.integer)) or not 0 <= day <= 6 for day in days):
            raise ValueError("Initialization weekdays must be integers from 0 (Monday) to 6 (Sunday), with at least one selected.")
        dates = dates[dates.weekday.isin(days)]
        # A leap-day initialization in the reference calendar may not exist in
        # the target historical year. Skip it rather than fail the whole run.
        dates = pd.to_datetime(dates.strftime(f"{year}-%m-%d"), errors="coerce").dropna()
    return pd.DatetimeIndex(dates)


def get_initialization_dates_monthly(year=None, *, date_filter_year=None, init_days=None, start_date=None, end_date=None, **kwargs):
    """Select valid month days within the configured inclusive seasonal bounds."""
    kwargs_clean = kwargs.copy()
    kw_year = kwargs_clean.pop("year", None)
    year = year if year is not None else kw_year
    year = int(year if year is not None else (date_filter_year or getattr(params, "date_filter_year", 2024)))

    if start_date is None:
        start_date = getattr(params, "start_date", (2003, 5, 1))
    if end_date is None:
        end_date = getattr(params, "end_date", (2023, 7, 31))
    if init_days is None:
        init_days = (getattr(params, "init_days", (1, 15))
                     if getattr(params, "init_type", "weekly") == "monthly" else (1, 15))
    try:
        days = tuple(init_days)
    except TypeError:
        raise ValueError("Monthly initialization days must be a nonempty sequence of integers from 1 to 31.") from None
    if not days or any(isinstance(day, bool) or not isinstance(day, (int, np.integer)) or not 1 <= day <= 31 for day in days):
        raise ValueError("Monthly initialization days must be integers from 1 to 31, with at least one selected.")
    datetime(2000, *start_date[1:])
    datetime(2000, *end_date[1:])
    start_md = start_date[1] * 100 + start_date[2]
    end_md = end_date[1] * 100 + end_date[2]
    dates = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    month_days = dates.month * 100 + dates.day
    in_season = ((month_days >= start_md) & (month_days <= end_md) if start_md <= end_md
                 else (month_days >= start_md) | (month_days <= end_md))
    return dates[in_season & dates.day.isin(days)]


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

    rainfall_var = _resolve_data_variable(ds, obs_var, obs_file)
    rainfall = ds[rainfall_var]
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
    admin_dim = next(
        (name for name in ("adm3_name", "adm2_name", "adm1_name", "woreda", "district") if name in ds.dims),
        None,
    )
    ds = dim_fmt_model(ds, preserve_admin=admin_dim is not None)

    available_init_times = pd.to_datetime(ds.init_time.values)
    matching_times = available_init_times[available_init_times.isin(filtered_dates_yr)]
    if len(matching_times) == 0:
        raise NoMatchingInitializationTimesError(
            f"No matching initialization times found for year {year} in {file_path}"
        )
    ds = ds.sel(init_time=matching_times)

    if admin_dim is None:
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

    try:
        p_model = precipitation_to_mm(p_model, unit_cvt)
    finally:
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
    admin_dim = next(
        (name for name in ("adm3_name", "adm2_name", "adm1_name", "woreda", "district") if name in ds.dims),
        None,
    )
    ds = dim_fmt_model_ensemble(ds, preserve_admin=admin_dim is not None)

    if "member" not in ds.dims:
        dimensions = tuple(ds.dims)
        ds.close()
        raise MissingEnsembleDimensionError(
            f"Probabilistic forecast file has no ensemble member dimension: {file_path} "
            f"(dimensions: {dimensions})"
        )

    available_init_times = pd.to_datetime(ds.init_time.values)
    matching_times = available_init_times[available_init_times.isin(filtered_dates_yr)]

    if len(matching_times) == 0:
        ds.close()
        raise NoMatchingInitializationTimesError(f"No matching initialization times found for year {year} in {file_path}")

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

    if ds.sizes.get("member", 0) < 2:
        member_count = ds.sizes.get("member", 0)
        ds.close()
        raise ValueError(
            f"Probabilistic forecast requires at least two ensemble members; "
            f"found {member_count} in {file_path}"
        )

    if admin_dim is None:
        ds = region_select(ds, region=region, **kwargs_clean)

    p_model = ds[model_var]

    try:
        p_model = precipitation_to_mm(p_model, unit_cvt)
    finally:
        ds.close()
    return p_model
