#import importlib.resources
import os
from datetime import datetime
import pandas as pd
import xarray as xr
from momp.utils.standard import dim_fmt, dim_fmt_model, dim_fmt_model_ensemble
from momp.utils.region import region_select
from momp.utils import storage   # local folders or gs:// (Google bucket)
#from momp.lib.control import restore_args
from momp.utils.practical import restore_args

#def set_dir(folder):
#    """
#    set absolute directory path for a specific folder in MOMP
#    """
#    package = "MOMP"
#    base_dir = importlib.resources.files(package)
#    target_dir = (base_dir / folder).resolve()
#
#    return target_dir


def load_thresh_file(*, thresh_file, thresh_var, wet_threshold, region, **kwargs):
    if thresh_file:
        thresh_ds = storage.open_dataset(thresh_file)
        #thresh_da = thresh_ds[kwargs["thresh_var"]]
        thresh_da = thresh_ds[thresh_var]

        thresh_da = region_select(thresh_da, region=region,  **kwargs)

    #elif np.isscalar(thresh_file):
    else:
        #thresh_da = kwargs["wet_threshold"]
        thresh_da = wet_threshold

    return thresh_da


#def get_initialization_dates(year, *, date_filter_year, init_days, start_date, end_date, **kwargs):
def get_initialization_dates(year, *, date_filter_year, init_days, start_date, end_date, init_type="weekly", **kwargs):
    """
    Get initialization dates (Mondays and Thursdays from May-July) for a given year.
    """
    if init_type == "monthly":
        return get_initialization_dates_monthly(year, start_date=start_date, end_date=end_date, init_days=init_days, **kwargs)
    
    #date_filter_year = kwargs['date_filter_year']
    #init_days = kwargs["init_days"]
    #start_MMDD = kwargs["start_date"][1:]
    #end_MMDD = kwargs["end_date"][1:]

    start_MMDD = start_date[1:]
    end_MMDD = end_date[1:]

    start_date = datetime(date_filter_year, *start_MMDD)
    end_date = datetime(date_filter_year, *end_MMDD)
    date_range = pd.date_range(start_date, end_date, freq='D')

    filtered_dates = date_range[date_range.weekday.isin(init_days)]

    # Convert to the requested year
    filtered_dates_yr = pd.to_datetime(filtered_dates.strftime(f'{year}-%m-%d'))

    return filtered_dates_yr


def get_initialization_dates_monthly(year, *, date_filter_year, init_days, start_date, end_date, **kwargs):
    """
    Get initialization dates 
    Filters for the 1st, 7th, 13th, 19th, 25th of each month (approx. weekly) for a given year.
    """

    start_month = start_date[1]   # 5  (May)
    end_month   = end_date[1]     # 7  (July)
    
    filtered_dates_yr = pd.DatetimeIndex([
        pd.Timestamp(year, month, day)
        for month in range(start_month, end_month + 1)
        for day in init_days
    ])

    return filtered_dates_yr


def load_imd_rainfall(year, *, obs_dir, obs_file_pattern, obs_var, obs_unit_cvt, region, 
                      grid_point=False, lat_select=None, lon_select=None, **kwargs):
    """Load IMD daily rainfall NetCDF for a given year."""

    #file_patterns = [f"data_{year}.nc", f"{year}.nc"]
    file_patterns = [p.format(year) for p in obs_file_pattern]


    obs_file = None
    for pattern in file_patterns:
        test_path = storage.join(obs_dir, pattern)
        if storage.exists(test_path):
            obs_file = test_path
            break

    if obs_file is None:
        available_files = storage.list_nc(obs_dir)
        raise FileNotFoundError(
            f"No observation file found for year {year} in {obs_dir}. "
            f"Tried patterns: {file_patterns}. "
            f"Available files: {available_files}"
        )

    #print(f"Loading observation rainfall from: {obs_file}")

    ds = storage.open_dataset(obs_file)

    # Standardize dimension names
    ds = dim_fmt(ds)

    ds = region_select(ds, region=region, **kwargs)

    rainfall = ds[obs_var]

    if obs_unit_cvt:
        rainfall *= obs_unit_cvt

    if grid_point:
        rainfall = rainfall.sel(lat=[lat_select], lon=[lon_select])

    return rainfall


def get_forecast_deterministic_twice_weekly(year, *, model_dir, model_var, date_filter_year, init_days, 
                                            start_date, end_date, unit_cvt, file_pattern, region, **kwargs):
    """
    Loads model precip data for twice-weekly initializations from May to July.
    Filters for Mondays and Thursdays in the specified year.
    The forecast file is expected to be named as '{year}.nc' in the model_dir with
    variable "tp" being daily accumulated rainfall with dimensions (init_time, lat, lon, step).

    Parameters:
    year: int, year to load data for

    Returns:
    p_model: ndarray, precipitation data
    """

    kwargs = restore_args(get_forecast_deterministic_twice_weekly, kwargs, locals())

    fname = file_pattern.format(year)
    file_path = storage.join(model_dir, fname)

    if not storage.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Filter for twice weekly data from daily for the specified year based on 2024 Monday and Thursday dates (to match with IFS CY48R1 reforecasts)

    filtered_dates_yr = get_initialization_dates(year, **kwargs)

    # Load data using xarray
    ds = storage.open_dataset(file_path)

    ds = dim_fmt_model(ds)

    #print("filtered_dates_yr = ", filtered_dates_yr )
    #print("init_time  = ", ds.init_time.values)

    #ds = ds.sel(init_time=filtered_dates_yr)

    # Find common dates between desired dates and available dates
    # redundant, len(ds.init_time) or ds.sizes['init_time'] is same as len(matching_times)
    available_init_times = pd.to_datetime(ds.init_time.values)
    matching_times = available_init_times[available_init_times.isin(filtered_dates_yr)]
    if len(matching_times) == 0:
        raise ValueError(f"No matching initialization times found for year {year}")
    ds = ds.sel(init_time=matching_times)

    ds = region_select(ds, **kwargs)

    # Check if 'step' dimension exists and conditionally slice
    if 'step' in ds.dims:
        # Check if the first value of 'step' is 0, then slice to exclude it
        if ds['step'][0].values == 0:
            ds = ds.sel(step=slice(1, None))
    else:
        raise KeyError("'step' dimension not found in dataset")

    p_model = ds[model_var]

    if unit_cvt:
        p_model *= unit_cvt

    ds.close()

    return p_model

def get_forecast_probabilistic_twice_weekly(year, *, model_dir, model_var, date_filter_year, init_days,
                                            start_date, end_date, unit_cvt, file_pattern,
                                            members, region, **kwargs):
    """
    Loads model precip data for twice-weekly initializations from May to July.
    """

    kwargs = restore_args(get_forecast_probabilistic_twice_weekly, kwargs, locals())

#    print("\n file_pattern =  ", file_pattern)
    fname = file_pattern.format(year)
    file_path = storage.join(model_dir, fname)

#    print("AAAAA", file_path)
    if not storage.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    filtered_dates_yr = get_initialization_dates(year, **kwargs)
#    print("BBBBB")

    # Load data using xarray
    ds = storage.open_dataset(file_path)

    ds.load()
#    print("\n\nYYYYY ds raw ", ds.tp[1,1,1,...].values)
#    print("XXXX", ds)
    ds = dim_fmt_model_ensemble(ds)

#    print("1. after dim_fmt_model_ensemble:", type(list(ds.data_vars.values())[0].variable._data))

#    print("\n\nYYYYY ds raw ", ds.tp[1,1,1,...].values)
#    print("\n\nYYYYY ds raw ", ds.tp[0,0,0,...].values)
#    print("YYYY", ds)
    # Find common dates between desired dates and available dates
    # redundant
    available_init_times = pd.to_datetime(ds.init_time.values)
    matching_times = available_init_times[available_init_times.isin(filtered_dates_yr)]

    if len(matching_times) == 0:
        raise ValueError(f"No matching initialization times found for year {year}")

    # Select only the matching initialization times
    ds = ds.sel(init_time=matching_times)

#    print("2. after init_time isel:", type(list(ds.data_vars.values())[0].variable._data))
#    print("\n\nYYYYY ds raw ", ds.tp[1,1,1,...].values)
#    print("ZZZZZ", ds)
#    if "total_precipitation_24hr" in ds.data_vars:
#        ds = ds.rename({"total_precipitation_24hr": "tp"}) # For the quantile-mapped variable change the var name from total_precipitation_24hr to tp
#        ds = ds[['tp']]*1000  # Convert from m to mm

    if 'step' in ds.dims:
        # Check if the first value of 'step' is 0, then slice to exclude it
        if ds['step'][0].values == 0:
            ds = ds.sel(step=slice(1, None))
    else:
        raise KeyError("'step' dimension not found in dataset")

#    print("3. after step sel:", type(list(ds.data_vars.values())[0].variable._data))
    #ds = ds.isel(member =slice(0, mem_num))  # limit to first mem_num members (0-mem_num)

    #print("\n\n\n members = ", members)
    #print(type(members))
    #print(list(members))
    # members="All" (string) means "keep every member". Without this guard
    # list("All") == ['A', 'l', 'l'] and ds.sel(member=...) raises a KeyError.
    if isinstance(members, str) and members.strip().lower() in ("all", ""):
        members = None

    if members:
        #ds = ds.isel(member = members)
        #ds = ds.isel(member = list(members) )

        ds = ds.sel(member = list(members) )

#        print("members type:", type(members), "values:", list(members))
#        member_arr = ds.member.values
#        member_idx = [int(np.where(member_arr == m)[0][0]) for m in list(members)]
#        ds = ds.isel(member=member_idx)
#        print("4. after member isel:", type(list(ds.data_vars.values())[0].variable._data))
#        print("backing type after member isel:", type(list(ds.data_vars.values())[0].variable._data))

#    else:
#        print("4. members is falsy, skipped. members =", members)

    #print("\n\nYYYYY ds raw ", ds.tp[1,1,1,...].values)
    #print("\n\n\n ds = ", ds)

#    import numpy as np
#    print("backing type before region_select:", type(list(ds.data_vars.values())[0].variable._data))

    ds = region_select(ds, **kwargs)

    #print("\n\n\n ds = ", ds)
#    print("\n\nYYYYY ds raw ", ds.tp[1,1,1,...].values)
    p_model = ds[model_var]  # in mm

    if unit_cvt:
        p_model *= unit_cvt

    #print("\n\n\n p_model = ", p_model[1,1,1,...])
    #init_times = p_model.init_time.values

    ds.close()

    return p_model#, init_times