"""
momp/stats/climatology.py
Climatological monsoon rainfall onset computation and baseline forecast evaluation.
"""

from datetime import datetime, timedelta
import sys
import numpy as np
import pandas as pd
import xarray as xr

from momp.io.input import get_initialization_dates, load_imd_rainfall, load_thresh_file
from momp.stats.detect import detect_observed_onset
from momp.utils.practical import restore_args
import momp.params as params


def compute_climatological_onset(
    *,
    obs_dir=None,
    obs_file_pattern=None,
    obs_var=None,
    thresh_file=None,
    thresh_var=None,
    wet_threshold=None,
    wet_init=None,
    wet_spell=None,
    dry_spell=None,
    dry_threshold=None,
    dry_extent=None,
    start_date=None,
    fallback_date=None,
    mok=None,
    years_clim=None,
    grid_point=False,
    lat_select=None,
    lon_select=None,
    **kwargs,
):
    """
    Compute climatological onset dates from all available IMD files.

    Parameters:
    obs_dir: str, folder containing IMD NetCDF files
    thresh_file: str, path to threshold file
    mok: tuple or bool, date filter for onset
    years_clim: list of int, years to use for climatology

    Returns:
    climatological_onset_doy: xarray DataArray with climatological onset day of year
    """
    kwargs = restore_args(compute_climatological_onset, kwargs, locals())

    years_clim = kwargs.get("years_clim", years_clim) or getattr(params, "years_clim", list(range(2003, 2024)))
    grid_point = kwargs.get("grid_point", grid_point)

    thresh_da = load_thresh_file(**kwargs)

    print(f"Computing climatological onset from {len(years_clim)} years_clim: {min(years_clim)}-{max(years_clim)}")

    all_onset_days = []
    rainfall_all_years = []

    for year in years_clim:
        try:
            if grid_point:
                if np.ndim(thresh_da) > 0:
                    raise ValueError("thresh_da must be scalar.")

                rainfall_ds = load_imd_rainfall(year, **kwargs)
                rainfall_all_years.append(rainfall_ds)
            else:
                rainfall_ds = load_imd_rainfall(year, **kwargs)

            # Detect onset for this year
            onset_da = detect_observed_onset(rainfall_ds, thresh_da, year, **kwargs)

            # Convert onset dates to day of year
            if np.issubdtype(onset_da.dtype, np.datetime64):
                onset_doy = onset_da.dt.dayofyear.astype(float)
            else:
                onset_doy = onset_da.astype(float)

            onset_doy = onset_doy.where(~onset_da.isnull())
            all_onset_days.append(onset_doy)

        except Exception as e:
            print(f"Warning: Could not process year {year} in compute_climatological_onset: {e}")
            continue

    if not all_onset_days:
        raise ValueError("No valid years found for climatology computation")

    # Stack all years and compute mean day of year
    onset_stack = xr.concat(all_onset_days, dim="year")
    climatological_onset_doy = onset_stack.mean(dim="year", skipna=True)

    # Round to nearest integer day
    climatological_onset_doy = np.round(climatological_onset_doy)

    print(f"Climatological onset computed from {len(all_onset_days)} valid years")

    if grid_point:
        rainfall_stack = xr.concat(rainfall_all_years, dim="year")

        rainfall_stack = rainfall_stack.sel(
            time=~((rainfall_stack["time"].dt.is_leap_year) & (rainfall_stack["time"].dt.dayofyear == 366))
        )

        doy = rainfall_stack["time"].dt.dayofyear
        rainfall_stack = rainfall_stack.assign_coords(doy=doy)

        rainfall_clim_mean = rainfall_stack.groupby("doy").mean(dim="time").mean(dim="year")

        return climatological_onset_doy, rainfall_clim_mean

    return climatological_onset_doy


def compute_climatology_as_forecast(
    climatological_onset_doy,
    year,
    init_dates,
    observed_onset_da,
    *,
    max_forecast_day=None,
    mok=None,
    **kwargs,
):
    """
    Use climatology as a forecast model for the given initialization dates.
    Only processes forecasts initialized before the observed onset date.
    """
    if isinstance(climatological_onset_doy, tuple):
        climatological_onset_doy = climatological_onset_doy[0]

    kwargs = restore_args(compute_climatology_as_forecast, kwargs, locals())

    max_forecast_day = kwargs.get("max_forecast_day", max_forecast_day) or getattr(params, "max_forecast_day", 45)
    mok = kwargs.get("mok", mok) or getattr(params, "mok", None)

    results_list = []

    lats = climatological_onset_doy.lat.values
    lons = climatological_onset_doy.lon.values

    print(f"Processing climatology as forecast for {len(init_dates)} init times x {len(lats)} lats x {len(lons)} lons...")
    print(f"Year: {year}")

    total_potential_inits = 0
    valid_inits = 0
    skipped_no_obs = 0
    skipped_late_init = 0
    onsets_forecasted = 0

    for t_idx, init_time in enumerate(init_dates):
        init_date = pd.to_datetime(init_time)
        init_year = init_date.year

        if mok:
            mok_date = datetime(init_year, *mok)

        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                total_potential_inits += 1

                try:
                    obs_onset = observed_onset_da.isel(lat=i, lon=j).values
                except Exception:
                    skipped_no_obs += 1
                    continue

                if pd.isna(obs_onset):
                    skipped_no_obs += 1
                    continue

                obs_onset_dt = pd.to_datetime(obs_onset)

                if init_date >= obs_onset_dt:
                    skipped_late_init += 1
                    continue

                valid_inits += 1

                clim_onset_doy = climatological_onset_doy.isel(lat=i, lon=j).values

                if np.isnan(clim_onset_doy):
                    onset_day, onset_date = None, None
                    result = {
                        "init_time": init_time,
                        "lat": lat,
                        "lon": lon,
                        "onset_day": onset_day,
                        "onset_date": None,
                        "climatological_onset_doy": clim_onset_doy,
                        "climatological_onset_date": None,
                        "obs_onset_date": obs_onset_dt.strftime("%Y-%m-%d"),
                    }
                    results_list.append(result)
                    continue

                try:
                    clim_onset_date = datetime(init_year, 1, 1) + timedelta(days=int(clim_onset_doy) - 1)
                    clim_onset_date = pd.to_datetime(clim_onset_date)
                except Exception:
                    continue

                forecast_window_start = init_date + pd.Timedelta(days=1)
                forecast_window_end = init_date + pd.Timedelta(days=max_forecast_day)

                onset_day = None
                onset_date = None

                if forecast_window_start <= clim_onset_date <= forecast_window_end:
                    onset_day = (clim_onset_date - init_date).days

                    if mok:
                        if clim_onset_date.date() >= mok_date.date():
                            onset_date = clim_onset_date
                            onsets_forecasted += 1
                        else:
                            onset_day = None
                            onset_date = None
                    else:
                        onset_date = clim_onset_date
                        onsets_forecasted += 1

                result = {
                    "init_time": init_time,
                    "lat": lat,
                    "lon": lon,
                    "onset_day": onset_day,
                    "onset_date": onset_date.strftime("%Y-%m-%d") if onset_date is not None else None,
                    "climatological_onset_doy": clim_onset_doy,
                    "climatological_onset_date": clim_onset_date.strftime("%Y-%m-%d"),
                    "obs_onset_date": obs_onset_dt.strftime("%Y-%m-%d"),
                }
                results_list.append(result)

    climatology_forecast_df = pd.DataFrame(results_list)

    print(f"\nClimatology Forecast Summary:")
    print(f"Total potential initializations: {total_potential_inits}")
    print(f"Skipped (no observed onset): {skipped_no_obs}")
    print(f"Skipped (initialized after observed onset): {skipped_late_init}")
    print(f"Valid initializations processed: {valid_inits}")
    print(f"Onsets forecasted: {onsets_forecasted}")
    print(f"Forecast rate: {onsets_forecasted/valid_inits:.3f}" if valid_inits > 0 else "Forecast rate: 0.000")

    return climatology_forecast_df


def compute_climatological_onset_dataset(
    *,
    obs_dir=None,
    obs_file_pattern=None,
    obs_var=None,
    thresh_file=None,
    thresh_var=None,
    wet_threshold=None,
    wet_init=None,
    wet_spell=None,
    dry_spell=None,
    dry_threshold=None,
    dry_extent=None,
    start_date=None,
    fallback_date=None,
    mok=None,
    years_clim=None,
    **kwargs,
):
    """
    Compute onset dates for all available years in IMD folder and create a climatological dataset.
    """
    kwargs = restore_args(compute_climatological_onset_dataset, kwargs, locals())

    years_clim = kwargs.get("years_clim", years_clim) or getattr(params, "years_clim", list(range(2003, 2024)))
    mok = kwargs.get("mok", mok) or getattr(params, "mok", None)

    thresh_slice = load_thresh_file(**kwargs)

    print(f"Computing climatological onset from {len(years_clim)} years_clim: {min(years_clim)}-{max(years_clim)}")

    onset_arrays = []
    valid_years = []
    last_ds = None

    for year in years_clim:
        print(f"\nProcessing year {year}...")

        try:
            rainfall_ds = load_imd_rainfall(year, **kwargs)
            last_ds = rainfall_ds

            onset_da = detect_observed_onset(rainfall_ds, thresh_slice, year, **kwargs)

            valid_onsets = (~pd.isna(onset_da.values)).sum()
            total_points = onset_da.size

            print(f"Year {year}: Found onset in {valid_onsets}/{total_points} grid points ({valid_onsets/total_points:.1%})")

            onset_arrays.append(onset_da.values)
            valid_years.append(year)

        except Exception as e:
            print(f"Error processing year {year}: {e}")
            continue

    if not onset_arrays:
        raise ValueError("No years were successfully processed")

    onset_3d = np.stack(onset_arrays, axis=0)

    climatological_onset_da = xr.DataArray(
        onset_3d,
        coords=[
            ("year", valid_years),
            ("lat", last_ds.lat.values),
            ("lon", last_ds.lon.values),
        ],
        name="climatological_onset_dates",
        attrs={
            "description": "Onset dates for climatological ensemble",
            "method": f"MOK {mok} filter" if mok else "no date filter",
            "years_processed": valid_years,
            "total_years": len(valid_years),
        },
    )

    total_possible = len(valid_years) * last_ds[0].size
    total_valid = (~pd.isna(climatological_onset_da.values)).sum()

    print(f"\n{'='*60}")
    print("CLIMATOLOGICAL ONSET DATASET SUMMARY")
    print(f"{'='*60}")
    print(f"Years processed: {len(valid_years)} ({min(valid_years)}-{max(valid_years)})")
    print(f"Spatial domain: {len(last_ds.lat)} lats x {len(last_ds.lon)} lons")
    print(f"Total valid onsets: {total_valid:,}/{total_possible:,} ({total_valid/total_possible:.1%})")
    print(f"Method: {'MOK filter' if mok else 'No date filter'}")

    return climatological_onset_da
