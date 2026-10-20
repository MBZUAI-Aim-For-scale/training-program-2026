from momp.io.input import load_imd_rainfall, load_thresh_file, get_initialization_dates
from momp.stats.detect import detect_observed_onset
from momp.utils.practical import restore_args
#from momp.stats.benchmark import compute_onset_metrics_with_windows

import numpy as np
import xarray as xr
import pandas as pd
from datetime import datetime, timedelta
#import os
#import glob
#from pathlib import Path
#import warnings
#from matplotlib.patches import Polygon
#from matplotlib.path import Path
#import matplotlib.patches as patches
import sys


def compute_climatological_onset(*, obs_dir, obs_file_pattern, obs_var, thresh_file, thresh_var, wet_threshold, 
                                  wet_init, wet_spell, dry_spell, dry_threshold, dry_extent, start_date, 
                                 fallback_date, mok, years_clim,
                                 grid_point=False, lat_select=None, lon_select=None, **kwargs):
    """
    Compute climatological onset dates from all available IMD files.
    
    Parameters:
    obs_dir: str, folder containing IMD NetCDF files
    thresh_file: str, path to threshold file
    mok: bool, if True use June 2nd as start date (MOK), if False use May 1st
    
    Returns:
    climatological_onset_doy: xarray DataArray with climatological onset day of year
    """
    
    kwargs = restore_args(compute_climatological_onset, kwargs, locals())

    thresh_da = load_thresh_file(**kwargs)
    
    print(f"Computing climatological onset from {len(years_clim)} years_clim: {min(years_clim)}-{max(years_clim)}")
    
    all_onset_days = []
    rainfall_all_years = []
    
    for year in years_clim:       
        try:
            # Load rainfall data using the existing function that handles both patterns
            if grid_point:
                if np.ndim(thresh_da) > 0:
                    raise ValueError("thresh_da must be scalar.")

                rainfall_ds = load_imd_rainfall(year, **kwargs)
                #rainfall_ds = load_imd_rainfall(year, grid_point=grid_point, 
                #                                lat_select=lat_select, lon_select=lon_select, **kwargs)

                rainfall_all_years.append(rainfall_ds)
            
            else:
                rainfall_ds = load_imd_rainfall(year, **kwargs)


            # Detect onset for this year
            onset_da = detect_observed_onset(rainfall_ds, thresh_da, year, **kwargs)
            #print("\n\n\n year = ", year)
            #print("onset_da = ", onset_da)
            #print("onset_da = ", onset_da.values)
            #print("\n\n\nYYYYYY")
            
            # Convert onset dates to day of year
            onset_doy = onset_da.dt.dayofyear.astype(float)
            onset_doy = onset_doy.where(~onset_da.isnull())
            
            all_onset_days.append(onset_doy)
            
        except Exception as e:
            print(f"Warning: Could not process year {year} when compute_climagoloical_onset: {e}")
            raise
            continue
    
    if not all_onset_days:
        raise ValueError("No valid years found for climatology computation")
    
    # Stack all years and compute mean day of year
    onset_stack = xr.concat(all_onset_days, dim='year')
    climatological_onset_doy = onset_stack.mean(dim='year')
    
    # Round to nearest integer day
    climatological_onset_doy = np.round(climatological_onset_doy)
    
    print(f"Climatological onset computed from {len(all_onset_days)} valid years")

    if grid_point:
        #print("\n\n rainfall_all_years = ", rainfall_all_years)
        rainfall_stack = xr.concat(rainfall_all_years, dim='year')
        #rainfall_clim_mean = rainfall_stack.mean(dim='year') # doesn't work as time dim contain unique timestamp

        # create day-of-year coordinate (1-365)
        rainfall_stack = rainfall_stack.sel(time=~((rainfall_stack['time'].dt.is_leap_year) &
                                           (rainfall_stack['time'].dt.dayofyear == 366)))
        
        # Create day-of-year index (just integer 1-365)
        doy = rainfall_stack['time'].dt.dayofyear
        rainfall_stack = rainfall_stack.assign_coords(doy=doy)
        
        # Aggregate over years using groupby
        # explicitly reduce the 'time' dimension by taking mean
        rainfall_clim_mean = (
            rainfall_stack.groupby('doy')
            .mean(dim='time')  # <- reduce the concatenated time dimension
            .mean(dim='year')  # <- average across years
        )

        return climatological_onset_doy, rainfall_clim_mean
    
    return climatological_onset_doy


def compute_climatology_as_forecast(climatological_onset_doy, year, init_dates, observed_onset_da,
                                   *, max_forecast_day, mok, **kwargs):
    """
    Use climatology as a forecast model for the given initialization dates.
    Only processes forecasts initialized before the observed onset date.
    
    Parameters:
    climatological_onset_doy: xarray DataArray with climatological onset day of year
    year: int, year to evaluate
    init_dates: pandas DatetimeIndex with initialization dates
    observed_onset_da: xarray DataArray with observed onset dates for filtering
    max_forecast_day: int, maximum forecast day to consider
    mok: bool, if True only count onset after June 2nd (MOK date)
    
    Returns:
    pandas DataFrame with climatology forecast results
    """
    
    #mok = kwargs['mok']

    results_list = []
    
    # Get dimensions
    lats = climatological_onset_doy.lat.values
    lons = climatological_onset_doy.lon.values

    #end_MMDD = kwargs["end_date"][1:]
    
    print(f"Processing climatology as forecast for {len(init_dates)} init times x {len(lats)} lats x {len(lons)} lons...")
    print(f"Year: {year}")
    #print(f"Only processing forecasts initialized before observed onset dates")
    
    # Track statistics
    total_potential_inits = 0
    valid_inits = 0
    skipped_no_obs = 0
    skipped_late_init = 0
    onsets_forecasted = 0
    
    # Loop over all initialization dates and grid points
    for t_idx, init_time in enumerate(init_dates):
        #if t_idx % 5 == 0:  # Print progress every 5 init times
        #    print(f"Processing init time {t_idx+1}/{len(init_dates)}: {init_time.strftime('%Y-%m-%d')}")
        
        init_date = pd.to_datetime(init_time)
        year = init_date.year
        #end_date = datetime(year, *end_MMDD)
#        print(f"init_time {init_time}, {type(init_time)}")
#        print(f"init_date {init_date}, {type(init_date)}")
#        print(f"init_dates {init_dates}, {type(init_dates)}")
#        sys.exit()

        if mok:
            mok_date = datetime(year, *mok)  # June 2nd of the same year
        
        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                
                total_potential_inits += 1
#                print(f"available init {init_date}") if lat==11.75 and lon==40.5 else None
                
                # Get observed onset date for this grid point
                try:
                    obs_onset = observed_onset_da.isel(lat=i, lon=j).values
                except:
                    skipped_no_obs += 1
                    continue
                
#                print("1111111111") if lat==11.75 and lon==40.5 else None
                #if pd.to_datetime(obs_onset) > end_date:
                #    skipped_no_obs += 1
                #    continue
#                print("2222222") if lat==11.75 and lon==40.5 else None

                # Skip if no observed onset
                if pd.isna(obs_onset):
                    skipped_no_obs += 1
                    continue
#                print("3333333") if lat==11.75 and lon==40.5 else None
                
                # Convert observed onset to datetime
                obs_onset_dt = pd.to_datetime(obs_onset)
                
                # Only process if forecast was initialized before observed onset
                if init_date >= obs_onset_dt:
                    skipped_late_init += 1
                    continue
 #               print("4444444") if lat==11.75 and lon==40.5 else None
                
                valid_inits += 1
                
                # Get climatological onset day of year for this grid point
                clim_onset_doy = climatological_onset_doy.isel(lat=i, lon=j).values
                
                # Skip if no climatological onset available
                # Bug fix from original code which directly skip/continue if no clim onset
                if np.isnan(clim_onset_doy):
                    onset_day, onset_date = None, None
                    result = {
                        'init_time': init_time,
                        'lat': lat,
                        'lon': lon,
                        'onset_day': onset_day,  # None if no onset forecasted
                        'onset_date': onset_date.strftime('%Y-%m-%d') if onset_date is not None else None,
                        'climatological_onset_doy': clim_onset_doy,
                        'climatological_onset_date': clim_onset_date.strftime('%Y-%m-%d'),
                        'obs_onset_date': obs_onset_dt.strftime('%Y-%m-%d')  # Store observed onset for reference
                    }
                    results_list.append(result)
                    continue
                
#                print(f"5555 {clim_onset_doy}") if lat==11.75 and lon==40.5 else None
                # Convert climatological day of year to actual date for this year
                try:
                    clim_onset_date = datetime(year, 1, 1) + timedelta(days=int(clim_onset_doy) - 1)
                    clim_onset_date = pd.to_datetime(clim_onset_date)
                except:
                    continue  # Skip if invalid day of year
                
                # Check if climatological onset is within forecast window
                forecast_window_start = init_date + pd.Timedelta(days=1)
                forecast_window_end = init_date + pd.Timedelta(days=max_forecast_day)
                
                onset_day = None
                onset_date = None
                
#                print(f"XXXxx clim_onset_date {clim_onset_date}") if lat==11.75 and lon==40.5 else None
#                print(f"init {init_date}") if lat==11.75 and lon==40.5 else None
#                print(f"{forecast_window_start} {forecast_window_end}") if lat==11.75 and lon==40.5 else None
                if forecast_window_start <= clim_onset_date <= forecast_window_end:
                    # Climatological onset is within forecast window
                    onset_day = (clim_onset_date - init_date).days
                    
#                    print(f"XXXxx onset_day {onset_day}") if lat==11.75 and lon==40.5 else None
                    # Apply MOK filtering if requested
                    if mok:
                        if clim_onset_date.date() >= mok_date.date():
                            # Valid onset after MOK date
                            onset_date = clim_onset_date
                            onsets_forecasted += 1
#                            print(f"PPPPP  onset_day {onset_day}") if lat==11.75 and lon==40.5 else None
                        else:
                            # Reset if before MOK date
                            onset_day = None
                            onset_date = None
#                            print(f"YYYYY onset_day {onset_day}") if lat==11.75 and lon==40.5 else None
                    else:
                        # No MOK filtering
                        onset_date = clim_onset_date
                        onsets_forecasted += 1
#                        print(f"ZZZZZ onset_day {onset_day}") if lat==11.75 and lon==40.5 else None
                
                # Store result
                result = {
                    'init_time': init_time,
                    'lat': lat,
                    'lon': lon,
                    'onset_day': onset_day,  # None if no onset forecasted
                    'onset_date': onset_date.strftime('%Y-%m-%d') if onset_date is not None else None,
                    'climatological_onset_doy': clim_onset_doy,
                    'climatological_onset_date': clim_onset_date.strftime('%Y-%m-%d'),
                    'obs_onset_date': obs_onset_dt.strftime('%Y-%m-%d')  # Store observed onset for reference
                }
                results_list.append(result)
    
    # Convert to DataFrame
    climatology_forecast_df = pd.DataFrame(results_list)
    
    print(f"\nClimatology Forecast Summary:")
    print(f"Total potential initializations: {total_potential_inits}")
    print(f"Skipped (no observed onset): {skipped_no_obs}")
    print(f"Skipped (initialized after observed onset): {skipped_late_init}")
    print(f"Valid initializations processed: {valid_inits}")
    print(f"Onsets forecasted: {onsets_forecasted}")
    print(f"Forecast rate: {onsets_forecasted/valid_inits:.3f}" if valid_inits > 0 else "Forecast rate: 0.000")
    
    if mok:
        print(f"Note: Only onsets on or after June 2nd were counted due to MOK flag")
    
    return climatology_forecast_df

# def compute_climatology_metrics_with_windows same as stats.benchmark.compute_onset_metrics_with_windows
# def compute_climatology_baseline_multiple_years same as stats.benchmark.compute_metrics_multiple_years


###=========  for bin climatology ============

## This function computes onset dates for all available years in IMD folder and creates a climatological onset dataset
def compute_climatological_onset_dataset(*, obs_dir, obs_file_pattern, obs_var, thresh_file, thresh_var, wet_threshold, 
                                         wet_init, wet_spell, dry_spell, dry_threshold, dry_extent, start_date,
                                         fallback_date, mok, years_clim, **kwargs):
    """
    Compute onset dates for all available years in IMD folder and create a climatological dataset.

    Parameters:
    -----------
    obs_dir : str
        Folder containing IMD NetCDF files
    thresh_slice : xarray.DataArray
        Rainfall threshold for each grid point
    years_clim : list, optional
        Specific years to process. If None, will auto-detect available years
    mok : bool, default=True
        Whether to use MOK date filter (June 2nd)

    Returns:
    --------
    xarray.DataArray
        3D array with dimensions [year, lat, lon] containing onset dates
    """

    kwargs = restore_args(compute_climatological_onset_dataset, kwargs, locals())

    #years = kwargs['years']
    #years = years_clim

    thresh_slice = load_thresh_file(**kwargs)

    print(f"Computing climatological onset from {len(years_clim)} years_clim: {min(years_clim)}-{max(years_clim)}")

    # Initialize lists to store results
    onset_arrays = []
    valid_years = []
#    all_onset_days = []

    # Process each year
    for year in years_clim:
        print(f"\nProcessing year {year}...")

        try:
            # Load rainfall data for this year
            rainfall_ds = load_imd_rainfall(year, **kwargs)

            # Select the same spatial domain as thresh_slice
            rainfall_slice = rainfall_ds
            # Detect onset for this year
            onset_da = detect_observed_onset(rainfall_slice, thresh_slice, year, **kwargs)

            # Count valid onsets
            valid_onsets = (~pd.isna(onset_da.values)).sum()
            total_points = onset_da.size

            print(f"Year {year}: Found onset in {valid_onsets}/{total_points} grid points ({valid_onsets/total_points:.1%})")

            # Store the onset array
            onset_arrays.append(onset_da.values)
            valid_years.append(year)

        except Exception as e:
            print(f"Error processing year {year}: {e}")
            continue

    if not onset_arrays:
        raise ValueError("No years were successfully processed")

    # Stack all onset arrays into a 3D array
    onset_3d = np.stack(onset_arrays, axis=0)

#    print("\n onset_3d = ", onset_3d)
#    print("\n thresh_slice = ", thresh_slice)
#    print("\n onset_da = ", onset_da)
    # Create the final DataArray
    climatological_onset_da = xr.DataArray(
        onset_3d,
        coords=[
            ('year', valid_years),
            ('lat', rainfall_ds.lat.values),
            ('lon', rainfall_ds.lon.values)
        ],
        name='climatological_onset_dates',
        attrs={
            'description': 'Onset dates for climatological ensemble',
            'method': 'MOK {mok} filter' if mok else 'no date filter',
            'years_processed': valid_years,
            'total_years': len(valid_years)
        }
    )

    # Print summary statistics
    total_possible = len(valid_years) * rainfall_ds[0].size
    total_valid = (~pd.isna(climatological_onset_da.values)).sum()

    print(f"\n{'='*60}")
    print(f"CLIMATOLOGICAL ONSET DATASET SUMMARY")
    print(f"{'='*60}")
    print(f"Years processed: {len(valid_years)} ({min(valid_years)}-{max(valid_years)})")
    print(f"Spatial domain: {len(rainfall_ds.lat)} lats x {len(rainfall_ds.lon)} lons")
    print(f"Total valid onsets: {total_valid:,}/{total_possible:,} ({total_valid/total_possible:.1%})")
    print(f"Method: {'MOK ({mok} filter)' if mok else 'No date filter'}")

    # Show onset statistics by year
    #print(f"\nOnset statistics by year:")
    #for i, year in enumerate(valid_years):
    #    year_onsets = (~pd.isna(climatological_onset_da.isel(year=i).values)).sum()
    #    print(f"  {year}: {year_onsets}/{rainfall_ds[0].size} ({year_onsets/rainfall_ds[0].size:.1%})")

    return climatological_onset_da




