import os
from momp.io.input import load_imd_rainfall, load_thresh_file
from momp.stats.detect import detect_observed_onset
#from momp.lib.loader import cfg,setting
from momp.lib.loader import get_cfg, get_setting
from momp.graphics.rainfall_time_series import plot_rainfall_timeseries_with_onset_and_wetspell
from momp.utils.practical import restore_args
from momp.stats.climatology import compute_climatological_onset
from momp.lib.control import years_tuple_clim
import numpy as np
import pandas as pd
import sys


def obs_onset_analysis(year, lat_select=11, lon_select=39, incl_clim=True, **kwargs):

    kwargs = restore_args(obs_onset_analysis, kwargs, locals())
    years_clim = years_tuple_clim(kwargs['start_year_clim'], kwargs['end_year_clim'])
    kwargs["years_clim"] = years_clim
    #print("\n\n years_clim = ", years_clim)
    #sys.exit()

    # load onset precip threshold, 2-D or scalar
    thresh_da = load_thresh_file(**kwargs)

    #year = years[0]
    print(f"\n{'-'*50}")
    print(f"Processing year {year}")
    #print(f"{'='*50}")

    # obs onset
    da = load_imd_rainfall(year, **kwargs)
    onset_da = detect_observed_onset(da, thresh_da, year, **kwargs)

    start_date = kwargs["start_date"][1:]
    end_date = kwargs["end_date"][1:]

    da_sub = da.where(
        (
            (da.time.dt.month > start_date[0]) |
            ((da.time.dt.month == start_date[0]) & (da.time.dt.day >= start_date[1]))
        ) &
        (
            (da.time.dt.month < end_date[0]) |
            ((da.time.dt.month == end_date[0]) & (da.time.dt.day <= end_date[1]))
        ),
        drop=True
    )

    da_pr_clim_sub = None
    climatological_onset_datetime = None

    if incl_clim:
        climatological_onset_doy, da_pr_clim = compute_climatological_onset(grid_point=True, 
                                                lat_select=lat_select, lon_select=lon_select, **kwargs)

        #climatological_onset_datetime = (np.datetime64(year, "Y")
        #                                 + (climatological_onset_doy.astype(int) - 1).astype("timedelta64[D]"))

        doy = int(climatological_onset_doy.squeeze())

        climatological_onset_datetime = (
            pd.Timestamp(year=year, month=1, day=1)
            + pd.Timedelta(days=doy - 1)
        )

        #print(climatological_onset_datetime)
        #print(climatological_onset_doy)
        #print(da_pr_clim)
        #print(da_sub.sel(lat=lat_select, lon=lon_select, method="nearest"))
        #sys.exit()

        #da_pr_clim_sub = da_pr_clim.sel(time=da_sub.time, method="nearest").squeeze()

        # get day-of-year for each timestamp
        doy = da_sub['time'].dt.dayofyear
        
        # select values from rainfall_clim_mean using day-of-year
        da_pr_clim_sub = da_pr_clim.sel(doy=doy).copy()
        
        # assign the actual time from da_sub
        da_pr_clim_sub = da_pr_clim_sub.assign_coords(time=da_sub['time']).squeeze()


#    imd = da
#    print(imd)
#    print(onset_da)

#    fully_valid_mask = imd.notnull().all(dim="time")
#    
#    fully_valid_locations = (
#        fully_valid_mask
#        .where(fully_valid_mask, drop=True)
#        .stack(points=("lat", "lon"))
#        .to_dataframe(name="valid")
#        .reset_index()[["lat", "lon"]]
#    )
#    
#    print(fully_valid_locations)


    save_path = os.path.join(kwargs["dir_fig"], f"onset_time_series_lat{lat_select}_lon{lon_select}_{year}.png") 

    #plot_onset_time_series(lat=10, lon=40,)
    plot_rainfall_timeseries_with_onset_and_wetspell(da_sub, onset_da, None, 
                                                     #lat_select=32, lon_select=72, year_select=year)
                                                     #lat_select=20, lon_select=80, year_select=year, save_path=save_path)
                                                     lat_select=lat_select, lon_select=lon_select, 
                                                     year_select=year, save_path=save_path,
                                                     incl_clim=incl_clim, pr_clim=da_pr_clim_sub,
                                                     onset_clim=climatological_onset_datetime)

    return da_sub, onset_da


# ------------------------------------------------------------------------------
if __name__ == "__main__":
    from types import SimpleNamespace
    cfg, setting = get_cfg(), get_setting()
    cfg_setting = SimpleNamespace(**(vars(cfg) | vars(setting)))
    #print("\n\n years_clim = ", vars(cfg_setting)['start_year_clim'])
    #print("\n\n years_clim = ", vars(cfg_setting)['end_year_clim'])
    obs_onset_analysis(year=2020, lat_select=10, lon_select=40, incl_clim=True, **vars(cfg_setting))
    #obs_onset_analysis(year=2020, lat_select=10, lon_select=40, incl_clim=False, **vars(cfg_setting))

