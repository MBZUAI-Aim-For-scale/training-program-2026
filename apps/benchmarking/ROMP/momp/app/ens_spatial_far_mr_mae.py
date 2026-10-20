import os
import xarray as xr
from dataclasses import asdict
from itertools import product
import copy

from momp.metrics.error import create_spatial_far_mr_mae
from momp.stats.benchmark import compute_metrics_multiple_years
from momp.lib.control import iter_list, make_case
from momp.lib.convention import Case
#from momp.lib.loader import cfg,setting
from momp.lib.loader import get_cfg, get_setting
from momp.graphics.maps import plot_spatial_metrics
from momp.graphics.onset_map import plot_spatial_climatology_onset
from momp.graphics.panel_portrait_error import panel_portrait_mae_far_mr
from momp.io.output import save_metrics_to_netcdf
#from momp.io.output import file_path
from momp.io.output import set_nested
from momp.utils.printing import tuple_to_str_range
from momp.stats.benchmark import compute_onset_metrics_with_windows
from momp.utils.practical import restore_args
from momp.io.input import load_thresh_file, get_initialization_dates
from momp.io.input import get_forecast_probabilistic_twice_weekly, get_forecast_deterministic_twice_weekly
from momp.io.input import load_imd_rainfall
from momp.stats.detect import detect_observed_onset, compute_onset_for_deterministic_model, compute_onset_for_all_members
from momp.stats.climatology import compute_climatological_onset, compute_climatology_as_forecast
from momp.utils.practical import restore_args
from momp.utils.printing import tuple_to_str_range

from momp.stats.benchmark import ens_compute_metrics_multiple_years
from momp.stats.parallel import parallel_metrics_multiple_years

cfg, setting = get_cfg(), get_setting()


#def ens_compute_metrics_multiple_years(*, obs_dir, obs_file_pattern, obs_var, 
#                                   thresh_file, thresh_var, wet_threshold, 
#                                   wet_init, wet_spell, dry_spell, dry_threshold, dry_extent, 
#                                   start_date, end_date, fallback_date, mok, years, years_clim,
#                                   model_dir, model_var, ref_model, date_filter_year, init_days, 
#                                   unit_cvt, file_pattern, tolerance_days, verification_window, max_forecast_day, 
#                                   members,  onset_percentage_threshold, probabilistic, save_nc_climatology, **kwargs):
#
#    """Compute onset metrics for multiple years."""
#
#    kwargs = restore_args(ens_compute_metrics_multiple_years, kwargs, locals())
#
#    metrics_df_dict = {}
#    onset_da_dict = {}
#
#    # load onset precip threshold, 2-D or scalar
#    thresh_da = load_thresh_file(**kwargs)
#
#    #ref_clim = (ref_model == 'climatology')
#    ref_clim = (kwargs['model'] == 'climatology')
#    #print("\n ############ ref_clim = ", ref_clim)
#
#    # calculate obs climatology onset 
#    if ref_clim:
#        climatological_onset_doy = compute_climatological_onset(**kwargs)
#
#        # climatological onset as "obs" for climatology baseline metrics
#        onset_da_dict = {year: climatological_onset_doy for year in years}
#        #onset_da_dict = {year: climatological_onset_doy for year in years_clim}
#
#        #for year in years_clim:
#        #    print(f"\n{'-'*50}")
#        #    print(f"Processing year {year}")
#
#        #    # obs onset
#        #    imd = load_imd_rainfall(year, **kwargs)
#        #    onset_da = detect_observed_onset(imd, thresh_da, year, **kwargs)
#
#        #    # climatology as forecast
#        #    print("Computing onset for climatology as forecast ...")
#        #    init_dates = get_initialization_dates(year, **kwargs)
#        #    onset_df = compute_climatology_as_forecast(
#        #        climatological_onset_doy, year, init_dates, onset_da, **kwargs)
#
#        #    print("Computing onset-obs metrics TP,TN,FP,FN for all locations and init times ...")
#        #    metrics_df, summary_stats = compute_onset_metrics_with_windows(
#        #        onset_df, **kwargs
#        #    )
#
#        #    metrics_df_dict[year] = metrics_df
#
#
#        #return metrics_df_dict, onset_da_dict
#
#
#    for year in years:
#        print(f"{'-'*50}")
#        print(f"Processing year {year}")
#        #print(f"{'='*50}")
#
#        # obs onset
#        imd = load_imd_rainfall(year, **kwargs)
#        onset_da = detect_observed_onset(imd, thresh_da, year, **kwargs)
#
#        # extract forecast at approporiate init dates
#        if probabilistic and not ref_clim:
#            print("-------Extracting ensemble forecast data ...")
#            p_model = get_forecast_probabilistic_twice_weekly(year, **kwargs)
#            print("...")
#        elif not ref_clim:
#            print("-------Extracting model forecast data ...")
#            p_model = get_forecast_deterministic_twice_weekly(year, **kwargs)
#            print("...")
#
#
#        # detect onset dates
#        if probabilistic and not ref_clim: # emsemble forecast onset
#            print("-------Computing onset for ensemble forecast ...")
#            _, onset_df = compute_onset_for_all_members(
#                p_model, thresh_da, onset_da, **kwargs
#            )
#
#        elif not ref_clim: # deterministic model onset
#            print("-------Computing onset for model forecast ...")
#            onset_df = compute_onset_for_deterministic_model(
#                p_model, thresh_da, onset_da, **kwargs
#            )
#
#        elif ref_clim: # climatology as forecast
#            print("-------Computing onset for climatology as forecast ...")
#            init_dates = get_initialization_dates(year, **kwargs)
#            onset_df = compute_climatology_as_forecast(
#                climatological_onset_doy, year, init_dates, onset_da,
#                **kwargs
#            )
#
#
#        # onset-obs metrics TP,TN,FP,FN for all locations and init times
#        print("-------Computing onset-obs metrics TP,TN,FP,FN for all locations and init times ...")
#        metrics_df, summary_stats = compute_onset_metrics_with_windows(
#            onset_df, **kwargs
#        )
#
#        metrics_df_dict[year] = metrics_df
#
#        if not ref_clim:
#            onset_da_dict[year] = onset_da
#
#
#        print(f"Year {year} completed. Grid points processed: {len(metrics_df)}")
#        print(f"Summary stats: TP={summary_stats['overall_true_positive']}, "
#              f"FP={summary_stats['overall_false_positive']}, "
#              f"FN={summary_stats['overall_false_negative']}, "
#              f"TN={summary_stats['overall_true_negative']}")
#
#
#    return metrics_df_dict, onset_da_dict




def ens_spatial_far_mr_mae_map(cfg=cfg, setting=setting):

    if not cfg.probabilistic:
        return

    results = {}

    layout_pool = iter_list(vars(cfg))

    for combi in product(*layout_pool):
        case = make_case(Case, combi, vars(cfg))

        print(f"{'='*50}")
        print(f"processing {case.model} onset evaluation for verification window \
                {case.verification_window}{case.case_name}")
        #print(f"\n model = {case.model}\n")
        #print(f" verification window = {case.verification_window}\n")

        case_cfg = {**asdict(case), **asdict(setting)}

        # model-obs onset benchmarking
        if setting.parallel:
            metrics_df_dict, onset_da_dict = parallel_metrics_multiple_years(ens=True, **case_cfg)
        else:
            metrics_df_dict, onset_da_dict = ens_compute_metrics_multiple_years(**case_cfg)
        
        # Create spatial metrics
        spatial_metrics = create_spatial_far_mr_mae(metrics_df_dict, onset_da_dict)
        
        # log case result into combined multi-case results dictionary
        results = set_nested(results, combi, spatial_metrics)

        # Save spatial metrics to NetCDF
        if case_cfg["save_nc_spatial_far_mr_mae"]:
        
            desc_dict = {
                    'title': 'Monsoon Onset MAE, FAR, MR Analysis',
                    'description': """Spatial maps of Mean Absolute Error, False Alarm Rate, and Miss Rate 
                    for monsoon onset predictions""",
            }

            save_metrics_to_netcdf(spatial_metrics, case_cfg, desc_dict=desc_dict)


        # make spatial metrics plot
        if case_cfg['plot_spatial_far_mr_mae']:
            plot_spatial_metrics(spatial_metrics, **case_cfg)


    # ------------------------------------------------------------------------
    # baseline metrics (climatology or user specified model)

    if not cfg.ref_model:
        return results 
        #pass

    cfg_ref = copy.copy(cfg)
    cfg_ref.model_list = (cfg.ref_model,)
    #print("cfg_ref['model_list'] = ", cfg_ref['model_list'])
    layout_pool = iter_list(vars(cfg_ref))
    #print("cfg_ref layout_pool = ", layout_pool)

    for combi in product(*layout_pool):
        case = make_case(Case, combi, vars(cfg_ref))
        print(f"processing model onset evaluation for {case.case_name}")

        case_ref = {'model_dir': case_cfg['ref_model_dir'],
                    'model_var': case_cfg['ref_model_var'],
                    'file_pattern': case_cfg['ref_model_file_pattern'],
                    'unit_cvt': case_cfg['ref_model_unit_cvt']
                    }

        case.update(case_ref)

        #if case.model == 'climatology':
        #    case.years = case.years_clim

        case_cfg_ref = {**asdict(case), **asdict(setting)}

        from pprint import pprint
        pprint(case_cfg_ref)

        # model-obs onset benchmarking
        if setting.parallel:
            metrics_df_dict, onset_da_dict = parallel_metrics_multiple_years(ens=True, **case_cfg_ref)
            #metrics_df_dict, onset_da_dict = ens_compute_metrics_multiple_years(**case_cfg_ref)
        else:
            metrics_df_dict, onset_da_dict = ens_compute_metrics_multiple_years(**case_cfg_ref)
        
        # Create spatial metrics
        spatial_metrics = create_spatial_far_mr_mae(metrics_df_dict, onset_da_dict)
        
        # log case result into combined multi-case results dictionary
        results = set_nested(results, combi, spatial_metrics)

        # Save spatial metrics to NetCDF
        if case_cfg["save_nc_spatial_far_mr_mae"]:
            desc_dict = {
                    'title': 'Monsoon Onset MAE, FAR, MR Analysis',
                    'description': """Spatial maps of Mean Absolute Error, False Alarm Rate, and Miss Rate 
                    for monsoon onset predictions""",
            }
            save_metrics_to_netcdf(spatial_metrics, case_cfg_ref, desc_dict=desc_dict)


        # make spatial metrics plot
        if case_cfg['plot_spatial_far_mr_mae']:
            plot_spatial_metrics(spatial_metrics, **case_cfg_ref)


    # save climatological onset to netcdf
    if case.model == 'climatology' and case_cfg['save_nc_climatology']:
        fout = os.path.join(case_cfg_ref['dir_out'], "climatology_onset_doy_{}.nc")
        fout = fout.format(tuple_to_str_range(case_cfg_ref['years_clim']))
        climatological_onset_doy = next(iter(onset_da_dict.values()))
        climatological_onset_doy.attrs["years"] = case.years_clim
        climatological_onset_doy.to_netcdf(fout)

    # spatial map of climatology onset day
    if case_cfg['plot_climatology_onset']:
        plot_spatial_climatology_onset(onset_da_dict, **case_cfg_ref)

#    if 2 > 1:
#        import pickle
#        fout = os.path.join(cfg['dir_out'],"combi_error_results.pkl")
#        with open(fout, "wb") as f:
#            pickle.dump(results, f)
    

    # panel portrait plot of mae, far, mr
    if case_cfg['plot_panel_heatmap_error']:
       panel_portrait_mae_far_mr(results, **case_cfg_ref) 


    #print("\n\n\n results dict = ", results)
    return results

# ------------------------------------------------------------------------------
if __name__ == "__main__":

    import time
    start = time.perf_counter()

    results = ens_spatial_far_mr_mae_map()

    end = time.perf_counter()
    print(f"\n\n\n ens_spatial metrics Execution time: {end - start:.4f} seconds\n\n\n")

#    import pickle
#    fout = os.path.join(cfg['dir_out'],"combi_error_results.pkl")
#    with open(fout, "rb") as f:
#        my_dict = pickle.load(f)
    

