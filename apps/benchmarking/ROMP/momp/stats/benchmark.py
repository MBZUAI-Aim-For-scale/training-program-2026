"""
momp/stats/benchmark.py
Core benchmarking routines for monsoon onset metrics evaluation, 
spatial grid alignment, contingency matrix generation, and multi-year scoring.
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Union, Dict, Tuple, Optional
import numpy as np
import pandas as pd
import xarray as xr

# Safe import for config and params
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

import momp.params as params
from momp.io.input import (
    MissingEnsembleDimensionError,
    NoMatchingInitializationTimesError,
    load_thresh_file,
    get_initialization_dates,
    get_forecast_probabilistic_twice_weekly,
    get_forecast_deterministic_twice_weekly,
    load_imd_rainfall,
    available_forecast_years,
)
from momp.stats.detect import (
    detect_observed_onset,
    compute_onset_for_deterministic_model,
    compute_onset_for_all_members,
)
from momp.stats.climatology import (
    compute_climatological_onset,
    compute_climatology_as_forecast,
)
from momp.utils.practical import restore_args
from momp.utils.printing import tuple_to_str_range


def _clean_coords(obj: Union[xr.Dataset, xr.DataArray]) -> Union[xr.Dataset, xr.DataArray]:
    """Ensures spatial coordinates (lat/lon) are numeric float64 and deduplicated."""
    obj = obj.copy()
    coord_updates = {}
    for dim in ["lat", "lon"]:
        if dim in obj.coords:
            vals = obj[dim].values
            num_vals = pd.to_numeric(np.asarray(vals).ravel(), errors="coerce").astype(np.float64)
            if not np.all(np.isnan(num_vals)):
                coord_updates[dim] = num_vals

    if coord_updates:
        obj = obj.assign_coords(coord_updates)

    for dim in ["lat", "lon"]:
        if dim in obj.dims:
            vals = obj[dim].values
            if len(vals) > 0 and len(vals) != len(np.unique(vals)):
                _, uidx = np.unique(vals, return_index=True)
                obj = obj.isel({dim: np.sort(uidx)})

    return obj


def align_obs_to_model_grid(obs_da: xr.DataArray, model_da: Union[xr.Dataset, xr.DataArray]) -> xr.DataArray:
    """
    Standardizes latitude orientation and interpolates observation data
    to match the model forecast spatial resolution.
    """
    obs_da = _clean_coords(obs_da)
    model_da = _clean_coords(model_da)

    # Sort latitudes ascending
    if "lat" in obs_da.dims and len(obs_da.lat) > 1 and obs_da.lat[0] > obs_da.lat[-1]:
        obs_da = obs_da.sortby("lat", ascending=True)
    if "lat" in model_da.dims and len(model_da.lat) > 1 and model_da.lat[0] > model_da.lat[-1]:
        model_da = model_da.sortby("lat", ascending=True)

    # Interpolate observed onset grid to model grid coordinates
    if "lat" in model_da.dims and "lon" in model_da.dims:
        try:
            obs_aligned = obs_da.interp(
                lat=model_da.lat,
                lon=model_da.lon,
                method="nearest"
            )
            return obs_aligned
        except Exception as e:
            print(f"[WARNING] Nearest interpolation failed ({e}), using raw observations.")
            return obs_da

    return obs_da


def safe_spatial_metrics(hits: np.ndarray, false_alarms: np.ndarray, misses: np.ndarray, absolute_errors: np.ndarray):
    """
    Computes spatial MAE, FAR, and Miss Rate safely handling 0/0 division.
    """
    total_forecasts = hits + false_alarms
    total_events = hits + misses

    # FAR = False Alarms / (Hits + False Alarms)
    far = np.where(total_forecasts > 0, false_alarms / total_forecasts, np.nan)

    # Miss Rate = Misses / (Hits + Misses)
    miss_rate = np.where(total_events > 0, misses / total_events, np.nan)

    # Mean Absolute Error across valid hits
    mae = np.where(hits > 0, absolute_errors / hits, np.nan)

    return mae, far, miss_rate


def compute_onset_metrics_with_windows(
    onset_df: pd.DataFrame,
    *,
    tolerance_days: int = 3,
    verification_window: Tuple[int, int] = (1, 15),
    **kwargs
) -> Tuple[pd.DataFrame, Dict[str, Union[int, float, Tuple[int, int]]]]:
    """
    Computes contingency matrix metrics (TP, FP, FN, TN, MAE) following verification window logic.
    """
    forecast_bin_start = verification_window[0]
    forecast_bin_end = verification_window[1]

    results_list = []
    unique_locations = onset_df[["lat", "lon"]].drop_duplicates()

    for idx, (_, row) in enumerate(unique_locations.iterrows()):
        lat, lon = row["lat"], row["lon"]

        grid_data = onset_df[(onset_df["lat"] == lat) & (onset_df["lon"] == lon)].copy()

        grid_data["obs_onset_dt"] = pd.to_datetime(grid_data["obs_onset_date"])
        grid_data["model_onset_dt"] = pd.to_datetime(grid_data["onset_date"])
        grid_data["init_dt"] = pd.to_datetime(grid_data["init_time"])

        TP, FP, FN, TN = 0, 0, 0, 0
        num_onset, num_no_onset = 0, 0
        mae_tp, mae_fp = [], []

        for _, init_row in grid_data.iterrows():
            t_init = init_row["init_dt"]
            model_onset = init_row["model_onset_dt"]
            observed_onset = init_row["obs_onset_dt"]

            valid_window_start = t_init + pd.Timedelta(days=forecast_bin_start)
            valid_window_end = t_init + pd.Timedelta(days=forecast_bin_end)

            is_observed_event = (
                pd.notna(observed_onset) and valid_window_start <= observed_onset <= valid_window_end
            )
            is_model_event = pd.notna(model_onset) and valid_window_start <= model_onset <= valid_window_end

            if is_observed_event:
                num_onset += 1
            else:
                num_no_onset += 1

            # Every valid forecast is an event or non-event for this specific
            # interval. Earlier onsets are non-events here, not dropped rows.
            if is_model_event:
                abs_diff_days = abs((model_onset - observed_onset).days) if pd.notna(observed_onset) else None
                if is_observed_event and abs_diff_days <= tolerance_days:
                    TP += 1
                    mae_tp.append(abs_diff_days)
                else:
                    FP += 1
                    if abs_diff_days is not None:
                        mae_fp.append(abs_diff_days)
            elif is_observed_event:
                FN += 1
            else:
                TN += 1

        total_forecasts = len(grid_data)
        mae_combined = mae_tp + mae_fp
        mae = np.mean(mae_combined) if len(mae_combined) > 0 else np.nan
        mae_tp_only = np.mean(mae_tp) if len(mae_tp) > 0 else np.nan

        result = {
            "lat": lat,
            "lon": lon,
            "total_forecasts": total_forecasts,
            "true_positive": TP,
            "true_negative": TN,
            "false_positive": FP,
            "false_negative": FN,
            "num_onset": num_onset,
            "num_no_onset": num_no_onset,
            "mae_combined": mae,
            "mae_tp_only": mae_tp_only,
            "num_tp_errors": len(mae_tp),
            "num_fp_errors": len(mae_fp),
            "tolerance_days": tolerance_days,
            "verification_window": verification_window,
            "forecast_days": forecast_bin_end,
        }
        results_list.append(result)

    metrics_df = pd.DataFrame(results_list)

    summary_stats = {
        "total_grid_points": len(metrics_df),
        "total_forecasts": metrics_df["total_forecasts"].sum() if not metrics_df.empty else 0,
        "overall_true_positive": metrics_df["true_positive"].sum() if not metrics_df.empty else 0,
        "overall_true_negative": metrics_df["true_negative"].sum() if not metrics_df.empty else 0,
        "overall_false_positive": metrics_df["false_positive"].sum() if not metrics_df.empty else 0,
        "overall_false_negative": metrics_df["false_negative"].sum() if not metrics_df.empty else 0,
        "overall_num_onset": metrics_df["num_onset"].sum() if not metrics_df.empty else 0,
        "overall_num_no_onset": metrics_df["num_no_onset"].sum() if not metrics_df.empty else 0,
        "overall_mae_combined": metrics_df["mae_combined"].mean() if not metrics_df.empty else np.nan,
        "overall_mae_tp_only": metrics_df["mae_tp_only"].mean() if not metrics_df.empty else np.nan,
        "tolerance_days": tolerance_days,
        "verification_window": verification_window,
        "forecast_days": forecast_bin_end,
    }

    return metrics_df, summary_stats


def compute_metrics_multiple_years(
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
    end_date=None,
    fallback_date=None,
    mok=None,
    years=None,
    years_clim=None,
    model_dir=None,
    model_var=None,
    ref_model=None,
    date_filter_year=None,
    init_days=None,
    unit_cvt=None,
    file_pattern=None,
    tolerance_days=3,
    verification_window=(1, 15),
    max_forecast_day=45,
    members=None,
    onset_percentage_threshold=0.5,
    probabilistic=False,
    save_nc_climatology=False,
    **kwargs
) -> Tuple[Dict[int, pd.DataFrame], Dict[int, xr.DataArray]]:
    """Compute onset metrics for multiple years for deterministic or baseline models."""
    kwargs = restore_args(compute_metrics_multiple_years, kwargs, locals())

    metrics_df_dict = {}
    onset_da_dict = {}

    thresh_da = load_thresh_file(**kwargs)
    ref_clim = kwargs.get("model") == "climatology"

    if years is None:
        years = kwargs.get("years") or getattr(params, "years", [2020])

    if not ref_clim:
        years = available_forecast_years(
            years,
            model_dir=model_dir,
            file_pattern=file_pattern,
        )
        if not years:
            raise FileNotFoundError(f"No forecast files are available in {model_dir!r}")
        kwargs["years"] = years

    if ref_clim:
        print("Generating climatology onset doy for each year as onset_da_dict")
        climatological_onset_doy = compute_climatological_onset(**kwargs)
        onset_da_dict = {year: climatological_onset_doy for year in years}

    for year in years:
        print(f"\n{'-'*50}\nProcessing year {year}")

        if probabilistic and not ref_clim:
            try:
                p_model = get_forecast_probabilistic_twice_weekly(year, **kwargs)
            except (MissingEnsembleDimensionError, NoMatchingInitializationTimesError) as exc:
                print(f"Skipping year {year}: {exc}")
                continue

        imd = load_imd_rainfall(year, **kwargs)
        onset_da = detect_observed_onset(imd, thresh_da, year, **kwargs)

        if probabilistic and not ref_clim:
            print("Extracting ensemble forecast data ...")
            print("Computing onset for ensemble forecast ...")
            onset_df = compute_onset_for_all_members(p_model, thresh_da, onset_da, **kwargs)

        elif not ref_clim:
            print("Extracting model forecast data ...")
            p_model = get_forecast_deterministic_twice_weekly(year, **kwargs)
            print("Computing onset for model forecast ...")
            onset_df = compute_onset_for_deterministic_model(p_model, thresh_da, onset_da, **kwargs)

        else:
            print("Computing onset for climatology as forecast ...")
            init_dates = get_initialization_dates(year, **kwargs)
            onset_df = compute_climatology_as_forecast(
                climatological_onset_doy, year, init_dates, onset_da, **kwargs
            )

        print("Computing onset-obs metrics TP,TN,FP,FN for all locations and init times ...")
        
        # Clean kwargs to avoid duplicate keyword argument errors
        kwargs_clean = kwargs.copy()
        kwargs_clean.pop("tolerance_days", None)
        kwargs_clean.pop("verification_window", None)

        metrics_df, summary_stats = compute_onset_metrics_with_windows(
            onset_df,
            tolerance_days=tolerance_days,
            verification_window=verification_window,
            **kwargs_clean
        )

        metrics_df_dict[year] = metrics_df

        if not ref_clim:
            onset_da_dict[year] = onset_da

        print(f"Year {year} completed. Grid points processed: {len(metrics_df)}")
        print(
            f"Summary stats: TP={summary_stats['overall_true_positive']}, "
            f"FP={summary_stats['overall_false_positive']}, "
            f"FN={summary_stats['overall_false_negative']}, "
            f"TN={summary_stats['overall_true_negative']}"
        )

    return metrics_df_dict, onset_da_dict


def ens_compute_metrics_multiple_years(
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
    end_date=None,
    fallback_date=None,
    mok=None,
    years=None,
    years_clim=None,
    model_dir=None,
    model_var=None,
    ref_model=None,
    date_filter_year=None,
    init_days=None,
    unit_cvt=None,
    file_pattern=None,
    tolerance_days=3,
    verification_window=(1, 15),
    max_forecast_day=45,
    members=None,
    onset_percentage_threshold=0.5,
    probabilistic=True,
    save_nc_climatology=False,
    **kwargs
) -> Tuple[Dict[int, pd.DataFrame], Dict[int, xr.DataArray]]:
    """Compute onset metrics for multiple years for probabilistic / ensemble models."""
    kwargs = restore_args(ens_compute_metrics_multiple_years, kwargs, locals())

    metrics_df_dict = {}
    onset_da_dict = {}

    thresh_da = load_thresh_file(**kwargs)
    ref_clim = kwargs.get("model") == "climatology"

    if years is None:
        years = kwargs.get("years") or getattr(params, "years", [2020])

    if not ref_clim:
        years = available_forecast_years(
            years,
            model_dir=model_dir,
            file_pattern=file_pattern,
        )
        if not years:
            raise FileNotFoundError(f"No forecast files are available in {model_dir!r}")
        kwargs["years"] = years

    if ref_clim:
        climatological_onset_doy = compute_climatological_onset(**kwargs)
        onset_da_dict = {year: climatological_onset_doy for year in years}

    for year in years:
        print(f"\n{'-'*50}\nProcessing year {year}")

        if probabilistic and not ref_clim:
            try:
                p_model = get_forecast_probabilistic_twice_weekly(year, **kwargs)
            except (MissingEnsembleDimensionError, NoMatchingInitializationTimesError) as exc:
                print(f"Skipping year {year}: {exc}")
                continue

        imd = load_imd_rainfall(year, **kwargs)
        onset_da = detect_observed_onset(imd, thresh_da, year, **kwargs)

        if probabilistic and not ref_clim:
            print("-------Extracting ensemble forecast data ...")
            print("-------Computing onset for ensemble forecast ...")
            onset_df = compute_onset_for_all_members(p_model, thresh_da, onset_da, **kwargs)

        elif ref_clim:
            print("-------Computing onset for climatology as forecast ...")
            init_dates = get_initialization_dates(year, **kwargs)
            onset_df = compute_climatology_as_forecast(
                climatological_onset_doy, year, init_dates, onset_da, **kwargs
            )

        print("-------Computing onset-obs metrics TP,TN,FP,FN for all locations and init times ...")

        # Clean kwargs to avoid duplicate keyword argument errors
        kwargs_clean = kwargs.copy()
        kwargs_clean.pop("tolerance_days", None)
        kwargs_clean.pop("verification_window", None)

        metrics_df, summary_stats = compute_onset_metrics_with_windows(
            onset_df,
            tolerance_days=tolerance_days,
            verification_window=verification_window,
            **kwargs_clean
        )

        metrics_df_dict[year] = metrics_df

        if not ref_clim:
            onset_da_dict[year] = onset_da

        print(f"Year {year} completed. Grid points processed: {len(metrics_df)}")
        print(
            f"Summary stats: TP={summary_stats['overall_true_positive']}, "
            f"FP={summary_stats['overall_false_positive']}, "
            f"FN={summary_stats['overall_false_negative']}, "
            f"TN={summary_stats['overall_true_negative']}"
        )

    return metrics_df_dict, onset_da_dict
