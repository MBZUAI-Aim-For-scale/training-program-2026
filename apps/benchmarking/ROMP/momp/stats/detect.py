"""
momp/stats/detect.py
Monsoon rainfall onset detection algorithms for observed time-series and forecast model evaluations.
"""

from datetime import datetime, timedelta
from itertools import product
import sys
import numpy as np
import pandas as pd
import xarray as xr

from momp.utils.practical import restore_args
import momp.params as params


def find_first_true(arr):
    """
    Find first occurrence of onset condition for a 1D grid point array.
    Works if arr contains floats, integers, or NaNs.
    Returns the 0-based index of the first True value, or -1 if none.
    """
    arr_bool = np.asarray(arr, dtype=float)
    arr_bool = np.nan_to_num(arr_bool, nan=0.0)
    arr_bool = arr_bool.astype(bool)

    if arr_bool.any():
        return int(np.argmax(arr_bool))
    else:
        return -1


def detect_onset(
    day,
    forecast_series,
    thresh,
    *,
    wet_init=1.0,
    wet_spell=3,
    dry_spell=7,
    dry_threshold=1.0,
    dry_extent=21,
    **kwargs,
):
    """
    Detect onset for a specific forecast lead day.
    forecast_series: 1D array of daily forecast rainfall values starting at step 1.
    """
    start_idx = day - 1

    if dry_extent <= wet_spell:
        end_idx = start_idx + wet_spell
        if end_idx <= len(forecast_series):
            window_series = forecast_series[start_idx:end_idx]
            if np.all(window_series >= wet_init) and np.nansum(window_series) > thresh:
                return True
    else:
        end_idx = start_idx + wet_spell
        end_idx_dry = start_idx + dry_extent

        if end_idx_dry <= len(forecast_series):
            window_series = forecast_series[start_idx:end_idx]
            dry_series = forecast_series[start_idx:end_idx_dry]

            if np.all(window_series >= wet_init) and np.nansum(window_series) > thresh:
                dry_bool = dry_series < wet_init
                consec_dry = np.convolve(dry_bool, np.ones(dry_spell, dtype=int), "valid")
                has_dry_spell = np.any(consec_dry == dry_spell)

                if not has_dry_spell:
                    return True

    return False


def detect_forecast_onset(day_or_series, forecast_series=None, thresh=10.0, **kwargs):
    """
    Flexible onset detector wrapper supporting both positional calling signatures:
    1) detect_forecast_onset(day, forecast_series, thresh, **kwargs) -> bool
    2) detect_forecast_onset(forecast_series, thresh_val=10.0, **kwargs) -> int or None
    """
    wet_init = kwargs.get("wet_init", getattr(params, "wet_init", 1.0))
    wet_spell = kwargs.get("wet_spell", getattr(params, "wet_spell", 3))
    dry_spell = kwargs.get("dry_spell", getattr(params, "dry_spell", 7))
    dry_threshold = kwargs.get("dry_threshold", getattr(params, "dry_threshold", 1.0))
    dry_extent = kwargs.get("dry_extent", getattr(params, "dry_extent", 21))

    # Signature 1: Called with day as integer
    if isinstance(day_or_series, (int, np.integer)):
        day = int(day_or_series)
        return detect_onset(
            day,
            forecast_series,
            thresh,
            wet_init=wet_init,
            wet_spell=wet_spell,
            dry_spell=dry_spell,
            dry_threshold=dry_threshold,
            dry_extent=dry_extent,
            **kwargs,
        )

    # Signature 2: Called directly with 1D forecast time series
    series = day_or_series
    thresh_val = kwargs.get("thresh_val", thresh)
    max_forecast_day = kwargs.get("max_forecast_day", getattr(params, "max_forecast_day", 45))

    for day in range(1, max_forecast_day + 1):
        if detect_onset(
            day,
            series,
            thresh_val,
            wet_init=wet_init,
            wet_spell=wet_spell,
            dry_spell=dry_spell,
            dry_threshold=dry_threshold,
            dry_extent=dry_extent,
            **kwargs,
        ):
            return day

    return None


def detect_observed_onset(
    rain_slice,
    thresh_slice,
    year,
    *,
    wet_init=1.0,
    wet_spell=3,
    dry_spell=7,
    dry_threshold=1.0,
    dry_extent=21,
    start_date=(5, 1),
    end_date=(7, 31),
    fallback_date=(4, 1),
    mok=None,
    extend_end_day=47,
    **kwargs,
):
    """Detect observed onset dates for a given year from gridded rainfall DataArray."""
    start_MMDD = start_date[1:] if isinstance(start_date, (list, tuple)) and len(start_date) == 3 else start_date
    end_MMDD = end_date[1:] if isinstance(end_date, (list, tuple)) and len(end_date) == 3 else end_date
    fallback_MMDD = fallback_date

    end_dt = datetime(year, *end_MMDD)
    if extend_end_day:
        end_dt = end_dt + timedelta(days=extend_end_day)

    if mok:
        start_dt = datetime(year, *mok)
        date_label = f"{mok[0]:02d}-{mok[1]:02d}"
    else:
        start_dt = datetime(year, *start_MMDD)
        date_label = f"{start_MMDD[0]:02d}-{start_MMDD[1]:02d}"

    time_dates = pd.to_datetime(rain_slice.time.values)
    start_idx_candidates = np.where(time_dates >= start_dt)[0]

    if len(start_idx_candidates) == 0 and fallback_date:
        fallback_dt = datetime(year, *fallback_MMDD)
        start_idx = np.where(time_dates >= fallback_dt)[0][0]
    else:
        start_idx = start_idx_candidates[0]

    rain_subset = rain_slice.isel(time=slice(start_idx, None)).sel(time=slice(None, end_dt))

    rolling_sum = rain_subset.rolling(time=wet_spell, min_periods=wet_spell, center=False).sum()
    rolling_sum_aligned = rolling_sum.shift(time=-(wet_spell - 1))

    wet_day_condition = rain_subset >= wet_init
    wet_day_spell = wet_day_condition.rolling(time=wet_spell, min_periods=wet_spell, center=False).reduce(np.all)

    first_day_condition = wet_day_spell.shift(time=-(wet_spell - 1))
    first_day_condition = first_day_condition.fillna(False).astype(bool)

    sum_condition = rolling_sum_aligned > thresh_slice

    if dry_extent >= dry_spell and dry_extent > wet_spell:
        dry_day_condition = rain_subset < wet_init
        dry_day_spell = dry_day_condition.rolling(time=dry_spell, min_periods=dry_spell).sum() == dry_spell

        has_dry_after = (
            (dry_day_spell.rolling(time=dry_extent + 1, min_periods=1).sum() > 0)
            .shift(time=-dry_extent)
            .fillna(False)
            .astype(bool)
        )

        no_dry_after = xr.apply_ufunc(np.logical_not, has_dry_after)
        onset_condition = first_day_condition & sum_condition & no_dry_after
    else:
        onset_condition = first_day_condition & sum_condition

    onset_indices = xr.apply_ufunc(
        find_first_true,
        onset_condition,
        input_core_dims=[["time"]],
        output_dtypes=[int],
        vectorize=True,
    )

    valid_mask = onset_indices.values >= 0
    time_coords = rain_subset.time.values
    onset_dates_array = np.full(onset_indices.shape, np.datetime64("NaT"), dtype="datetime64[ns]")

    for i in range(onset_indices.shape[0]):
        for j in range(onset_indices.shape[1]):
            if valid_mask[i, j]:
                idx = int(onset_indices[i, j].values)
                if 0 <= idx < len(time_coords):
                    onset_dates_array[i, j] = time_coords[idx]

    onset_da = xr.DataArray(
        onset_dates_array,
        coords=[("lat", rain_slice.lat.values), ("lon", rain_slice.lon.values)],
        name="onset_date",
    )

    return onset_da


def compute_onset_for_deterministic_model(
    p_model,
    thresh_slice,
    onset_da,
    *,
    wet_init=1.0,
    wet_spell=3,
    dry_spell=7,
    dry_threshold=1.0,
    dry_extent=21,
    max_forecast_day=45,
    mok=None,
    end_date=(7, 31),
    **kwargs,
):
    """Compute onset dates for deterministic model forecast."""
    kwargs = restore_args(compute_onset_for_deterministic_model, kwargs, locals())

    results_list = []

    init_times = p_model.init_time.values
    lats = p_model.lat.values
    lons = p_model.lon.values

    print(f"Processing {len(init_times)} init times x {len(lats)} lats x {len(lons)} lons...")

    if dry_extent <= wet_spell:
        max_steps_needed = max_forecast_day + wet_spell - 1
    else:
        max_steps_needed = max_forecast_day + dry_extent

    full_steps = p_model.sizes["step"]
    if full_steps < max_steps_needed:
        raise ValueError(
            f"Not enough forecast time steps: model steps {full_steps} "
            f"< min steps required {max_steps_needed}."
        )

    total_potential_inits = 0
    valid_inits = 0
    skipped_no_obs = 0
    skipped_late_init = 0
    onsets_found = 0

    for t_idx, init_time in enumerate(init_times):
        init_date = pd.to_datetime(init_time)
        year = init_date.year

        if mok:
            mok_date = datetime(year, *mok)
        else:
            mok_date = None

        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                total_potential_inits += 1

                try:
                    obs_onset = onset_da.isel(lat=i, lon=j).values
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

                thresh = thresh_slice.isel(lat=i, lon=j).values if not np.isscalar(thresh_slice) else thresh_slice

                try:
                    forecast_series = (
                        p_model.isel(init_time=t_idx, lat=i, lon=j)
                        .sel(step=slice(1, max_steps_needed))
                        .values
                    )

                    onset_day = None
                    if len(forecast_series) >= max_steps_needed:
                        for day in range(1, max_forecast_day + 1):
                            isonset = detect_onset(
                                day,
                                forecast_series,
                                thresh,
                                wet_init=wet_init,
                                wet_spell=wet_spell,
                                dry_spell=dry_spell,
                                dry_threshold=dry_threshold,
                                dry_extent=dry_extent,
                                **kwargs,
                            )

                            if isonset:
                                forecast_date = init_date + pd.Timedelta(days=day)
                                if mok:
                                    if forecast_date.date() >= mok_date.date():
                                        onset_day = day
                                        break
                                else:
                                    onset_day = day
                                    break
                except Exception as e:
                    print(f"Error at init_time {t_idx}, lat {i}, lon {j}: {e}")
                    onset_day = None

                onset_date = None
                if onset_day is not None:
                    onsets_found += 1
                    onset_date = init_date + pd.Timedelta(days=onset_day)

                result = {
                    "init_time": init_time,
                    "lat": lat,
                    "lon": lon,
                    "onset_day": onset_day,
                    "onset_date": onset_date.strftime("%Y-%m-%d") if onset_date is not None else None,
                    "obs_onset_date": obs_onset_dt.strftime("%Y-%m-%d"),
                }
                results_list.append(result)

    onset_df = pd.DataFrame(results_list)

    print(f"\nProcessing Summary:")
    print(f"Total potential initializations: {total_potential_inits}")
    print(f"Skipped (no observed onset): {skipped_no_obs}")
    print(f"Skipped (initialized after observed onset): {skipped_late_init}")
    print(f"Valid initializations processed: {valid_inits}")
    print(f"Onsets found: {onsets_found}")
    print(f"Onset rate: {onsets_found/valid_inits:.3f}" if valid_inits > 0 else "Onset rate: 0.000")

    return onset_df


def compute_onset_for_all_members(
    p_model,
    thresh_slice,
    onset_da,
    *,
    wet_init=1.0,
    wet_spell=3,
    dry_spell=7,
    dry_threshold=1.0,
    dry_extent=21,
    members=None,
    onset_percentage_threshold=0.5,
    max_forecast_day=45,
    mok=None,
    end_date=(7, 31),
    **kwargs,
):
    """Compute onset dates for each ensemble member, initialization time, and grid point."""
    kwargs = restore_args(compute_onset_for_all_members, kwargs, locals())

    results_list = []

    init_times = p_model.init_time.values

    if not members:
        members = p_model.member.values if "member" in p_model.dims else [0]

    lats = p_model.lat.values
    lons = p_model.lon.values

    print(f"Processing {len(init_times)} init times x {len(lats)*len(lons)} unique locations x {len(members)} members...")

    if dry_extent <= wet_spell:
        max_steps_needed = max_forecast_day + wet_spell - 1
    else:
        max_steps_needed = max_forecast_day + dry_extent

    full_steps = p_model.sizes["step"]
    if full_steps < max_steps_needed:
        raise ValueError(
            f"Not enough forecast time steps: model steps {full_steps} "
            f"< min steps required {max_steps_needed}."
        )

    total_potential_forecasts = 0
    valid_forecasts = 0
    skipped_no_obs = 0
    skipped_late_init = 0
    ensemble_onsets_found = 0

    for t_idx, init_time in enumerate(init_times):
        init_date = pd.to_datetime(init_time)
        year = init_date.year

        if mok:
            mok_date = datetime(year, *mok)
        else:
            mok_date = None

        for lon_idx, lat_idx in product(range(len(lons)), range(len(lats))):
            lon = lons[lon_idx]
            lat = lats[lat_idx]

            total_potential_forecasts += len(members)

            try:
                obs_onset = onset_da.isel(lat=lat_idx, lon=lon_idx).values
            except Exception:
                skipped_no_obs += len(members)
                continue

            if pd.isna(obs_onset):
                skipped_no_obs += len(members)
                continue

            obs_onset_dt = pd.to_datetime(obs_onset)

            if init_date >= obs_onset_dt:
                skipped_late_init += len(members)
                continue

            thresh = thresh_slice.isel(lat=lat_idx, lon=lon_idx).values if not np.isscalar(thresh_slice) else thresh_slice

            for m_idx, member in enumerate(members):
                valid_forecasts += 1

                try:
                    if "member" in p_model.dims:
                        forecast_series = (
                            p_model.isel(init_time=t_idx, lat=lat_idx, lon=lon_idx)
                            .sel(step=slice(1, max_steps_needed))
                            .sel(member=member)
                            .values
                        )
                    else:
                        forecast_series = (
                            p_model.isel(init_time=t_idx, lat=lat_idx, lon=lon_idx)
                            .sel(step=slice(1, max_steps_needed))
                            .values
                        )

                    if len(forecast_series) < max_steps_needed:
                        onset_day = None
                    else:
                        onset_day = None
                        for day in range(1, max_forecast_day + 1):
                            isonset = detect_onset(
                                day,
                                forecast_series,
                                thresh,
                                wet_init=wet_init,
                                wet_spell=wet_spell,
                                dry_spell=dry_spell,
                                dry_threshold=dry_threshold,
                                dry_extent=dry_extent,
                                **kwargs,
                            )

                            if isonset:
                                forecast_date = init_date + pd.Timedelta(days=day)
                                if mok:
                                    if forecast_date.date() >= mok_date.date():
                                        onset_day = day
                                        break
                                else:
                                    onset_day = day
                                    break
                except Exception:
                    onset_day = None

                onset_date = None
                if onset_day is not None:
                    ensemble_onsets_found += 1
                    onset_date = init_date + pd.Timedelta(days=onset_day)

                result = {
                    "init_time": init_time,
                    "lat": lat,
                    "lon": lon,
                    "member": member,
                    "onset_day": onset_day,
                    "onset_date": onset_date.strftime("%Y-%m-%d") if onset_date is not None else None,
                    "obs_onset_date": obs_onset_dt.strftime("%Y-%m-%d"),
                }
                results_list.append(result)

    onset_df = pd.DataFrame(results_list)

    print(f"\nProcessing Ensemble Summary:")
    print(f"Total potential forecasts: {total_potential_forecasts}")
    print(f"Skipped (no observed onset): {skipped_no_obs}")
    print(f"Skipped (initialized after observed onset): {skipped_late_init}")
    print(f"Valid forecasts processed: {valid_forecasts}")
    print(f"Ensemble onsets found: {ensemble_onsets_found}")
    print(f"Onset rate: {ensemble_onsets_found/valid_forecasts:.3f}" if valid_forecasts > 0 else "Onset rate: 0.000")

    return onset_df
