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
from momp.utils.standard import align_spatial_grid
from momp.io.input import admin_forecast_grid_index
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


def _fit_forecast_horizon(p_model, max_forecast_day, dry_extent, wet_spell, day_bins, dry_spell=7):
    step_count = int(p_model.sizes.get("step", 0))
    if step_count == 0:
        raise ValueError("Forecast data has no 'step' dimension.")

    available_day = step_count
    if "step" in p_model.coords:
        values = np.asarray(p_model["step"].values)
        if np.issubdtype(values.dtype, np.timedelta64):
            values = values / np.timedelta64(1, "D")
        else:
            try:
                values = values.astype(float)
            except (TypeError, ValueError):
                values = np.asarray([], dtype=float)

            units = str(p_model["step"].attrs.get("units", "")).lower()
            if "hour" in units:
                values = values / 24.0
            elif "minute" in units:
                values = values / 1440.0
            elif "second" in units:
                values = values / 86400.0

        finite_steps = values[np.isfinite(values)]
        if finite_steps.size:
            available_day = int(np.floor(np.max(finite_steps)))

    window_end = max((int(end) for _, end in (day_bins or ())), default=int(max_forecast_day))
    max_forecast_day = min(int(max_forecast_day), window_end)
    if max_forecast_day > available_day:
        raise ValueError(
            f"Forecast data provides steps only through day {available_day}, "
            f"but the verification window requires day {max_forecast_day}."
        )

    dry_extent = max(0, int(dry_extent or 0))
    # The confirmation interval includes the candidate day itself.
    confirmation = dry_extent if dry_extent > wet_spell and dry_spell > 0 else wet_spell
    required_tail = max(confirmation, wet_spell) - 1
    required_steps = max_forecast_day + required_tail
    if required_steps > available_day:
        raise ValueError(
            f"Forecast data provides steps only through day {available_day}, "
            f"but onset detection requires day {required_steps} with the configured "
            f"wet_spell={wet_spell} and dry_extent={dry_extent}. "
            f"The latest verifiable onset day with these criteria is {max(0, available_day - required_tail)}. "
            "Supply a longer forecast or explicitly change the verification window/onset criteria."
        )

    return max_forecast_day, dry_extent, required_steps


def detect_onset(
    day,
    forecast_series,
    thresh,
    *,
    wet_init=1.0,
    wet_spell=3,
    dry_spell=7,
    dry_threshold=1.0,
    dry_extent=0,
    **kwargs,
):
    """
    Detect onset for a specific forecast lead day.
    forecast_series: 1D array of daily forecast rainfall values starting at step 1.
    """
    series = np.asarray(forecast_series, dtype=float)
    if series.ndim != 1:
        raise ValueError("Onset detection requires a one-dimensional daily rainfall series.")
    if day < 1 or wet_spell < 1 or dry_spell < 0 or dry_extent < 0:
        raise ValueError("Onset day and wet_spell must be positive; dry windows cannot be negative.")
    start_idx = day - 1
    end_idx = start_idx + wet_spell
    if end_idx > len(series):
        return False
    wet = series[start_idx:end_idx]
    if not (np.isfinite(wet).all() and np.all(wet >= wet_init) and wet.sum() > thresh):
        return False
    if dry_extent > wet_spell and dry_spell > 0:
        dry_end = start_idx + dry_extent
        if dry_end > len(series):
            return False
        confirmation = series[start_idx:dry_end]
        if not np.isfinite(confirmation).all():
            return False
        if dry_spell <= dry_extent:
            dry = confirmation < dry_threshold
            if np.any(np.convolve(dry, np.ones(dry_spell, dtype=int), "valid") == dry_spell):
                return False
    return True


def detect_forecast_onset(day_or_series, forecast_series=None, thresh=10.0, **kwargs):
    """
    Flexible onset detector wrapper supporting both positional calling signatures:
    1) detect_forecast_onset(day, forecast_series, thresh, **kwargs) -> bool
    2) detect_forecast_onset(forecast_series, thresh_val=10.0, **kwargs) -> int or None
    """
    kwargs = kwargs.copy()
    wet_init = kwargs.pop("wet_init", getattr(params, "wet_init", 1.0))
    wet_spell = kwargs.pop("wet_spell", getattr(params, "wet_spell", 3))
    dry_spell = kwargs.pop("dry_spell", getattr(params, "dry_spell", 7))
    dry_threshold = kwargs.pop("dry_threshold", getattr(params, "dry_threshold", 1.0))
    dry_extent = kwargs.pop("dry_extent", getattr(params, "dry_extent", 0))

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
    dry_extent=0,
    start_date=(5, 1),
    end_date=(7, 31),
    fallback_date=(4, 1),
    mok=None,
    extend_end_day=47,
    **kwargs,
):
    """Detect observed onset dates for a given year from gridded rainfall DataArray."""
    thresh_slice = align_spatial_grid(thresh_slice, rain_slice, label='onset threshold')
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

    if wet_spell < 1 or dry_spell < 0 or dry_extent < 0:
        raise ValueError("wet_spell must be positive and dry windows cannot be negative.")
    time_dates = pd.to_datetime(rain_slice.time.values)
    if time_dates.hasnans or (len(time_dates) > 1 and
            not np.all(np.diff(time_dates.values) == np.timedelta64(1, "D"))):
        raise ValueError("Observed rainfall must have unique, consecutive daily timestamps.")
    start_idx_candidates = np.where(time_dates >= start_dt)[0]
    if len(start_idx_candidates) == 0 and fallback_date:
        fallback_dt = datetime(year, *fallback_MMDD)
        start_idx_candidates = np.where(time_dates >= fallback_dt)[0]
    start_idx = start_idx_candidates[0] if len(start_idx_candidates) else len(time_dates)
    rain_subset = rain_slice.isel(time=slice(start_idx, None)).sel(time=slice(None, end_dt))
    if rain_subset.sizes["time"] == 0:
        return xr.DataArray(
            np.full((rain_slice.sizes["lat"], rain_slice.sizes["lon"]), np.datetime64("NaT", "ns")),
            dims=("lat", "lon"), coords={"lat": rain_slice.lat, "lon": rain_slice.lon}, name="onset_date")

    rolling_sum = rain_subset.rolling(time=wet_spell, min_periods=wet_spell, center=False).sum()
    rolling_sum_aligned = rolling_sum.shift(time=-(wet_spell - 1))

    wet_day_condition = xr.apply_ufunc(np.isfinite, rain_subset) & (rain_subset >= wet_init)
    wet_day_spell = wet_day_condition.rolling(time=wet_spell, min_periods=wet_spell, center=False).reduce(np.all)

    first_day_condition = wet_day_spell.shift(time=-(wet_spell - 1))
    first_day_condition = first_day_condition.fillna(False).astype(bool)

    sum_condition = rolling_sum_aligned > thresh_slice

    if dry_extent > wet_spell and dry_spell > 0:
        # A dry run must fit wholly inside [candidate, candidate + dry_extent).
        # Rolling runs are labelled at their first day, matching detect_onset.
        finite = xr.apply_ufunc(np.isfinite, rain_subset)
        complete = finite.rolling(time=dry_extent, min_periods=dry_extent).sum().shift(
            time=-(dry_extent - 1)) == dry_extent
        if dry_spell <= dry_extent:
            dry = rain_subset < dry_threshold
            dry_run = (dry.rolling(time=dry_spell, min_periods=dry_spell).sum() == dry_spell).shift(
                time=-(dry_spell - 1), fill_value=False)
            positions = dry_extent - dry_spell + 1
            has_dry = dry_run.rolling(time=positions, min_periods=positions).max().shift(
                time=-(positions - 1)).fillna(False).astype(bool)
            complete = complete & ~has_dry
        onset_condition = first_day_condition & sum_condition & complete
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
    onset_dates_array = np.full(onset_indices.shape, np.datetime64("NaT", "ns"), dtype="datetime64[ns]")

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
    dry_extent=0,
    max_forecast_day=45,
    mok=None,
    end_date=(7, 31),
    **kwargs,
):
    """Compute onset dates for deterministic model forecast."""
    kwargs = restore_args(compute_onset_for_deterministic_model, kwargs, locals())
    onset_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key not in {"wet_init", "wet_spell", "dry_spell", "dry_threshold", "dry_extent"}
    }

    results_list = []

    init_times = p_model.init_time.values
    admin_dim = next(
        (name for name in ("adm3_name", "adm2_name", "adm1_name", "woreda", "district") if name in p_model.dims),
        None,
    )
    if admin_dim is None:
        onset_da = align_spatial_grid(onset_da, p_model, label='observed onset')
    thresh_slice = align_spatial_grid(thresh_slice, onset_da, label='onset threshold')
    if admin_dim:
        lats = onset_da.lat.values
        lons = onset_da.lon.values
        admin_grid = admin_forecast_grid_index(
            p_model[admin_dim].values,
            onset_da,
            kwargs.get("shpfile_dir"),
            region=kwargs.get("region", "Ethiopia"),
        )
    else:
        lats = p_model.lat.values
        lons = p_model.lon.values
        admin_grid = None

    print(f"Processing {len(init_times)} init times x {len(lats)} lats x {len(lons)} lons...")

    max_forecast_day, dry_extent, max_steps_needed = _fit_forecast_horizon(
        p_model, max_forecast_day, dry_extent, wet_spell, kwargs.get("day_bins"), dry_spell
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
                admin_index = admin_grid[i, j] if admin_grid is not None else None
                if admin_index is not None and admin_index < 0:
                    skipped_no_obs += 1
                    continue

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
                    forecast_indexers = {"init_time": t_idx}
                    if admin_index is None:
                        forecast_indexers.update(lat=i, lon=j)
                    else:
                        forecast_indexers[admin_dim] = int(admin_index)
                    forecast_series = p_model.isel(forecast_indexers).sel(
                        step=slice(1, max_steps_needed)
                    ).values

                    if not np.isfinite(forecast_series).all():
                        valid_inits -= 1
                        continue
                    if len(forecast_series) < max_steps_needed:
                        raise ValueError("Forecast series is shorter than the required onset detection horizon.")
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
                                **onset_kwargs,
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
                except Exception as exc:
                    raise ValueError(f"Could not detect onset for initialization {init_time}, lat={lat}, lon={lon}: {exc}") from exc

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

    onset_df = pd.DataFrame(results_list, columns=["init_time", "lat", "lon", "onset_day", "onset_date", "obs_onset_date"])

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
    dry_extent=0,
    members=None,
    onset_percentage_threshold=0.5,
    max_forecast_day=45,
    mok=None,
    end_date=(7, 31),
    **kwargs,
):
    """Compute onset dates for each ensemble member, initialization time, and grid point."""
    kwargs = restore_args(compute_onset_for_all_members, kwargs, locals())
    onset_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key not in {"wet_init", "wet_spell", "dry_spell", "dry_threshold", "dry_extent"}
    }

    results_list = []

    init_times = p_model.init_time.values

    if isinstance(members, str) and members.strip().lower() in ("all", ""):
        members = None
    if members is None or len(members) == 0:
        members = p_model.member.values if "member" in p_model.dims else [0]
    if "member" in p_model.dims and not np.isin(members, p_model.member.values).all():
        raise ValueError(f"Requested ensemble members {members} are not present in the forecast.")

    admin_dim = next(
        (name for name in ("adm3_name", "adm2_name", "adm1_name", "woreda", "district") if name in p_model.dims),
        None,
    )
    if admin_dim is None:
        onset_da = align_spatial_grid(onset_da, p_model, label='observed onset')
    thresh_slice = align_spatial_grid(thresh_slice, onset_da, label='onset threshold')
    if admin_dim:
        lats = onset_da.lat.values
        lons = onset_da.lon.values
        admin_grid = admin_forecast_grid_index(
            p_model[admin_dim].values,
            onset_da,
            kwargs.get("shpfile_dir"),
            region=kwargs.get("region", "Ethiopia"),
        )
    else:
        lats = p_model.lat.values
        lons = p_model.lon.values
        admin_grid = None

    print(f"Processing {len(init_times)} init times x {len(lats)*len(lons)} unique locations x {len(members)} members...")

    max_forecast_day, dry_extent, max_steps_needed = _fit_forecast_horizon(
        p_model, max_forecast_day, dry_extent, wet_spell, kwargs.get("day_bins"), dry_spell
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
            admin_index = admin_grid[lat_idx, lon_idx] if admin_grid is not None else None
            if admin_index is not None and admin_index < 0:
                skipped_no_obs += len(members)
                continue

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
                    forecast_indexers = {"init_time": t_idx}
                    if admin_index is None:
                        forecast_indexers.update(lat=lat_idx, lon=lon_idx)
                    else:
                        forecast_indexers[admin_dim] = int(admin_index)
                    forecast_data = p_model.isel(forecast_indexers).sel(
                        step=slice(1, max_steps_needed)
                    )
                    if "member" in p_model.dims:
                        forecast_series = forecast_data.sel(member=member).values
                    else:
                        forecast_series = forecast_data.values

                    if not np.isfinite(forecast_series).all():
                        valid_forecasts -= 1
                        continue
                    if len(forecast_series) < max_steps_needed:
                        raise ValueError("Forecast series is shorter than the required onset detection horizon.")
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
                                **onset_kwargs,
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
                except Exception as exc:
                    raise ValueError(f"Could not detect onset for member {member}, initialization {init_time}, lat={lat}, lon={lon}: {exc}") from exc

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

    onset_df = pd.DataFrame(results_list, columns=["init_time", "lat", "lon", "member", "onset_day", "onset_date", "obs_onset_date"])

    print(f"\nProcessing Ensemble Summary:")
    print(f"Total potential forecasts: {total_potential_forecasts}")
    print(f"Skipped (no observed onset): {skipped_no_obs}")
    print(f"Skipped (initialized after observed onset): {skipped_late_init}")
    print(f"Valid forecasts processed: {valid_forecasts}")
    print(f"Ensemble onsets found: {ensemble_onsets_found}")
    print(f"Onset rate: {ensemble_onsets_found/valid_forecasts:.3f}" if valid_forecasts > 0 else "Onset rate: 0.000")

    return onset_df


def aggregate_ensemble_onset(member_onsets, *, onset_percentage_threshold=0.5):
    """Apply reference ROMP consensus and rounded-mean timing for spatial scores.

    Probabilistic bin scores still use individual member rows. Spatial metrics
    instead verify one consensus forecast for each initialization and grid cell.
    """
    if not np.isfinite(onset_percentage_threshold) or not 0 <= onset_percentage_threshold <= 1:
        raise ValueError('Ensemble onset percentage threshold must be between zero and one.')
    columns = ['init_time', 'lat', 'lon', 'onset_day', 'onset_date',
               'member_onset_count', 'total_members', 'onset_percentage', 'obs_onset_date']
    rows = []
    for (initialization, lat, lon), group in member_onsets.groupby(['init_time', 'lat', 'lon']):
        if group['member'].duplicated().any():
            raise ValueError('Duplicate ensemble member rows for one forecast.')
        if group['obs_onset_date'].nunique(dropna=False) != 1:
            raise ValueError('Ensemble members must share the same observed onset.')
        days = pd.to_numeric(group['onset_day'], errors='raise').dropna()
        if not np.isfinite(days).all():
            raise ValueError('Ensemble onset days must be finite or missing.')
        count, total = len(days), len(group)
        fraction = count / total
        onset_day = int(round(days.mean())) if count and fraction >= onset_percentage_threshold else None
        onset_date = (pd.Timestamp(initialization) + pd.Timedelta(days=onset_day)
                      if onset_day is not None else None)
        rows.append({
            'init_time': initialization, 'lat': lat, 'lon': lon,
            'onset_day': onset_day,
            'onset_date': onset_date.strftime('%Y-%m-%d') if onset_date is not None else None,
            'member_onset_count': count, 'total_members': total,
            'onset_percentage': fraction, 'obs_onset_date': group['obs_onset_date'].iloc[0],
        })
    return pd.DataFrame(rows, columns=columns)
