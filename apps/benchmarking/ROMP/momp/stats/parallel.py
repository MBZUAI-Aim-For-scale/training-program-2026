"""
momp/stats/parallel.py
Parallel execution routines for ROMP/MOMP onset evaluation, climatology, and forecast metrics.
Uses 'spawn' multiprocessing context to ensure thread-safety with gRPC / GCS bucket streaming.
"""

import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
import numpy as np
import pandas as pd
import xarray as xr

# Force gRPC fork-safety settings for multiprocessing
os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "1"
os.environ["GRPC_POLL_STRATEGY"] = "poll"

from momp.stats.bins import (
    multi_year_climatological_forecast_obs_pairs,
    multi_year_forecast_obs_pairs,
)
from momp.stats.climatology import (
    compute_climatological_onset_dataset,
    compute_climatological_onset,
)
from momp.stats.benchmark import (
    compute_metrics_multiple_years,
    ens_compute_metrics_multiple_years,
)
from momp.io.input import available_forecast_years


def func_wrapper(chunk_and_args, target_func=None):
    """
    Wrapper function to handle chunked argument unpacking across worker processes.
    """
    if target_func == "clim":
        chunk, clim_onset, kwargs = chunk_and_args
    elif target_func == "forecast":
        chunk, kwargs = chunk_and_args

    local_kwargs = kwargs.copy()
    local_kwargs["years"] = chunk.tolist() if hasattr(chunk, "tolist") else chunk

    if target_func == "clim":
        return multi_year_climatological_forecast_obs_pairs(
            clim_onset,
            **local_kwargs
        )

    if target_func == "forecast":
        return multi_year_forecast_obs_pairs(
            **local_kwargs
        )


def parallel_climatological_forecast_obs_pairs(clim_onset, **kwargs):
    """
    Parallel version of multi_year_climatological_forecast_obs_pairs.
    """
    all_years = kwargs.get("years")
    if all_years is None:
        raise ValueError("The 'years' argument must be provided in kwargs.")

    cpu_count = os.cpu_count() or 1
    task_count = len(all_years)
    n_workers = min(cpu_count, task_count, 6)

    print(f"\n\ncpu_count is {cpu_count}")
    print(f"task count is {task_count}")
    print(f"Running on {n_workers} cores for {task_count} years...\n\n")

    year_chunks = np.array_split(all_years, n_workers)
    tasks = [(chunk, clim_onset, kwargs) for chunk in year_chunks]

    print(f"Executing on {n_workers} cores...")

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as executor:
        func_wrapper_clim = partial(func_wrapper, target_func="clim")
        list_of_results = list(executor.map(func_wrapper_clim, tasks))

    combined_forecast_obs = pd.concat(list_of_results, ignore_index=True)
    return combined_forecast_obs


def parallel_forecast_obs_pairs(**kwargs):
    """
    Parallel version of multi_year_forecast_obs_pairs.
    """
    all_years = kwargs.get("years")
    if all_years is None:
        raise ValueError("The 'years' argument must be provided in kwargs.")

    cpu_count = os.cpu_count() or 1
    task_count = len(all_years)
    n_workers = min(cpu_count, task_count, 6)

    print(f"\n\ncpu_count is {cpu_count}")
    print(f"task count is {task_count}")
    print(f"Running on {n_workers} cores for {task_count} years...\n\n")

    year_chunks = np.array_split(all_years, n_workers)
    tasks = [(chunk, kwargs) for chunk in year_chunks]

    print(f"Executing on {n_workers} cores...")

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as executor:
        func_wrapper_forecast = partial(func_wrapper, target_func="forecast")
        list_of_results = list(executor.map(func_wrapper_forecast, tasks))

    combined_forecast_obs = pd.concat(list_of_results, ignore_index=True)
    return combined_forecast_obs


def func_wrapper_climatology(chunk_and_args, dataset=False):
    """
    Wrapper function for parallel climatological onset calculation.
    """
    chunk, kwargs = chunk_and_args

    local_kwargs = kwargs.copy()
    local_kwargs["years_clim"] = chunk.tolist() if hasattr(chunk, "tolist") else chunk

    if dataset:
        return compute_climatological_onset_dataset(**local_kwargs)
    else:
        return compute_climatological_onset(**local_kwargs)


def parallel_climatological_onset(**kwargs):
    """
    Parallel version of compute_climatological_onset.
    """
    grid_point = kwargs.get("grid_point", False)
    years_clim = kwargs.get("years_clim")

    if not grid_point and len(years_clim) > 1:
        n_workers = min(os.cpu_count() or 1, len(years_clim), 6)
        year_chunks = np.array_split(years_clim, n_workers)

        print(f"\n\ncpu_count is {os.cpu_count()}")
        print(f"task count is {len(years_clim)}")
        print(f"Running on {n_workers} cores for {len(years_clim)} years...\n\n")

        tasks = [(chunk, kwargs) for chunk in year_chunks]

        print(f"Running in PARALLEL ({n_workers} cores) for non-grid-point task...")
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            results = list(executor.map(func_wrapper_climatology, tasks))

        climatological_onset_doy = xr.concat(results, dim="chunk_idx").mean(dim="chunk_idx")
        climatological_onset_doy = np.round(climatological_onset_doy)
        return climatological_onset_doy
    else:
        print("Running in SINGLE-PROCESS mode (Grid Point = True or single year)...")
        return compute_climatological_onset(**kwargs)


def parallel_climatological_onset_dataset(**kwargs):
    """
    Parallel version of compute_climatological_onset_dataset.
    """
    years_clim = kwargs.get("years_clim")

    n_workers = min(os.cpu_count() or 1, len(years_clim), 6)
    year_chunks = np.array_split(years_clim, n_workers)

    print(f"\n\ncpu_count is {os.cpu_count()}")
    print(f"task count is {len(years_clim)}")
    print(f"Running on {n_workers} cores for {len(years_clim)} years...\n\n")

    tasks = [(chunk, kwargs) for chunk in year_chunks]

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        func_wrapper_forecast = partial(func_wrapper_climatology, dataset=True)
        list_of_results = list(executor.map(func_wrapper_forecast, tasks))

    combined_da = xr.concat(list_of_results, dim="year")
    return combined_da


def func_wrapper_spatial(chunk_and_args, target_func=None):
    """
    Wrapper function for deterministic or ensemble metrics calculation.
    """
    chunk, kwargs = chunk_and_args

    local_kwargs = kwargs.copy()
    local_kwargs["years"] = chunk.tolist() if hasattr(chunk, "tolist") else chunk

    if target_func == "ens":
        return ens_compute_metrics_multiple_years(**local_kwargs)
    else:
        return compute_metrics_multiple_years(**local_kwargs)


def parallel_metrics_multiple_years(ens=False, **kwargs):
    """
    Parallel version of compute_metrics_multiple_years.
    """
    all_years = kwargs.get("years")

    if all_years is None:
        raise ValueError("The 'years' argument must be provided in kwargs.")

    if kwargs.get("model") != "climatology":
        all_years = available_forecast_years(
            all_years,
            model_dir=kwargs.get("model_dir"),
            file_pattern=kwargs.get("file_pattern"),
        )
        if not all_years:
            raise FileNotFoundError(
                f"No forecast files are available in {kwargs.get('model_dir')!r}"
            )
        kwargs["years"] = all_years

    cpu_count = os.cpu_count() or 1
    task_count = len(all_years)
    n_workers = min(cpu_count, task_count, 6)

    print(f"\n\ncpu_count is {cpu_count}")
    print(f"task count is {task_count}")
    print(f"Running on {n_workers} cores for {task_count} years...\n\n")

    year_chunks = np.array_split(all_years, n_workers)
    tasks = [(chunk, kwargs) for chunk in year_chunks]

    print(f"Executing on {n_workers} cores...")

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as executor:
        if ens:
            func_wrapper_ens = partial(func_wrapper_spatial, target_func="ens")
            list_of_results = list(executor.map(func_wrapper_ens, tasks))
        else:
            func_wrapper_det = partial(func_wrapper_spatial, target_func=None)
            list_of_results = list(executor.map(func_wrapper_det, tasks))

    final_metrics_df_dict = {}
    final_onset_da_dict = {}

    for metrics_part, onset_part in list_of_results:
        final_metrics_df_dict.update(metrics_part)
        final_onset_da_dict.update(onset_part)

    return final_metrics_df_dict, final_onset_da_dict
