"""Per-cell probabilistic scores using the same estimators as pooled ROMP scores."""

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from momp.stats.score import (
    calculate_auc,
    calculate_paired_brier,
    calculate_paired_rps,
    match_samples,
)


def _cell_groups(frame):
    if frame is None or frame.empty:
        return None
    # Share the probability columns; normalize only the small coordinate columns.
    # Fetch groups lazily rather than retaining another full copy of every cell.
    frame = frame.copy(deep=False)
    for column in ("lat", "lon"):
        frame[column] = pd.to_numeric(frame[column], errors="raise").round(8)
        if not np.isfinite(frame[column]).all():
            raise ValueError("Spatial probability scores require finite coordinates")
    return frame.groupby(["lat", "lon"], sort=True)


def _skill(forecast, reference):
    return 1 - forecast / reference if np.isfinite(reference) and reference != 0 else np.nan


def calculate_spatial_probability_scores(results):
    """Score each cell against its paired reference, preserving complete RPS bins.

    These are distributions over cells, not a decomposition of pooled AUC or
    skill: averaging per-cell scores generally differs from pooling all cases.
    Undefined reference denominators and single-class AUC remain NaN.
    """
    forecasts = _cell_groups(results.get("forecast_obs_df"))
    if forecasts is None:
        return xr.Dataset()
    references = _cell_groups(results.get("reference_obs_df"))
    rps_forecasts = _cell_groups(results.get("rps_forecast_obs_df"))
    rps_references = _cell_groups(results.get("rps_reference_obs_df"))
    columns = ("fair_brier_score", "reference_fair_brier_score", "fair_brier_skill_score",
               "fair_rps", "reference_fair_rps", "fair_rps_skill_score", "auc", "reference_auc")
    rows = []
    print(f"Computing spatial probabilistic scores for {forecasts.ngroups} cells...", flush=True)
    for index, cell in enumerate(forecasts.indices, 1):
        forecast = forecasts.get_group(cell)
        row = {column: np.nan for column in columns}
        row.update(lat=cell[0], lon=cell[1])
        reference = (references.get_group(cell) if references is not None
                     and cell in references.indices else None)
        # Scorers print pooled diagnostics; retain a single grid-level log below.
        # This runs inside the benchmark subprocess, never the dashboard thread.
        with redirect_stdout(StringIO()):
            if reference is not None:
                forecast, reference = match_samples(forecast, reference)
                if results.get("BS") is not None:
                    bs, bs_ref = calculate_paired_brier(forecast, reference)
                    row["fair_brier_score"] = bs["fair_brier_score"]
                    row["reference_fair_brier_score"] = bs_ref["fair_brier_score"]
                    if results.get("skill_results") is not None:
                        row["fair_brier_skill_score"] = _skill(
                            row["fair_brier_score"], row["reference_fair_brier_score"])
                if results.get("AUC") is not None:
                    row["auc"] = calculate_auc(forecast)["auc"]
                    row["reference_auc"] = calculate_auc(reference)["auc"]
            if (results.get("RPS") is not None and rps_forecasts is not None
                    and rps_references is not None and cell in rps_forecasts.indices
                    and cell in rps_references.indices):
                rps, rps_ref = calculate_paired_rps(
                    rps_forecasts.get_group(cell), rps_references.get_group(cell))
                row["fair_rps"] = rps["fair_rps"]
                row["reference_fair_rps"] = rps_ref["fair_rps"]
                if results.get("skill_results") is not None:
                    row["fair_rps_skill_score"] = _skill(row["fair_rps"], row["reference_fair_rps"])
        rows.append(row)
        if index % 50 == 0:
            print(f"Spatial probabilistic scores: {index}/{forecasts.ngroups} cells", flush=True)
    if not rows:
        return xr.Dataset()
    dataset = pd.DataFrame(rows).set_index(["lat", "lon"]).to_xarray()
    for name in columns:
        dataset[name].attrs["units"] = "1"
    dataset.attrs.update(
        score_definition="paired_fair_rps_v2",
        aggregation="scores computed independently per grid cell over paired forecast cases",
        rps_category_order="chronological_complete_partition",
        pooled_scores="Overall CSV scores pool cases; spatial score means are different statistics.",
    )
    return dataset


def save_spatial_probability_scores(results, *, model, verification_window, dir_out, **kwargs):
    dataset = calculate_spatial_probability_scores(results)
    if not dataset.data_vars:
        return None
    start, end = verification_window
    dataset.attrs.update(model=str(model), region=str(kwargs.get("region", "Ethiopia")),
                         verification_window=f"{start}-{end}",
                         evaluation_years=str(kwargs.get("years", "")),
                         reference_years=str(kwargs.get("years_clim", "")))
    destination = Path(dir_out) / f"probabilistic_metrics_{model}_{start}-{end}.nc"
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Write a complete file before replacing a prior run's grid.
    temporary = destination.with_suffix(".nc.tmp")
    try:
        dataset.to_netcdf(temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Spatial probabilistic scores saved to '{destination}' ({len(dataset.lat)} x {len(dataset.lon)} cells)")
    return destination
