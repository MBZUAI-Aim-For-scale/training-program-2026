import numpy as np
import pandas as pd
import xarray as xr


def create_spatial_far_mr_mae(metrics_df_dict, onset_da_dict):
    """
    Create spatial maps of False Alarm Rate (FAR), Miss Rate (MR), yearly MAE,
    and mean MAE across evaluated years.

    Fixes applied:
    1. Corrected meteorological FAR formula: sum(FP) / sum(TP + FP).
    2. Retains np.nan for grid points without valid forecasts or onsets instead of defaulting to 0.
    3. Handles float precision matching on lat/lon to prevent empty query matches.
    4. Vectorized with Pandas grouping for fast execution without nested loops.
    """
    if not metrics_df_dict or not onset_da_dict:
        return {}

    first_year = list(onset_da_dict.keys())[0]
    ref_da = onset_da_dict[first_year]
    lats = ref_da.lat.values
    lons = ref_da.lon.values

    print("Creating spatial FAR, Miss Rate, yearly MAE, and mean MAE maps...")
    print(f"Grid dimensions: {len(lats)} lats x {len(lons)} lons")
    print(f"Years: {list(metrics_df_dict.keys())}")

    # Combine all year dataframes with an added 'year' column
    all_dfs = []
    for yr, df in metrics_df_dict.items():
        if df is not None and not df.empty:
            df_copy = df.copy()
            df_copy["year"] = yr
            all_dfs.append(df_copy)

    if not all_dfs:
        raise ValueError("metrics_df_dict contains no valid DataFrames.")

    combined_df = pd.concat(all_dfs, ignore_index=True)

    # Standardize column key names dynamically across variations
    col_tp = next((c for c in ["true_positive", "TP", "tp"] if c in combined_df.columns), None)
    col_fp = next((c for c in ["false_positive", "FP", "fp"] if c in combined_df.columns), None)
    col_fn = next((c for c in ["false_negative", "FN", "fn"] if c in combined_df.columns), None)
    col_onset = next((c for c in ["num_onset", "num_onsets", "onsets"] if c in combined_df.columns), None)
    col_mae = next((c for c in ["mae_combined", "mae", "mae_tp_only"] if c in combined_df.columns), None)

    # Clean coordinate rounding to prevent floating-point mismatch
    combined_df["lat"] = np.round(combined_df["lat"].astype(np.float64), 4)
    combined_df["lon"] = np.round(combined_df["lon"].astype(np.float64), 4)
    lats_rounded = np.round(lats.astype(np.float64), 4)
    lons_rounded = np.round(lons.astype(np.float64), 4)

    # Group spatially across all years
    grouped = combined_df.groupby(["lat", "lon"])

    tp_sum = grouped[col_tp].sum() if col_tp else 0
    fp_sum = grouped[col_fp].sum() if col_fp else 0
    fn_sum = grouped[col_fn].sum() if col_fn else 0

    if col_onset:
        onset_sum = grouped[col_onset].sum()
    else:
        onset_sum = tp_sum + fn_sum

    mae_mean = grouped[col_mae].mean() if col_mae else np.nan

    # Calculate contingency metrics; retain np.nan when denominators are zero
    total_forecasts = tp_sum + fp_sum
    far_series = np.where(total_forecasts > 0, fp_sum / total_forecasts, np.nan)
    mr_series = np.where(onset_sum > 0, fn_sum / onset_sum, np.nan)

    # Reindex against target grid template to ensure exact alignment
    grid_index = pd.MultiIndex.from_product([lats_rounded, lons_rounded], names=["lat", "lon"])
    
    stats_df = pd.DataFrame(
        {
            "false_alarm_rate": far_series,
            "miss_rate": mr_series,
            "mean_mae": mae_mean,
        },
        index=tp_sum.index if isinstance(tp_sum, pd.Series) else grouped.size().index
    )

    stats_grid = stats_df.reindex(grid_index)

    # Reshape 1D series back to 2D grid arrays (lats, lons)
    far_map = stats_grid["false_alarm_rate"].values.reshape(len(lats), len(lons))
    mr_map = stats_grid["miss_rate"].values.reshape(len(lats), len(lons))
    mean_mae_map = stats_grid["mean_mae"].values.reshape(len(lats), len(lons))

    # Process yearly MAE maps
    yearly_mae_maps = {}
    for yr in metrics_df_dict.keys():
        yr_df = combined_df[combined_df["year"] == yr]
        if not yr_df.empty and col_mae:
            yr_grouped = yr_df.groupby(["lat", "lon"])[col_mae].mean()
            yr_grid = yr_grouped.reindex(grid_index)
            yearly_mae_maps[yr] = yr_grid.values.reshape(len(lats), len(lons))
        else:
            yearly_mae_maps[yr] = np.full((len(lats), len(lons)), np.nan)

    # Construct final DataArrays
    spatial_metrics = {
        "false_alarm_rate": xr.DataArray(
            far_map,
            coords=[("lat", lats), ("lon", lons)],
            name="false_alarm_rate",
            attrs={"description": "False Alarm Rate = sum(FP) / sum(TP + FP) across all valid years"},
        ),
        "miss_rate": xr.DataArray(
            mr_map,
            coords=[("lat", lats), ("lon", lons)],
            name="miss_rate",
            attrs={"description": "Miss Rate = sum(FN) / sum(TP + FN) across all valid years"},
        ),
        "mean_mae": xr.DataArray(
            mean_mae_map,
            coords=[("lat", lats), ("lon", lons)],
            name="mean_mae",
            attrs={"description": "Mean MAE across all valid years (omitting NaN values)"},
        ),
    }

    for yr, mae_map in yearly_mae_maps.items():
        spatial_metrics[f"mae_{yr}"] = xr.DataArray(
            mae_map,
            coords=[("lat", lats), ("lon", lons)],
            name=f"mae_{yr}",
            attrs={"description": f"Mean Absolute Error for year {yr}"},
        )

    return spatial_metrics
