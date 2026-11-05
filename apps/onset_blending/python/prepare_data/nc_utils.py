# ==============================================================================
# File: apps/onset_blending/python/prepare_data/nc_utils.py
# ==============================================================================
# Purpose
# Shared utilities for the single-pass NetCDF -> onset-output pipeline.
# Supports both "rainfall_forecast" and "ground_truth_rainfall" pipeline modes.
# Key conventions
# - Spatial key `id` = adm3_name string (replaces former lat_lon id).
# - The NetCDF spatial dimension is named "adm3_name" (string coordinate).
# - Filename years must agree with decoded NetCDF forecast dates.
# - Rainfall variable name is specified by spec["input"]["value_col"].
# ==============================================================================

import os
import sys
import re
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
from itertools import product
from pathlib import Path

# ==============================================================================
# Project Root Resolution
# ==============================================================================
# Import PROJECT_ROOT from config.py for consistent path resolution

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))



from ai_weather.config import PROJECT_ROOT

# Updated imports to use full absolute paths
from apps.onset_blending.python.pipelines._shared.misc import coalesce, validate_time_year
from apps.onset_blending.python.pipelines._shared.read_spec import load_spec, validate_spec
from apps.onset_blending.python.prepare_data.onset_utils import (
    read_mok_dates, read_thresholds,
    read_onset_params,
    roll_sum_na_rm_left, roll_sum_na_propagate_left,
    find_onset_precomp,
)

# ==============================================================================
# Spec helpers
# ==============================================================================
def validate_spec_single(spec):
    """Validate the YAML spec for required sections and fields."""
    def _type_check(s):
        if s["type"] == "rainfall_forecast":
            for nm in ("min_day", "max_day", "window"):
                if s.get("options", {}).get(nm) is None:
                    raise ValueError(f"Missing options.{nm}")
        if s["type"] == "ground_truth_rainfall":
            if s.get("options", {}).get("window") is None:
                raise ValueError("Missing options.window")
            if s.get("options", {}).get("cutoff_month_day") is None:
                raise ValueError("Missing options.cutoff_month_day")
        
        folder = s["input"]["nc_folder"]
        if not os.path.isabs(folder):
            folder = str(PROJECT_ROOT / folder)
        if not os.path.isdir(folder):
            raise ValueError(f"input.nc_folder does not exist: {folder}")
            
    return validate_spec(
        spec,
        required_top=["input", "dimensions", "output", "options", "type"],
        required_paths=["input.nc_folder", "input.file_regex", "input.value_col", "output.out_dir"],
        checks=[_type_check],
    )

def get_value_var(spec):
    v = spec.get("input", {}).get("value_col")
    if v:
        return str(v)
    raise ValueError("Missing spec['input']['value_col'] in YAML.")

def rename_dimensions(df, rename_map):
    """Case-insensitive rename of column names according to rename_map (old->new)."""
    if not rename_map:
        return df
    rename = {}
    for old, new in rename_map.items():
        for col in df.columns:
            if col.lower() == old.lower():
                rename[col] = new
                break
    return df.rename(columns=rename)

# ==============================================================================
# File discovery
# ==============================================================================
def list_nc_files_with_year(spec):
    """
    List NetCDF files in input.nc_folder matching file_regex, extract year.
    Returns DataFrame: nc_path (str), year (int).
    """
    folder = spec["input"]["nc_folder"]
    if not os.path.isabs(folder):
        folder = str(PROJECT_ROOT / folder)
        
    regex = spec["input"]["file_regex"]
    
    if not os.path.exists(folder):
        raise FileNotFoundError(f"input.nc_folder does not exist: {folder}")
        
    files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if re.search(regex, f)
    ]
    if not files:
        raise ValueError(f"No files matched file_regex '{regex}' in {folder}")
        
    rows = []
    for f in files:
        m = re.search(r"(19|20)\d{2}", os.path.basename(f))
        if not m:
            raise ValueError(f"Could not extract year from filename: {f}")
        rows.append({"nc_path": f, "year": int(m.group())})
        
    df = pd.DataFrame(rows)
    min_year = spec.get("options", {}).get("min_year")
    max_year = spec.get("options", {}).get("max_year")
    if min_year is not None:
        df = df[df["year"] >= int(min_year)]
    if max_year is not None:
        df = df[df["year"] <= int(max_year)]
    if df.empty:
        raise ValueError("After year filtering, no files remain.")
        
    return df.reset_index(drop=True)

def read_forecast_dates(nc_path):
    """Read only forecast initialization metadata, without loading rainfall arrays."""
    import netCDF4 as nc4

    with nc4.Dataset(nc_path) as dataset:
        info = get_nc_time(dataset)
        return nc_time_to_dates(info["values"], info["units"])


def validate_forecast_inputs(spec, issue_date=None):
    """Check source calendars before running expensive preparation stages."""
    available_dates = set()
    for row in list_nc_files_with_year(spec).itertuples(index=False):
        dates = read_forecast_dates(row.nc_path)
        frame = pd.DataFrame({"time": dates, "year": dates.year})
        validate_time_year(frame, row.nc_path, expected_year=int(row.year))
        available_dates.update(dates.date)
    if issue_date is not None and pd.Timestamp(issue_date).date() not in available_dates:
        nearby = sorted(available_dates, key=lambda value: abs((value - pd.Timestamp(issue_date).date()).days))[:5]
        raise ValueError(
            f"No forecast initialization for issue date {issue_date} in {spec['input']['nc_folder']}. "
            f"Nearby available dates: {', '.join(str(value) for value in sorted(nearby))}"
        )


# ==============================================================================
# NetCDF coordinate + time utilities
# ==============================================================================
def get_nc_adm3_names(ds):
    """
    Read the adm3_name string coordinate from a netCDF4 Dataset.
    Tries common capitalisation variants. Returns a list of strings.
    """
    import netCDF4 as nc4
    variants = ["adm3_name", "ADM3_NAME", "Adm3_Name", "adm3"]
    for v in variants:
        if v in ds.variables:
            var = ds.variables[v]
            try:
                items = var[:].tolist()
            except (NotImplementedError, TypeError):
                import netCDF4 as nc4
                items = nc4.chartostring(var[:]).tolist()
            names = []
            for item in items:
                if isinstance(item, (bytes, bytearray)):
                    names.append(item.decode("utf-8").strip())
                else:
                    names.append(str(item).strip())
            return names
    raise ValueError(
        "Could not find adm3_name variable/dimension in NetCDF file. "
        f"Available variables: {list(ds.variables.keys())}"
    )

def get_nc_time(ds):
    """
    Robust time getter from a netCDF4 Dataset.
    Returns dict: values, units, name, source
    """
    variants = ["TIME", "time", "Time", "t", "T"]
    for v in variants:
        if v in ds.variables:
            vals = np.array(ds[v][:]).flatten()
            units = getattr(ds[v], "units", None)
            return {"values": vals, "units": units, "name": v, "source": "var"}
    for v in variants:
        vl = v.lower()
        for name in ds.variables:
            if name.lower() == vl:
                vals = np.array(ds[name][:]).flatten()
                units = getattr(ds[name], "units", None)
                return {"values": vals, "units": units, "name": name, "source": "var_ci"}
    for v in variants:
        if v in ds.dimensions:
            if v in ds.variables:
                vals = np.array(ds[v][:]).flatten()
                units = getattr(ds[v], "units", None)
            else:
                vals = np.arange(ds.dimensions[v].size, dtype=float)
                units = None
            return {"values": vals, "units": units, "name": v, "source": "dim"}
    raise ValueError("Could not find time variable/dimension in NetCDF file")

def nc_time_to_dates(time_num, time_units):
    """
    Convert numeric NetCDF time values to pandas DatetimeSeries using CF units.
    """
    m = re.match(
        r"^\s*(seconds?|minutes?|hours?|days?)\s+since\s+(.+)\s*$",
        str(time_units),
        re.IGNORECASE,
    )
    if not m:
        raise ValueError(f"Unrecognized NetCDF time units: {time_units}")
    unit = m.group(1).lower().rstrip("s")
    origin_str = m.group(2).strip().replace("T", " ")
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            origin = datetime.strptime(origin_str, fmt)
            break
        except ValueError:
            continue
    else:
        raise ValueError(f"Could not parse NetCDF time origin: {origin_str}")
    mult = {"second": 1, "minute": 60, "hour": 3600, "day": 86400}[unit]
    timestamps = [origin + timedelta(seconds=float(t) * mult) for t in time_num]
    return pd.DatetimeIndex(timestamps)

# ==============================================================================
# ID helpers (adm3_name-based, replacing lat/lon)
# ==============================================================================
def ensure_id_col(df, id_col="id"):
    """
    Ensure the DataFrame has an `id` column.
    For adm3-based data the column is already named `adm3_name`; this renames
    it to `id` (or leaves it if `id` already exists).
    """
    df = df.copy()
    if id_col in df.columns:
        return df
    if "adm3_name" in df.columns:
        df[id_col] = df["adm3_name"].astype(str)
    else:
        raise ValueError(
            "Cannot create id column: neither 'id' nor 'adm3_name' found in DataFrame."
        )
    return df

def add_id_from_latlon(df, lat_col="lat", lon_col="lon", id_col="id"):
    """
    Backward-compat wrapper. Delegates to ensure_id_col().
    """
    return ensure_id_col(df, id_col=id_col)

def prep_thresholds_id(thr_df):
    """
    Normalize thresholds DataFrame to be keyed by id (adm3_name).
    """
    if thr_df is None:
        return None
    if isinstance(thr_df, (int, float, np.floating, np.integer)):
        return thr_df
    thr_df = thr_df.copy()
    thr_df.columns = thr_df.columns.str.lower()
    if "id" not in thr_df.columns:
        if "adm3_name" in thr_df.columns:
            thr_df["id"] = thr_df["adm3_name"].astype(str)
        else:
            raise ValueError(
                "Thresholds table must have either 'id' or 'adm3_name' column."
            )
    thr_df["onset_thresh"] = thr_df["onset_thresh"].astype(float)
    return thr_df[["id", "onset_thresh"]].drop_duplicates().set_index("id")

def attach_thresholds_id(df, thr_df):
    """Left-join onset_thresh from thr_df by id."""
    df = ensure_id_col(df)
    if thr_df is None:
        df["onset_thresh"] = np.nan
        return df
    if isinstance(thr_df, (int, float, np.floating, np.integer)):
        df["onset_thresh"] = float(thr_df)
        return df
    thr_idx = prep_thresholds_id(thr_df)
    if isinstance(thr_idx, (int, float)):
        df["onset_thresh"] = float(thr_idx)
        return df
    df = df.merge(thr_idx.reset_index(), on="id", how="left")
    return df

# ==============================================================================
# Cell filtering by dissemination cells list
# ==============================================================================
def filter_by_dissemination_cells(df, spec):
    """
    If spec['filter']['dissemination_cells_file'] is set, keep only rows whose
    id appears in that CSV. Handles various CSV formats and ensures robust matching.
    """
    filt = spec.get("filter") or {}
    dc_file = filt.get("dissemination_cells_file")
    if not dc_file:
        return df
        
    if not os.path.isabs(dc_file):
        dc_file = str(PROJECT_ROOT / dc_file)
        
    if not os.path.exists(dc_file):
        raise FileNotFoundError(f"dissemination_cells_file not found: {dc_file}")
        
    dc = pd.read_csv(dc_file)
    
    # Determine the ID column in the dissemination cells file
    # Priority: adm3_name > id > Unnamed: 0
    id_col = None
    if "adm3_name" in dc.columns:
        id_col = "adm3_name"
    elif "id" in dc.columns:
        id_col = "id"
    elif "Unnamed: 0" in dc.columns:
        # Check if Unnamed: 0 contains actual names or just indices
        sample_val = str(dc["Unnamed: 0"].iloc[0])
        if sample_val.isnumeric():
            # Try to find a column with actual names
            for col in dc.columns:
                if col not in ["Unnamed: 0", "lat", "lon", "adm3_pcode"]:
                    sample = str(dc[col].iloc[0])
                    if not sample.isnumeric() and len(sample) > 3:
                        id_col = col
                        break
            if id_col is None:
                id_col = "Unnamed: 0"
        else:
            id_col = "Unnamed: 0"
    else:
        raise ValueError(
            f"dissemination_cells_file must contain an 'adm3_name', 'id', or 'Unnamed: 0' column. "
            f"Found: {dc.columns.tolist()}"
        )
    
    # Create the set of valid IDs with robust string cleaning
    valid_ids = set()
    for val in dc[id_col]:
        if pd.notna(val):
            cleaned = str(val).strip()
            cleaned = ''.join(c for c in cleaned if c.isprintable())
            if cleaned:
                valid_ids.add(cleaned)
    
    # Ensure the DataFrame has an id column
    df = ensure_id_col(df)
    
    # Clean the id column in the DataFrame
    df['id'] = df['id'].astype(str).str.strip()
    df['id'] = df['id'].apply(lambda x: ''.join(c for c in x if c.isprintable()))
    
    before = len(df)
    df = df[df["id"].isin(valid_ids)]
    print(f"  dissemination_cells filter: {before} -> {len(df)} rows "
          f"({before - len(df)} removed)")
    
    if df.empty:
        raise ValueError(
            f"All rows removed by dissemination_cells filter. "
            f"This usually means the 'id' column in the NetCDF doesn't match "
            f"the '{id_col}' column in the CSV. Check for whitespace or encoding differences."
        )
    
    return df

# ==============================================================================
# Cell transform (optional regridding)
# ==============================================================================
def read_cell_transform(spec):
    """Read weights file for linear cell regridding. Returns DataFrame or None."""
    if not spec.get("options", {}).get("cell_transform_enabled", False):
        return None
    f = spec["options"].get("cell_transform_file")
    if not f:
        raise ValueError("cell_transform_enabled=True but options.cell_transform_file is empty.")
        
    if not os.path.isabs(f):
        f = str(PROJECT_ROOT / f)
        
    if not os.path.exists(f):
        raise FileNotFoundError(f"cell_transform_file not found: {f}")
        
    w = pd.read_csv(f)
    for col in ("target_id", "source_id", "weight"):
        if col not in w.columns:
            raise ValueError(f"Transform file must have: target_id, source_id, weight")
    w["weight"] = w["weight"].astype(float)
    return w

def transform_forecast_wide(df, weights_df):
    """Apply cell transform to forecast wide-by-day DataFrame."""
    if weights_df is None:
        return df
    df = ensure_id_col(df)
    day_cols = [c for c in df.columns if re.search(r"day\d+$", c)]
    if not day_cols:
        raise ValueError("No forecast day columns found to transform.")
    meta_cols = [c for c in df.columns if c not in day_cols + ["id", "adm3_name"]]
    long = df.melt(id_vars=["id"] + meta_cols, value_vars=day_cols,
                   var_name="day_col", value_name="rain")
    long = long.merge(weights_df.rename(columns={"source_id": "id"}), on="id", how="inner")
    long["rain"] = long["weight"] * long["rain"]
    trans = long.groupby(meta_cols + ["target_id", "day_col"], as_index=False)["rain"].sum()
    trans = trans.rename(columns={"target_id": "id"})
    wide = trans.pivot_table(index=meta_cols + ["id"], columns="day_col", values="rain").reset_index()
    wide.columns.name = None
    return wide

def transform_groundtruth_long(df, weights_df, value_col):
    """Apply cell transform to ground-truth long DataFrame."""
    if weights_df is None:
        return df
    df = ensure_id_col(df)
    if value_col not in df.columns:
        raise ValueError(f"Value col not found: {value_col}")
    meta_cols = [c for c in df.columns if c not in ["id", value_col, "adm3_name"]]
    x = df.merge(weights_df.rename(columns={"source_id": "id"}), on="id", how="inner")
    x[value_col] = x["weight"] * x[value_col].astype(float)
    trans = x.groupby(meta_cols + ["target_id"], as_index=False)[value_col].sum()
    trans = trans.rename(columns={"target_id": "id"})
    return trans

# ==============================================================================
# NetCDF readers
# ==============================================================================
def nc_read_forecast_wide(nc_path, var_name, dim_rename_map, spec,
                          day_dim="day", prefix=None, add_year=True):
    """
    Read a forecast NetCDF (adm3_name-indexed) into a wide-by-lead-day DataFrame.
    """
    import netCDF4 as nc4
    prefix = prefix or var_name
    ds = nc4.Dataset(nc_path)
    try:
        if var_name not in ds.variables:
            return None
        v = ds.variables[var_name]
        dim_names = list(v.dimensions)
        dim_vals = {}
        adm3_dim_name = None
        for d in dim_names:
            dl = d.lower()
            if dl in ("adm3_name", "adm3"):
                adm3_dim_name = d
                dim_vals[d] = get_nc_adm3_names(ds)
            elif dl == "time":
                tinfo = get_nc_time(ds)
                dim_vals[d] = tinfo["values"]
            else:
                if d in ds.variables:
                    dim_vals[d] = np.array(ds[d][:]).flatten()
                elif d in ds.dimensions:
                    dim_vals[d] = np.arange(ds.dimensions[d].size)
                else:
                    dim_vals[d] = np.arange(len(v))
                    
        dummy = pd.DataFrame({d: [dim_vals[d][0]] for d in dim_names})
        dummy_renamed = rename_dimensions(dummy, dim_rename_map or {})
        renamed_names = [c.lower() for c in dummy_renamed.columns]
        try:
            day_idx = renamed_names.index(day_dim.lower())
        except ValueError:
            raise ValueError(
                f"Could not identify day dimension '{day_dim}' (after rename). "
                f"Available: {renamed_names}"
            )
            
        day_vals = dim_vals[dim_names[day_idx]]
        day_vals_num = []
        for dv in day_vals:
            try:
                day_vals_num.append(int(dv))
            except (TypeError, ValueError):
                day_vals_num.append(None)
                
        min_day = spec.get("options", {}).get("min_day")
        max_day = spec.get("options", {}).get("max_day")
        keep = np.ones(len(day_vals), dtype=bool)
        if min_day is not None:
            keep &= np.array([(x is not None and x >= int(min_day)) for x in day_vals_num])
        if max_day is not None:
            keep &= np.array([(x is not None and x <= int(max_day)) for x in day_vals_num])
        if not np.any(keep):
            raise ValueError(f"Day filtering removed all columns for: {os.path.basename(nc_path)}")
            
        day_vals = [dv for dv, k in zip(day_vals, keep) if k]
        day_vals_num = [dv for dv, k in zip(day_vals_num, keep) if k]
        
        arr = np.array(v[:])
        other_idx = [i for i in range(len(dim_names)) if i != day_idx]
        perm = other_idx + [day_idx]
        arr = np.transpose(arr, perm)
        arr = arr[..., np.where(keep)[0]]
        
        other_dim_names = [dim_names[i] for i in other_idx]
        other_dim_vals = [dim_vals[dim_names[i]] for i in other_idx]
        grid_rows = list(product(*other_dim_vals))
        grid = pd.DataFrame(grid_rows, columns=other_dim_names)
        grid = rename_dimensions(grid, dim_rename_map or {})
        grid.columns = grid.columns.str.lower()
        
        adm3_col_renamed = adm3_dim_name.lower() if adm3_dim_name else None
        if adm3_col_renamed and adm3_col_renamed in grid.columns:
            grid = grid.rename(columns={adm3_col_renamed: "id"})
        if "adm3_name" in grid.columns:
            grid = grid.rename(columns={"adm3_name": "id"})
            
        n_rows = len(grid_rows)
        n_days = len(day_vals)
        mat = arr.reshape(n_rows, n_days)
        day_labels = [
            str(dv_num) if dv_num is not None else str(dv)
            for dv, dv_num in zip(day_vals, day_vals_num)
        ]
        day_col_names = [f"{prefix}_day_{lab}" for lab in day_labels]
        day_df = pd.DataFrame(mat, columns=day_col_names)
        out = pd.concat([grid.reset_index(drop=True), day_df], axis=1)
        
        if add_year and "time" in out.columns:
            tinfo = get_nc_time(ds)
            if pd.api.types.is_float_dtype(out["time"]) or pd.api.types.is_integer_dtype(out["time"]):
                dates = nc_time_to_dates(out["time"].values, tinfo["units"])
            else:
                dates = pd.to_datetime(out["time"])
            out["time"] = dates.date
            out["year"] = pd.DatetimeIndex(dates).year
        return out
    finally:
        ds.close()

def nc_read_groundtruth_long(nc_path, var_name, dim_rename_map, add_year=True):
    """
    Read a ground-truth NetCDF (adm3_name-indexed) into a long (tidy) DataFrame.
    """
    import netCDF4 as nc4
    ds = nc4.Dataset(nc_path)
    try:
        if var_name not in ds.variables:
            return None
        v = ds.variables[var_name]
        dim_names = list(v.dimensions)
        dim_vals = {}
        adm3_dim_name = None
        for d in dim_names:
            dl = d.lower()
            if dl in ("adm3_name", "adm3"):
                adm3_dim_name = d
                dim_vals[d] = get_nc_adm3_names(ds)
            elif dl == "time":
                tinfo = get_nc_time(ds)
                dim_vals[d] = tinfo["values"]
            else:
                if d in ds.variables:
                    dim_vals[d] = np.array(ds[d][:]).flatten()
                elif d in ds.dimensions:
                    dim_vals[d] = np.arange(ds.dimensions[d].size)
                else:
                    dim_vals[d] = np.arange(len(v))
                    
        arr = np.array(v[:])
        grid_rows = list(product(*[dim_vals[d] for d in dim_names]))
        grid = pd.DataFrame(grid_rows, columns=dim_names)
        grid = rename_dimensions(grid, dim_rename_map or {})
        grid.columns = grid.columns.str.lower()
        
        adm3_col_renamed = adm3_dim_name.lower() if adm3_dim_name else None
        if adm3_col_renamed and adm3_col_renamed in grid.columns:
            grid = grid.rename(columns={adm3_col_renamed: "id"})
        if "adm3_name" in grid.columns:
            grid = grid.rename(columns={"adm3_name": "id"})
            
        vcol = var_name.lower()
        grid[vcol] = arr.flatten()
        
        if add_year and "time" in grid.columns:
            tinfo = get_nc_time(ds)
            if pd.api.types.is_float_dtype(grid["time"]) or pd.api.types.is_integer_dtype(grid["time"]):
                dates = nc_time_to_dates(grid["time"].values, tinfo["units"])
                grid["time"] = dates.date
            else:
                grid["time"] = pd.to_datetime(grid["time"]).dt.date
            grid["year"] = pd.to_datetime(grid["time"]).dt.year
        return grid
    finally:
        ds.close()

# ==============================================================================
# Stage-2 onset helpers
# ==============================================================================
def order_day_cols(df, key_cols):
    """Return ordered day column names and their integer values."""
    cand = [c for c in df.columns if c not in key_cols]
    day_ints = []
    valid_cols = []
    for c in cand:
        try:
            day_ints.append(int(c))
            valid_cols.append(c)
        except (ValueError, TypeError):
            pass
    order = np.argsort(day_ints)
    return [valid_cols[i] for i in order], [day_ints[i] for i in order]

def calc_onsets_rowwise(df, day_cols, day_ints, win, params=None):
    """
    Compute per-row onset indices under three restriction rules:
    raw, clim_mok_date (after June 2), mok (after MOK date).
    """
    X = df[day_cols].values.astype(float)
    t0 = pd.to_datetime(df["time"]).values
    th = df["onset_thresh"].values.astype(float)
    yr = df["year"].values.astype(int)
    june2 = np.array([np.datetime64(f"{y}-06-02") for y in yr])
    need_clim_offset = (june2 - t0).astype("timedelta64[D]").astype(int)
    day_ints_arr = np.array(day_ints)
    need_clim = np.searchsorted(day_ints_arr, need_clim_offset - 1, side='right') + 1
    need_clim = np.where(need_clim > len(day_ints_arr), 9999, need_clim).astype(int)
    mok_dates = pd.to_datetime(df["mok_date"]).values if "mok_date" in df.columns else np.array([pd.NaT] * len(df))
    start_mok = np.ones(len(df), dtype=int)
    n = len(df)
    onset_raw = []
    onset_clim_mok_date = []
    onset_mok = []
    for i in range(n):
        s = X[i]
        wsum_all = roll_sum_na_rm_left(s, win)
        if len(s) >= 10:
            sum10 = roll_sum_na_propagate_left(s, 10)
            bad10 = (~np.isnan(sum10)) & (sum10 < 5)
            pre_bad = np.concatenate([[0], np.cumsum(bad10.astype(int))])
            last10start = len(s) - 10 + 1
        else:
            pre_bad = np.array([0])
            last10start = 0
        onset_raw.append(find_onset_precomp(s, win, th[i], wsum_all, pre_bad, last10start, start_day=0, params=params))
        onset_clim_mok_date.append(find_onset_precomp(s, win, th[i], wsum_all, pre_bad, last10start,
                                                       start_day=int(need_clim[i]), params=params))
        mk = mok_dates[i]
        if pd.isnull(mk):
            sd_mok = 1
        else:
            offset = int((mk - t0[i]).astype("timedelta64[D]").astype(int))
            sd_mok = int(np.searchsorted(day_ints_arr, offset - 1, side='right')) + 1
            if sd_mok > len(day_ints_arr):
                sd_mok = 9999
        onset_mok.append(find_onset_precomp(s, win, th[i], wsum_all, pre_bad, last10start, start_day=sd_mok, params=params))
    return {"onset_raw": onset_raw, "onset_clim_mok_date": onset_clim_mok_date, "onset_mok": onset_mok}

def process_rainfall_forecast_id(df, spec, mok_dt=None, thr_dt=None):
    """Forecast pipeline: compute ensemble onset probabilities per (id, time, year)."""
    df = df.copy()
    df = ensure_id_col(df)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"]).dt.date
    if "year" in df.columns:
        df["year"] = df["year"].astype(int)
    if "number" in df.columns:
        df["number"] = df["number"].astype(int)
    has_number = "number" in df.columns
    filter_cfg = spec.get("filter") or {}
    max_number = filter_cfg.get("max_number")
    df = attach_thresholds_id(df, thr_dt)
    if mok_dt is not None and "year" in df.columns:
        df = df.merge(mok_dt, on="year", how="left")
    else:
        df["mok_date"] = pd.NaT
    if has_number and max_number is not None:
        df = df[df["number"] <= int(max_number)]
        if df.empty:
            raise ValueError(f"After filtering number <= {max_number}, no rows remain.")
            
    wide_prefix = spec.get("input", {}).get("wide_prefix") or spec["input"]["value_col"].lower()
    day_pattern = re.compile(rf"^{re.escape(wide_prefix)}_day_(\d+)$")
    day_cols_pref = [c for c in df.columns if day_pattern.match(c)]
    if not day_cols_pref:
        raise ValueError(f"No wide day columns matched pattern '{wide_prefix}_day_<n>'")
    day_nums = {c: int(day_pattern.match(c).group(1)) for c in day_cols_pref}
    rename_map = {c: str(day_nums[c]) for c in day_cols_pref}
    df = df.rename(columns=rename_map)
    
    key_base = ["id", "time"]
    if "year" in df.columns:
        key_base.append("year")
    key_base += ["onset_thresh", "mok_date"]
    key_member = key_base + (["number"] if has_number else [])
    
    cols_to_drop = [c for c in df.columns
                    if c not in key_member
                    and not any(c == str(i) for i in range(0, 100))]
    df = df.drop(columns=cols_to_drop)
    
    day_cols, day_ints = order_day_cols(df, key_member)
    onset_params = read_onset_params(spec)
    win = onset_params.win
    min_day = int(spec["options"]["min_day"])
    max_day = int(spec["options"]["max_day"])
    keep_days = [dc for dc, di in zip(day_cols, day_ints) if min_day <= di <= max_day + win - 1]
    keep_ints = [di for di in day_ints if min_day <= di <= max_day + win - 1]
    if not keep_days:
        raise ValueError("After min_day/max_day filtering, no day columns remain.")
        
    on = calc_onsets_rowwise(df, keep_days, keep_ints, win, params=onset_params)
    df["onset_raw"] = on["onset_raw"]
    df["onset_clim_mok_date"] = on["onset_clim_mok_date"]
    df["onset_mok"] = on["onset_mok"]
    D = len(keep_ints)
    
    def prob_from_idx(idxs):
        valid = [x for x in idxs if x is not None and 1 <= x <= D]
        if not valid:
            return [0.0] * D
        counts = np.zeros(D)
        for x in valid:
            counts[int(x) - 1] += 1
        return list(counts / len(idxs))
        
    stats_agg = df.groupby(key_base).apply(
        lambda g: pd.Series({
            **{f"forecast_rain_day_{day_ints[j]}": g[keep_days[j]].mean() for j in range(len(keep_days))},
            **{f"forecast_rain_sd_day_{day_ints[j]}": g[keep_days[j]].std() for j in range(len(keep_days))},
            **{f"frac_raining_day_{day_ints[j]}": (g[keep_days[j]] > 1).mean() for j in range(len(keep_days))},
        })
    ).reset_index()
    
    prob_agg = df.groupby(key_base).apply(
        lambda g: pd.Series({
            **{f"predicted_prob_day_{keep_ints[j]}": p
               for j, p in enumerate(prob_from_idx(list(g["onset_raw"])))},
            **{f"predicted_prob_clim_mok_date_day_{keep_ints[j]}": p
               for j, p in enumerate(prob_from_idx(list(g["onset_clim_mok_date"])))},
            **{f"predicted_prob_mok_day_{keep_ints[j]}": p
               for j, p in enumerate(prob_from_idx(list(g["onset_mok"])))},
        })
    ).reset_index()
    
    wide = stats_agg.merge(prob_agg, on=key_base)
    return {"wide": wide}

def process_ground_truth_rainfall_id(df, spec, mok_dt=None, thr_dt=None, value_col=None):
    """Ground-truth pipeline: compute onset dates per (id, year)."""
    df = df.copy()
    df = ensure_id_col(df)
    df["time"] = pd.to_datetime(df["time"]).dt.date
    df["year"] = df["year"].astype(int)
    df[value_col] = df[value_col].astype(float)
    df = attach_thresholds_id(df, thr_dt)
    if mok_dt is not None:
        df = df.merge(mok_dt, on="year", how="left")
    else:
        df["mok_date"] = pd.NaT
        
    cutoff_md = spec["options"]["cutoff_month_day"]
    df["cutoff_date"] = df["year"].apply(lambda y: pd.Timestamp(f"{y}-{cutoff_md}").date())
    df["start_date"] = df["cutoff_date"]
    mask = ~df["mok_date"].isna()
    df.loc[mask, "start_date"] = df.loc[mask].apply(
        lambda r: max(r["cutoff_date"], r["mok_date"]), axis=1
    )
    df = df.sort_values(["id", "year", "time"])
    
    onset_params = read_onset_params(spec)
    win = onset_params.win
    wide_rows = []
    long_rows = []
    for (cell_id, yr), g in df.groupby(["id", "year"]):
        series = g[value_col].values
        dates = g["time"].values
        sd = g["start_date"].iloc[0]
        th = g["onset_thresh"].iloc[0]
        start_pos_arr = np.where(pd.to_datetime(dates) >= pd.Timestamp(sd))[0]
        start_pos = int(start_pos_arr[0]) + 1 if len(start_pos_arr) > 0 else 9999
        if np.all(np.isnan(series)) or len(series) < win or np.isnan(th):
            mr_idx = None
            mr_date = None
        else:
            wsum_all = roll_sum_na_rm_left(series, win)
            if len(series) >= 10:
                sum10 = roll_sum_na_propagate_left(series, 10)
                bad10 = (~np.isnan(sum10)) & (sum10 < 5)
                pre_bad = np.concatenate([[0], np.cumsum(bad10.astype(int))])
                last10start = len(series) - 10 + 1
            else:
                pre_bad = np.array([0])
                last10start = 0
            mr_idx = find_onset_precomp(series, win, th, wsum_all, pre_bad, last10start,
                            start_day=start_pos, reject_if_short_followup=True,
                            params=onset_params)
            mr_date = dates[mr_idx - 1] if mr_idx is not None else None
            
        cutoff_date = g["cutoff_date"].iloc[0]
        if mr_date is not None:
            mr_onset_day = (pd.Timestamp(mr_date) - pd.Timestamp(cutoff_date)).days
        else:
            mr_onset_day = np.nan
            
        wide_rows.append({
            "id": cell_id,
            "year": yr,
            "mr_onset_idx": mr_idx if mr_idx is not None else np.nan,
            "mr_onset_date": mr_date,
            "mr_onset_day": mr_onset_day,
            "cutoff_date": cutoff_date,
        })
        mok_date = g["mok_date"].iloc[0]
        for _, row in g.iterrows():
            long_rows.append({
                "id": cell_id,
                "time": row["time"],
                "year": yr,
                value_col: row[value_col],
                "onset_thresh": th,
                "mok_date": mok_date,
                "mr_onset_date": mr_date,
                "mr_onset_flag": row["time"] == mr_date,
            })
            
    wide = pd.DataFrame(wide_rows)
    long = pd.DataFrame(long_rows)
    return {"wide": wide, "long": long}

# ==============================================================================
# Pipeline entrypoint
# ==============================================================================
def run_single_pipeline(spec):
    """
    Main driver: process all NetCDF years, write outputs.
    Accepts a spec dict directly (from config.py).
    """
    out_dir = spec["output"]["out_dir"]
    if not os.path.isabs(out_dir):
        out_dir = str(PROJECT_ROOT / out_dir)
    os.makedirs(out_dir, exist_ok=True)
    
    # Use spec_id from spec dict if available, otherwise generate one
    spec_id = spec.get("id", "unknown")
    basename = spec["output"].get("basename", spec_id)
    var_name = get_value_var(spec)
    dim_rename_map = spec.get("dimensions", {}).get("rename") or {}
    mok_dt = read_mok_dates(spec)
    thr_dt = read_thresholds(spec)
    weights_df = read_cell_transform(spec)
    files_df = list_nc_files_with_year(spec)
    wide_all = []
    long_all = []
    
    for _, row in files_df.iterrows():
        nc_path = row["nc_path"]
        yr = row["year"]
        print(f"Processing year {yr}: {nc_path}")
        if spec["type"] == "rainfall_forecast":
            wide_prefix = spec["input"].get("wide_prefix") or var_name.lower()
            day_dim = spec["input"].get("wide_day_dim", "day")
            dt = nc_read_forecast_wide(nc_path, var_name, dim_rename_map, spec,
                                       day_dim=day_dim, prefix=wide_prefix)
            if dt is None:
                print(f"  Skipping {nc_path}: variable '{var_name}' not found.")
                continue
            validate_time_year(dt, nc_path, expected_year=int(yr))
            dt = filter_by_dissemination_cells(dt, spec)
            if weights_df is not None:
                dt = transform_forecast_wide(dt, weights_df)
            result = process_rainfall_forecast_id(dt, spec, mok_dt=mok_dt, thr_dt=thr_dt)
            wide_all.append(result["wide"])
        elif spec["type"] == "ground_truth_rainfall":
            dt = nc_read_groundtruth_long(nc_path, var_name, dim_rename_map)
            if dt is None:
                print(f"  Skipping {nc_path}: variable '{var_name}' not found.")
                continue
            dt["year"] = yr
            dt = filter_by_dissemination_cells(dt, spec)
            if weights_df is not None:
                dt = transform_groundtruth_long(dt, weights_df, var_name.lower())
            result = process_ground_truth_rainfall_id(dt, spec, mok_dt=mok_dt, thr_dt=thr_dt,
                                                       value_col=var_name.lower())
            wide_all.append(result["wide"])
            long_all.append(result["long"])
            
    if wide_all:
        wide_out = pd.concat(wide_all, ignore_index=True)
        wide_path = os.path.join(out_dir, f"{basename}_wide.pkl")
        with open(wide_path, "wb") as f:
            pickle.dump(wide_out, f)
        print(f"Wrote wide: {wide_path}")
    if long_all:
        long_out = pd.concat(long_all, ignore_index=True)
        long_path = os.path.join(out_dir, f"{basename}_long.pkl")
        with open(long_path, "wb") as f:
            pickle.dump(long_out, f)
        print(f"Wrote long: {long_path}")