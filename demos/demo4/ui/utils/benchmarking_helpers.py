# ui/utils/benchmarking_helpers.py
"""
Benchmarking helpers for ROMP (Rainfall Onset Model Performance) integration.
All paths are relative to APP_DIR (etagrometaix directory).
"""
from __future__ import annotations

import os
import sys
import re
import shutil
import subprocess
import textwrap
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import pandas as pd
import xarray as xr
import streamlit as st

# Import from config
from config import (
    APP_DIR,
    PROJECT_ROOT,
    ROMP_ROOT,
    ROMP_DEMO_ET_DIR,
    ROMP_DEMO_FIG_DIR,
    ROMP_DEMO_OUT_DIR,
    ROMP_RUNTIME_ROOT,
    STATION_CSV_PATH,
    BENCHMARK_OBS_CATALOG,
    BENCHMARK_MODEL_CATALOG,
)
from data.loader import _years_from_netcdf_dir, ensure_loaded, load_station_csv
from ui.utils.onset import detect_onset

# ----------------------------------------------------------------------
# Locate the mask file (inside APP_DIR/data/external/)
# ----------------------------------------------------------------------
def _get_mask_path() -> Path:
    """Locate the existing mask file, or create a dummy one if missing."""
    possible_paths = [
        APP_DIR / "data" / "external" / "jjas_seasonal_mask_0p25.nc",
        APP_DIR / "data" / "jjas_seasonal_mask_0p25.nc",
        APP_DIR / "data" / "mask.nc",
    ]
    for p in possible_paths:
        if p.exists():
            return p

    # If still not found, create a dummy mask (Ethiopia bounding box)
    st.warning(
        f"Mask file not found. Creating a dummy mask at {APP_DIR / 'data' / 'mask.nc'}.\n"
        "Please replace with the correct mask file if needed."
    )
    lat = np.arange(3, 15.25, 0.25)
    lon = np.arange(33, 48.25, 0.25)
    mask = xr.DataArray(
        np.ones((len(lat), len(lon))),
        coords={'lat': lat, 'lon': lon},
        dims=('lat', 'lon')
    )
    save_path = APP_DIR / "data" / "mask.nc"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    mask.to_netcdf(save_path)
    return save_path

MASK_PATH = _get_mask_path()

# ----------------------------------------------------------------------
# Benchmarking method constants with the mask path
# ----------------------------------------------------------------------
BENCHMARK_METHOD = {
    "climatology_years": tuple(range(2015, 2023)),
    "test_years": tuple(range(2019, 2023)),
    "resolution_label": "0.25° (~25 km)",
    "mask_path": str(MASK_PATH.resolve()),
    "mask_label": "JJAS seasonal 0.25° mask (dummy if missing)",
    "wet_init": 1,
    "wet_threshold": 20,
    "wet_spell": 3,
    "dry_threshold": 1,
    "dry_spell": 7,
    "dry_extent": 21,
    "start_month_day": (5, 1),
    "end_month_day": (9, 30),
    "mok": None,
}

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def _benchmark_eval_years(overlap_years):
    target_years = set(BENCHMARK_METHOD["test_years"])
    return [year for year in overlap_years if year in target_years]

def _benchmark_climatology_years(obs_years):
    target_years = set(BENCHMARK_METHOD["climatology_years"])
    return [year for year in obs_years if year in target_years]

@st.cache_data(show_spinner=False)
def get_benchmark_reference_sources():
    sources = {}
    for key, meta in BENCHMARK_OBS_CATALOG.items():
        years = _years_from_netcdf_dir(meta["dir"])
        if not years:
            continue
        clim_years = _benchmark_climatology_years(years)
        if clim_years != list(BENCHMARK_METHOD["climatology_years"]):
            continue
        sources[key] = {
            **meta,
            "dir": str(meta["dir"].resolve()),
            "years": years,
            "year_min": years[0],
            "year_max": years[-1],
        }
    return sources

@st.cache_data(show_spinner=False)
def get_benchmark_model_sources():
    models = {}
    for key, meta in BENCHMARK_MODEL_CATALOG.items():
        years = _years_from_netcdf_dir(meta["dir"])
        if not years:
            continue
        models[key] = {
            **meta,
            "dir": str(meta["dir"].resolve()),
            "years": years,
            "year_min": years[0],
            "year_max": years[-1],
        }
    return models

@st.cache_data(show_spinner=False)
def get_benchmark_presets():
    """Discover runnable benchmark presets based on available observations and models."""
    mask_path = Path(BENCHMARK_METHOD["mask_path"])
    if not mask_path.exists():
        return {}
    obs_sources = get_benchmark_reference_sources()
    model_sources = get_benchmark_model_sources()
    presets = {}
    for obs_key, obs_meta in obs_sources.items():
        obs_years = obs_meta["years"]
        clim_years = _benchmark_climatology_years(obs_years)
        if not clim_years:
            continue
        for model_key, model_meta in model_sources.items():
            overlap_years = sorted(set(obs_years) & set(model_meta["years"]))
            eval_years = _benchmark_eval_years(overlap_years)
            if eval_years != list(BENCHMARK_METHOD["test_years"]):
                continue
            category = model_meta["category"]
            preset_key = f"{category.lower()}::{model_key}::{obs_key}"
            presets[preset_key] = {
                "label": f"{model_meta['label']} vs {obs_key}",
                "description": f"ICPAC-style onset benchmark for {model_meta['model_name']} against {obs_meta['label']} at {BENCHMARK_METHOD['resolution_label']} using clean test years {eval_years[0]}–{eval_years[-1]} and climatology years {clim_years[0]}–{clim_years[-1]}.",
                "project_name": f"Dashboard {model_meta['model_name']} benchmark vs {obs_key}",
                "category": category,
                "probabilistic": model_meta["probabilistic"],
                "slow": model_meta["slow"],
                "model_key": model_key,
                "obs_key": obs_key,
                "obs_label": obs_meta["label"],
                "obs_dir": obs_meta["dir"],
                "obs_var": obs_meta["var"],
                "obs_file_pattern": obs_meta["file_pattern"],
                "ref_model_dir": obs_meta["dir"],
                "ref_model_var": obs_meta["var"],
                "ref_model_file_pattern": obs_meta["file_pattern"],
                "model_list": (model_meta["model_name"],),
                "model_dir_list": (model_meta["dir"],),
                "model_var_list": ("tp",),
                "unit_cvt_list": (None,),
                "file_pattern_list": ("{}.nc",),
                "verification_window_list": ((1, 15),),
                "tolerance_days_list": (3,),
                "max_forecast_day": 15,
                "day_bins": ((1, 5), (6, 10), (11, 15)),
                "save_csv_score": model_meta["probabilistic"],
                "plot_heatmap_bss_auc": model_meta["probabilistic"],
                "plot_reliability": False,
                "plot_panel_heatmap_error": False,
                "plot_panel_heatmap_skill": False,
                "plot_bar_bss_rpss_auc": model_meta["probabilistic"],
                "BS": model_meta["probabilistic"],
                "RPS": False,
                "AUC": model_meta["probabilistic"],
                "Reliability": False,
                "skill_score": model_meta["probabilistic"],
                "nc_mask": str(mask_path),
                "mask_label": BENCHMARK_METHOD["mask_label"],
                "eval_years": eval_years,
                "clim_years": clim_years,
            }
    return presets

def get_runtime_dir(preset_key):
    return ROMP_RUNTIME_ROOT / preset_key

def format_window_list(windows):
    return ", ".join(f"{a}-{b}" for a, b in windows)

def _python_import_available(module_name, env=None):
    try:
        proc = subprocess.run(
            [sys.executable, "-c", f"import {module_name}"],
            capture_output=True, text=True, env=env or os.environ.copy(), timeout=20
        )
        return proc.returncode == 0
    except Exception:
        return False

def ensure_regionmask_shim(runtime_dir):
    shim_dir = runtime_dir / "shim"
    shim_dir.mkdir(parents=True, exist_ok=True)
    shim_path = shim_dir / "regionmask.py"
    shim_code = textwrap.dedent("""
    import numpy as np, xarray as xr, geopandas as gpd
    from shapely.geometry import Point
    from functools import lru_cache
    import cartopy.io.shapereader as shpreader
    @lru_cache(maxsize=1)
    def _land_union():
        shp = shpreader.natural_earth(resolution="110m",category="physical",name="land")
        gdf = gpd.read_file(shp)
        return gdf.union_all() if hasattr(gdf,"union_all") else gdf.unary_union
    class _Land110:
        def mask(self,lon,lat=None):
            if lat is None: raise ValueError("lat required")
            lon_vals=np.asarray(getattr(lon,"values",lon),dtype=float)
            lat_vals=np.asarray(getattr(lat,"values",lat),dtype=float)
            lon_dim=getattr(lon,"dims",("lon",))[0]; lat_dim=getattr(lat,"dims",("lat",))[0]
            union=_land_union(); data=np.full((lat_vals.size,lon_vals.size),np.nan,dtype=float)
            for i,la in enumerate(lat_vals):
                for j,lo in enumerate(lon_vals):
                    if union.covers(Point(float(lo),float(la))): data[i,j]=0.0
            return xr.DataArray(data,dims=(lat_dim,lon_dim),coords={lat_dim:lat_vals,lon_dim:lon_vals})
    class _NaturalEarth: land_110=_Land110()
    class _DefinedRegions: natural_earth_v5_0_0=_NaturalEarth()
    defined_regions=_DefinedRegions()
    """).strip() + "\n"
    shim_path.write_text(shim_code)
    return shim_dir

def build_romp_runtime_config(preset_key):
    preset = get_benchmark_presets()[preset_key]
    runtime_dir = get_runtime_dir(preset_key)
    output_dir = runtime_dir / "output"
    figure_dir = runtime_dir / "figure"
    config_path = runtime_dir / "config_runtime.in"
    log_path = runtime_dir / "run.log"
    eval_years = preset["eval_years"]
    clim_years = preset["clim_years"]
    return {
        "project_name": preset["project_name"],
        "work_dir": str(ROMP_ROOT.resolve()),
        "pkg_dir": str((ROMP_ROOT / "momp").resolve()),
        "layout": ("model", "verification_window"),
        "model_list": preset["model_list"],
        "obs": preset["obs_key"],
        "obs_dir": preset["obs_dir"],
        "obs_file_pattern": (preset["obs_file_pattern"],),
        "obs_var": preset["obs_var"],
        "obs_unit_cvt": None,
        "ref_model": "climatology",
        "ref_model_dir": preset["ref_model_dir"],
        "ref_model_file_pattern": preset["ref_model_file_pattern"],
        "ref_model_var": preset["ref_model_var"],
        "ref_model_unit_cvt": None,
        "model_dir_list": preset["model_dir_list"],
        "model_var_list": preset["model_var_list"],
        "unit_cvt_list": preset["unit_cvt_list"],
        "file_pattern_list": preset["file_pattern_list"],
        "region": "Ethiopia",
        "nc_mask": preset["nc_mask"],
        "shpfile_dir": None,
        "polygon": False,
        "wet_init": BENCHMARK_METHOD["wet_init"],
        "wet_threshold": BENCHMARK_METHOD["wet_threshold"],
        "wet_spell": BENCHMARK_METHOD["wet_spell"],
        "dry_threshold": BENCHMARK_METHOD["dry_threshold"],
        "dry_spell": BENCHMARK_METHOD["dry_spell"],
        "dry_extent": BENCHMARK_METHOD["dry_extent"],
        "thresh_file": None,
        "thresh_var": None,
        "onset_percentage_threshold": 0.5,
        "start_date": (eval_years[0], *BENCHMARK_METHOD["start_month_day"]),
        "end_date": (eval_years[-1], *BENCHMARK_METHOD["end_month_day"]),
        "start_year_clim": clim_years[0],
        "end_year_clim": clim_years[-1],
        "init_days": (1, 7, 13, 19, 25),
        "init_type": "monthly",
        "date_filter_year": 2024,
        "verification_window_list": preset["verification_window_list"],
        "tolerance_days_list": preset["tolerance_days_list"],
        "max_forecast_day": preset["max_forecast_day"],
        "day_bins": preset["day_bins"],
        "FAR": True,
        "MAE": True,
        "MR": True,
        "probabilistic": preset["probabilistic"],
        "members": "All",
        "BS": preset["BS"],
        "RPS": preset["RPS"],
        "AUC": preset["AUC"],
        "Reliability": preset["Reliability"],
        "skill_score": preset["skill_score"],
        "dir_out": str(output_dir.resolve()),
        "dir_fig": str(figure_dir.resolve()),
        "save_fig": True,
        "save_nc_spatial_far_mr_mae": True,
        "save_csv_score": preset["save_csv_score"],
        "save_nc_climatology": True,
        "plot_spatial_far_mr_mae": True,
        "plot_heatmap_bss_auc": preset["plot_heatmap_bss_auc"],
        "plot_reliability": preset["plot_reliability"],
        "plot_climatology_onset": True,
        "plot_panel_heatmap_error": preset["plot_panel_heatmap_error"],
        "plot_panel_heatmap_skill": preset["plot_panel_heatmap_skill"],
        "plot_bar_bss_rpss_auc": preset["plot_bar_bss_rpss_auc"],
        "show_plot": False,
        "show_panel": False,
        "parallel": False,
        "debug": False,
        "mok": BENCHMARK_METHOD["mok"],
        "years": tuple(eval_years),
        "years_clim": tuple(clim_years),
        "fallback_date": None,
        "_runtime_dir": runtime_dir,
        "_config_path": config_path,
        "_log_path": log_path,
    }

def write_romp_runtime_config(cfg):
    runtime_dir = cfg["_runtime_dir"]
    runtime_dir.mkdir(parents=True, exist_ok=True)
    (runtime_dir / "output").mkdir(exist_ok=True)
    (runtime_dir / "figure").mkdir(exist_ok=True)
    lines = [f"{k} = {repr(v)}" for k, v in cfg.items() if not k.startswith("_")]
    cfg["_config_path"].write_text("\n".join(lines) + "\n")
    return cfg["_config_path"]

def run_romp_benchmark(preset_key):
    cfg = build_romp_runtime_config(preset_key)
    runtime_dir = cfg["_runtime_dir"]
    if runtime_dir.exists():
        shutil.rmtree(runtime_dir)
    cfg = build_romp_runtime_config(preset_key)
    config_path = write_romp_runtime_config(cfg)
    env = os.environ.copy()
    existing_pythonpath = [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p]
    python_paths = [str(ROMP_ROOT.resolve()), *existing_pythonpath]
    shim_used = False
    env["PYTHONPATH"] = os.pathsep.join(python_paths)
    if not _python_import_available("regionmask", env=env):
        shim_dir = ensure_regionmask_shim(cfg["_runtime_dir"])
        python_paths = [str(shim_dir.resolve()), *python_paths]
        env["PYTHONPATH"] = os.pathsep.join(python_paths)
        shim_used = True
    dep_status = {dep: _python_import_available(dep, env=env) for dep in ["regionmask", "geopandas", "cartopy"]}
    env["MPLCONFIGDIR"] = str((cfg["_runtime_dir"] / "mplcache").resolve())
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "momp.driver", "-p", str(config_path)],
            cwd=str(ROMP_ROOT.resolve()), env=env, capture_output=True, text=True, timeout=900
        )
        ok = proc.returncode == 0
        cfg["_log_path"].write_text((proc.stdout or "") + (("\n" + proc.stderr) if proc.stderr else ""))
        return {
            "preset_key": preset_key,
            "returncode": proc.returncode,
            "runtime_dir": str(runtime_dir),
            "config_path": str(config_path),
            "log_path": str(cfg["_log_path"]),
            "stdout": proc.stdout or "",
            "stderr": proc.stderr or "",
            "ok": ok,
            "dep_status": dep_status,
            "regionmask_shim_used": shim_used,
        }
    except subprocess.TimeoutExpired as e:
        log = (e.stdout or "") + (e.stderr or "") + "\nROMP timed out after 900s."
        cfg["_log_path"].write_text(log)
        return {
            "preset_key": preset_key,
            "returncode": 124,
            "runtime_dir": str(runtime_dir),
            "config_path": str(config_path),
            "log_path": str(cfg["_log_path"]),
            "stdout": e.stdout or "",
            "stderr": log,
            "ok": False,
            "dep_status": dep_status,
            "regionmask_shim_used": shim_used,
        }
    except Exception as e:
        cfg["_log_path"].write_text(str(e))
        return {
            "preset_key": preset_key,
            "returncode": 1,
            "runtime_dir": str(runtime_dir),
            "config_path": str(config_path),
            "log_path": str(cfg["_log_path"]),
            "stdout": "",
            "stderr": str(e),
            "ok": False,
            "dep_status": dep_status,
            "regionmask_shim_used": shim_used,
        }

def runtime_result_inventory(preset_key):
    runtime_dir = get_runtime_dir(preset_key)
    output_dir = runtime_dir / "output"
    figure_dir = runtime_dir / "figure"
    figures = sorted(str(p.resolve()) for p in figure_dir.glob("*.png")) if figure_dir.exists() else []
    outputs = sorted(str(p.resolve()) for p in output_dir.glob("*.nc")) if output_dir.exists() else []
    return {"figures": figures, "outputs": outputs}

def parse_romp_summary(log_text):
    if not log_text:
        return []
    lines = [l.strip() for l in log_text.splitlines()
             if l.strip().startswith("Summary stats:") or l.strip().startswith("Processing year") or "saved to:" in l]
    return lines[-18:]

def summarize_benchmark_dataset(nc_path):
    ds = xr.open_dataset(Path(nc_path))
    variables = list(ds.data_vars)
    summary = {"path": nc_path, "variables": variables, "dims": {k: int(v) for k, v in ds.sizes.items()}, "means": {}}
    for var in variables:
        arr = np.asarray(ds[var].values, dtype=float)
        if np.isfinite(arr).any():
            summary["means"][var] = float(np.nanmean(arr))
    return summary

def _metric_percent(arr):
    val = float(np.nanmean(np.asarray(arr, dtype=float)))
    return val * 100.0 if val <= 1.5 else val

def summarize_spatial_metrics_output(nc_path):
    ds = xr.open_dataset(Path(nc_path))
    if not {"mean_mae", "false_alarm_rate", "miss_rate"}.issubset(ds.data_vars):
        return None
    mae = ds["mean_mae"]
    far = ds["false_alarm_rate"]
    mr = ds["miss_rate"]
    far_pct = far * 100.0 if float(far.max(skipna=True)) <= 1.5 else far
    mr_pct = mr * 100.0 if float(mr.max(skipna=True)) <= 1.5 else mr
    candidate_mask = (mae <= 7.0) & (far_pct <= 35.0) & (mr_pct <= 60.0)
    model_name = Path(nc_path).stem.replace("spatial_metrics_", "").replace("_1-15", "")
    return {
        "model": model_name,
        "mean_mae_days": float(mae.mean(skipna=True)),
        "mean_far_pct": _metric_percent(far),
        "mean_miss_rate_pct": _metric_percent(mr),
        "candidate_cells": int(candidate_mask.fillna(False).astype(int).sum().item()),
        "grid_cells": int(mae.count().item()),
        "nc_path": str(Path(nc_path).resolve()),
    }

def identify_skill_locations(nc_path, top_n=12):
    ds = xr.open_dataset(Path(nc_path))
    if not {"mean_mae", "false_alarm_rate", "miss_rate"}.issubset(ds.data_vars):
        return pd.DataFrame()
    lat_name = "lat" if "lat" in ds.coords else "latitude"
    lon_name = "lon" if "lon" in ds.coords else "longitude"
    mae = ds["mean_mae"]
    far = ds["false_alarm_rate"] * 100.0 if float(ds["false_alarm_rate"].max(skipna=True)) <= 1.5 else ds["false_alarm_rate"]
    mr = ds["miss_rate"] * 100.0 if float(ds["miss_rate"].max(skipna=True)) <= 1.5 else ds["miss_rate"]
    frame = pd.DataFrame({
        "lat": mae[lat_name].values.repeat(mae[lon_name].size),
        "lon": np.tile(mae[lon_name].values, mae[lat_name].size),
        "mean_mae_days": np.asarray(mae.values).reshape(-1),
        "false_alarm_rate_pct": np.asarray(far.values).reshape(-1),
        "miss_rate_pct": np.asarray(mr.values).reshape(-1),
    }).dropna()
    if frame.empty:
        return frame
    frame["composite_skill"] = frame["mean_mae_days"] + 0.08 * frame["false_alarm_rate_pct"] + 0.06 * frame["miss_rate_pct"]
    frame = frame[
        (frame["mean_mae_days"] <= 7.0)
        & (frame["false_alarm_rate_pct"] <= 35.0)
        & (frame["miss_rate_pct"] <= 60.0)
    ].sort_values(["composite_skill", "mean_mae_days", "false_alarm_rate_pct"]).head(top_n)
    return frame.reset_index(drop=True)

def runtime_benchmark_scoreboard():
    rows = []
    for nc_path in sorted(ROMP_RUNTIME_ROOT.glob("**/output/spatial_metrics_*.nc")):
        if "climatology" in nc_path.name.lower():
            continue
        summary = summarize_spatial_metrics_output(str(nc_path.resolve()))
        if summary:
            rows.append(summary)
    if not rows:
        return pd.DataFrame()
    score_df = pd.DataFrame(rows)
    score_df["composite_rank_score"] = (
        score_df["mean_mae_days"] +
        0.08 * score_df["mean_far_pct"] +
        0.06 * score_df["mean_miss_rate_pct"] -
        0.01 * score_df["candidate_cells"]
    )
    return score_df.sort_values(["composite_rank_score", "mean_mae_days"]).reset_index(drop=True)

def compute_point_onset_history(dataset_key, lat0, lon0, onset_params, years):
    from data.loader import COORDS_BY_KEY, FILES_BY_KEY, extract_point_series
    la, lo, ti = COORDS_BY_KEY[dataset_key]
    point_df = extract_point_series(dataset_key, tuple(FILES_BY_KEY[dataset_key]), la, lo, ti, lat0, lon0)
    history = []
    for year in years:
        df_year = point_df[point_df["year"] == year]
        if df_year.empty:
            continue
        od, _ = detect_onset(
            df_year["time"], df_year["rain"].values,
            onset_params["wet_day_mm"], onset_params["accum_days"], onset_params["accum_mm"],
            onset_params["dry_spell_days"], onset_params["lookahead_days"],
            onset_params["start_month"], onset_params["start_day"],
            onset_params["end_month"], onset_params["end_day"]
        )
        from config import get_operational_season_context
        season = get_operational_season_context()
        seas_total = float(df_year[df_year["month"].isin(season["months"])]["rain"].sum())
        history.append({"year": year, "onset_doy": int(pd.Timestamp(od).dayofyear) if od else np.nan, "season_total_mm": seas_total})
    return history

def history_rows_to_df(history_rows):
    expected_cols = ["year", "onset_doy", "season_total_mm"]
    if not history_rows:
        return pd.DataFrame(columns=expected_cols)
    df = pd.DataFrame(history_rows)
    for col in expected_cols:
        if col not in df.columns:
            df[col] = np.nan
    return df

def build_benchmark_observation_crosscheck(onset_params, lat0, lon0, emi_meta, emi_ts):
    rows = []
    for dataset_key in ["ENACTS", "CHIRPS", "IMERG"]:
        directory = BENCHMARK_OBS_CATALOG[dataset_key]["dir"]
        if not directory.exists():
            continue
        try:
            ensure_loaded(dataset_key)
            years = _years_from_netcdf_dir(directory)
            history = compute_point_onset_history(dataset_key, lat0, lon0, onset_params, years)
        except Exception:
            continue
        hist_df = history_rows_to_df(history).dropna(subset=["onset_doy"])
        if hist_df.empty:
            continue
        latest = hist_df.sort_values("year").iloc[-1]
        rows.append({
            "Source": dataset_key,
            "Coverage": f"{years[0]}-{years[-1]}",
            "Median onset DOY": int(hist_df["onset_doy"].median()),
            "Latest year": int(latest["year"]),
            "Latest onset DOY": int(latest["onset_doy"]),
            "Latest season total (mm)": round(float(latest["season_total_mm"]), 1),
        })
    if (emi_meta is None or emi_ts is None) and STATION_CSV_PATH.exists():
        try:
            emi_meta, emi_ts = load_station_csv(STATION_CSV_PATH)
        except Exception:
            emi_meta, emi_ts = None, None
    years = list(range(int(emi_ts.index.min().year), int(emi_ts.index.max().year) + 1)) if emi_ts is not None else []
    station_name, station_history = compute_station_onset_history(lat0, lon0, onset_params, emi_meta, emi_ts, years)
    station_df = history_rows_to_df(station_history).dropna(subset=["onset_doy"])
    if station_name and not station_df.empty:
        latest = station_df.sort_values("year").iloc[-1]
        rows.append({
            "Source": f"EMI station · {station_name}",
            "Coverage": f"{years[0]}-{years[-1]}",
            "Median onset DOY": int(station_df["onset_doy"].median()),
            "Latest year": int(latest["year"]),
            "Latest onset DOY": int(latest["onset_doy"]),
            "Latest season total (mm)": round(float(latest["season_total_mm"]), 1),
        })
    return pd.DataFrame(rows)

def compute_station_onset_history(lat0, lon0, onset_params, emi_meta, emi_ts, years):
    if emi_meta is None or emi_ts is None or emi_meta.empty:
        return None, []
    dist = np.sqrt((emi_meta["lat"] - lat0) ** 2 + (emi_meta["lon"] - lon0) ** 2)
    station_name = dist.idxmin()
    series = emi_ts[station_name].dropna()
    history = []
    for year in years:
        df_year = series.loc[f"{year}-01-01":f"{year}-12-31"].rename("rain").to_frame().reset_index()
        if df_year.empty:
            continue
        df_year.columns = ["time", "rain"]
        od, _ = detect_onset(
            df_year["time"], df_year["rain"].values,
            onset_params["wet_day_mm"], onset_params["accum_days"], onset_params["accum_mm"],
            onset_params["dry_spell_days"], onset_params["lookahead_days"],
            onset_params["start_month"], onset_params["start_day"],
            onset_params["end_month"], onset_params["end_day"]
        )
        from config import get_operational_season_context
        season = get_operational_season_context()
        seas_total = float(df_year[pd.to_datetime(df_year["time"]).dt.month.isin(season["months"])]["rain"].sum())
        history.append({"year": year, "onset_doy": int(pd.Timestamp(od).dayofyear) if od else np.nan, "season_total_mm": seas_total})
    return station_name, history

# ----------------------------------------------------------------------
# UI helper functions (used by benchmarking page)
# ----------------------------------------------------------------------

def info_box(text: str, kind: str = "") -> None:
    """Render an info box with optional styling."""
    st.markdown(f'<div class="ibox {kind}">{text}</div>', unsafe_allow_html=True)

def section_header(text: str, icon: str = "", accent: str = "") -> None:
    """Render a section header with an optional icon and accent color."""
    ico = f"<span style='font-size:.95rem'>{icon}</span> " if icon else ""
    cls = f"sec-hd {accent}".strip()
    st.markdown(f'<div class="{cls}"><span class="sec-hd-accent"></span>{ico}{text}</div>', unsafe_allow_html=True)

def kpi(label: str, value: str, unit: str = "", color: str = "") -> str:
    """Return HTML for a KPI card."""
    cls = f"kpi {color}".strip()
    u = f"<div class='kpi-unit'>{unit}</div>" if unit else ""
    return f"<div class='{cls}'><div class='kpi-label'>{label}</div><div class='kpi-value'>{value}</div>{u}</div>"

def render_source_status(title: str, status: str, detail: str, kind: str = "sky") -> None:
    """Render a status box for a data source."""
    info_box(f"<b>{title}</b><br>{status}<br><span style='font-size:.74rem'>{detail}</span>", kind)

def _render_benchmark_figure_grid(fig_paths, climatology_years=None, height_px=360) -> None:
    """Render a grid of benchmark figures (cards)."""
    if not fig_paths:
        info_box("No figures available.", "amber")
        return
    n_cols = 2 if len(fig_paths) >= 2 else 1
    for i in range(0, len(fig_paths), n_cols):
        cols = st.columns(n_cols, gap="large")
        for j, col in enumerate(cols):
            idx = i + j
            if idx >= len(fig_paths):
                continue
            fpath = fig_paths[idx]
            fname = Path(fpath).name
            clean_name = fname.replace(".png", "").replace("_", " ").title()
            with col:
                st.markdown('<div class="bench-card">', unsafe_allow_html=True)
                st.image(str(fpath), width='stretch')
                caption = f'<div class="bench-card-caption">{clean_name}<br><small>{Path(fpath).parent.name}</small>'
                if climatology_years:
                    caption += f'<br><span style="font-size:0.7rem;">📅 Climatology: {climatology_years[0]}–{climatology_years[-1]}</span>'
                caption += '</div>'
                st.markdown(caption, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)