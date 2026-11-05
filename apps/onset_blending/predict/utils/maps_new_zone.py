#!/usr/bin/env python3
"""
file: etagrometaix/apps/onset_forecast/predict/utils/maps_new_zone.py 
Generate forecast maps using adm3-level polygon geometries.
Each dissemination cell is a named woreda (adm3_name) rendered as a filled polygon.
"""
from datetime import timedelta
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib.patches import Patch
from matplotlib.colors import BoundaryNorm, ListedColormap
import matplotlib.colors as mcolors
import geopandas as gpd
from pathlib import Path
import logging
import xarray as xr

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(message)s")


# ==============================================================================
# Boundary loader
# ==============================================================================
def load_boundary(base=None, use_cartopy=False, region='Ethiopia',
                  resolution='10m', category='cultural',
                  name='admin_0_countries', shapefile_path=None):
    """Load country boundary. Uses local shapefile if provided, else cartopy."""
    if shapefile_path and Path(shapefile_path).exists():
        gdf = gpd.read_file(shapefile_path).to_crs("EPSG:4326")
        # Dissolve to get single country boundary
        if len(gdf) > 1:
            gdf = gdf.dissolve().reset_index()
        return gdf
    
    if use_cartopy:
        import cartopy.io.shapereader as shpreader
        ne_path = shpreader.natural_earth(
            resolution=resolution, category=category, name=name
        )
        geom = None
        for country in shpreader.Reader(ne_path).records():
            if country.attributes.get('NAME') == region:
                geom = country.geometry
                break
        if geom is None:
            raise ValueError(f"Region '{region}' not found in cartopy Natural Earth.")
        gdf = gpd.GeoDataFrame(geometry=[geom], crs="EPSG:4326")
    else:
        # Fallback: try to find a country boundary shapefile
        if base:
            shp_path = base / "data" / "shapefile" / "Country_Boundary.shp"
            if shp_path.exists():
                gdf = gpd.read_file(shp_path).to_crs("EPSG:4326")
                return gdf
        raise ValueError("No boundary source available. Provide shapefile_path or use_cartopy=True.")
    return gdf


# ==============================================================================
# Period label helper
# ==============================================================================
def _period_labels(t):
    fmt = '%m/%d/%Y'
    def date(offset): return (t + timedelta(days=offset)).strftime(fmt)
    return {
        'week1': f"{date(1)} - {date(7)}",
        'week2': f"{date(8)} - {date(14)}",
        'week3': f"{date(15)} - {date(21)}",
        'week4': f"{date(22)} - {date(28)}",
        'later': f"{date(29)}+",
    }


# ==============================================================================
# Woreda geometry loader
# ==============================================================================
def load_woreda_geometries(shapefile_path):
    """Load the adm3-level woreda shapefile."""
    gdf = gpd.read_file(shapefile_path).to_crs("EPSG:4326")
    
    # Find the name column
    name_col = None
    for col in ["adm3_name", "ADM3_EN", "NAME_3", "Woreda", "name"]:
        if col in gdf.columns:
            name_col = col
            break
    
    if name_col is None:
        raise ValueError(
            f"Woreda shapefile must contain an 'adm3_name' column. "
            f"Found: {gdf.columns.tolist()}"
        )
    
    # Rename to adm3_name if needed
    if name_col != "adm3_name":
        gdf = gdf.rename(columns={name_col: "adm3_name"})
    
    return gdf[["adm3_name", "geometry"]].copy()


# ==============================================================================
# ADM2 boundary loader
# ==============================================================================
def load_adm2_geometries(shapefile_path):
    """Load the adm2-level shapefile for boundary overlays and name labels."""
    gdf = gpd.read_file(shapefile_path).to_crs("EPSG:4326")
    
    # Find adm2 column
    adm2_col = None
    for col in ["adm2_name", "ADM2_EN", "NAME_2", "Zone", "ZONE_NAME"]:
        if col in gdf.columns:
            adm2_col = col
            break
    
    if adm2_col is None:
        logging.warning("No adm2_name column found. Skipping ADM2 overlays.")
        return None
    
    # Rename if needed
    if adm2_col != "adm2_name":
        gdf = gdf.rename(columns={adm2_col: "adm2_name"})
    
    # Dissolve adm3 polygons up to adm2
    adm2 = gdf.dissolve(by="adm2_name", as_index=False)[["adm2_name", "geometry"]]
    return adm2


# ==============================================================================
# Main map generator
# ==============================================================================
def make_maps(summary, output_path, mok=False, all_cells_file=None,
              woreda_shapefile=None, adm2_shapefile=None,
              use_cartopy=False, region='Ethiopia',
              resolution='10m', category='cultural',
              name='admin_0_countries', zoom_to_data=False,
              onset_file=None):
    """
    Generate forecast maps from a blended summary DataFrame (adm3_name-based).
    
    Parameters
    ----------
    summary : pd.DataFrame
        Blended forecast output with columns:
        id (adm3_name), time, week1-week4, clim_week1-clim_week4, later (optional).
    output_path : Path
        Directory where output PNGs will be saved.
    mok : bool
        If True, save into maps_mok/ subfolder with _mok suffix.
    all_cells_file : Path or None
        Path to all_cells.csv (adm3_name column).
    woreda_shapefile : Path or None
        Path to the adm3-level shapefile.
    adm2_shapefile : Path or None
        Path to the adm2-level shapefile (defaults to same as woreda).
    onset_file : Path or None
        Path to onset CSV (adm3_name column).
    """
    output_path = Path(output_path)
    
    # Load country boundary
    country_gdf = load_boundary(
        use_cartopy=use_cartopy, region=region,
        resolution=resolution, category=category, name=name,
        shapefile_path=woreda_shapefile
    )
    logging.info(f"Country boundary loaded: {len(country_gdf)} features")
    
    # Load all_cells if provided
    valid_ids = None
    exclude_set = set()
    if all_cells_file and Path(all_cells_file).exists():
        all_cells = pd.read_csv(all_cells_file)
        if "adm3_name" in all_cells.columns:
            valid_ids = set(all_cells["adm3_name"].astype(str).str.strip())
            logging.info(f"Loaded {len(valid_ids)} valid cells from {all_cells_file}")
        
        # Check for exclude flag
        if "flag" in all_cells.columns:
            exclude_set = set(
                all_cells.loc[all_cells["flag"] == "exclude", "adm3_name"].astype(str)
            )
            logging.info(f"Excluding {len(exclude_set)} cells")
    
    # Load onset districts
    onset_set = set()
    if onset_file and Path(onset_file).exists():
        df_onset = pd.read_csv(onset_file, dtype=str)
        if "adm3_name" in df_onset.columns:
            onset_set = set(df_onset["adm3_name"].astype(str).str.strip())
            logging.info(f"Loaded {len(onset_set)} onset-occurred districts")
    
    # Load woreda polygons
    if not woreda_shapefile or not Path(woreda_shapefile).exists():
        raise ValueError(f"Woreda shapefile not found: {woreda_shapefile}")
    
    woredas = load_woreda_geometries(woreda_shapefile)
    if valid_ids:
        woredas = woredas[woredas["adm3_name"].isin(valid_ids - exclude_set)]
    logging.info(f"Woreda polygons loaded: {len(woredas)} cells")
    
    # Load ADM2 boundaries
    adm2_gdf = None
    if adm2_shapefile and Path(adm2_shapefile).exists():
        adm2_gdf = load_adm2_geometries(adm2_shapefile)
    elif woreda_shapefile:
        # Try to load ADM2 from same shapefile
        try:
            adm2_gdf = load_adm2_geometries(woreda_shapefile)
        except Exception as e:
            logging.warning(f"Could not load ADM2 from woreda shapefile: {e}")
    
    # Get date from summary
    if "time" not in summary.columns or len(summary) == 0:
        raise ValueError("Summary must have 'time' column with at least one row")
    
    first_date = pd.to_datetime(summary["time"].iloc[0])
    date_str_fmt = first_date.strftime("%Y%m%d")
    
    output_dir = output_path / ("maps_mok" if mok else "maps")
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare forecast data
    preds_df = summary.copy()
    if "id" not in preds_df.columns and "adm3_name" in preds_df.columns:
        preds_df["id"] = preds_df["adm3_name"]
    
    if exclude_set:
        preds_df = preds_df[~preds_df["id"].isin(exclude_set)]
    
    # Create Forecast_p_* and Climatology_p_* columns
    for i in range(1, 5):
        preds_df[f"Climatology_p_{i}"] = preds_df.get(f"clim_week{i}", 0)
        preds_df[f"Forecast_p_{i}"] = preds_df.get(f"week{i}", 0)
    
    preds_df["Forecast_p_later"] = preds_df.get(
        "later", 1 - preds_df[[f"Forecast_p_{i}" for i in range(1, 5)]].sum(axis=1)
    )
    preds_df["Climatology_p_later"] = (
        1 - preds_df[[f"Climatology_p_{i}" for i in range(1, 5)]].sum(axis=1)
    )
    
    # Map extent
    minx, miny, maxx, maxy = country_gdf.total_bounds
    x_min, x_max, y_min, y_max = minx, maxx, miny, maxy
    
    if zoom_to_data:
        wb = woredas.total_bounds
        x_min, x_max, y_min, y_max = wb[0], wb[2], wb[1], wb[3]
        logging.info("Map extent: zoomed to woreda bounding box")
    
    # Color schemes
    period_order = ['week1', 'week2', 'week3', 'week4', 'later']
    colors = [
        '#e3eef9',  # 0.0-0.3
        '#bdd7e7',  # 0.3-0.5
        '#6baed6',  # 0.5-0.6
        '#2171b5',  # 0.6-0.7
        '#08519c',  # 0.7-0.8
        '#041f4a',  # 0.8-1.0
    ]
    
    idx = [0, 2, 4, 5, 5]
    selected = [colors[i] for i in idx]
    period_colors = {k: c for k, c in zip(period_order, selected)}
    period_colors['none'] = '#d3d3d3'
    period_colors['onset'] = '#6e6e6e'
    
    prob_bins = [0, 0.3, 0.5, 0.6, 0.7, 0.8, 1.0]
    prob_cmap = ListedColormap(plt.get_cmap('Blues')(np.linspace(0.05, 1, len(prob_bins) - 1)))
    prob_norm = BoundaryNorm(prob_bins, ncolors=len(prob_bins) - 1, clip=True)
    
    def max_period(vf):
        period_keys = ['week1', 'week2', 'week3', 'week4', 'later']
        return period_keys[int(np.argmax(vf))]
    
    # ========================================================================
    # SECTION: Weekly Probability Maps
    # ========================================================================
    for t, grp in preds_df.groupby('time'):
        ds_str = pd.Timestamp(t).strftime('%Y-%m-%d')
        week_titles = {
            i: (f"{(pd.Timestamp(t) + timedelta(days=(i-1)*7+1)).strftime('%m/%d/%Y')} - "
                f"{(pd.Timestamp(t) + timedelta(days=i*7)).strftime('%m/%d/%Y')}")
            for i in range(1, 5)
        }
        
        # Join forecast probs onto woreda geometries
        merged = woredas.merge(
            grp[["id"] + [f"Forecast_p_{i}" for i in range(1, 5)] +
                [f"Climatology_p_{i}" for i in range(1, 5)]],
            left_on="adm3_name", right_on="id", how="left"
        )
        
        fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharex=True, sharey=True,
                                 gridspec_kw={'wspace': 0.1})
        
        for i, ax in enumerate(axes, 1):
            country_gdf.boundary.plot(ax=ax, linewidth=0.5, edgecolor='black')
            
            col = f"Forecast_p_{i}"
            clim_col = f"Climatology_p_{i}"
            merged[col] = pd.to_numeric(merged[col], errors='coerce')
            merged[clim_col] = pd.to_numeric(merged[clim_col], errors='coerce')
            
            merged_na = merged[merged[col].isna()]
            merged_ok = merged[merged[col].notna()]
            
            if not merged_na.empty:
                merged_na.plot(ax=ax, color=period_colors['none'], edgecolor='none', zorder=1)
            
            if not merged_ok.empty:
                merged_ok.plot(ax=ax, column=col, cmap=prob_cmap, norm=prob_norm,
                               edgecolor='none', zorder=2)
                
                # Onset overlay
                if onset_set:
                    merged_onset = merged[merged["adm3_name"].isin(onset_set)]
                    if not merged_onset.empty:
                        merged_onset.plot(ax=ax, color=period_colors['onset'],
                                          edgecolor='none', zorder=2.5)
                
                # Climatology border overlay
                lo = merged_ok[merged_ok[col] <= merged_ok[clim_col] - 0.10]
                hi = merged_ok[merged_ok[col] >= merged_ok[clim_col] + 0.10]
                if not lo.empty:
                    lo.plot(ax=ax, facecolor='none', edgecolor='red', linewidth=0.05, zorder=3)
                if not hi.empty:
                    hi.plot(ax=ax, facecolor='none', edgecolor='green', linewidth=0.2, zorder=3)
            
            ax.set_title(week_titles[i], fontsize=10, pad=6)
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.grid(True, linestyle='--', linewidth=0.4, color='gray', alpha=0.5)
            
            # ADM2 boundary overlay
            if adm2_gdf is not None:
                adm2_gdf.boundary.plot(ax=ax, linewidth=0.5, edgecolor='dimgray',
                                       linestyle='-', zorder=4)
        
        sm = plt.cm.ScalarMappable(norm=prob_norm, cmap=prob_cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=list(axes), orientation='horizontal', fraction=0.04, pad=0.08)
        cbar.set_label('Probability')
        cbar.set_ticks(prob_bins)
        cbar_tick_labels = [str(v) for v in prob_bins]
        cbar_tick_labels[-2] = '>0.8'
        cbar_tick_labels[-1] = ''
        cbar.set_ticklabels(cbar_tick_labels)
        
        legend_handles = [
            Patch(facecolor='none', edgecolor=c, linewidth=1.5, label=l)
            for c, l in [('red', '≥10% lower than climatology'),
                         ('green', '≥10% higher than climatology')]
        ]
        axes[-1].legend(handles=legend_handles, loc='lower right',
                        fontsize=7, handlelength=1.5, handleheight=1.5,
                        borderpad=0.5, labelspacing=0.3, framealpha=0.8)
        
        fig.subplots_adjust(left=0.03, right=0.98, top=0.90, bottom=0.22)
        suffix = '_mok' if mok else ''
        fname = output_dir / f"prob_weeks1-4_{ds_str}{suffix}_zone.png"
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        logging.info(f"Saved weekly probability map: {fname}")
        
        # NetCDF export
        prob_cols = [f"Forecast_p_{i}" for i in range(1, 5)] + ["Forecast_p_later"]
        nc_path = os.path.join(output_dir, f"weekly_probs_{date_str_fmt}{suffix}.nc")
        export_to_netcdf(grp, prob_cols, nc_path, pd.Timestamp(t).date())
        
        plt.close(fig)
    
    # ========================================================================
    # SECTION: Max-Period Map
    # ========================================================================
    for t, grp in preds_df.groupby('time'):
        ds_str = pd.Timestamp(t).strftime('%Y-%m-%d')
        merged = woredas.merge(
            grp[["id"] + [f"Forecast_p_{i}" for i in range(1, 5)] + ["Forecast_p_later"]],
            left_on="adm3_name", right_on="id", how="left"
        )
        
        def _row_max_period(row):
            vf = [row.get(f"Forecast_p_{i}", np.nan) for i in range(1, 5)] + \
                 [row.get("Forecast_p_later", np.nan)]
            if any(pd.isna(vf)):
                return 'none'
            return max_period(vf)
        
        merged["period"] = merged.apply(_row_max_period, axis=1)
        
        # Onset districts override
        if onset_set:
            merged.loc[merged["adm3_name"].isin(onset_set), "period"] = "onset"
        
        # Deduplicate
        merged["_is_none"] = (merged["period"] == "none").astype(int)
        merged = merged.sort_values("_is_none").drop_duplicates(subset="adm3_name", keep="first")
        merged = merged.drop(columns="_is_none")
        
        labels = _period_labels(pd.Timestamp(t))
        week_labels = ["week1", "week2", "week3", "week4+"]
        handles = [
            Patch(facecolor=period_colors[k], edgecolor='none', label=week_labels[p])
            for k, p in zip(period_order, range(0, 4))
        ]
        handles.append(Patch(facecolor=period_colors['none'], edgecolor='none',
                             label='No forecast/onset'))
        handles.append(Patch(facecolor=period_colors['onset'], edgecolor='none',
                             label='Onset occurred'))
        
        fig, ax = plt.subplots(figsize=(6, 6))
        country_gdf.boundary.plot(ax=ax, linewidth=0.5, edgecolor='black')
        
        plot_order = ['none'] + [k for k in period_colors if k != 'none']
        for period_key in plot_order:
            color = period_colors[period_key]
            subset = merged[merged["period"] == period_key]
            if not subset.empty:
                from matplotlib.colors import to_rgb
                rgb = to_rgb(color)
                l = 0.299*rgb[0] + 0.587*rgb[1] + 0.114*rgb[2]
                contrast_edge = 'black' if l > 0.5 else 'white'
                subset.plot(ax=ax, color=color, edgecolor=contrast_edge, linewidth=0.01, zorder=1)
        
        ax.legend(handles=handles, title='Period with Max Probability of Onset',
                  loc='lower left', ncol=2, fontsize=9, title_fontsize=7,
                  handlelength=1.5, handleheight=1.5,
                  borderpad=0.5, labelspacing=0.3, framealpha=0.8)
        
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.grid(True, linestyle='--', linewidth=0.4, color='gray', alpha=0.5)
        
        # ADM2 overlay
        if adm2_gdf is not None:
            adm2_gdf.boundary.plot(ax=ax, linewidth=0.5, edgecolor='dimgray',
                                   linestyle='-', zorder=4)
        
        suffix = '_mok' if mok else ''
        fname = output_dir / f"map_max_period_{ds_str}{suffix}_zone.png"
        plt.tight_layout()
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logging.info(f"Saved max-period map: {fname}")
        
        # NetCDF export
        def _identify_max_period_idx(row):
            probs = [row.get(f"Forecast_p_{i}", np.nan) for i in range(1, 5)] + \
                    [row.get("Forecast_p_later", np.nan)]
            if any(pd.isna(probs)):
                return np.nan
            return float(np.argmax(probs) + 1)
        
        grp = grp.copy()
        grp["max_period_index"] = grp.apply(_identify_max_period_idx, axis=1)
        nc_max_path = os.path.join(output_dir, f"max_period_index_{date_str_fmt}{suffix}.nc")
        export_to_netcdf(grp, ["max_period_index"], nc_max_path, pd.Timestamp(t).date())
    
    logging.info(f"All maps saved under {output_dir}")


# ==============================================================================
# NetCDF export
# ==============================================================================
def export_to_netcdf(df, columns, output_path, issue_date):
    """Export a forecast DataFrame to NetCDF with adm3_name as the spatial dimension."""
    df = df.copy()
    if "id" not in df.columns and "adm3_name" in df.columns:
        df["id"] = df["adm3_name"]
    
    keep = ["id"] + [c for c in columns if c in df.columns]
    df = df[keep].drop_duplicates(subset=["id"])
    
    adm3_names = df["id"].astype(str).values
    data_vars = {}
    for col in columns:
        if col in df.columns:
            data_vars[col] = xr.DataArray(
                df[col].values.astype(float),
                dims=["adm3_name"],
                attrs={"issue_date": str(issue_date)},
            )
    
    ds = xr.Dataset(data_vars, coords={"adm3_name": adm3_names})
    ds.attrs["issue_date"] = str(issue_date)
    ds.to_netcdf(output_path)
    logging.info(f"NetCDF saved: {output_path}")


def save_to_netcdf(df, columns, output_path):
    """Convenience wrapper."""
    export_to_netcdf(df, columns, output_path, issue_date="unknown")