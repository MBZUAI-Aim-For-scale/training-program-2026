#!/usr/bin/env python3
"""
run_maps.py
Generate forecast maps from a blended output summary CSV.
Uses config.py paths for shapefiles and ensures the country boundary is
always overlaid correctly on both weekly and max-period maps.
"""
import argparse
import os
import sys
import re
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap, to_rgb
from matplotlib.patches import Patch
from datetime import timedelta
from pathlib import Path
# ==============================================================================
# Project Root Resolution
# ==============================================================================
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Import the exact paths from your config.py
from ai_weather.config import (
    ETH_GEOJSON_PATH,       # Country boundary
    ZONE_WOREDA_SHP,        # Woreda/Zone geometries
)

def make_maps(summary, output_path, date_str="unknown", mok=False,
              region="Ethiopia", zoom_to_data=False, onset_file=None,
              country_boundary=None, district_boundary=None):
    """Generate 4-panel weekly probability maps and max-period map."""
    output_path = Path(output_path)
    country_path = Path(country_boundary or ETH_GEOJSON_PATH).expanduser()
    district_path = Path(district_boundary or ZONE_WOREDA_SHP).expanduser()
    maps_dir = output_path / ("maps_mok" if mok else "maps")
    maps_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Country Boundary (ETH_GEOJSON_PATH)
    country_gdf = None
    if country_path.exists():
        country_gdf = gpd.read_file(country_path).to_crs("EPSG:4326")
        # Dissolve to ensure we have a single continuous outline
        if len(country_gdf) > 1:
            country_gdf = country_gdf.dissolve().reset_index()
        print(f"Loaded country boundary: {country_path}")
    else:
        print(f"WARNING: Country shapefile not found at {country_path}")

    # 2. Load Woreda Geometries (ZONE_WOREDA_SHP)
    if not district_path.exists():
        raise FileNotFoundError(f"Woreda shapefile not found at {district_path}")
        
    woreda_gdf = gpd.read_file(district_path).to_crs("EPSG:4326")
    print(f"Loaded woreda shapefile: {len(woreda_gdf)} features")

    # Find the name column in the woreda shapefile
    name_col = None
    for col in ["adm3_name", "ADM3_EN", "NAME_3", "Woreda", "name", "ZONE_NAME"]:
        if col in woreda_gdf.columns:
            name_col = col
            break
    if name_col is None:
        raise ValueError("Could not find a name column in the woreda shapefile.")

    # 3. Identify probability columns
    week_cols = [c for c in summary.columns if c in ["week1", "week2", "week3", "week4"]]
    cv_week_cols = [c for c in summary.columns if c in ["cv_week1", "cv_week2", "cv_week3", "cv_week4"]]
    
    if week_cols and len(week_cols) == 4:
        prob_cols = week_cols
        has_clim = all(f"clim_{wc}" in summary.columns for wc in week_cols)
    elif cv_week_cols and len(cv_week_cols) == 4:
        prob_cols = cv_week_cols
        has_clim = False
    else:
        raise ValueError("No complete weekly probability columns found")

    # 4. Load onset districts
    onset_districts = set()
    if onset_file and Path(onset_file).exists():
        onset_df = pd.read_csv(onset_file)
        if "adm3_name" in onset_df.columns:
            onset_districts = set(onset_df["adm3_name"].astype(str))
        elif "id" in onset_df.columns:
            onset_districts = set(onset_df["id"].astype(str))
        print(f"Loaded {len(onset_districts)} onset districts")

    # 5. Map extent
    if country_gdf is not None:
        minx, miny, maxx, maxy = country_gdf.total_bounds
    else:
        minx, miny, maxx, maxy = woreda_gdf.total_bounds
        
    if zoom_to_data:
        wb = woreda_gdf.total_bounds
        minx, maxx, miny, maxy = wb[0], wb[2], wb[1], wb[3]

    # Color schemes (matching maps_new_zone.py)
    period_order = ['week1', 'week2', 'week3', 'week4', 'later']
    colors = ['#e3eef9', '#bdd7e7', '#6baed6', '#2171b5', '#08519c', '#041f4a']
    idx = [0, 2, 4, 5, 5]
    selected = [colors[i] for i in idx]
    period_colors = {k: c for k, c in zip(period_order, selected)}
    period_colors['none'] = '#d3d3d3'
    period_colors['onset'] = '#6e6e6e'

    prob_bins = [0, 0.3, 0.5, 0.6, 0.7, 0.8, 1.0]
    prob_cmap = ListedColormap(plt.get_cmap('Blues')(np.linspace(0.05, 1, len(prob_bins) - 1)))
    prob_norm = BoundaryNorm(prob_bins, ncolors=len(prob_bins) - 1, clip=True)

    # ==========================================================================
    # SECTION 1: 4-PANEL WEEKLY PROBABILITY MAPS
    # ==========================================================================
    time_groups = list(summary.groupby('time'))
    if not time_groups:
        raise ValueError("No forecast rows available for maps")
        
    t, grp = time_groups[0]
    ds_str = pd.Timestamp(t).strftime('%Y-%m-%d')
    
    week_titles = {
        i: f"{(pd.Timestamp(t) + timedelta(days=(i-1)*7+1)).strftime('%m/%d/%Y')} - "
           f"{(pd.Timestamp(t) + timedelta(days=i*7)).strftime('%m/%d/%Y')}"
        for i in range(1, 5)
    }

    merge_cols = ["id"] + prob_cols
    if has_clim:
        merge_cols += [f"clim_{wc}" for wc in week_cols]
        
    merge_df = grp[merge_cols].copy()
    merge_df["id"] = merge_df["id"].astype(str)
    woreda_gdf[name_col] = woreda_gdf[name_col].astype(str)
    matching = woreda_gdf.loc[woreda_gdf[name_col].isin(merge_df["id"])]
    if matching.empty or not (matching.geometry.notna() & ~matching.geometry.is_empty).any():
        raise ValueError("No forecast districts match a nonempty district geometry")
    merged = woreda_gdf.merge(merge_df, left_on=name_col, right_on="id", how="left")

    fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharex=True, sharey=True,
                             gridspec_kw={'wspace': 0.08})
    
    for i, ax in enumerate(axes, 1):
        if week_cols:
            wc = f"week{i}"
            clim_col = f"clim_week{i}" if has_clim else None
        else:
            wc = f"cv_week{i}"
            clim_col = None
            
        # Plot country boundary FIRST (thin, as a base)
        if country_gdf is not None:
            country_gdf.boundary.plot(ax=ax, linewidth=0.5, edgecolor='black', zorder=1)
            
        merged[wc] = pd.to_numeric(merged[wc], errors='coerce')
        if clim_col:
            merged[clim_col] = pd.to_numeric(merged[clim_col], errors='coerce')
        
        merged_na = merged[merged[wc].isna()]
        merged_ok = merged[merged[wc].notna()]
        
        if not merged_na.empty:
            merged_na.plot(ax=ax, color=period_colors['none'], edgecolor='none', zorder=2)
        
        if not merged_ok.empty:
            merged_ok.plot(ax=ax, column=wc, cmap=prob_cmap, norm=prob_norm,
                          edgecolor='none', zorder=3)
            
            # Luminance-based edges for woredas
            face_colors = prob_cmap(prob_norm(merged_ok[wc]))
            luminance = (0.299 * face_colors[:, 0] + 0.587 * face_colors[:, 1] + 0.114 * face_colors[:, 2])
            edge_colors = ['black' if l > 0.5 else 'white' for l in luminance]
            merged_ok.plot(ax=ax, column=wc, cmap=prob_cmap, norm=prob_norm,
                          edgecolor=edge_colors, linewidth=0.1, zorder=3)
            
            if clim_col and clim_col in merged_ok.columns:
                lo = merged_ok[merged_ok[wc] <= merged_ok[clim_col] - 0.10]
                hi = merged_ok[merged_ok[wc] >= merged_ok[clim_col] + 0.10]
                if not lo.empty:
                    lo.plot(ax=ax, facecolor='none', edgecolor='red', linewidth=0.05, zorder=4)
                if not hi.empty:
                    hi.plot(ax=ax, facecolor='none', edgecolor='green', linewidth=0.2, zorder=4)
                    
            if onset_districts:
                onset_gdf = merged[merged[name_col].isin(onset_districts)]
                if not onset_gdf.empty:
                    onset_gdf.plot(ax=ax, color=period_colors['onset'], edgecolor='none', zorder=5)
        
        # Plot country boundary AGAIN on top (thick, to ensure it's visible)
        if country_gdf is not None:
            country_gdf.boundary.plot(ax=ax, linewidth=1.5, edgecolor='black', zorder=10)

        ax.set_title(week_titles[i], fontsize=10, pad=6)
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)
        ax.set_axis_off()

    # Colorbar
    sm = plt.cm.ScalarMappable(norm=prob_norm, cmap=prob_cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=list(axes), orientation='horizontal', 
                       fraction=0.04, pad=0.08, shrink=0.8)
    cbar.set_label('Probability', fontsize=9)
    cbar.set_ticks(prob_bins)
    cbar_tick_labels = [str(v) for v in prob_bins]
    cbar_tick_labels[-2] = '>0.8'
    cbar_tick_labels[-1] = ''
    cbar.set_ticklabels(cbar_tick_labels, fontsize=8)

    # Legend
    legend_handles = []
    if has_clim:
        legend_handles = [
            Patch(facecolor='none', edgecolor=c, linewidth=1.5, label=l)
            for c, l in [('red', '≥10% lower than climatology'),
                         ('green', '≥10% higher than climatology')]
        ]
    if onset_districts:
        legend_handles.append(Patch(facecolor=period_colors['onset'], edgecolor='none', label='Onset occurred'))
        
    if legend_handles:
        axes[-1].legend(handles=legend_handles, loc='lower right', 
                       fontsize=7, framealpha=0.9, handlelength=1.5, handleheight=1.5)

    plt.suptitle(f"{region} - Weekly Onset Probability\nIssue: {ds_str}", fontsize=12, y=0.98)
    fig.subplots_adjust(left=0.03, right=0.98, top=0.90, bottom=0.22)
    
    out_file = maps_dir / f"prob_weeks1-4_{date_str}.png"
    fig.savefig(out_file, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"Saved 4-panel weekly map: {out_file}")

    # ==========================================================================
    # SECTION 2: MAX-PERIOD MAP
    # ==========================================================================
    later_col = "later" if "later" in summary.columns else ("cv_later" if "cv_later" in summary.columns else None)
    prob_cols_with_later = prob_cols + [later_col] if later_col else prob_cols

    max_week = grp[prob_cols_with_later].idxmax(axis=1)
    summary_map = grp[["id"]].copy()
    summary_map["max_week"] = max_week
    summary_map["id"] = summary_map["id"].astype(str)
    
    merged_max = woreda_gdf.merge(summary_map, left_on=name_col, right_on="id", how="left")

    if onset_districts:
        merged_max.loc[merged_max[name_col].isin(onset_districts), "max_week"] = "onset"
    merged_max.loc[merged_max["max_week"].isna(), "max_week"] = "none"
    merged_max["color"] = merged_max["max_week"].map(period_colors).fillna(period_colors['none'])

    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot country boundary base
    if country_gdf is not None:
        country_gdf.boundary.plot(ax=ax, linewidth=0.5, edgecolor='black', zorder=1)
        
    # Plot woredas (NO edgecolor here, so the outer boundary isn't hidden by white lines)
    merged_max.plot(ax=ax, color=merged_max["color"], edgecolor='none', linewidth=0, zorder=2)

    # Plot country boundary ON TOP (thick black line)
    if country_gdf is not None:
        country_gdf.boundary.plot(ax=ax, linewidth=2.0, edgecolor='black', zorder=10)

    # Legend
    week_labels = ["week1", "week2", "week3", "week4+"]
    handles = [
        Patch(facecolor=period_colors[k], edgecolor='none', label=week_labels[p])
        for k, p in zip(period_order, range(0, 4))
    ]
    handles.append(Patch(facecolor=period_colors['none'], edgecolor='none', label='No data'))
    handles.append(Patch(facecolor=period_colors['onset'], edgecolor='none', label='Onset occurred'))
    
    ax.legend(handles=handles, title='Period with Max Probability of Onset',
             loc='lower left', ncol=2, fontsize=9, title_fontsize=7,
             handlelength=1.5, handleheight=1.5,
             borderpad=0.5, labelspacing=0.3, framealpha=0.8)
    
    ax.set_title(f"{region} - Most Likely Onset Window\nIssue: {ds_str}", fontsize=12, pad=15)
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_axis_off()

    out_file = maps_dir / f"map_max_period_{date_str}.png"
    fig.savefig(out_file, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"Saved max-period map: {out_file}")

    print(f"\nAll maps saved to: {maps_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate forecast maps.")
    parser.add_argument("--input_file", required=True, help="Path to CSV")
    parser.add_argument("--output_path", required=True, help="Output directory")
    parser.add_argument("--region", default="Ethiopia", help="Region name")
    parser.add_argument("--mok", action="store_true", help="Save to maps_mok/")
    parser.add_argument("--zoom_to_data", action="store_true", help="Zoom to data bounds")
    parser.add_argument("--onset_file", default=None, help="Path to onset CSV")
    parser.add_argument("--country_boundary", help="Override country boundary geometry")
    parser.add_argument("--district_boundary", help="Override district boundary geometry")
    args = parser.parse_args()

    input_file = Path(args.input_file).resolve()
    output_path = Path(args.output_path).resolve()

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    output_path.mkdir(parents=True, exist_ok=True)

    # Extract date from filename
    date_match = re.search(r"(\d{8})", input_file.stem)
    date_str = date_match.group(1) if date_match else "unknown"
    print(f"Date string: {date_str}")

    summary = pd.read_csv(input_file)
    if "time" in summary.columns:
        if date_str == "unknown":
            raise ValueError("Input filename must include the forecast issue date YYYYMMDD")
        target_date = pd.to_datetime(date_str, format="%Y%m%d").date()
        dates = pd.to_datetime(summary["time"], errors="raise").dt.date
        summary = summary.loc[dates == target_date].copy()
        if summary.empty:
            raise ValueError(f"No forecast rows found for issue date {target_date}")

    make_maps(
        summary,
        output_path=output_path,
        date_str=date_str,
        mok=args.mok,
        region=args.region,
        zoom_to_data=args.zoom_to_data,
        onset_file=args.onset_file,
        country_boundary=args.country_boundary,
        district_boundary=args.district_boundary,
    )

if __name__ == "__main__":
    main()