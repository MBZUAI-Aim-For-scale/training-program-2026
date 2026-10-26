"""
momp/graphics/maps.py
Spatial metric map plotting routines with dynamic region boundary support.
"""

import os
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
import xarray as xr

from momp.params.region_def import polygon_boundary
from momp.utils.land_mask import get_region_outline, shp_outline
from momp.utils.visual import box_boundary


def calculate_cmz_averages(da, polygon_lon, polygon_lat):
    """Calculate spatial average within the CMZ polygon."""
    from matplotlib.path import Path

    lats = da.lat.values
    lons = da.lon.values

    polygon_path = Path(list(zip(polygon_lon, polygon_lat)))
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    points = np.column_stack((lon_grid.ravel(), lat_grid.ravel()))
    inside_polygon = polygon_path.contains_points(points).reshape(lon_grid.shape)

    values_inside = da.values[inside_polygon]

    if len(values_inside) > 0 and not np.all(np.isnan(values_inside)):
        return np.nanmean(values_inside)
    else:
        return np.nan


def plot_spatial_metrics(
    spatial_metrics,
    *,
    case_name=None,
    shpfile_dir=None,
    polygon=False,
    dir_fig=None,
    region="Ethiopia",
    figsize=(18, 6),
    show_plot=False,
    rect_box=True,
    **kwargs
):
    """
    Plot spatial maps of Mean MAE, False Alarm Rate, and Miss Rate in a 1x3 panel layout.
    Dynamically renders boundary shapefiles for the configured region.
    """
    if case_name is None:
        model_name = kwargs.get("model") or kwargs.get("model_name") or "Model"
        verification_window = kwargs.get("verification_window") or (1, 15)
        case_name = f"{model_name}_{verification_window[0]}-{verification_window[1]}"

    if dir_fig is None:
        dir_fig = kwargs.get("dir_out") or "/app/data/ROMP_OUT/et/figure"

    os.makedirs(dir_fig, exist_ok=True)

    mean_mae = spatial_metrics["mean_mae"]
    far = spatial_metrics["false_alarm_rate"] * 100
    miss_rate = spatial_metrics["miss_rate"] * 100

    lats = mean_mae.lat.values
    lons = mean_mae.lon.values

    lat_diff = abs(lats[1] - lats[0]) if len(lats) > 1 else 0.1
    print(f"Detected resolution: {lat_diff:.1f} degrees")

    polygon_defined = False
    polygon1_lat, polygon1_lon = [], []

    if polygon:
        polygon1_lat, polygon1_lon = polygon_boundary(mean_mae)
        if len(polygon1_lat) > 0 and len(polygon1_lon) > 0:
            polygon_defined = True

    def calculate_mae_stats_across_years(spatial_metrics, lons, lats, polygon_lon, polygon_lat):
        yearly_mae_keys = [
            key for key in spatial_metrics.keys()
            if key.startswith("mae_") and key != "mae_combined"
        ]

        if not yearly_mae_keys:
            return np.nan, np.nan, np.nan, np.nan

        cmz_yearly_averages = []
        overall_yearly_averages = []

        for mae_key in yearly_mae_keys:
            year_mae_map = spatial_metrics[mae_key]

            if polygon_defined and polygon_lon is not None:
                cmz_avg = calculate_cmz_averages(year_mae_map, polygon_lon, polygon_lat)
                if not np.isnan(cmz_avg):
                    cmz_yearly_averages.append(cmz_avg)

            valid_vals = year_mae_map.values[~np.isnan(year_mae_map.values)]
            if len(valid_vals) > 0:
                overall_avg = np.mean(valid_vals)
                overall_yearly_averages.append(overall_avg)

        if len(cmz_yearly_averages) > 0 and polygon_defined:
            cmz_mean = np.mean(cmz_yearly_averages)
            cmz_se = np.std(cmz_yearly_averages, ddof=1) / np.sqrt(len(cmz_yearly_averages)) if len(cmz_yearly_averages) > 1 else 0
        else:
            cmz_mean, cmz_se = np.nan, np.nan

        if len(overall_yearly_averages) > 0:
            overall_mean = np.mean(overall_yearly_averages)
            overall_se = np.std(overall_yearly_averages, ddof=1) / np.sqrt(len(overall_yearly_averages)) if len(overall_yearly_averages) > 1 else 0
        else:
            overall_mean, overall_se = np.nan, np.nan

        return cmz_mean, cmz_se, overall_mean, overall_se

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    if polygon_defined:
        cmz_mae_mean, cmz_mae_se, overall_mae_mean, overall_mae_se = calculate_mae_stats_across_years(
            spatial_metrics, lons, lats, polygon1_lon, polygon1_lat
        )
        cmz_far = calculate_cmz_averages(spatial_metrics["false_alarm_rate"] * 100, polygon1_lon, polygon1_lat)
        cmz_mr = calculate_cmz_averages(spatial_metrics["miss_rate"] * 100, polygon1_lon, polygon1_lat)
    else:
        cmz_mae_mean, cmz_mae_se, overall_mae_mean, overall_mae_se = calculate_mae_stats_across_years(
            spatial_metrics, lons, lats, None, None
        )
        cmz_far = np.nan
        cmz_mr = np.nan

    lon_edges = np.concatenate([lons - (lons[1] - lons[0]) / 2, [lons[-1] + (lons[1] - lons[0]) / 2]])
    lat_edges = np.concatenate([lats - (lats[1] - lats[0]) / 2, [lats[-1] + (lats[1] - lats[0]) / 2]])
    LON_edges, LAT_edges = np.meshgrid(lon_edges, lat_edges)

    map_lw = 0.75
    polygon_lw = 1.25
    panel_linewidth = 0.5
    tick_length = 3
    tick_width = 0.8
    txt_fsize = 8 if abs(lat_diff) >= 0.99 else None

    # Load shapefile outlines dynamically for requested region
    region_boundaries = get_region_outline(shpfile_dir, region=region) if shpfile_dir else []

    # Panel 1: Mean MAE
    masked_mae = np.ma.masked_invalid(mean_mae.values)
    im1 = axes[0].pcolormesh(LON_edges, LAT_edges, masked_mae, cmap="OrRd", vmin=0, vmax=15, shading="flat")

    if region_boundaries:
        for boundary in region_boundaries:
            axes[0].plot(boundary[0], boundary[1], color="black", linewidth=map_lw)
    else:
        shp_outline(axes[0], region=region)

    if polygon_defined:
        poly_patch = Polygon(list(zip(polygon1_lon, polygon1_lat)), fill=False, edgecolor="black", linewidth=polygon_lw)
        axes[0].add_patch(poly_patch)

    if txt_fsize:
        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                val = mean_mae.values[i, j]
                if not np.isnan(val):
                    tc = "white" if val > 7.5 else "black"
                    axes[0].text(lon, lat, f"{val:.1f}", ha="center", va="center", color=tc, fontsize=txt_fsize)

    if polygon_defined and not np.isnan(cmz_mae_mean):
        cmz_text = f"MAE: {cmz_mae_mean:.1f}±{cmz_mae_se:.1f} days" if cmz_mae_se > 0 else f"MAE: {cmz_mae_mean:.1f} days"
        axes[0].text(0.98, 0.02, cmz_text, transform=axes[0].transAxes, color="black", fontsize=14, ha="right", va="bottom")

    axes[0].text(0.98, 0.98, "MAE (in days)", transform=axes[0].transAxes, color="black", fontsize=14, ha="right", va="top")
    axes[0].set_xlabel("Longitude", fontsize=12)
    axes[0].set_ylabel("Latitude", fontsize=12)

    # Panel 2: False Alarm Rate
    masked_far = np.ma.masked_invalid(far.values)
    im2 = axes[1].pcolormesh(LON_edges, LAT_edges, masked_far, cmap="Reds", vmin=0, vmax=100, shading="flat")

    if region_boundaries:
        for boundary in region_boundaries:
            axes[1].plot(boundary[0], boundary[1], color="black", linewidth=map_lw)
    else:
        shp_outline(axes[1], region=region)

    if polygon_defined:
        poly_patch = Polygon(list(zip(polygon1_lon, polygon1_lat)), fill=False, edgecolor="black", linewidth=polygon_lw)
        axes[1].add_patch(poly_patch)

    if txt_fsize:
        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                val = far.values[i, j]
                if not np.isnan(val):
                    tc = "white" if val > 50 else "black"
                    axes[1].text(lon, lat, f"{val:.0f}", ha="center", va="center", color=tc, fontsize=txt_fsize)

    if polygon_defined and not np.isnan(cmz_far):
        axes[1].text(0.98, 0.02, f"FAR: {cmz_far:.1f}%", transform=axes[1].transAxes, color="black", fontsize=14, ha="right", va="bottom")

    axes[1].text(0.98, 0.98, "False Alarm Rate (%)", transform=axes[1].transAxes, color="black", fontsize=14, ha="right", va="top")
    axes[1].set_xlabel("Longitude", fontsize=12)

    # Panel 3: Miss Rate
    masked_mr = np.ma.masked_invalid(miss_rate.values)
    im3 = axes[2].pcolormesh(LON_edges, LAT_edges, masked_mr, cmap="Blues", vmin=0, vmax=100, shading="flat")

    if region_boundaries:
        for boundary in region_boundaries:
            axes[2].plot(boundary[0], boundary[1], color="black", linewidth=map_lw)
    else:
        shp_outline(axes[2], region=region)

    if polygon_defined:
        poly_patch = Polygon(list(zip(polygon1_lon, polygon1_lat)), fill=False, edgecolor="black", linewidth=polygon_lw)
        axes[2].add_patch(poly_patch)

    if txt_fsize:
        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                val = miss_rate.values[i, j]
                if not np.isnan(val):
                    tc = "white" if val > 50 else "black"
                    axes[2].text(lon, lat, f"{val:.0f}", ha="center", va="center", color=tc, fontsize=txt_fsize)

    if polygon_defined and not np.isnan(cmz_mr):
        axes[2].text(0.98, 0.02, f"MR: {cmz_mr:.1f}%", transform=axes[2].transAxes, color="black", fontsize=14, ha="right", va="bottom")

    axes[2].text(0.98, 0.98, "Miss Rate (%)", transform=axes[2].transAxes, color="black", fontsize=14, ha="right", va="top")
    axes[2].set_xlabel("Longitude", fontsize=12)

    # Colorbars
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    for ax, im in zip(axes, [im1, im2, im3]):
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("bottom", size="5%", pad=0.3)
        cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
        pos = cax.get_position()
        new_width = pos.width * 0.6
        new_x = pos.x0 + (pos.width - new_width) / 2
        cax.set_position([new_x, pos.y0, new_width, pos.height])
        cax.set_anchor("C")
        cax.set_box_aspect(0.05)

    # Axis styling
    for i, ax in enumerate(axes):
        ax.set_xlim([lons.min() - 2, lons.max() + 2])
        ax.set_ylim([lats.min() - 2, lats.max() + 2])

        xticks = np.arange(lons.min(), lons.max() + 1, 4)
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"{int(x)}°E" for x in xticks])

        if i == 0:
            yticks = np.arange(lats.min(), lats.max() + 1, 4)
            ax.set_yticks(yticks)
            ax.set_yticklabels([f"{int(y)}°N" for y in yticks])
        else:
            ax.set_yticks([])
            ax.set_yticklabels([])

        ax.tick_params(axis="both", which="major", labelsize=10, length=tick_length, width=tick_width)
        for side in ["top", "right", "bottom", "left"]:
            ax.spines[side].set_linewidth(panel_linewidth)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(False)

        if rect_box:
            box_boundary("rect_boundary", ax, edgecolor="black", linewidth=2, linestyle="-", fill=False, alpha=1.0, zorder=20)

    plt.tight_layout()

    plot_filename = f"spatial_metrics_{case_name}.png"
    plot_path = os.path.join(dir_fig, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Figure saved to: {plot_path}")

    if show_plot:
        plt.show()

    plt.close(fig)
    return fig, axes
