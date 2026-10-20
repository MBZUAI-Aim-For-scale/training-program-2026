#apps/benchmarking/ROMP/momp/graphics/onset_map.py
import copy
import os
from datetime import datetime, timedelta

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors as mcolors

from momp.graphics.func_map import spatial_metrics_map
from momp.graphics.maps import calculate_cmz_averages
from momp.utils.land_mask import add_polygon, shp_mask
from momp.utils.printing import tuple_to_str_range
from momp.utils.visual import cbar_season, set_basemap


def doy_to_date_string(doy, date_filter_year=2024):
    """Convert day of year to dd/mm format string."""
    date = datetime(date_filter_year, 1, 1) + timedelta(days=int(doy) - 1)
    return date.strftime("%d/%m")


def doy_to_mmm_dd(doy, date_filter_year=2024):
    """Convert day of year to 'MMM DD' format string."""
    date = pd.to_datetime(f"{date_filter_year}-{int(doy):03d}", format="%Y-%j")
    return date.strftime("%b %d")


def plot_spatial_climatology_onset(
    onset_da_dict,
    *,
    years_clim,
    shpfile_dir,
    polygon,
    dir_fig,
    region,
    figsize=(8, 6),
    cbar_ssn=False,
    domain_mask=False,
    show_plot=True,
    rect_box=False,
    **kwargs,
):
    """Plot spatial climatological onset date map across specified region."""
    # Extract DataArray
    climatological_onset_doy = next(iter(onset_da_dict.values()))

    # Coordinate extraction
    lats = climatological_onset_doy.lat.values
    lons = climatological_onset_doy.lon.values

    # Determine text overlay step based on spatial resolution
    lat_diff = abs(lats[1] - lats[0]) if len(lats) > 1 else 1.0
    txt_fsize = None if abs(lat_diff) < 0.99 else 8

    levels = np.arange(135, 245, 3)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())

    # --- GEOSException Workaround for Cartopy Title Updater ---
    ax._update_title_position = lambda renderer: None

    # Set spatial extent
    lon_min, lon_max = float(np.nanmin(lons)), float(np.nanmax(lons))
    lat_min, lat_max = float(np.nanmin(lats)), float(np.nanmax(lats))
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

    # Apply base map configuration
    set_basemap(ax, region, shpfile_dir, polygon, **kwargs)

    if polygon:
        ax, polygon1_lat, polygon1_lon, polygon_defined = add_polygon(
            ax, climatological_onset_doy, polygon, return_polygon=True
        )
        if polygon_defined:
            cmz_onset_mean = calculate_cmz_averages(
                climatological_onset_doy, polygon1_lon, polygon1_lat
            )
        else:
            cmz_onset_mean = np.nan

    if cbar_ssn:
        cmap_jjas, norm_jjas, bounds = cbar_season()
    else:
        cmap_jjas = plt.cm.Spectral
        norm_jjas = mcolors.BoundaryNorm(levels, cmap_jjas.N, extend="max")

    if domain_mask:
        climatological_onset_doy = shp_mask(climatological_onset_doy, region=region)

    im = ax.pcolormesh(
        climatological_onset_doy.lon,
        climatological_onset_doy.lat,
        climatological_onset_doy.values,
        cmap=cmap_jjas,
        norm=norm_jjas,
        transform=ccrs.PlateCarree(),
        shading="auto",
    )

    cbar = plt.colorbar(im, ax=ax, orientation="vertical", pad=0.02, shrink=0.6, aspect=20)

    if cbar_ssn:
        tick_positions = bounds[::2]
        tick_labels = [doy_to_mmm_dd(doy) for doy in tick_positions[:-1]]
        cbar.set_ticks(bounds, minor=True)
        cbar.set_ticks(tick_positions[:-1])
    else:
        tick_levels = levels[::4]
        tick_labels = [doy_to_mmm_dd(doy) for doy in tick_levels]
        cbar.set_ticks(tick_levels)

    cbar.set_ticklabels(tick_labels)
    cbar.set_label("Mean onset date", fontsize=12, fontweight="normal")
    cbar.ax.tick_params(labelsize=10)

    # Values overlay
    if txt_fsize:
        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                value = climatological_onset_doy.values[i, j]
                if not np.isnan(value):
                    text_color = "white" if value > 200 else "black"
                    ax.text(
                        lon,
                        lat,
                        f"{value:.0f}",
                        ha="center",
                        va="center",
                        color=text_color,
                        fontsize=txt_fsize,
                        fontweight="normal",
                        transform=ccrs.PlateCarree(),
                    )

    ax.text(
        0.98,
        0.98,
        "onset (day of year)",
        transform=ax.transAxes,
        color="black",
        fontsize=14,
        fontweight="normal",
        verticalalignment="top",
        horizontalalignment="right",
    )

    plt.tight_layout()

    plot_path = None
    if dir_fig:
        os.makedirs(dir_fig, exist_ok=True)
        plot_filename = f"climatology_onset_{tuple_to_str_range(years_clim)}.png"
        plot_path = os.path.join(dir_fig, plot_filename)
        fig.savefig(plot_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {plot_path}")

    # Non-interactive backend check for display vs close
    backend = plt.get_backend().lower()
    non_interactive = ["agg", "pdf", "ps", "svg", "cairo"]

    if show_plot and (backend not in non_interactive):
        plt.show()
    else:
        plt.close(fig)

    return fig, ax, plot_path


if __name__ == "__main__":
    from dataclasses import asdict
    from itertools import product

    from momp.lib.control import iter_list, make_case
    from momp.lib.convention import Case
    from momp.lib.loader import get_cfg, get_setting
    from momp.stats.benchmark import compute_metrics_multiple_years

    cfg, setting = get_cfg(), get_setting()

    cfg.ref_model = "climatology"
    cfg.probabilistic = False

    cfg_ref = copy.copy(cfg)
    cfg_ref.model_list = (cfg.ref_model,)
    layout_pool = iter_list(vars(cfg_ref))

    for combi in product(*layout_pool):
        case = make_case(Case, combi, vars(cfg_ref))
        print(f"Processing model onset evaluation for {case.case_name}")

        case_ref = {
            "model_dir": setting.ref_model_dir,
            "model_var": case.ref_model_var,
            "file_pattern": setting.ref_model_file_pattern,
            "unit_cvt": setting.ref_model_unit_cvt,
        }

        case.update(case_ref)

        if case.model == "climatology":
            case.years = case.years_clim

        case_cfg_ref = {**asdict(case), **asdict(setting)}

        metrics_df_dict, onset_da_dict = compute_metrics_multiple_years(**case_cfg_ref)
        break

    da = next(iter(onset_da_dict.values()))

    fig, _, _, _ = spatial_metrics_map(
        da,
        case.model,
        fig=None,
        ax=None,
        figsize=(8, 6),
        cmap="YlOrRd",
        n_colors=10,
        onset_plot=True,
        cbar_ssn=True,
        domain_mask=True,
        polygon_only=False,
        show_ylabel=True,
        title="climatology onset",
        **case_cfg_ref,
    )

    years_clim = case_cfg_ref.get("years_clim")
    dir_fig = case_cfg_ref.get("dir_fig")
    if dir_fig and years_clim:
        os.makedirs(dir_fig, exist_ok=True)
        plot_filename = f"climatology_onset_{tuple_to_str_range(years_clim)}.png"
        plot_path = os.path.join(dir_fig, plot_filename)
        fig.savefig(plot_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {plot_path}")

    plt.close(fig)