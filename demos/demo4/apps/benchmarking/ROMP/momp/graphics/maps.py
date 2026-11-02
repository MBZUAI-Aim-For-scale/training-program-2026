"""
momp/graphics/maps.py
Spatial metric map plotting routines with dynamic region boundary support.
"""

from ai_weather.spatial import climatology_boundary_path
from ai_weather.spatial import apply_jjas_mask
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
import xarray as xr
from ai_weather import config as weather_config
from ai_weather.spatial import boundary_geometry, clip_map_artist, clip_to_boundary, load_boundary_frame

from momp.params.region_def import domain, polygon_boundary
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


def _coordinate_spacing(values):
    numeric = np.asarray(values, dtype=float)
    numeric = np.unique(numeric[np.isfinite(numeric)])
    differences = np.diff(np.sort(numeric))
    differences = differences[differences > 0]
    if differences.size:
        return float(np.median(differences))
    return max(float(np.ptp(numeric)) * 0.02 if numeric.size else 0.0, 0.1)


def _region_geometry(shpfile_dir, region):
    if shpfile_dir:
        import geopandas as gpd
        from shapely.ops import unary_union

        source = os.path.expanduser(str(shpfile_dir))
        if os.path.isfile(source):
            boundary_files = [source]
        elif os.path.isdir(source):
            boundary_files = [
                os.path.join(root, filename)
                for root, _, files in os.walk(source)
                for filename in files
                if filename.lower().endswith((".geojson", ".json", ".shp"))
            ]
        else:
            boundary_files = []

        boundary_files.sort(
            key=lambda path: (
                not any(token in os.path.basename(path).lower() for token in ("adm3", "admin3", "woreda")),
                path.lower(),
            )
        )
        for boundary_file in boundary_files:
            try:
                frame = gpd.read_file(boundary_file)
                if frame.empty or frame.geometry.isna().all():
                    continue
                if frame.crs is not None and str(frame.crs).upper() not in {"EPSG:4326", "OGC:CRS84"}:
                    frame = frame.to_crs("EPSG:4326")

                region_key = str(region).strip().casefold()
                for field in ("ADM0_EN", "NAME_0", "COUNTRY", "ADMIN", "ADM1_EN", "NAME_1", "ADM2_EN", "NAME_2", "ADM3_EN", "NAME_3"):
                    if field in frame:
                        selected = frame[frame[field].astype(str).str.strip().str.casefold() == region_key]
                        if not selected.empty:
                            frame = selected
                            break
                return unary_union(frame.geometry)
            except Exception:
                continue

    try:
        from momp.utils.land_mask import get_shp

        return get_shp(region=region)
    except Exception:
        return None


def _region_grid_mask(latitudes, longitudes, region, geometry):
    lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)
    mask = np.ones(lon_grid.shape, dtype=bool)
    try:
        south, north, west, east = map(float, domain(region))
        if south < north and west < east:
            mask &= (lat_grid >= south) & (lat_grid <= north)
            mask &= (lon_grid >= west) & (lon_grid <= east)
            extent = (west, east, south, north)
        else:
            extent = None
    except (TypeError, ValueError, UnboundLocalError):
        extent = None

    if geometry is not None:
        try:
            from shapely.vectorized import contains

            mask &= contains(geometry, lon_grid, lat_grid)
        except ImportError:
            from shapely.geometry import Point

            mask &= np.array(
                [geometry.contains(Point(lon, lat)) for lon, lat in zip(lon_grid.ravel(), lat_grid.ravel())]
            ).reshape(lon_grid.shape)

    return mask, extent


def _geometry_boundaries(geometry):
    if geometry is None:
        return []
    boundaries = []
    for polygon in getattr(geometry, "geoms", [geometry]):
        if getattr(polygon, "geom_type", "") == "Polygon":
            coordinates = np.asarray(polygon.exterior.coords)
            boundaries.append((coordinates[:, 0], coordinates[:, 1]))
    return boundaries


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

    is_ethiopia = str(region).strip().casefold() == "ethiopia"
    if is_ethiopia:
        shpfile_dir = climatology_boundary_path(weather_config)
    nc_mask = kwargs.get("nc_mask", getattr(weather_config, "nc_mask", None))
    spatial_metrics = {
        key: (clip_to_boundary(apply_jjas_mask(value, nc_mask), shpfile_dir)
              if is_ethiopia else apply_jjas_mask(value, nc_mask))
        if isinstance(value, xr.DataArray) else value
        for key, value in spatial_metrics.items()
    }
    mean_mae = spatial_metrics["mean_mae"]
    far = spatial_metrics["false_alarm_rate"] * 100
    miss_rate = spatial_metrics["miss_rate"] * 100

    lats = mean_mae.lat.values
    lons = mean_mae.lon.values
    region_geometry = boundary_geometry(shpfile_dir) if is_ethiopia else _region_geometry(shpfile_dir, region)
    if not is_ethiopia:
        region_mask, _ = _region_grid_mask(lats, lons, region, region_geometry)
        mean_mae = mean_mae.where(region_mask)
        far = far.where(region_mask)
        miss_rate = miss_rate.where(region_mask)

    from ai_weather.map_extent import geographic_extent
    map_extent = kwargs.get("map_extent")
    if map_extent is None:
        support = np.isfinite(mean_mae.transpose("lat", "lon").values) | np.isfinite(far.transpose("lat", "lon").values) | np.isfinite(miss_rate.transpose("lat", "lon").values)
        region_bounds = None
        if region_geometry is not None:
            west, south, east, north = region_geometry.bounds
            region_bounds = (west, east, south, north)
        map_extent = geographic_extent(lons, lats, np.where(support, 1.0, np.nan), region_bounds=region_bounds)

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

    lon_step = _coordinate_spacing(lons) * (-1 if lons[-1] < lons[0] else 1)
    lat_step = _coordinate_spacing(lats) * (-1 if lats[-1] < lats[0] else 1)
    lon_edges = np.concatenate([lons - lon_step / 2, [lons[-1] + lon_step / 2]])
    lat_edges = np.concatenate([lats - lat_step / 2, [lats[-1] + lat_step / 2]])
    LON_edges, LAT_edges = np.meshgrid(lon_edges, lat_edges)

    map_lw = 0.75
    polygon_lw = 1.25
    panel_linewidth = 0.5
    tick_length = 3
    tick_width = 0.8
    txt_fsize = 8 if abs(lat_diff) >= 0.99 else None

    # Load shapefile outlines dynamically for requested region
    if is_ethiopia:
        region_boundaries = [boundary for geometry in load_boundary_frame(shpfile_dir).geometry
                             for boundary in _geometry_boundaries(geometry)]
    else:
        region_boundaries = _geometry_boundaries(region_geometry)
    if not region_boundaries:
        region_boundaries = get_region_outline(shpfile_dir, region=region) if shpfile_dir else []

    # Panel 1: Mean MAE
    masked_mae = np.ma.masked_invalid(mean_mae.values)
    im1 = axes[0].pcolormesh(LON_edges, LAT_edges, masked_mae, cmap="OrRd", vmin=0, vmax=15, shading="flat")

    if region_boundaries:
        for boundary in region_boundaries:
            axes[0].plot(boundary[0], boundary[1], color="black", linewidth=map_lw, zorder=10)
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
            axes[1].plot(boundary[0], boundary[1], color="black", linewidth=map_lw, zorder=10)
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
    if is_ethiopia:
        for target_ax, image in zip(axes, (im1, im2, im3)):
            clip_map_artist(image, target_ax, shpfile_dir)

    if region_boundaries:
        for boundary in region_boundaries:
            axes[2].plot(boundary[0], boundary[1], color="black", linewidth=map_lw, zorder=10)
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
        ax.set_xlim(map_extent[0], map_extent[1])
        ax.set_ylim(map_extent[2], map_extent[3])

        from matplotlib.ticker import MaxNLocator
        xticks = MaxNLocator(nbins=4).tick_values(map_extent[0], map_extent[1])
        xticks = xticks[(xticks >= map_extent[0]) & (xticks <= map_extent[1])]
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"{abs(x):g}°{'E' if x >= 0 else 'W'}" for x in xticks])

        if i == 0:
            yticks = MaxNLocator(nbins=4).tick_values(map_extent[2], map_extent[3])
            yticks = yticks[(yticks >= map_extent[2]) & (yticks <= map_extent[3])]
            ax.set_yticks(yticks)
            ax.set_yticklabels([f"{abs(y):g}°{'N' if y >= 0 else 'S'}" for y in yticks])
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
