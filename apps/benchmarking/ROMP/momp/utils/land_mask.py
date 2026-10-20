import numpy as np
from matplotlib.path import Path
from typing import Union
import regionmask
import xarray as xr
from momp.params.region_def import polygon_boundary
from matplotlib.patches import Polygon
from momp.utils.standard import dim_fmt


def _drop_duplicate_coords(obj: Union[xr.Dataset, xr.DataArray]) -> Union[xr.Dataset, xr.DataArray]:
    """Drop duplicate coordinate values along 'lon' and 'lat' dimensions if present."""
    for dim in ['lon', 'lat', 'longitude', 'latitude']:
        if dim in obj.coords and not obj.indexes[dim].is_unique:
            _, index = np.unique(obj[dim].values, return_index=True)
            obj = obj.isel({dim: np.sort(index)})
    return obj


# Function to find grid points inside a polygon (For core-monsoon zone analysis)
def points_inside_polygon(polygon_lon, polygon_lat, grid_lons, grid_lats):
    """
    Find grid points that are inside a polygon.

    Parameters:
    polygon_lon: array of polygon longitude vertices
    polygon_lat: array of polygon latitude vertices
    grid_lons: array of grid longitude points
    grid_lats: array of grid latitude points

    Returns:
    inside_mask: boolean array indicating which points are inside
    inside_lons: longitude coordinates of points inside polygon
    inside_lats: latitude coordinates of points inside polygon
    """
    polygon_vertices = np.column_stack((polygon_lon, polygon_lat))
    polygon_path = Path(polygon_vertices)

    # Create meshgrid if needed
    if grid_lons.ndim == 1 and grid_lats.ndim == 1:
        lon_grid, lat_grid = np.meshgrid(grid_lons, grid_lats)
    else:
        lon_grid, lat_grid = grid_lons, grid_lats

    # Flatten the grids to test each point
    points = np.column_stack((lon_grid.ravel(), lat_grid.ravel()))

    # Test which points are inside the polygon
    inside_mask = polygon_path.contains_points(points)
    inside_mask = inside_mask.reshape(lon_grid.shape)

    # Get coordinates of points inside polygon
    inside_lons = lon_grid[inside_mask]
    inside_lats = lat_grid[inside_mask]

    return inside_mask, inside_lons, inside_lats


def polygon_mask(da_model):
    """mask data based on polygon boundary"""

    orig_lat = da_model.lat.values
    orig_lon = da_model.lon.values

    polygon1_lat, polygon1_lon = polygon_boundary(da_model)

    inside_mask, inside_lons, inside_lats = points_inside_polygon(polygon1_lon, polygon1_lat, orig_lon, orig_lat)

    da_model_slice = da_model.where(inside_mask)

    return da_model_slice


def polygon_outline(ax, polygon1_lon, polygon1_lat, linewidth=1.25):
    """ add polygon boundary to basemap """
    import cartopy.crs as ccrs
    from shapely.geometry import Polygon as ShapelyPolygon
    from cartopy.feature import ShapelyFeature
    
    poly_geom = ShapelyPolygon(list(zip(polygon1_lon, polygon1_lat)))
    
    # wrap as a Cartopy feature
    poly_feature = ShapelyFeature(
        [poly_geom],
        crs=ccrs.PlateCarree(),
        edgecolor='black',
        facecolor='none',
        linewidth=linewidth
    )
    
    # add to axes above gridlines
    ax.add_feature(poly_feature, zorder=10)

    return ax


def add_polygon(ax, da, polygon, return_polygon=False, linewidth=1.25):
    polygon_defined = False

    if polygon:
        polygon1_lat, polygon1_lon = polygon_boundary(da)

        if len(polygon1_lat) > 0 and len(polygon1_lon) > 0:
            polygon_defined = True

    # Add CMZ polygon only if defined
    if polygon_defined:
        ax = polygon_outline(ax, polygon1_lon, polygon1_lat, linewidth=linewidth)

    if return_polygon:
        return ax, polygon1_lat, polygon1_lon, polygon_defined
    else:
        return ax


def get_india_outline(shpfile_path):
    """
    Get region outline coordinates from shapefile.
    """
    import geopandas as gpd
    india_gdf = gpd.read_file(shpfile_path)

    boundaries = []
    for geom in india_gdf.geometry:
        if hasattr(geom, 'exterior'):
            coords = list(geom.exterior.coords)
            lon_coords = [coord[0] for coord in coords]
            lat_coords = [coord[1] for coord in coords]
            boundaries.append((lon_coords, lat_coords))
        elif hasattr(geom, 'geoms'):
            for sub_geom in geom.geoms:
                if hasattr(sub_geom, 'exterior'):
                    coords = list(sub_geom.exterior.coords)
                    lon_coords = [coord[0] for coord in coords]
                    lat_coords = [coord[1] for coord in coords]
                    boundaries.append((lon_coords, lat_coords))
    return boundaries


def create_land_sea_mask(
    obj: Union[xr.Dataset, xr.DataArray],
    as_boolean: bool = False,
) -> xr.DataArray:
    """Generate a land-sea mask (1 for land, 0 for sea) for a given xarray Dataset or DataArray."""

    land_mask = regionmask.defined_regions.natural_earth_v5_0_0.land_110

    lon = obj["lon"]
    lat = obj["lat"]

    # Mask the land-sea mask to match the dataset's coordinates
    land_sea_mask = land_mask.mask(lon, lat=lat)

    if as_boolean:
        land_sea_mask = xr.where(land_sea_mask, False, True)
    else:
        land_sea_mask = xr.where(land_sea_mask, 0, 1)

    return land_sea_mask


def mask_land(da, land=True):
    """
    Mask a DataArray to select either land or sea areas.
    """
    land_sea_mask = create_land_sea_mask(da)

    if land:
        da_masked = da.where(land_sea_mask == 1)
    else:
        da_masked = da.where(land_sea_mask == 0)

    return da_masked


def get_shp(region='Ethiopia', resolution='10m', category='cultural', name='admin_0_countries'):
    """ Create country boundaries"""
    import cartopy.io.shapereader as shpreader

    ethiopia_shp = shpreader.natural_earth(resolution=resolution, category=category, name=name)

    region_geom = None
    for country in shpreader.Reader(ethiopia_shp).records():
        if country.attributes['NAME'] == region:
            region_geom = country.geometry
            return region_geom


def shp_mask(da, region='Ethiopia', resolution='10m', category='cultural', name='admin_0_countries', 
             return_mask=False, **kwargs):
    """ Create mask based on country boundaries"""
    from shapely.vectorized import contains

    # Deduplicate coordinates prior to masking
    da = _drop_duplicate_coords(da)

    region_geom = get_shp(region=region, resolution=resolution, category=category, name=name)

    if region_geom is None:
        print("    WARNING - specified region is not in cartopy.io.shapereader")
        return da if not return_mask else (da, None)

    lons, lats = np.meshgrid(da.lon, da.lat)
    points = np.column_stack((lons.ravel(), lats.ravel()))
    mask = contains(region_geom, points[:, 0], points[:, 1])
    mask = mask.reshape(lons.shape)
    mask_da = xr.DataArray(mask, dims=['lat', 'lon'], coords={'lat': da.lat, 'lon': da.lon})

    da_masked = da.where(mask_da)

    if return_mask:
        return da_masked, mask_da
    else:
        return da_masked


def shp_outline(ax, region='Ethiopia', resolution='10m', category='cultural', name='admin_0_countries'):
    """ add shapefile country boundaries to plot"""
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    region_geom = get_shp(region=region, resolution=resolution, category=category, name=name)

    if region_geom is None:
        print("WARNING - specified region is not in cartopy.io.shapereader")
        return ax

    if hasattr(ax, "add_geometries"):
        ax.add_geometries(
            [region_geom],
            crs=ccrs.PlateCarree(),
            facecolor="none",
            edgecolor="black",
            linewidth=1.5,
            zorder=10
        )
    else:
        geoms = getattr(region_geom, "geoms", [region_geom])
        for geom in geoms:
            exterior = getattr(geom, "exterior", None)
            if exterior is None:
                continue
            x, y = exterior.xy
            ax.plot(x, y, color="black", linewidth=1.5, zorder=10)

    if hasattr(ax, "add_feature"):
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=":")

    return ax


from pathlib import Path
import xarray as xr
import numpy as np

def apply_nc_mask(ds, nc_mask, mask_var=None, keep_value=1):
    """
    Mask an xarray Dataset/DataArray using a 0/1 mask stored in a NetCDF file or Dataset/DataArray object.
    """
    # 1. Clean coordinate duplicate issues on target dataset
    ds = _drop_duplicate_coords(ds)

    # 2. Check if nc_mask is a file path (str or Path object) or already an xarray object
    if isinstance(nc_mask, (str, Path)):
        mask_ds = xr.open_dataset(nc_mask)
        should_close = True
    else:
        mask_ds = nc_mask
        should_close = False

    try:
        # Format dimensions AFTER opening the file
        mask_ds = dim_fmt(mask_ds)
        mask_ds = _drop_duplicate_coords(mask_ds)

        # Pick mask variable
        if isinstance(mask_ds, xr.Dataset):
            if mask_var is None:
                if len(mask_ds.data_vars) != 1:
                    raise ValueError(
                        f"Mask file has multiple variables {list(mask_ds.data_vars)}; "
                        "please specify mask_var."
                    )
                mask = next(iter(mask_ds.data_vars.values()))
            else:
                mask = mask_ds[mask_var]
        else:
            mask = mask_ds

        # Convert mask to boolean
        if mask.dtype == bool:
            mask_bool = mask
        else:
            mask_bool = (mask == keep_value)

        # Reindex mask to match target grid safely
        try:
            mask_bool = mask_bool.reindex_like(ds, method="nearest")
        except Exception:
            mask_bool = mask_bool.assign_coords(lon=ds.lon, lat=ds.lat)

        # Apply mask
        if isinstance(ds, xr.Dataset):
            ds_masked = ds.copy()
            for var in ds_masked.data_vars:
                ds_masked[var] = ds_masked[var].where(mask_bool)
            return ds_masked
        else:
            return ds.where(mask_bool)

    finally:
        if should_close:
            mask_ds.close()