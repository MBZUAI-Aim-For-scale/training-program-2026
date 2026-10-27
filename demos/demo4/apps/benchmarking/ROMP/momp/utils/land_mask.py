"""
momp/utils/land_mask.py
Land, polygon, and dynamic boundary shapefile masking routines with GCS auto-download support.
"""

import os
import glob
import sys
from pathlib import Path
from typing import Union
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.path import Path as MplPath
import regionmask

# Safe import for config
try:
    import config
except ImportError:
    try:
        from ai_weather import config
    except ImportError:
        _curr = Path(__file__).resolve()
        for _parent in _curr.parents:
            if (_parent / "ai_weather" / "config.py").exists() or (_parent / "config.py").exists():
                sys.path.insert(0, str(_parent / "ai_weather"))
                sys.path.insert(0, str(_parent))
                break
        import config

# Safe import for region definition
try:
    from momp.params.region_def import polygon_boundary
except ImportError:
    def polygon_boundary(da):
        return [], []

try:
    from momp.utils.standard import dim_fmt
except ImportError:
    def dim_fmt(ds):
        if "latitude" in ds.dims:
            ds = ds.rename({"latitude": "lat"})
        if "longitude" in ds.dims:
            ds = ds.rename({"longitude": "lon"})
        return ds


def _clean_coord_array(da_coord) -> np.ndarray:
    """Coerces a coordinate DataArray or array to a 1D float64 NumPy array."""
    vals = da_coord.values if hasattr(da_coord, "values") else da_coord
    return pd.to_numeric(np.asarray(vals).ravel(), errors="coerce").astype(np.float64)


def _dedup_coords(ds_or_da):
    """Removes duplicate coordinate indices along lat and lon dimensions."""
    obj = ds_or_da
    for dim in ["lat", "lon"]:
        if dim in obj.dims:
            vals = obj[dim].values
            if len(vals) > 0 and len(vals) != len(np.unique(vals)):
                _, uidx = np.unique(vals, return_index=True)
                obj = obj.isel({dim: np.sort(uidx)})
    return obj


def _sanitize_dataset_coords(ds: xr.Dataset) -> xr.Dataset:
    """Ensures lat/lon coordinates are clean, deduplicated, and numeric floats."""
    ds = _dedup_coords(ds)
    coord_updates = {}
    for dim in ["lat", "lon"]:
        if dim in ds.coords:
            clean_vals = _clean_coord_array(ds[dim])
            if not np.all(np.isnan(clean_vals)):
                coord_updates[dim] = clean_vals
    if coord_updates:
        ds = ds.assign_coords(coord_updates)
    return ds


def ensure_local_shapefile(shpfile_path=None, region="Ethiopia") -> str:
    """
    Ensures region shapefile exists locally. Downloads shapefile components
    from GCS bucket to /tmp/shapefile_<region> if needed.
    """
    region_clean = str(region).replace(" ", "_")
    local_tmp_dir = f"/tmp/shapefile_{region_clean}"
    os.makedirs(local_tmp_dir, exist_ok=True)

    if shpfile_path:
        p_str = str(shpfile_path).strip()
        if os.path.exists(p_str):
            if os.path.isdir(p_str):
                shps = glob.glob(os.path.join(p_str, "*.shp"))
                if shps:
                    return shps[0]
            elif p_str.endswith(".shp"):
                return p_str

    existing = glob.glob(os.path.join(local_tmp_dir, "*.shp"))
    if existing and (not shpfile_path or not str(shpfile_path).startswith("gs://")):
        return existing[0]

    gcs_uri = None
    if shpfile_path and str(shpfile_path).startswith("gs:"):
        gcs_uri = str(shpfile_path)
    else:
        try:
            if hasattr(config, "resolve_path"):
                gcs_uri = config.resolve_path(f"demo4/DATA/{region}/shapefile")
        except Exception:
            pass

        if not gcs_uri or not gcs_uri.startswith("gs://"):
            bucket = os.environ.get("DEMO_BUCKET", "aim4scale_training_2026")
            gcs_uri = f"gs://{bucket}/demo4/DATA/{region}/shapefile"

    gcs_clean = gcs_uri[5:] if gcs_uri.startswith("gs://") else gcs_uri.lstrip("gs:/")

    try:
        import gcsfs
        fs = None
        for tok in [None, "anon"]:
            try:
                t_fs = gcsfs.GCSFileSystem(token=tok)
                if t_fs.exists(gcs_clean):
                    fs = t_fs
                    break
            except Exception:
                continue

        if fs and fs.exists(gcs_clean):
            for gfile in fs.ls(gcs_clean):
                fname = os.path.basename(gfile)
                if fname:
                    target = os.path.join(local_tmp_dir, fname)
                    if not os.path.exists(target):
                        fs.get(gfile, target)

            downloaded = glob.glob(os.path.join(local_tmp_dir, "*.shp"))
            if downloaded:
                return downloaded[0]
    except Exception as e:
        print(f"[WARNING] Could not download shapefile for '{region}' from GCS ({gcs_uri}): {e}")

    downloaded = glob.glob(os.path.join(local_tmp_dir, "*.shp"))
    return downloaded[0] if downloaded else (str(shpfile_path) if shpfile_path else "")


def get_region_outline(shpfile_path=None, region="Ethiopia"):
    """Loads polygon coordinate boundaries for the target region from shapefile."""
    import geopandas as gpd

    try:
        real_path = ensure_local_shapefile(shpfile_path, region=region)
        if not os.path.exists(real_path) and not str(real_path).endswith(".shp"):
            return []

        gdf = gpd.read_file(real_path)
        if region and hasattr(gdf, "columns"):
            name_cols = [c for c in gdf.columns if c.upper() in ["NAME", "NAME_0", "COUNTRY", "ADMIN"]]
            if name_cols:
                filt = gdf[gdf[name_cols[0]].astype(str).str.lower() == str(region).lower()]
                if not filt.empty:
                    gdf = filt

        boundaries = []
        for geom in gdf.geometry:
            geoms = getattr(geom, "geoms", [geom])
            for g in geoms:
                if hasattr(g, "exterior"):
                    coords = list(g.exterior.coords)
                    boundaries.append(([c[0] for c in coords], [c[1] for c in coords]))
        return boundaries
    except Exception as e:
        print(f"[WARNING] Failed to load shapefile outline for '{region}': {e}")
        return []


get_india_outline = get_region_outline


def apply_nc_mask(ds, nc_mask, mask_var=None, keep_value=1):
    """
    Masks an xarray Dataset or DataArray using a NetCDF/GCS mask file.
    Safely converts string/object spatial coordinates to float64 to avoid ufunc errors.
    """
    from momp.utils import storage

    close_needed = False

    if isinstance(nc_mask, (str, Path)):
        mask_path = str(nc_mask).strip()
        if config.is_uri(mask_path):
            mask_path = f"gs://{config._strip(mask_path)}"
        mask_ds = storage.open_dataset(mask_path)
        close_needed = True
    elif isinstance(nc_mask, (xr.Dataset, xr.DataArray)):
        mask_ds = nc_mask
    else:
        mask_ds = storage.open_dataset(str(nc_mask))
        close_needed = True

    try:
        ds = _sanitize_dataset_coords(ds)

        if isinstance(mask_ds, xr.DataArray):
            mask_da = mask_ds
        else:
            mask_ds = dim_fmt(mask_ds)
            mask_var = mask_var or (list(mask_ds.data_vars)[0] if len(mask_ds.data_vars) > 0 else None)
            mask_da = mask_ds[mask_var] if mask_var else mask_ds

        mask_da = _dedup_coords(mask_da)

        if "lat" in mask_da.dims and "lon" in mask_da.dims and "lat" in ds.dims and "lon" in ds.dims:
            ds_lat_f = _clean_coord_array(ds.lat)
            ds_lon_f = _clean_coord_array(ds.lon)
            mask_lat_f = _clean_coord_array(mask_da.lat)
            mask_lon_f = _clean_coord_array(mask_da.lon)

            # Calculate nearest indices numerically
            lat_idx = [int(np.argmin(np.abs(mask_lat_f - v))) for v in ds_lat_f]
            lon_idx = [int(np.argmin(np.abs(mask_lon_f - v))) for v in ds_lon_f]

            mask_aligned = mask_da.isel(lat=lat_idx, lon=lon_idx)
            raw_vals = mask_aligned.values
            bool_vals = (raw_vals == keep_value) if raw_vals.dtype != bool else raw_vals

            mask_bool = xr.DataArray(
                bool_vals,
                coords={"lat": ds.lat, "lon": ds.lon},
                dims=["lat", "lon"]
            )
        else:
            mask_bool = (mask_da == keep_value) if mask_da.dtype != bool else mask_da

        ds_out = ds.copy()
        for v in ds_out.data_vars:
            ds_out[v] = ds_out[v].where(mask_bool)

        return ds_out

    finally:
        if close_needed and hasattr(mask_ds, "close"):
            mask_ds.close()


def points_inside_polygon(polygon_lon, polygon_lat, grid_lons, grid_lats):
    """Find grid points inside a polygon boundary."""
    polygon_path = MplPath(np.column_stack((polygon_lon, polygon_lat)))
    if grid_lons.ndim == 1 and grid_lats.ndim == 1:
        lon_grid, lat_grid = np.meshgrid(grid_lons, grid_lats)
    else:
        lon_grid, lat_grid = grid_lons, grid_lats

    points = np.column_stack((lon_grid.ravel(), lat_grid.ravel()))
    inside_mask = polygon_path.contains_points(points).reshape(lon_grid.shape)
    return inside_mask, lon_grid[inside_mask], lat_grid[inside_mask]


def polygon_mask(da_model):
    """Masks DataArray based on CMZ polygon boundary."""
    p_lat, p_lon = polygon_boundary(da_model)
    inside_mask, _, _ = points_inside_polygon(p_lon, p_lat, da_model.lon.values, da_model.lat.values)
    return da_model.where(inside_mask)


def polygon_outline(ax, polygon1_lon, polygon1_lat, linewidth=1.25):
    """Add polygon boundary outline to basemap / cartopy / matplotlib axes."""
    import cartopy.crs as ccrs
    from shapely.geometry import Polygon as ShapelyPolygon
    from cartopy.feature import ShapelyFeature

    if len(polygon1_lon) > 0 and len(polygon1_lat) > 0:
        poly_geom = ShapelyPolygon(list(zip(polygon1_lon, polygon1_lat)))
        poly_feature = ShapelyFeature(
            [poly_geom],
            crs=ccrs.PlateCarree(),
            edgecolor="black",
            facecolor="none",
            linewidth=linewidth,
        )
        ax.add_feature(poly_feature, zorder=10)
    return ax


def add_polygon(ax, da, polygon, return_polygon=False, linewidth=1.25):
    """Overlay polygon boundary onto spatial plot axes."""
    polygon_defined = False
    polygon1_lat, polygon1_lon = [], []

    if polygon:
        polygon1_lat, polygon1_lon = polygon_boundary(da)
        if len(polygon1_lat) > 0 and len(polygon1_lon) > 0:
            polygon_defined = True

    if polygon_defined:
        ax = polygon_outline(ax, polygon1_lon, polygon1_lat, linewidth=linewidth)

    if return_polygon:
        return ax, polygon1_lat, polygon1_lon, polygon_defined
    return ax


def create_land_sea_mask(obj: Union[xr.Dataset, xr.DataArray], as_boolean: bool = False) -> xr.DataArray:
    """Generates a Natural Earth land-sea mask aligned to dataset coordinates."""
    land_region = regionmask.defined_regions.natural_earth_v5_0_0.land_110
    mask = land_region.mask(obj["lon"], lat=obj["lat"])
    return xr.where(mask, False, True) if as_boolean else xr.where(mask, 0, 1)


def mask_land(da, land=True):
    """Masks DataArray to retain land or ocean points."""
    mask = create_land_sea_mask(da)
    return da.where(mask == 1 if land else mask == 0)


def get_shp(region="Ethiopia", resolution="10m", category="cultural", name="admin_0_countries"):
    """Fetches geometry from Cartopy NaturalEarth reader."""
    import cartopy.io.shapereader as shpreader
    shp = shpreader.natural_earth(resolution=resolution, category=category, name=name)
    for country in shpreader.Reader(shp).records():
        c_name = str(country.attributes.get("NAME", "")).lower()
        if c_name == str(region).lower() or str(region).lower() in c_name:
            return country.geometry
    return None


def shp_mask(da, region="Ethiopia", **kwargs):
    """Masks DataArray using country polygon geometry."""
    from shapely.vectorized import contains
    geom = get_shp(region=region)
    if geom is None:
        return da

    lons, lats = np.meshgrid(da.lon.values, da.lat.values)
    points = np.column_stack((lons.ravel(), lats.ravel()))
    mask = contains(geom, points[:, 0], points[:, 1]).reshape(lons.shape)
    mask_da = xr.DataArray(mask, dims=["lat", "lon"], coords={"lat": da.lat, "lon": da.lon})
    return da.where(mask_da)


def shp_outline(ax, region="Ethiopia", **kwargs):
    """Plots country boundaries onto matplotlib/cartopy axes."""
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    geom = get_shp(region=region)
    if geom is None:
        return ax

    if hasattr(ax, "add_geometries"):
        ax.add_geometries([geom], crs=ccrs.PlateCarree(), facecolor="none", edgecolor="black", linewidth=1.5, zorder=10)
    if hasattr(ax, "add_feature"):
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=":")
    return ax
