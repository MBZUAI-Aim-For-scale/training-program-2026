"""
momp/utils/region.py
Region trimming, coordinate axis orientation formatting, and spatial mask application routines.
"""

import sys
from pathlib import Path
import numpy as np
import xarray as xr

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

import momp.params as params
from momp.utils import storage

try:
    from momp.params.region_def import domain
except ImportError:
    def domain(region=None, **kwargs):
        return 3, 15, 33, 48

from momp.utils.land_mask import polygon_mask, mask_land, shp_mask, apply_nc_mask, _dedup_coords


def swap_lon_axis(
    dataset: xr.Dataset, to: tuple[float, float], sort_ascending: bool = True
) -> xr.Dataset:
    """Swaps orientation of dataset longitude axis."""
    ds = dataset.copy()

    if "lon" not in ds.coords and "longitude" in ds.coords:
        ds = ds.rename({"longitude": "lon"})
    if "lat" not in ds.coords and "latitude" in ds.coords:
        ds = ds.rename({"latitude": "lat"})

    if "lon" not in ds.coords:
        return ds

    lon_vals = ds["lon"].values

    if to == (0, 360):
        new_lon = np.where(lon_vals < 0, lon_vals + 360, lon_vals)
    elif to == (-180, 180):
        new_lon = np.where(lon_vals > 180, lon_vals - 360, lon_vals)
    else:
        return ds

    ds = ds.assign_coords(lon=new_lon)

    if sort_ascending and np.issubdtype(ds.lon.dtype, np.number):
        ds = ds.sortby("lon", ascending=True)

    return ds


def lon_swap(ds_tag, *, region=None, **kwargs):
    """Standardizes longitude coordinate names and adjusts bounding box for requested region."""
    kwargs_clean = kwargs.copy()
    kwargs_clean.pop("region", None)

    if region is None:
        region = kwargs.get("region") or getattr(params, "region", "Ethiopia")

    lats, latn, lonw, lone = domain(region, **kwargs_clean)

    coord_list = list(ds_tag.coords.keys())

    if "lon" not in coord_list and "longitude" in coord_list:
        ds_tag = ds_tag.rename({"longitude": "lon"})
    if "lat" not in coord_list and "latitude" in coord_list:
        ds_tag = ds_tag.rename({"latitude": "lat"})

    coord_list = list(ds_tag.coords.keys())
    if "lon" not in coord_list:
        lat_coords = [variable for variable in coord_list if "lat" in variable]
        lon_coords = [variable for variable in coord_list if "lon" in variable]
        if lat_coords and lon_coords:
            ds_tag = ds_tag.rename({lat_coords[0]: "lat", lon_coords[0]: "lon"})

    if "lon" in ds_tag.coords and np.issubdtype(ds_tag.lon.dtype, np.number) and np.min(ds_tag.lon) < 0:
        lonw = ((lonw + 180) % 360) - 180
        lone = ((lone + 180) % 360) - 180

    if lonw > lone:
        if "lon" in ds_tag.coords and np.issubdtype(ds_tag.lon.dtype, np.number) and np.min(ds_tag.lon) < 0:
            ds_tag = swap_lon_axis(ds_tag, (0, 360)).compute()
            lonw = lonw % 360
            lone = lone % 360
        else:
            ds_tag = swap_lon_axis(ds_tag, (-180, 180)).compute()
            lonw = ((lonw + 180) % 360) - 180
            lone = ((lone + 180) % 360) - 180

    return lats, latn, lonw, lone, ds_tag


def lat_swap(ds_tag):
    """Ensures latitude orientation is sorted in ascending order (S to N)."""
    if "lat" in ds_tag.coords and len(ds_tag.lat) > 1:
        if np.issubdtype(ds_tag.lat.dtype, np.number):
            if ds_tag.lat[0] > ds_tag.lat[-1]:
                ds_tag = ds_tag.sortby("lat", ascending=True)
    return ds_tag


def coords_fmt(ds_tag, *, region=None, **kwargs):
    """Formats latitude and longitude coordinates for spatial slicing."""
    kwargs_clean = kwargs.copy()
    kwargs_clean.pop("region", None)

    # Cast lat and lon coordinates to float if stored as object/string
    for dim in ["lat", "lon"]:
        if dim in ds_tag.coords:
            try:
                if not np.issubdtype(ds_tag[dim].dtype, np.number):
                    ds_tag = ds_tag.assign_coords({dim: ds_tag[dim].astype(float)})
            except Exception:
                pass

    lats, latn, lonw, lone, ds_tag = lon_swap(ds_tag, region=region, **kwargs_clean)
    ds_tag = lat_swap(ds_tag)
    return lats, latn, lonw, lone, ds_tag


def region_select(ds, *, region=None, nc_mask=None, land_only=False, shp_only=False, **kwargs):
    """
    Slices dataset to region domain and applies NetCDF, land, or shapefile masks.
    """
    kwargs_clean = kwargs.copy()
    kwargs_clean.pop("region", None)
    kwargs_clean.pop("nc_mask", None)
    kwargs_clean.pop("land_only", None)
    kwargs_clean.pop("shp_only", None)

    if region is None:
        region = kwargs.get("region") or getattr(params, "region", "Ethiopia")
    if nc_mask is None:
        nc_mask = kwargs.get("nc_mask") or getattr(params, "nc_mask", None)

    ds = _dedup_coords(ds)
    lats, latn, lonw, lone, ds = coords_fmt(ds, region=region, **kwargs_clean)

    # Slice spatial domain with fallback
    if "lat" in ds.dims and "lon" in ds.dims:
        if np.issubdtype(ds.lat.dtype, np.number) and np.issubdtype(ds.lon.dtype, np.number):
            min_lat, max_lat = min(float(lats), float(latn)), max(float(lats), float(latn))
            min_lon, max_lon = min(float(lonw), float(lone)), max(float(lonw), float(lone))
            try:
                ds_reg = ds.sel(lat=slice(min_lat, max_lat), lon=slice(min_lon, max_lon))
            except (TypeError, KeyError):
                lat_mask = (ds.lat >= min_lat) & (ds.lat <= max_lat)
                lon_mask = (ds.lon >= min_lon) & (ds.lon <= max_lon)
                ds_reg = ds.sel(lat=ds.lat[lat_mask], lon=ds.lon[lon_mask])
        else:
            ds_reg = ds
    else:
        ds_reg = ds

    # Apply NetCDF Rainfall Mask if specified
    if nc_mask:
        try:
            ds_reg = apply_nc_mask(ds_reg, nc_mask)
        except Exception as e:
            print(f"[WARNING] Could not apply nc_mask ({nc_mask}): {e}")

    if land_only:
        try:
            ds_reg = mask_land(ds_reg)
        except Exception as e:
            print(f"[WARNING] Could not apply land_mask: {e}")

    if shp_only:
        try:
            ds_reg = shp_mask(ds_reg, region=region)
        except Exception as e:
            print(f"[WARNING] Could not apply shp_mask for region '{region}': {e}")

    return ds_reg


def domain_average(da, polygon=False, weighted=False, **kwargs):
    """Computes spatial domain average across latitude and longitude dimensions."""
    if polygon:
        da_slice = polygon_mask(da)
    else:
        da_slice = da

    if weighted and "lat" in da_slice.coords and np.issubdtype(da_slice.lat.dtype, np.number):
        weights = np.cos(np.deg2rad(da_slice.lat))
        weights.name = "weights"
        domain_avg = da_slice.weighted(weights).mean(dim=("lat", "lon"), skipna=True)
    else:
        domain_avg = da_slice.mean(dim=("lat", "lon"), skipna=True)

    return domain_avg
