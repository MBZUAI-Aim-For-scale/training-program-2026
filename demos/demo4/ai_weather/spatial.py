"""Shared geographic masking, exact plot clipping and cached boundary reads."""

from functools import lru_cache
from io import BytesIO
from pathlib import Path

import numpy as np
import xarray as xr


def climatology_boundary_path(config):
    """Derive Ethiopia zones from canonical config.shpfile_dir, even in old sessions."""
    folder = getattr(config, "shpfile_dir", None)
    if folder:
        folder = str(folder).rstrip("/")
        if folder.lower().endswith((".shp", ".geojson", ".json")):
            folder = folder.rsplit("/", 1)[0] if "/" in folder else "."
        return folder + "/ETH_Zones.geojson"
    explicit = getattr(config, "CLIMATOLOGY_SHAPEFILE", None)
    if explicit:
        return str(explicit)
    raise ValueError("Set shpfile_dir in config.py to the directory containing ETH_Zones.geojson.")


@lru_cache(maxsize=4)
def _jjas_mask(path, modified_ns):
    opener = xr.open_dataset
    if str(path).startswith("gs://"):
        from ai_weather import config
        opener = config.open_dataset
    with opener(path) as ds:
        ds = ds.rename({name: target for name, target in (("latitude", "lat"), ("longitude", "lon"))
                        if name in ds.dims and target not in ds.dims})
        candidates = [v for v in ds.data_vars.values() if set(v.dims) == {"lat", "lon"}]
        if not candidates:
            raise ValueError(f"JJAS mask has no two-dimensional latitude/longitude variable: {path}")
        return candidates[0].transpose("lat", "lon").load()


def apply_jjas_mask(data, mask_path):
    """Apply configured JJAS keep/exclusion flags without extrapolating coverage."""
    if not mask_path or not {"lat", "lon"}.issubset(data.dims):
        return data
    if str(mask_path).startswith("gs://"):
        mask = _jjas_mask(str(mask_path), None)
    else:
        source = Path(mask_path).expanduser()
        mask = _jjas_mask(str(source), source.stat().st_mtime_ns)

    def indices_and_coverage(values, targets):
        values = np.asarray(values, dtype=float)
        targets = np.asarray(targets, dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("JJAS mask coordinates must be finite")
        spacing = np.diff(np.unique(values))
        half_cell = float(np.median(spacing)) / 2 if spacing.size else .5
        indices = np.abs(values[:, None] - targets[None, :]).argmin(axis=0)
        covered = (targets >= values.min() - half_cell) & (targets <= values.max() + half_cell)
        return indices, covered

    rows, lat_covered = indices_and_coverage(mask.lat, data.lat)
    cols, lon_covered = indices_and_coverage(mask.lon, data.lon)
    values = mask.values[np.ix_(rows, cols)]
    keep = (values == 1) & lat_covered[:, None] & lon_covered[None, :]
    keep = xr.DataArray(keep, dims=("lat", "lon"), coords={"lat": data.lat, "lon": data.lon})
    return data.where(keep)


@lru_cache(maxsize=4)
def _boundary_frame(path, modified_ns):
    import geopandas as gpd

    if str(path).startswith("gs://") and Path(path).suffix.lower() in {".geojson", ".json"}:
        from ai_weather import config

        # GDAL does not inherit the authenticated gcsfs client used by config.exists.
        with config.gcs_fs().open(config._strip(path), "rb") as source:
            frame = gpd.read_file(BytesIO(source.read()))
    else:
        frame = gpd.read_file(path)
    if frame.crs is None:
        raise ValueError(f"Boundary has no coordinate reference system: {path}")
    frame = frame.to_crs("EPSG:4326")
    frame = frame[frame.geometry.notna() & ~frame.geometry.is_empty].copy()
    if frame.empty:
        raise ValueError(f"Boundary contains no usable polygons: {path}")
    return frame


def load_boundary_frame(path):
    source = Path(path).expanduser()
    modified_ns = source.stat().st_mtime_ns if source.is_file() else None
    return _boundary_frame(str(path), modified_ns)


@lru_cache(maxsize=4)
def _boundary_union(path, modified_ns):
    return _boundary_frame(path, modified_ns).geometry.union_all()


def boundary_geometry(path):
    source = Path(path).expanduser()
    modified_ns = source.stat().st_mtime_ns if source.is_file() else None
    return _boundary_union(str(path), modified_ns)


def clip_to_boundary(data, path):
    """Mask outside polygon centers; preserve missing cells inside the region."""
    from shapely import intersects_xy

    lon_name = next((name for name in ("lon", "longitude") if name in data.coords), None)
    lat_name = next((name for name in ("lat", "latitude") if name in data.coords), None)
    if not lon_name or not lat_name:
        return data
    lon, lat = data[lon_name], data[lat_name]
    if lon.ndim != 1 or lat.ndim != 1:
        raise ValueError("Boundary masking requires rectilinear longitude/latitude coordinates")
    xx, yy = np.meshgrid(lon.values, lat.values)
    inside = intersects_xy(boundary_geometry(path), xx, yy)
    mask = xr.DataArray(inside, dims=(lat.dims[0], lon.dims[0]),
                        coords={dim: data.coords[dim] for dim in (lat.dims[0], lon.dims[0])
                                if dim in data.coords})
    return data.where(mask)


def clip_map_artist(artist, ax, path):
    """Clip rendered cell edges exactly to polygons, including interior holes.

    Callers use longitude/latitude axes (plain Matplotlib or PlateCarree).
    """
    from cartopy.mpl.path import shapely_to_path
    from matplotlib.patches import PathPatch

    patch = PathPatch(shapely_to_path(boundary_geometry(path)), transform=ax.transData)
    artist.set_clip_path(patch)
