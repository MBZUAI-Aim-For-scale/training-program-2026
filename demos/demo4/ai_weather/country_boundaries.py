"""Download and validate reusable country boundaries outside the UI thread."""

from io import BytesIO
import json
from pathlib import Path
import ssl
from tempfile import NamedTemporaryFile
from urllib.request import Request, urlopen

import certifi

COUNTRY_ISO3 = {
    "Senegal": "SEN", "Rwanda": "RWA", "Colombia": "COL", "Peru": "PER",
    "Bolivia": "BOL", "Philippines": "PHL", "Indonesia": "IDN", "Pakistan": "PAK",
    "Nigeria": "NGA", "Ethiopia": "ETH", "Kenya": "KEN", "Sierra Leone": "SLE",
    "Chile": "CHL", "Bangladesh": "BGD",
}
BOUNDARY_LEVELS = ("ADM0", "ADM1", "ADM2")


def cached_boundary_path(country, level, cache_root):
    """Only inspect local files; never perform network access on a page rerun."""
    iso = COUNTRY_ISO3.get(country)
    if iso is None or level not in BOUNDARY_LEVELS:
        return None
    path = Path(cache_root) / iso / f"{level}.geojson"
    return path if path.is_file() else None


def _read_url(url):
    # Fetch geoBoundaries LFS data directly, avoiding GitHub's redirect endpoint.
    github_prefix = "https://github.com/wmgeolab/geoBoundaries/raw/"
    if url.startswith(github_prefix):
        url = "https://media.githubusercontent.com/media/wmgeolab/geoBoundaries/" + url[len(github_prefix):]
    request = Request(url, headers={"User-Agent": "AI-Weather-Benchmark/1.0"})
    context = ssl.create_default_context()
    # Supplement platform/custom roots for Python installs with incomplete CA stores.
    context.load_verify_locations(cafile=certifi.where())
    with urlopen(request, timeout=30, context=context) as response:
        return response.read()


def _atomic_write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def download_country_boundaries(country, cache_root):
    """Fetch each available level independently; retain successful partial downloads.

    Call from BackgroundTask. GeoJSON avoids shapefile sidecar files and retains
    full administrative names. Metadata preserves source and attribution details.
    """
    import geopandas as gpd

    from ai_weather.spatial import load_boundary_frame

    iso = COUNTRY_ISO3[country]
    result = {"paths": {}, "errors": {}}
    for level in BOUNDARY_LEVELS:
        try:
            path = cached_boundary_path(country, level, cache_root)
            if path is not None:
                try:
                    load_boundary_frame(str(path))
                except (OSError, ValueError):
                    # A damaged local cache can be replaced by an explicit retry.
                    path = None
            if path is None:
                api = f"https://www.geoboundaries.org/api/current/gbOpen/{iso}/{level}/"
                metadata = json.loads(_read_url(api))
                if not isinstance(metadata, dict) or not metadata.get("gjDownloadURL"):
                    raise ValueError("No boundary available from geoBoundaries")
                geometry_url = metadata.get("simplifiedGeometryGeoJSON") or metadata["gjDownloadURL"]
                frame = gpd.read_file(BytesIO(_read_url(geometry_url)))
                if frame.crs is None:
                    raise ValueError("Boundary has no coordinate reference system")
                frame = frame.to_crs("EPSG:4326")
                frame = frame[frame.geometry.notna() & ~frame.geometry.is_empty].copy()
                if not frame.geometry.is_valid.all():
                    frame.geometry = frame.geometry.make_valid()
                frame = frame[frame.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
                if frame.empty:
                    raise ValueError("Boundary contains no usable polygons")
                if level != "ADM0":
                    if "shapeName" not in frame or frame["shapeName"].dropna().empty:
                        raise ValueError("Boundary contains no administrative names")
                    frame[f"Admin{level[-1]}"] = frame["shapeName"]
                path = Path(cache_root) / iso / f"{level}.geojson"
                _atomic_write(path.with_suffix(".metadata.json"),
                              json.dumps(metadata, indent=2).encode("utf-8"))
                _atomic_write(path, frame.to_json().encode("utf-8"))
            # Warm the shared geometry cache in the worker, never in Streamlit.
            load_boundary_frame(str(path))
            result["paths"][level] = str(path)
        except Exception as exc:  # noqa: BLE001 -- retain other levels after any provider/GDAL failure
            result["errors"][level] = str(exc)
    return result
