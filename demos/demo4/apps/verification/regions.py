# Authors: The africlimate AI team
# SPDX-License-Identifier: MIT
"""Region definitions and spatial geometry (shared by the pipeline and the UI).

The region→countries table, the admin-1 subregion shapefiles, and the pure masking
geometry (keep-grids for gridded fields, point-in-polygon for stations) live here — in
the ``verification`` layer — so **both** the diagnostics pipeline (which subsets the IDR
to a region before pooling) and the Streamlit UI (which masks a field at display time)
can use them. The UI's ``ui.utils.regions`` re-exports these and adds display-time
concerns (mask caching, prewarming).

``regionmask`` and the Natural Earth country set load lazily on first mask, so reading
``REGION_COUNTRIES`` (e.g. to populate a dropdown) does not pull the heavy geodata
dependency. Country names must match the Natural Earth v5 50m dataset exactly. ``None``
means the full loaded domain (no mask).

A deployment country (``VERIFICATION_COUNTRY``) can additionally expose its **admin-1
subregions** (see ``_SUBREGION_SOURCES``): those are masked with the country's own
shapefile polygons rather than Natural Earth (which only knows whole countries).
"""
from __future__ import annotations

import threading as _threading
import warnings
from pathlib import Path

import numpy as np
import xarray as xr

from apps.verification import config

REGION_COUNTRIES: dict[str, list[str] | None] = {
    # ── Full domain ────────────────────────────────────────────────────────
    "Africa": None,

    # ── Multi-national regions ─────────────────────────────────────────────
    "SADEC": [
        "Angola", "Botswana", "Comoros", "Dem. Rep. Congo", "eSwatini",
        "Lesotho", "Madagascar", "Malawi", "Mauritius", "Mozambique",
        "Namibia", "Seychelles", "South Africa", "Tanzania", "Zambia",
        "Zimbabwe",
    ],
    "GHAF": [
        "Ethiopia", "Eritrea", "Djibouti", "Somalia", "Kenya",
        "Uganda", "Sudan", "S. Sudan",
    ],
    "West Africa": [
        "Senegal", "Gambia", "Guinea-Bissau", "Guinea", "Sierra Leone",
        "Liberia", "Côte d'Ivoire", "Ghana", "Togo", "Benin", "Nigeria",
        "Niger", "Burkina Faso", "Mali", "Mauritania", "Cabo Verde",
    ],
    "East Africa": [
        "Kenya", "Tanzania", "Uganda", "Rwanda", "Burundi",
        "Ethiopia", "Somalia", "Djibouti", "Eritrea",
    ],
    "Sahel": [
        "Senegal", "Gambia", "Mauritania", "Mali", "Burkina Faso",
        "Niger", "Nigeria", "Chad", "Sudan",
    ],
    "Central Africa": [
        "Dem. Rep. Congo", "Congo", "Cameroon", "Central African Rep.",
        "Gabon", "Eq. Guinea", "São Tomé and Principe", "Chad",
    ],
    "North Africa": [
        "Morocco", "Algeria", "Tunisia", "Libya", "Egypt", "Sudan",
        "Mauritania", "W. Sahara",
    ],

    # ── Individual countries ───────────────────────────────────────────────
    "Nigeria":               ["Nigeria"],
    "Ethiopia":              ["Ethiopia"],
    "Kenya":                 ["Kenya"],
    "Tanzania":              ["Tanzania"],
    "South Africa":          ["South Africa"],
    "Ghana":                 ["Ghana"],
    "Mozambique":            ["Mozambique"],
    "Madagascar":            ["Madagascar"],
    "Zambia":                ["Zambia"],
    "Zimbabwe":              ["Zimbabwe"],
    "Mali":                  ["Mali"],
    "Niger":                 ["Niger"],
    "Senegal":               ["Senegal"],
    "Cameroon":              ["Cameroon"],
    "Angola":                ["Angola"],
    "Sudan":                 ["Sudan"],
    "S. Sudan":              ["S. Sudan"],
    "Uganda":                ["Uganda"],
    "Burkina Faso":          ["Burkina Faso"],
    "Somalia":               ["Somalia"],
    "Chad":                  ["Chad"],
    "Rwanda":                ["Rwanda"],
    "Burundi":               ["Burundi"],
    "Malawi":                ["Malawi"],
    "Namibia":               ["Namibia"],
    "Botswana":              ["Botswana"],
    "Lesotho":               ["Lesotho"],
    "eSwatini":              ["eSwatini"],
    "Mauritius":             ["Mauritius"],
    "Seychelles":            ["Seychelles"],
    "Comoros":               ["Comoros"],
    "Cabo Verde":            ["Cabo Verde"],
    "São Tomé and Principe": ["São Tomé and Principe"],
    "Dem. Rep. Congo":       ["Dem. Rep. Congo"],
    "Congo":                 ["Congo"],
    "Gabon":                 ["Gabon"],
    "Eq. Guinea":            ["Eq. Guinea"],
    "Central African Rep.":  ["Central African Rep."],
    "Eritrea":               ["Eritrea"],
    "Djibouti":              ["Djibouti"],
    "Egypt":                 ["Egypt"],
    "Libya":                 ["Libya"],
    "Tunisia":               ["Tunisia"],
    "Algeria":               ["Algeria"],
    "Morocco":               ["Morocco"],
    "Mauritania":            ["Mauritania"],
    "Guinea":                ["Guinea"],
    "Guinea-Bissau":         ["Guinea-Bissau"],
    "Sierra Leone":          ["Sierra Leone"],
    "Liberia":               ["Liberia"],
    "Côte d'Ivoire":         ["Côte d'Ivoire"],
    "Togo":                  ["Togo"],
    "Benin":                 ["Benin"],
    "Gambia":                ["Gambia"],
}


def _country_match() -> str | None:
    """The REGION_COUNTRIES key matching the deployment country (case-insensitive), or None."""
    c = config.country()
    if not c:
        return None
    target = c.strip().lower()
    for key in REGION_COUNTRIES:
        if key.lower() == target:
            return key
    return None


def available_regions() -> list[str]:
    """Regions to offer in the UI.

    Under a country deployment: the country itself plus its admin-1 subregions (when a
    shapefile exists for it), so a user can subset from the whole country down to one
    region. Unscoped (pan-African): the full REGION_COUNTRIES list, no subregions.
    """
    match = _country_match()
    if match:
        return [match] + sorted(_get_subregions())
    return list(REGION_COUNTRIES)


def default_region() -> str:
    """The region selected by default: the deployment country, or 'Africa' when unscoped."""
    return _country_match() or "Africa"


def regions_for_domain() -> list[str]:
    """The region names the verification CLI may target for the current deployment.

    Same set as :func:`available_regions` — the deployment country + its admin-1 subregions
    (or the pan-African list when unscoped). This is the ``--region`` choice list; the
    whole-country entry (``default_region()``) is the region-less default.
    """
    return available_regions()


# Natural Earth country set, built once on first mask (kept out of import
# time so the dict above stays cheap to read).
_COUNTRIES = None


# GEOS geometry queries are NOT thread-safe the way plain shapely calls look:
# prepared geometries (used internally by regionmask masking and contains_xy)
# build their spatial index lazily on the FIRST query, and two threads racing
# that build segfault the process (observed: GEOS PreparedPolygon::contains
# crashing in the dashboard's precache pool once admin-1 masks multiplied the
# concurrent queries). The lazy shapefile/Natural-Earth loads race the same
# way. Every GEOS-touching entry point below therefore serializes on this
# reentrant lock; callers cache the resulting masks, so the serialization
# costs one pass per (region, grid) — not per render.
_GEOS_LOCK = _threading.RLock()


def _get_countries():
    global _COUNTRIES
    with _GEOS_LOCK:
        if _COUNTRIES is None:
            import regionmask  # type: ignore
            _COUNTRIES = regionmask.defined_regions.natural_earth_v5_1_2.countries_50
        return _COUNTRIES


# ── Sub-national admin regions (per deployment country) ───────────────────
# When VERIFICATION_COUNTRY names a country listed here, that country's admin-1
# units are offered in the Region dropdown alongside the country and masked with the
# shapefile polygons below (Natural Earth only knows whole countries). Add a row —
# country name (a REGION_COUNTRIES key) -> shapefile + admin-1 field — per agency.
_ADMIN_DIR = Path(__file__).resolve().parent / "assets" / "admin_boundaries"
_SUBREGION_SOURCES: dict[str, dict] = {
    "Ethiopia": {"shapefile": _ADMIN_DIR / "ethiopia" / "ETH_Regions.shp",
                 "name_field": "admin1Name"},
}

# {subregion name -> shapely geometry}, cached per deployment country (_SUBREGIONS_FOR).
_SUBREGIONS: dict | None = None
_SUBREGIONS_FOR: str | None = None


def _get_subregions() -> dict:
    """Admin-1 subregions of the deployment country as {name: geometry}, or {} if none.

    Empty when VERIFICATION_COUNTRY is unset (pan-African) or names a country with no
    shapefile row. Cached, but re-read if the deployment country changes (so tests that
    flip VERIFICATION_COUNTRY see the right set). ``geopandas`` is imported lazily (like
    ``regionmask`` above) so merely reading the region list for the sidebar stays cheap.
    """
    global _SUBREGIONS, _SUBREGIONS_FOR
    match = _country_match()
    with _GEOS_LOCK:
      if _SUBREGIONS is None or _SUBREGIONS_FOR != match:
          _SUBREGIONS_FOR = match
          source = _SUBREGION_SOURCES.get(match)
          if source is None:
              _SUBREGIONS = {}
          else:
              import geopandas as gpd  # type: ignore
              gdf = gpd.read_file(source["shapefile"])
              field = source["name_field"]
              _SUBREGIONS = {str(row[field]): row.geometry for _, row in gdf.iterrows()}
      return _SUBREGIONS


def _frac_mask(subset, lon: np.ndarray, lat: np.ndarray) -> np.ndarray | None:
    """Try mask_3D_frac_approx with regularised coords; return None if grid is irregular."""
    def _regularise(arr: np.ndarray) -> np.ndarray | None:
        if arr.ndim != 1 or len(arr) < 2:
            return None
        diffs = np.diff(arr.astype(float))
        if np.max(np.abs(diffs - diffs[0])) > 1e-4 * abs(diffs[0]):
            return None  # genuinely unequal spacing
        return np.linspace(float(arr[0]), float(arr[-1]), len(arr))

    reg_lon = _regularise(lon)
    reg_lat = _regularise(lat)
    if reg_lon is None or reg_lat is None:
        return None
    try:
        with _GEOS_LOCK, warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            frac = subset.mask_3D_frac_approx(reg_lon, reg_lat)
            return (frac > 0).any(dim="region").values
    except Exception:
        return None


def _country_keep(region, lon, lat, lat_name, lon_name):
    """Boolean keep grid for a REGION_COUNTRIES region, or None if no polygons match.

    Uses the fast fractional-overlap mask on a regular grid, falling back to the Natural
    Earth center-point mask on a genuinely irregular one.
    """
    countries = _get_countries()
    ids = set()
    for name in REGION_COUNTRIES[region]:
        try:
            ids.add(countries.map_keys(name))
        except KeyError:
            print(f"[mask_region] country '{name}' not found in Natural Earth — skipped")
    if not ids:
        return None

    # Select only the relevant countries so masking is fast
    positions = [i for i, n in enumerate(countries.numbers) if n in ids]
    keep_raw = _frac_mask(countries[positions], lon, lat)
    if keep_raw is None:
        # Truly irregular grid — fall back to center-point mask
        with _GEOS_LOCK:
            cm = countries.mask(lon, lat)
        rename = {}
        if lat_name != "lat" and "lat" in cm.dims:
            rename["lat"] = lat_name
        if lon_name != "lon" and "lon" in cm.dims:
            rename["lon"] = lon_name
        if rename:
            cm = cm.rename(rename)
        keep_raw = cm.isin(list(ids)).values
    return keep_raw


def _subregion_keep(region, lon, lat):
    """Boolean keep grid for an admin-1 subregion, by point-in-polygon on centers."""
    import shapely

    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    lon2d, lat2d = np.meshgrid(lon, lat)
    with _GEOS_LOCK:
        return shapely.contains_xy(_get_subregions()[region], lon2d, lat2d)


def stations_in_region(lons, lats, region: str) -> np.ndarray:
    """Boolean mask: which station points (lons, lats) fall inside ``region``.

    The point analogue of a grid mask. An admin-1 subregion uses containment against its
    shapefile polygon; a REGION_COUNTRIES key uses the union of its member-country
    polygons; a ``None`` member list ("Africa" / full domain) keeps every station.
    """
    import shapely

    lons = np.asarray(lons, dtype=float)
    lats = np.asarray(lats, dtype=float)

    subregions = _get_subregions()
    if region in subregions:
        with _GEOS_LOCK:
            return shapely.contains_xy(subregions[region], lons, lats)

    members = REGION_COUNTRIES.get(region)
    if members is None:
        return np.ones(lons.shape, dtype=bool)

    countries = _get_countries()
    ids = set()
    for name in members:
        try:
            ids.add(countries.map_keys(name))
        except KeyError:
            print(f"[stations_in_region] country '{name}' not in Natural Earth — skipped")
    if not ids:
        return np.ones(lons.shape, dtype=bool)

    with _GEOS_LOCK:
        geom = shapely.union_all(
            [p for p, n in zip(countries.polygons, countries.numbers) if n in ids]
        )
        return shapely.contains_xy(geom, lons, lats)


def region_boundary(region: str):
    """(lon, lat) polyline of a subregion's boundary for map overlay, or None.

    Flat arrays with ``np.nan`` gaps between rings, so one line trace draws each ring
    separately (exterior + any holes, across a MultiPolygon's parts). None when ``region``
    is not a shapefile subregion — whole-country and continent borders are already drawn by
    the map's Natural Earth geo layer.
    """
    geom = _get_subregions().get(region)
    if geom is None:
        return None

    lons: list[float] = []
    lats: list[float] = []

    def _add(ring):
        x, y = ring.coords.xy
        lons.extend(x); lons.append(np.nan)
        lats.extend(y); lats.append(np.nan)

    polys = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
    for poly in polys:
        _add(poly.exterior)
        for hole in poly.interiors:
            _add(hole)
    return np.asarray(lons, dtype=float), np.asarray(lats, dtype=float)
