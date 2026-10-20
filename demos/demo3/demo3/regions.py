"""Program countries and their plotting boxes.

Raw extents come from Natural Earth 1:110m admin-0 boundaries. Each box is
padded by max(2 deg, 10% of the country's span) on every side and snapped
outward to the 0.25 deg grid, so plots show context around the border
instead of cutting at the country's edge.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# (lon_min, lat_min, lon_max, lat_max) from ne_110m_admin_0_countries
_RAW_EXTENTS = {
    # Africa
    "Ethiopia": (32.95, 3.42, 47.79, 14.96),
    "Kenya": (33.89, -4.68, 41.86, 5.51),
    "Nigeria": (2.69, 4.24, 14.58, 13.87),
    "Rwanda": (29.02, -2.92, 30.82, -1.13),
    "Senegal": (-17.63, 12.33, -11.47, 16.60),
    "Sierra Leone": (-13.25, 6.79, -10.23, 10.05),
    # Asia
    "Bangladesh": (88.08, 20.67, 92.67, 26.45),
    "Indonesia": (95.29, -10.36, 141.03, 5.48),
    "Pakistan": (60.87, 23.69, 77.84, 37.13),
    "Philippines": (117.17, 5.58, 126.54, 18.51),
    # Latin America
    "Chile": (-75.64, -55.61, -66.96, -17.58),
    "Colombia": (-78.99, -4.30, -66.88, 12.44),
    "Peru": (-81.41, -18.35, -68.67, -0.06),
    # Onset reference region (not a participating country)
    "India": (68.18, 7.97, 97.40, 35.49),
}

GROUPS = {
    "Africa": ["Ethiopia", "Kenya", "Nigeria", "Rwanda", "Senegal", "Sierra Leone"],
    "Asia": ["Bangladesh", "Indonesia", "Pakistan", "Philippines"],
    "Latin America": ["Chile", "Colombia", "Peru"],
    "Reference": ["India"],
}

MIN_PAD_DEG = 2.0
PAD_FRACTION = 0.10
GRID = 0.25


@dataclass(frozen=True)
class Box:
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float

    def label(self) -> str:
        def ns(v): return f"{abs(v):g}°{'N' if v >= 0 else 'S'}"
        def ew(v): return f"{abs(v):g}°{'E' if v >= 0 else 'W'}"
        return f"{ns(self.lat_min)}–{ns(self.lat_max)}, {ew(self.lon_min)}–{ew(self.lon_max)}"

    def is_valid(self) -> bool:
        return -90 <= self.lat_min < self.lat_max <= 90 and -180 <= self.lon_min < self.lon_max <= 180


def _pad(lo: float, hi: float, limit: float) -> tuple[float, float]:
    p = max(MIN_PAD_DEG, PAD_FRACTION * (hi - lo))
    lo = max(-limit, math.floor((lo - p) / GRID) * GRID)
    hi = min(limit, math.ceil((hi + p) / GRID) * GRID)
    return lo, hi


def country_box(name: str) -> Box:
    lon0, lat0, lon1, lat1 = _RAW_EXTENTS[name]
    lat_min, lat_max = _pad(lat0, lat1, 90)
    lon_min, lon_max = _pad(lon0, lon1, 180)
    return Box(lat_min, lat_max, lon_min, lon_max)


COUNTRIES = [c for g in GROUPS.values() for c in g]
