# Authors: The africlimate AI team
# SPDX-License-Identifier: MIT
"""Season definitions (shared by the pipeline and the UI).

The Ethiopian calendar seasons and the calendar months they span live here — in the
``verification`` layer — so **both** the diagnostics pipeline (which subsets the calibrated
forecast to a season's months before pooling the threshold curves) and the Streamlit UI
(dropdowns + on-the-fly seasonal means) share one definition and never drift. The UI's
``ui.app_config.settings`` re-exports these as a label-keyed map for its selectors.

Each season is keyed by a short filesystem-safe **slug** — the segment written into the
threshold-curve path (``.../{year}/[{region}/][{slug}/]{lead}h.nc``). The slugs are a
controlled vocabulary disjoint from admin-1 region names, which is what lets discovery tell
a season path segment apart from a region one. ``months`` is a tuple of 1-based calendar
months (``None`` for "Yearly" = the whole year, i.e. no month filter). A cross-year season
such as Bega (Oct–Jan) means Oct/Nov/Dec + Jan of the *same* verification year — the curves
are computed per calendar year.
"""
from __future__ import annotations

# slug -> (display label, 1-based months | None)
SEASONS: dict[str, tuple[str, tuple[int, ...] | None]] = {
    "yearly": ("Yearly",           None),
    "bega":   ("Bega (Oct–Jan)",   (10, 11, 12, 1)),
    "belg":   ("Belg (Feb–May)",   (2, 3, 4, 5)),
    "kiremt": ("Kiremt (Jun–Sep)", (6, 7, 8, 9)),
}

DEFAULT_SEASON_SLUG: str = "yearly"

#: Every season slug — a controlled vocabulary used to disambiguate a season path segment
#: from an admin-1 region segment during curve discovery.
SEASON_SLUGS: frozenset[str] = frozenset(SEASONS)


def season_months(slug: str) -> tuple[int, ...] | None:
    """1-based calendar months for a season slug; None = Yearly (no month filter)."""
    entry = SEASONS.get(slug)
    return entry[1] if entry else None


def season_label(slug: str) -> str:
    """Display label for a season slug (falls back to the slug itself if unknown)."""
    entry = SEASONS.get(slug)
    return entry[0] if entry else slug
