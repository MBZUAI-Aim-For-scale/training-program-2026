# Authors: The africlimate AI team
# SPDX-License-Identifier: MIT
"""Presentation constants for the dashboard.

What is *available* (which forecasts / targets / metrics / variables /
years) is discovered from disk via
:class:`ui.services.menu.MenuIndex` — not listed here. This
module holds only cosmetics: the canonical lead-time axis, the brand
palette, units, and the token→label maps that turn engine tokens
(``graphcast``, ``rfe2``, ``mae``, ``precip``) into pretty display names.

Unknown tokens fall back to a derived label, so a newly-onboarded
forecast / target / metric still shows up (just less prettily) instead of
vanishing from the menu.
"""
from __future__ import annotations

from ui.utils.regions import available_regions as _available_regions
from ui.utils.regions import default_region as _default_region
from apps.verification.seasons import SEASONS as _CANON_SEASONS

# ── Canonical lead-time axis (hours) ──────────────────────────────────────
# The day-N×24 axis shared by gridded and station results. The loader
# requests these leads; any a given run lacks are skipped.
LEAD_TIMES: list[int] = [24, 36, 48, 60, 72, 84, 96, 108, 120, 132, 144,
                         156, 168, 180, 192, 204, 216, 228, 240]

# ── Seasons — Ethiopian calendar seasons for the global Season selector ────
# Derived from the canonical registry in verification/seasons.py (shared with the
# diagnostics pipeline so definitions never drift). Here they are re-keyed by display
# label — what the sidebar dropdown offers and what FilterState.season carries; the
# Short & Medium-Range tab aggregates the stored monthly means over these months.
# Cross-year Bega uses Oct–Dec + Jan of the *same* verification year.
SEASONS: dict[str, tuple[int, ...] | None] = {
    label: months for (label, months) in _CANON_SEASONS.values()
}
DEFAULT_SEASON: str = "Yearly"


def season_months(label: str) -> tuple[int, ...] | None:
    """1-based month numbers for a season label; None = Yearly (all months)."""
    return SEASONS.get(label)

# ── Available regions — driven by utils/regions.py (narrowed to the ───────
# deployment country when VERIFICATION_COUNTRY is set, else the full list) ──
REGIONS: list[str] = _available_regions()
DEFAULT_REGION: str = _default_region()

# ── Brand palette — 8 colors from AfriClimate AI identity guide ───────────
MODEL_PALETTE: list[str] = [
    "#0a9938",  # Jungle Green   (primary)
    "#213b7d",  # Blue           (primary)
    "#f2b029",  # Sun Gold       (primary)
    "#2e7d33",  # Fern Green     (primary)
    "#305429",  # Dark Green     (secondary)
    "#f0d161",  # Lightning      (secondary)
    "#70ba6e",  # Moss Green     (secondary)
    "#bac7c9",  # Silver         (secondary)
]

# ── Direction of goodness — the single source of truth ────────────────────
# Every component that ranks models, colors a difference map, or runs a
# one-sided test derives its direction from these three sets (via the helpers
# below), so "which way is better" can never disagree between tabs.
LOWER_IS_BETTER_METRICS: set[str] = {"mae", "crps", "rmse", "seeps"}
# Correlation metrics: higher = better, and their spatial map / daily series follow the
# correlation "two-flavor" design (temporal map + spatial-pattern daily/monthly series).
CORRELATION_METRICS: set[str] = {"pearsonr", "spearmanr", "acc"}
# Signed metrics whose ideal value is 0 (bias): neither end of the scale is
# "good", so maps need a diverging scale centred at 0, and model-vs-model
# comparison must use the magnitude (|bias| lower = better).
ZERO_IS_BEST_METRICS: set[str] = {"bias"}


def higher_is_better(metric_key: str) -> bool:
    """True when a larger value means a better forecast (correlation family)."""
    return metric_key.lower() in CORRELATION_METRICS


def zero_is_best(metric_key: str) -> bool:
    """True for signed metrics whose ideal value is 0 (bias). Compare by |value|."""
    return metric_key.lower() in ZERO_IS_BEST_METRICS


def direction_note(metric_key: str) -> str:
    """Short user-facing reminder of which direction means better skill."""
    if zero_is_best(metric_key):
        return "closer to 0 is better"
    return "higher is better" if higher_is_better(metric_key) else "lower is better"

# ── Units per variable *kind* (the engine fact: precip is a flux, temp °C) ─
# Keyed by the kind FilterState carries (viz.runs.kind_of output), so
# existing consumers' ``VARIABLE_UNITS.get(f.variable)`` keeps working.
VARIABLE_UNITS: dict[str, str] = {"precip": "mm/hr", "Tmax": "°C", "Tmin": "°C"}

# ── Token → display label maps (presentation only) ────────────────────────
_VARIABLE_LABELS: dict[str, str] = {
    "precip": "Rainfall", "Tmax": "Temperature (MAX)", "Tmin": "Temperature (MIN)",
}
_FORECAST_LABELS: dict[str, str] = {
    "graphcast": "GraphCast", "ifs": "IFS", "fuxi": "FUXI", "pangu": "Pangu",
    "chirps_gefs": "CHIRPS-GEFS", "ifs_ens": "IFS-ENS",
    "ifs_ens_dynamical": "IFS-ENS-Dynamical",
}
_TARGET_LABELS: dict[str, str] = {
    "rfe2": "RFE2", "chirps": "CHIRPS", "imerg": "IMERG",
    "chirts_tmax": "CHIRTS-TMAX", "chirts_tmin": "CHIRTS-TMIN",
    "zambia_enacts": "Zambia ENACTS", "ethiopia_enacts": "Ethiopia ENACTS",
    "ethiopia_stations": "Ethiopia Stations",
}
_METRIC_LABELS: dict[str, str] = {
    "mae": "MAE", "rmse": "RMSE", "bias": "Bias", "seeps": "SEEPS",
    "crps": "CRPS", "acc": "ACC", "acc_per_cell": "ACC (per-cell)",
    "pearsonr": "Pearson r", "spearmanr": "Spearman ρ",
    # Spatial = pattern correlation per time (a time series); temporal =
    # per-cell correlation over the whole period (a map). Labelled distinctly
    # so they're never read as the same quantity.
    "pearson_spatial":  "Pearson (spatial pattern)",
    "pearson_temporal": "Pearson (temporal, per-cell)",
    "spearman_spatial":  "Spearman (spatial pattern)",
    "spearman_temporal": "Spearman (temporal, per-cell)",
    "roc_auc": "ROC AUC",
}


def variable_label(kind: str) -> str:
    return _VARIABLE_LABELS.get(kind, kind)


def forecast_label(token: str) -> str:
    # An ``-idr`` suffix marks an IDR-calibrated variant: label the base model and
    # tag it ``(IDR)`` so raw and calibrated read as the same model, distinctly named.
    if token.endswith("-idr"):
        return f"{forecast_label(token[:-4])} (IDR)"
    return _FORECAST_LABELS.get(token, token.replace("_", " ").title())


def target_label(token: str) -> str:
    return _TARGET_LABELS.get(token, token.replace("_", " ").upper())


def metric_label(token: str) -> str:
    return _METRIC_LABELS.get(token, token.replace("_", " ").upper())




