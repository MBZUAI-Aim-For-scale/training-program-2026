"""Discovery + loading of threshold (curve) diagnostics for the dashboard.

The curve metrics (reliability / ROC / cost-loss) are stored flat at
``{root}/{forecast}/{native_var}/{target}/{metric}/{year}/{lead}h.nc`` — i.e. the file
sits directly under the year, with no ``annual_monthly_averages`` subdir — so this
discovery never collides with the scalar-summary discovery in :mod:`ui.services.engine`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import pandas as pd
import xarray as xr

from ui.backend.nc_io import read_dataset
from ui.services.engine import LEAD_FILE_RE, RUN_KEYS, TRANSFORM_SUBDIR
from apps.verification.regions import default_region
from apps.verification.seasons import DEFAULT_SEASON_SLUG, SEASON_SLUGS

CURVE_METRICS = ("reliability", "roc", "cost_loss")
# Facet columns of the discovered curve index: the run keys, plus the region and season a
# file belongs to (whole-country files are region-less and yearly files are season-less on
# disk, labelled default_region() / DEFAULT_SEASON_SLUG respectively).
CURVE_KEYS = (*RUN_KEYS, "region", "season")


def _parse_curve_path(path, root) -> Optional[dict]:
    """Parse a curve-file path under ``root`` into run tokens + region + season + lead, or None.

    After the five run tokens (``forecast/native_var/target/metric/year``) come 0–2 optional
    middle segments before the ``{lead}h.nc`` file: an admin-1 region and/or a season slug.
    Season slugs are a controlled vocabulary (:data:`verification.seasons.SEASON_SLUGS`)
    disjoint from region names, so each middle segment is classified unambiguously —
    a known season slug ⇒ season (region = whole country); anything else ⇒ region
    (season = yearly). Whole-country + yearly is the legacy region-less/season-less layout.
    The scalar summaries under ``annual_monthly_averages`` are still excluded.
    """
    p = Path(path)
    m = LEAD_FILE_RE.match(p.name)
    if m is None or p.parent.name == TRANSFORM_SUBDIR:
        return None
    try:
        rel = p.parent.resolve().relative_to(Path(root).resolve()).parts
    except ValueError:
        return None
    if len(rel) < 5 or len(rel) > 7:
        return None
    forecast, native_var, target, metric, year = rel[:5]
    region, season = default_region(), DEFAULT_SEASON_SLUG
    for seg in rel[5:]:
        if seg in SEASON_SLUGS:
            season = seg
        else:
            region = seg
    if metric not in CURVE_METRICS:
        return None
    return {"forecast": forecast, "native_var": native_var, "target": target,
            "metric": metric, "year": year, "region": region, "season": season,
            "lead": int(m.group(1))}


def discover_curves(root: Union[Path, str, None] = None) -> pd.DataFrame:
    """Index the curve diagnostics under ``root`` as a faceted DataFrame (one row per run+region)."""
    columns = [*CURVE_KEYS, "leads"]
    root_path = Path(root) if root is not None else Path(".")
    if not root_path.exists():
        return pd.DataFrame(columns=columns)
    by_run: dict = {}
    for nc in sorted(root_path.rglob("*h.nc")):
        idx = _parse_curve_path(nc, root_path)
        if idx is None:
            continue
        by_run.setdefault(tuple(idx[k] for k in CURVE_KEYS), set()).add(idx["lead"])
    rows = [{**dict(zip(CURVE_KEYS, key)), "leads": sorted(leads)}
            for key, leads in by_run.items()]
    rows.sort(key=lambda r: tuple(r[k] for k in CURVE_KEYS))
    return pd.DataFrame(rows, columns=columns)


def load_curve(forecast, native_var, target, metric, year, lead,
               root: Union[Path, str, None] = None, region=None,
               season=None) -> xr.Dataset:
    """Open the stored curve Dataset for one run + lead (+ optional admin-1 region + season).

    Checks existence first and raises FileNotFoundError before touching the netCDF backend,
    so a lead a run lacks fails cleanly (no C-level error stack) instead of killing the page —
    the selector already restricts choices to existing leads; this is the safety net.
    ``region`` None / the whole country (``default_region()``) reads the region-less path;
    ``season`` None / ``"yearly"`` reads the season-less path. A subregion and/or season insert
    their segments after the year: ``.../{year}/[{region}/][{season}/]{lead}h.nc``.
    """
    base = str(root) if root is not None else "."
    reg_seg = f"/{region}" if region and region != default_region() else ""
    sea_seg = f"/{season}" if season and season != DEFAULT_SEASON_SLUG else ""
    path = f"{base}/{forecast}/{native_var}/{target}/{metric}/{year}{reg_seg}{sea_seg}/{lead}h.nc"
    if not Path(path).exists():
        raise FileNotFoundError(path)
    return read_dataset(path)
