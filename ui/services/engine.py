"""Self-contained discovery + loading of annual/monthly summaries (vendored, light).

Replaces the former external engine. Provides exactly what the dashboard needs:
the result-path scheme, ``open_summary`` to load one run's summary, ``SummarySet.discover``
to build a faceted index of scalar-metric runs, and ``kind_of`` for the variable kind.
On-disk layout:
``{root}/{forecast}/{native_var}/{target}/{metric}/{year}/annual_monthly_averages/{lead}h.nc``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import pandas as pd
import xarray as xr

from ui.backend.nc_io import read_dataset

TRANSFORM_SUBDIR = "annual_monthly_averages"
LEAD_FILE_RE = re.compile(r"^(\d+)h\.nc$")
RUN_KEYS = ("forecast", "native_var", "target", "metric", "year")
KNOWN_METRICS = {"mae", "crps", "seeps", "rmse", "bias", "pearsonr", "spearmanr", "acc"}
_INDEX_COLUMNS = (*RUN_KEYS, "leads", "target_kind")


def kind_of(native_var: str) -> str:
    """Map a native variable name to its display kind: Tmax / Tmin / precip / unknown."""
    n = native_var.lower()
    if "tmax" in n or "temperature_max" in n or n.endswith("_max"):
        return "Tmax"
    if "tmin" in n or "temperature_min" in n or n.endswith("_min"):
        return "Tmin"
    if "precip" in n or n in ("tp", "rfe", "rfe_merged"):
        return "precip"
    return "unknown"


def transform_path(forecast, native_var, target, metric, year, lead, *, root=None) -> str:
    """Path of the annual/monthly summary file for one run + lead."""
    base = str(root) if root is not None else "."
    return (f"{base}/{forecast}/{native_var}/{target}/{metric}/{year}"
            f"/{TRANSFORM_SUBDIR}/{lead}h.nc")


def parse_transform_path(path, root) -> Optional[dict]:
    """Parse a summary file path under ``root`` into its run tokens + lead, or None."""
    p = Path(path)
    m = LEAD_FILE_RE.match(p.name)
    if m is None or p.parent.name != TRANSFORM_SUBDIR:
        return None
    try:
        rel = p.parent.parent.resolve().relative_to(Path(root).resolve()).parts
    except ValueError:
        return None
    if len(rel) != 5:
        return None
    forecast, native_var, target, metric, year = rel
    return {"forecast": forecast, "native_var": native_var, "target": target,
            "metric": metric, "year": year, "lead": int(m.group(1))}


def open_summary(forecast, native_var, target, metric, year, lead, *, root=None) -> xr.Dataset:
    """Open the annual/monthly summary Dataset for one run + lead.

    Checks the path exists first and raises FileNotFoundError before touching the netCDF
    backend — otherwise the h5 backend dumps its C error stack to stderr for missing leads
    (the prewarm probes canonical leads a run may not have).
    """
    path = transform_path(forecast, native_var, target, metric, year, lead,
                          root=str(root) if root is not None else None)
    if not Path(path).exists():
        raise FileNotFoundError(path)
    return read_dataset(path)


def _classify_target_kind(target_name: str, summary_file: Path) -> str:
    """Infer whether a target is gridded or station from the summary file's dims."""
    try:
        dims = set(read_dataset(summary_file).dims)
    except Exception:
        return "unknown"
    if "station" in dims:
        return "station"
    if {"lat", "lon"} <= dims or {"latitude", "longitude"} <= dims:
        return "gridded"
    return "unknown"


@dataclass(frozen=True)
class SummarySet:
    """The set of discovered scalar-metric runs, exposed as a faceted index."""

    rows: tuple

    @classmethod
    def discover(cls, root: Union[Path, str, None] = None) -> "SummarySet":
        """Scan ``root`` for ``annual_monthly_averages/*h.nc`` summaries of known metrics."""
        root_path = Path(root) if root is not None else Path(".")
        if not root_path.exists():
            return cls(rows=())
        by_run: dict = {}
        sample: dict = {}
        for nc in sorted(root_path.rglob(f"{TRANSFORM_SUBDIR}/*h.nc")):
            idx = parse_transform_path(nc, root_path)
            if idx is None or idx["metric"] not in KNOWN_METRICS:
                continue
            key = tuple(idx[k] for k in RUN_KEYS)
            by_run.setdefault(key, set()).add(idx["lead"])
            sample.setdefault(key, nc)
        rows = []
        for key, leads in by_run.items():
            row = dict(zip(RUN_KEYS, key))
            row["leads"] = sorted(leads)
            row["target_kind"] = _classify_target_kind(row["target"], sample[key])
            rows.append(row)
        return cls(rows=tuple(sorted(rows, key=lambda r: tuple(r[k] for k in RUN_KEYS))))

    @property
    def index(self) -> pd.DataFrame:
        """The discovered runs as a DataFrame (one row per run, with its lead list)."""
        if not self.rows:
            return pd.DataFrame(columns=list(_INDEX_COLUMNS))
        return pd.DataFrame(list(self.rows))
