"""Where model outputs live on the Spark, and how saved runs are reused.

Layout: {OUTPUT_DIR}/{model}/{YYYYMMDDTHH}_{lead}h_m{members}.nc
A saved run is reused for any request with the same model, init and member
count whose lead time is <= the saved one (the extra lead times are sliced off).

Shared runs: if DEMO3_BUCKET is set (e.g. gs://<bucket>/demo3), runs found there
with the same layout can be downloaded into OUTPUT_DIR from the page. Runs made
on the Spark stay on it. runners/sync_bucket.py pulls others' in bulk.
"""
from __future__ import annotations

import os
import sys
import re
import time
from dataclasses import dataclass
from pathlib import Path
import pandas as pd

APP_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_ROOT))

# <repo>/outputs, which docker/run.sh mounts at /app/outputs (so runs survive a new container)
OUTPUT_DIR = Path(os.environ.get("DEMO3_OUTPUT_DIR", Path(__file__).resolve().parents[2] / "outputs"))
JOBS_DIR = OUTPUT_DIR / "_jobs"
BUCKET = os.environ.get("DEMO3_BUCKET", "").rstrip("/")

_NAME = re.compile(r"^(\d{8}T\d{2})_(\d+)h_m(\d+)(?:_synthetic)?\.nc$")


@dataclass(frozen=True)
class RunRequest:
    model: str
    init: pd.Timestamp
    lead_hours: int
    members: int
    synthetic: bool = False

    def path(self) -> Path:
        tag = "_synthetic" if self.synthetic else ""
        return OUTPUT_DIR / self.model / f"{self.init:%Y%m%dT%H}_{self.lead_hours}h_m{self.members}{tag}.nc"


def find_saved(req: RunRequest) -> Path | None:
    """Shortest saved run that covers the request, or None."""
    folder = OUTPUT_DIR / req.model
    if not folder.is_dir():
        return None
    best = None
    for p in folder.glob("*.nc"):
        m = _NAME.match(p.name)
        if not m or ("_synthetic" in p.name) != req.synthetic:
            continue
        init, lead, members = m.group(1), int(m.group(2)), int(m.group(3))
        if init == f"{req.init:%Y%m%dT%H}" and members == req.members and lead >= req.lead_hours:
            if best is None or lead < best[0]:
                best = (lead, p)
    return best[1] if best else None


def find_any(model: str, init: pd.Timestamp, lead_hours: int, synthetic: bool = False) -> Path | None:
    """A saved run of this model and start time covering the lead time, with any member
    count (most members first, then the shortest lead). Used when comparing models."""
    folder = OUTPUT_DIR / model
    if not folder.is_dir():
        return None
    best = None
    for p in folder.glob("*.nc"):
        m = _NAME.match(p.name)
        if not m or ("_synthetic" in p.name) != synthetic:
            continue
        lead, members = int(m.group(2)), int(m.group(3))
        if m.group(1) == f"{init:%Y%m%dT%H}" and lead >= lead_hours:
            rank = (-members, lead)
            if best is None or rank < best[0]:
                best = (rank, p)
    return best[1] if best else None


def list_saved() -> pd.DataFrame:
    rows = []
    if OUTPUT_DIR.is_dir():
        for p in OUTPUT_DIR.glob("*/*.nc"):
            m = _NAME.match(p.name)
            if m:
                rows.append(dict(model=p.parent.name, init=m.group(1), lead_h=int(m.group(2)),
                                 members=int(m.group(3)), synthetic="_synthetic" in p.name,
                                 size_mb=round(p.stat().st_size / 1e6, 1), path=str(p)))
    return pd.DataFrame(rows)


# ── shared runs in a bucket ─────────────────────────────────────────────────

_remote_cache: dict = {}


def bucket_fs():
    """(filesystem, root path) for DEMO3_BUCKET. Public buckets work without credentials."""
    import fsspec
    return fsspec.core.url_to_fs(BUCKET)


def remote_runs(max_age_s: float = 300) -> list[tuple[str, str, int]]:
    """[(model, file name, size in bytes)] in the bucket, cached for a few minutes."""
    if not BUCKET:
        return []
    hit = _remote_cache.get(BUCKET)
    if hit and time.time() - hit[0] < max_age_s:
        return hit[1]
    try:
        fs, root = bucket_fs()
        found = []
        for path, info in fs.find(root, detail=True).items():
            parts = path[len(root):].strip("/").split("/")
            if len(parts) == 2 and _NAME.match(parts[1]):
                found.append((parts[0], parts[1], int(info.get("size", 0))))
    except Exception:
        found = []
    _remote_cache[BUCKET] = (time.time(), found)
    return found


def find_remote(req: RunRequest) -> tuple[str, int] | None:
    """(file name, size) of the shortest run in the bucket covering the request."""
    best = None
    for model, name, size in remote_runs():
        m = _NAME.match(name)
        if model != req.model or ("_synthetic" in name) != req.synthetic:
            continue
        if (m.group(1) == f"{req.init:%Y%m%dT%H}" and int(m.group(3)) == req.members
                and int(m.group(2)) >= req.lead_hours):
            if best is None or int(m.group(2)) < best[0]:
                best = (int(m.group(2)), name, size)
    return (best[1], best[2]) if best else None


def in_bucket(path: Path) -> bool:
    """True if this saved run is already in the bucket (same name and size)."""
    size = path.stat().st_size
    return any(m == path.parent.name and n == path.name and s == size for m, n, s in remote_runs())


def download(model: str, name: str) -> Path:
    """Copy a run (and its timings file) from the bucket into OUTPUT_DIR."""
    fs, root = bucket_fs()
    dest = OUTPUT_DIR / model / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    side = name[:-3] + ".json"
    if fs.exists(f"{root}/{model}/{side}"):
        fs.get(f"{root}/{model}/{side}", str(dest.with_suffix(".json")))
    part = dest.with_suffix(".part")
    fs.get(f"{root}/{model}/{name}", str(part))
    part.replace(dest)
    return dest
