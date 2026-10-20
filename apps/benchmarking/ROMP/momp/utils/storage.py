"""Read input data from local folders or from the Google bucket (gs://), the same way.

Self-contained on purpose (no import of config.py), so ROMP workers can use it.
Credentials: GOOGLE_APPLICATION_CREDENTIALS (service-account key), else
`gcloud auth application-default login`. DEMO_BUCKET_ANON=1 for a public bucket.
Nothing is downloaded to disk: files are streamed from the bucket.
"""
from __future__ import annotations

import functools
import glob as _glob
import os
from pathlib import Path
from typing import Any, List

import xarray as xr


def is_uri(p: Any) -> bool:
    return str(p).startswith(("gs://", "gcs://"))


def _strip(uri: Any) -> str:
    return str(uri).split("://", 1)[1]


def join(base: Any, name: str) -> str:
    """os.path.join that keeps 'gs://' intact (Path() would turn it into 'gs:/')."""
    base = str(base)
    return f"{base.rstrip('/')}/{name}" if is_uri(base) else os.path.join(base, name)


@functools.lru_cache(maxsize=1)
def gcs_fs():
    import gcsfs
    if os.getenv("DEMO_BUCKET_ANON", "0").strip().lower() in ("1", "true", "yes"):
        return gcsfs.GCSFileSystem(token="anon")
    key = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
    if key:
        key_path = Path(key).expanduser()
        if not key_path.is_file():
            raise FileNotFoundError(f"GOOGLE_APPLICATION_CREDENTIALS points to a missing file: {key_path}")
        return gcsfs.GCSFileSystem(token=str(key_path))
    return gcsfs.GCSFileSystem(token="google_default")


def exists(path: Any) -> bool:
    return gcs_fs().exists(_strip(path)) if is_uri(path) else os.path.exists(path)


def glob(pattern: Any) -> List[str]:
    """Sorted matches (use '**' for recursive), as gs:// URIs or local paths."""
    if is_uri(pattern):
        return sorted("gs://" + m for m in gcs_fs().glob(_strip(pattern)))
    return sorted(_glob.glob(str(pattern), recursive=True))


def list_nc(folder: Any) -> List[str]:
    """File names (not paths) of the .nc files directly in a folder."""
    if is_uri(folder):
        return sorted(m.rsplit("/", 1)[-1] for m in gcs_fs().glob(_strip(folder).rstrip("/") + "/*.nc"))
    return sorted(f for f in os.listdir(folder) if f.endswith(".nc"))


def open_dataset(path: Any, **kwargs):
    """xarray.open_dataset that also reads straight from gs://."""
    if not is_uri(path):
        return xr.open_dataset(path, **kwargs)
    f = gcs_fs().open(_strip(path), "rb")
    try:
        return xr.open_dataset(f, **{"engine": "h5netcdf", **kwargs})      # NetCDF4 / HDF5
    except Exception:
        f.seek(0)
        return xr.open_dataset(f, **{**kwargs, "engine": "scipy"})         # NetCDF3