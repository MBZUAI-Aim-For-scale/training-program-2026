"""
momp/utils/storage.py
File system and GCS storage abstraction for MOMP benchmarking.
Supports local filesystem access and keyless gs:// GCS streaming via gcsfs.
Streams NetCDF data directly from GCS buckets without downloading to disk.
"""

from __future__ import annotations

import glob as _glob
import os
import sys
import time
from pathlib import Path
from typing import Any, List

import xarray as xr

# Safe import for config.py located in ai_weather/ or top level
try:
    import config
except ImportError:
    try:
        from ai_weather import config
    except ImportError:
        _curr = Path(__file__).resolve()
        for _parent in _curr.parents:
            if (_parent / "ai_weather" / "config.py").exists() or (_parent / "config.py").exists():
                sys.path.insert(0, str(_parent / "ai_weather"))
                sys.path.insert(0, str(_parent))
                break
        import config


def is_uri(p: Any) -> bool:
    """True if path string represents a GCS URI (e.g. gs://, gcs://, /app/gs:/)."""
    if not p:
        return False
    s = str(p).strip()
    return "gs:" in s or "gcs:" in s or "gs:/" in s or "gcs:/" in s or s.startswith(("gs://", "gcs://"))


def _strip(uri: Any) -> str:
    """Strip local directory prefixes (e.g., /app/gs:/) and normalize URI to bucket/path."""
    s = str(uri).strip()
    if "gs:/" in s:
        s = "gs:/" + s.split("gs:/", 1)[1]
    elif "gcs:/" in s:
        s = "gcs:/" + s.split("gcs:/", 1)[1]
    elif "gs:" in s:
        s = "gs:/" + s.split("gs:", 1)[1]

    if "://" in s:
        return s.split("://", 1)[1].strip("/")
    if ":/" in s:
        return s.split(":/", 1)[1].strip("/")
    return s.strip("/")


_fs_cache = [0, 0.0, None]  # [pid, timestamp, fs_instance]


def gcs_fs():
    """Process-isolated GCSFileSystem instance (re-created on PID change or timeout)."""
    if hasattr(config, "gcs_fs"):
        return config.gcs_fs()

    current_pid = os.getpid()
    if (
        _fs_cache[2] is None
        or _fs_cache[0] != current_pid
        or time.time() - _fs_cache[1] > 1500
    ):
        import gcsfs

        token = "google_default"
        if os.getenv("DEMO_BUCKET_ANON", "0").strip().lower() in ("1", "true", "yes"):
            token = "anon"
        else:
            key = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
            if key:
                key_path = Path(key).expanduser()
                if not key_path.is_file():
                    raise FileNotFoundError(f"GOOGLE_APPLICATION_CREDENTIALS points to missing file: {key_path}")
                token = str(key_path)

        _fs_cache[:] = [current_pid, time.time(), gcsfs.GCSFileSystem(token=token)]
    return _fs_cache[2]


def join(base: Any, name: str) -> str:
    """Safely join paths preserving gs:// URIs without path corruption."""
    base_str = str(base).strip() if base else ""
    if is_uri(base_str):
        clean_base = f"gs://{_strip(base_str)}"
        return f"{clean_base}/{name.lstrip('/')}"
    return os.path.join(base_str, name)


def exists(path: Any) -> bool:
    """Check existence for local filesystem paths or gs:// GCS objects."""
    if not path:
        return False
    path_str = str(path).strip()
    if is_uri(path_str):
        if hasattr(config, "exists"):
            return config.exists(path_str)
        try:
            return gcs_fs().exists(_strip(path_str))
        except Exception:
            return False
    return Path(path_str).expanduser().exists()


def glob(pattern: Any) -> List[str]:
    """Sorted matches as gs:// URIs or local paths."""
    if not pattern:
        return []
    pattern_str = str(pattern).strip()
    if is_uri(pattern_str):
        if hasattr(config, "glob"):
            return config.glob(pattern_str)
        try:
            fs = gcs_fs()
            return sorted("gs://" + m for m in fs.glob(_strip(pattern_str))) if fs else []
        except Exception:
            return []
    return sorted(_glob.glob(pattern_str, recursive=True))


def list_nc(folder: Any) -> List[str]:
    """List .nc filenames directly inside a local folder or gs:// directory."""
    if not folder:
        return []
    folder_str = str(folder).strip()
    if is_uri(folder_str):
        try:
            matches = glob(join(folder_str, "*.nc"))
            return sorted(os.path.basename(m) for m in matches)
        except Exception as err:
            print(f"[storage.py] Error listing GCS bucket {folder_str}: {err}")
            return []

    if not os.path.exists(folder_str):
        return []

    return sorted(f for f in os.listdir(folder_str) if f.endswith(".nc"))


def open_dataset(path: Any, **kwargs) -> xr.Dataset:
    """
    Open NetCDF dataset from local filesystem or gs:// URI stream.
    Streams NetCDF4 via h5netcdf (requires h5py) with legacy scipy fallback for NetCDF3.
    """
    if not path:
        raise ValueError("Cannot open dataset: path is empty or None")

    path_str = str(path).strip()

    # Detect GCS URI even if corrupted with local prefix (e.g., /app/gs:/)
    if is_uri(path_str):
        fs = gcs_fs()
        if not fs:
            raise RuntimeError(f"GCS FileSystem not initialized for path: {path_str}")

        clean_p = _strip(path_str)
        f = fs.open(clean_p, "rb")

        try:
            return xr.open_dataset(f, **{"engine": "h5netcdf", **kwargs})
        except ImportError as err:
            if "h5py" in str(err):
                raise ImportError(
                    "Failed to stream NetCDF4 file from GCS because 'h5py' is missing. "
                    "Run `uv pip install h5py` inside your python environment."
                ) from err
            raise
        except Exception:
            try:
                f.seek(0)
                return xr.open_dataset(f, **{"engine": "scipy", **kwargs})
            except Exception as fallback_err:
                raise RuntimeError(
                    f"Failed to open NetCDF dataset from GCS ({path_str}). "
                    f"Error: {fallback_err}"
                ) from fallback_err

    # Local Filesystem Opening
    try:
        return xr.open_dataset(path_str, **kwargs)
    except Exception:
        return xr.open_dataset(path_str, engine="netcdf4", **kwargs)
