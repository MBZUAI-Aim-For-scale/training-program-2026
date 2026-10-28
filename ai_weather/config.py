import os
import logging
# Fix gRPC C-core fork safety issues when using multiprocessing with gcsfs
os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "1"
os.environ["GRPC_POLL_STRATEGY"] = "poll"

import re
import shutil
import subprocess
import sys
import threading
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

# Silence third-party deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="google.api_core")
warnings.filterwarnings("ignore", category=FutureWarning, module="google.cloud")

# ==============================================================================
# 1. DYNAMIC PROJECT ROOT & DIRECTORY RESOLUTION
# ==============================================================================
CURRENT_FILE = Path(__file__).resolve()
PKG_DIR = CURRENT_FILE.parent.parent


def _find_base_dir(start: Path) -> Path:
    for p in [start] + list(start.parents):
        if (p / "data").exists() and (p / "apps").exists():
            return p
        if p == p.parent or p == Path.home():
            break
    return start


BASE_DIR = _find_base_dir(PKG_DIR)

# ------------------------------------------------------------------------------
# Load .env (ai_weather/.env). Real environment variables win over file values.
# ------------------------------------------------------------------------------
_ENV_FILE_CANDIDATES = [
    BASE_DIR / ".env",
    PKG_DIR / ".env",
    CURRENT_FILE.parent / ".env",
    Path.cwd() / ".env",
    Path.home() / "ai_weather" / ".env",
]
ENV_FILE: Optional[Path] = None
for _cand in _ENV_FILE_CANDIDATES:
    if _cand.is_file():
        load_dotenv(_cand, override=False)
        if ENV_FILE is None:
            ENV_FILE = _cand
del _cand

_env_root = os.getenv("ai_weather")
PROJ_ROOT = Path(_env_root).expanduser().resolve() if _env_root else BASE_DIR
PROJECT_ROOT = PROJ_ROOT
APP_DIR = PROJ_ROOT


def _pick_data_dir() -> Path:
    for name in ("DEMO_DATA_DIR", "ROMP_DATA_DIR", "DATA_DIR"):
        v = os.getenv(name)
        if v:
            return Path(v).expanduser().resolve()

    candidates = [PROJ_ROOT / "data", Path.home() / "data", Path("/app/data")]

    for c in candidates:
        external = c / "external"
        if external.is_dir() and next(external.rglob("*.nc"), None) is not None:
            return c.resolve()
    for c in candidates:
        if c.is_dir():
            return c.resolve()
    return (PROJ_ROOT / "data").resolve()


DATA_DIR = _pick_data_dir()
EXTERNAL_DATA_DIR = DATA_DIR / "external"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ENACTS_DIR = EXTERNAL_DATA_DIR / "ENACTS_regridded_025"
CHIRPS_025_DIR = EXTERNAL_DATA_DIR / "CHIRPS_regridded_025"

# Region (country) and observation dataset
region = os.getenv("ROMP_REGION", "Ethiopia").strip()
obs = os.getenv("ROMP_OBS", "CHIRPS").upper()

# ==============================================================================
# 2. BUCKET CONFIGURATION (STRICTLY READ FROM .ENV)
# ==============================================================================
DATA_MODE = os.getenv("DEMO_DATA_SOURCE", "auto").strip().lower()
if DATA_MODE not in ("auto", "local", "bucket"):
    raise ValueError(f"DEMO_DATA_SOURCE must be auto, local or bucket, got {DATA_MODE!r}")

BUCKET_PREFIXES = {"demo2": "demo2", "demo3": "demo3", "demo4": "demo4", "demo5": "demo5"}


def _clean_bucket(raw: str) -> str:
    raw = (raw or "").strip()
    if raw.startswith("gs://"):
        raw = raw[len("gs://"):]
    return raw.strip("/")


# Read bucket strictly from .env / environment without exposing bucket name
DEMO_BUCKET = _clean_bucket(os.getenv("DEMO_BUCKET", ""))
BUCKET_PREFIX = os.getenv("DEMO_BUCKET_PREFIX", BUCKET_PREFIXES["demo4"]).strip("/")

BUCKET_OK = bool(re.fullmatch(r"[a-z0-9][a-z0-9._-]{1,220}[a-z0-9]", DEMO_BUCKET)) and \
    DEMO_BUCKET not in ("your-bucket", "your-bucket-name", "bucket-name")
BUCKET_NOTE = "" if BUCKET_OK else (
    "DEMO_BUCKET is not set in .env or environment" if not DEMO_BUCKET else
    f"DEMO_BUCKET is {DEMO_BUCKET!r}, not a valid Google bucket name")

if DATA_MODE == "bucket" and not BUCKET_OK:
    raise ValueError(f"{BUCKET_NOTE}. Put the real bucket name in ai_weather/.env "
                     f"(DEMO_BUCKET=<bucket-name>, lowercase, no gs://).")
if DATA_MODE == "local":
    BUCKET_OK = False
    BUCKET_NOTE = "DEMO_DATA_SOURCE=local"

BUCKET_ROOT = f"gs://{DEMO_BUCKET}/{BUCKET_PREFIX}"

MODEL_FOLDERS = {
    "AIFS": "aifs_single",
    "AIFS_ENS": "aifs_ens",
    "FuXi": "fuxi",
    "GraphCast": "graphcast",
    "GenCast": "gencast",
}


def bucket_uri(demo: str = "demo4", rel: str = "") -> str:
    prefix = BUCKET_PREFIXES.get(demo, demo)
    return "gs://" + "/".join(p for p in (DEMO_BUCKET, prefix, rel.strip("/")) if p)


def console_url(demo: str = "demo4", rel: str = "") -> str:
    return bucket_uri(demo, rel).replace("gs://", "https://console.cloud.google.com/storage/browser/", 1)


MODEL_URI_TEMPLATE = "{root}/MODEL/{region}/{folder}"


def _model_uri(key: str) -> str:
    return MODEL_URI_TEMPLATE.format(
        root=BUCKET_ROOT,
        region=region,
        folder=MODEL_FOLDERS.get(key, key.lower())
    )


# Standard CHIRPS observation path in GCS (0.25-degree regridded)
OBS_URI = f"gs://{DEMO_BUCKET}/demo4/DATA/{region}/{obs}"

# ==============================================================================
# 3. URI SANITIZATION & GCS STORAGE HELPERS
# ==============================================================================
def is_uri(p: Any) -> bool:
    if not p:
        return False
    s = str(p).strip()
    return "gs:" in s or "gcs:" in s or "gs:/" in s or "gcs:/" in s or s.startswith(("gs://", "gcs://"))


def _strip(uri: str) -> str:
    """Strip local directory prefixes (e.g., /app/gs:/) and normalize URI."""
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


def resolve_path(p: Any, root: Optional[Path] = None) -> str:
    if not p:
        return ""
    s = str(p).strip()
    if is_uri(s):
        clean_path = _strip(s)
        return f"gs://{clean_path}"

    path_obj = Path(s).expanduser()
    if path_obj.is_absolute():
        return str(path_obj.resolve())

    parts = path_obj.parts
    if parts and parts[0].lower() == "data":
        return str(DATA_DIR.joinpath(*parts[1:]).resolve())

    if root is None:
        root = PROJ_ROOT
    return str((Path(root) / path_obj).resolve())


_token_cache = [0.0, ""]


def _gcloud_token() -> str:
    tok = os.environ.get("DEMO_GCS_ACCESS_TOKEN", "").strip()
    if tok:
        return tok
    if time.time() - _token_cache[0] < 1500 and _token_cache[1]:
        return _token_cache[1]
    exe = shutil.which("gcloud")
    if not exe:
        return ""
    try:
        out = subprocess.run([exe, "auth", "print-access-token"], capture_output=True, text=True, timeout=30)
    except Exception:
        return ""
    tok = out.stdout.strip() if out.returncode == 0 else ""
    _token_cache[:] = [time.time(), tok]
    return tok


def _auth_token():
    if os.environ.get("DEMO_BUCKET_ANON", "0").strip().lower() in ("1", "true", "yes"):
        return "anon"
    key = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
    if key:
        if not Path(key).expanduser().is_file():
            raise FileNotFoundError(f"GOOGLE_APPLICATION_CREDENTIALS points to a missing file: {key}")
        return "google_default"
    if (Path.home() / ".config" / "gcloud" / "application_default_credentials.json").is_file():
        return "google_default"
    tok = _gcloud_token()
    if tok:
        from google.oauth2.credentials import Credentials
        return Credentials(tok)
    return "anon"


def _auth_hint(err: Exception) -> str:
    t = str(err)
    if "401" in t or "Anonymous" in t or "403" in t or "credentials" in t.lower():
        return (" No Google login was found and the bucket is not public. Run `gcloud auth login` "
                "(or `gcloud auth application-default login`) on this machine, or set "
                "GOOGLE_APPLICATION_CREDENTIALS.")
    return ""


_fs_cache = [0, 0.0, None]


def gcs_fs():
    """Process-isolated GCSFileSystem instance (re-created on PID change)."""
    current_pid = os.getpid()
    if (
        _fs_cache[2] is None 
        or _fs_cache[0] != current_pid 
        or time.time() - _fs_cache[1] > 1500
    ):
        import gcsfs
        _fs_cache[:] = [current_pid, time.time(), gcsfs.GCSFileSystem(token=_auth_token())]
    return _fs_cache[2]


def join(base: Any, name: str) -> str:
    base_str = str(base).strip()
    if is_uri(base_str):
        clean_base = f"gs://{_strip(base_str)}"
        return f"{clean_base}/{name.lstrip('/')}"
    return os.path.join(base_str, name)


def exists(path: Any) -> bool:
    if not path:
        return False
    path_str = str(path).strip()
    if is_uri(path_str):
        try:
            return gcs_fs().exists(_strip(path_str))
        except Exception:
            return False
    return Path(path_str).expanduser().exists()


def glob(pattern: str) -> List[str]:
    if is_uri(pattern):
        try:
            return sorted("gs://" + m for m in gcs_fs().glob(_strip(pattern)))
        except Exception:
            return []
    import glob as _glob
    return sorted(_glob.glob(str(pattern)))


def open_dataset(path: Any, **kwargs):
    import xarray as xr
    if not is_uri(path):
        return xr.open_dataset(path, **kwargs)
    f = gcs_fs().open(_strip(path), "rb")
    try:
        return xr.open_dataset(f, **{"engine": "h5netcdf", **kwargs})
    except Exception:
        f.seek(0)
        return xr.open_dataset(f, **{**kwargs, "engine": "scipy"})

# ==============================================================================
# 4. DATA SELECTION: FIRST LOOK AT BUCKET, OTHERWISE LOCAL
# ==============================================================================
DATA_SOURCES: Dict[str, str] = {}
_MISSING_ITEMS: List[str] = []


def _match_case(p: Path) -> Path:
    p = Path(p)
    if p.exists():
        return p
    try:
        rel = p.relative_to(EXTERNAL_DATA_DIR)
    except ValueError:
        return p
    cur = EXTERNAL_DATA_DIR
    for part in rel.parts:
        nxt = cur / part
        if not nxt.exists() and cur.is_dir():
            nxt = next((c for c in cur.iterdir() if c.name.lower() == part.lower()), nxt)
        cur = nxt
    return cur


def _has_nc(p: Path, recursive: bool = True) -> bool:
    p = Path(p)
    if p.suffix == ".nc":
        return p.is_file()
    if not p.is_dir():
        return False
    return next(p.rglob("*.nc") if recursive else p.glob("*.nc"), None) is not None


_REMOTE_PROBE_UNAVAILABLE = False


def _remote_has_data(remote_uri: str) -> bool:
    global _REMOTE_PROBE_UNAVAILABLE
    if not is_uri(remote_uri):
        return False
    if _REMOTE_PROBE_UNAVAILABLE:
        return False

    result = []
    finished = threading.Event()
    gcs_logger = logging.getLogger("gcsfs")
    previous_log_level = gcs_logger.level
    gcs_logger.setLevel(logging.CRITICAL + 1)

    def probe():
        try:
            fs = gcs_fs()
            if remote_uri.lower().endswith(".nc"):
                result.append((fs.exists(_strip(remote_uri)), True))
            else:
                pattern = remote_uri.rstrip("/") + "/*.nc"
                result.append((bool(fs.glob(_strip(pattern))), True))
        except Exception:
            result.append((False, False))
        finally:
            finished.set()

    threading.Thread(target=probe, daemon=True).start()
    try:
        timeout = max(0.1, float(os.getenv("DEMO_GCS_PROBE_TIMEOUT", "3")))
    except ValueError:
        timeout = 3.0
    if not finished.wait(timeout):
        _REMOTE_PROBE_UNAVAILABLE = True
        return False

    available, connected = result[0]
    if connected:
        gcs_logger.setLevel(previous_log_level)
    else:
        _REMOTE_PROBE_UNAVAILABLE = True
    return available


def _pick(label: str, local_candidates: List[tuple], remote_uri: str) -> str:
    """
    Selection strategy:
    1. First check Bucket (if BUCKET_OK and DATA_MODE != 'local').
    2. Fall back to local storage candidates.
    3. If neither is available, mark as missing.
    """
    # 1. First look at Bucket (unless local mode is explicitly specified)
    if DATA_MODE != "local" and BUCKET_OK and _remote_has_data(remote_uri):
        DATA_SOURCES[label] = "bucket"
        return remote_uri

    # 2. Fall back to local candidates
    for cand, recursive in local_candidates:
        c = _match_case(Path(cand))
        if _has_nc(c, recursive):
            DATA_SOURCES[label] = "local"
            return str(c)

    # 3. Handle missing items
    DATA_SOURCES[label] = "missing"
    first = Path(local_candidates[0][0])
    _MISSING_ITEMS.append(f"{label}: {first}")
    return str(first)


def _pick_shapefile_dir(local_dir: Path, remote_uri: str) -> str:
    """Choose a local administrative boundary directory or its configured GCS location."""
    local_dir = _match_case(Path(local_dir))
    if DATA_MODE != "bucket" and local_dir.is_dir() and any(
        path.is_file() and path.suffix.lower() in {".shp", ".geojson", ".json"}
        for path in local_dir.rglob("*")
    ):
        DATA_SOURCES["administrative boundary"] = "local"
        return str(local_dir)
    if DATA_MODE != "local" and (BUCKET_OK or bool(os.getenv("ROMP_SHAPEFILE_URI"))):
        DATA_SOURCES["administrative boundary"] = "bucket"
        return remote_uri
    DATA_SOURCES["administrative boundary"] = "missing"
    _MISSING_ITEMS.append(f"administrative boundary: {local_dir}")
    return str(local_dir)


LOCAL_MODEL_DIRS = {
    "aifs_single": EXTERNAL_DATA_DIR / "aifs_single",
    "fuxi": EXTERNAL_DATA_DIR / "fuxi",
    "graphcast": EXTERNAL_DATA_DIR / "graphcast",
    "aifs_ens": EXTERNAL_DATA_DIR / "aifs_ens",
    "gencast": EXTERNAL_DATA_DIR / "gencast",
}


def _model_dir(key: str):
    return _pick(f"model {key}", [(LOCAL_MODEL_DIRS[key], True)], _model_uri(key))


# Prioritize 0.25-degree regridded CHIRPS data directories
OBS_DIR = _pick(
    "observations",
    [
        (EXTERNAL_DATA_DIR / f"{obs}_regridded_025" / region, True),
        (EXTERNAL_DATA_DIR / f"{obs}_regridded_025", False),
        (EXTERNAL_DATA_DIR / "CHIRPS_regridded_025" / region, True),
        (EXTERNAL_DATA_DIR / "CHIRPS_regridded_025", False),
        (EXTERNAL_DATA_DIR / "CHIRPS_025" / region, True),
        (EXTERNAL_DATA_DIR / "CHIRPS_025", False),
        (EXTERNAL_DATA_DIR / obs / region, True),
        (EXTERNAL_DATA_DIR / obs, False),
        (EXTERNAL_DATA_DIR / "CHIRPS", False),
    ],
    OBS_URI
)
CHIRPS_DIR = OBS_DIR

JJAS_MASK_FILE = _pick(
    "region mask",
    [(EXTERNAL_DATA_DIR / "jjas_100mm_rainfall_mask_0p25.nc", False)],
    os.getenv("ROMP_MASK_URI") or f"{BUCKET_ROOT}/DATA/{region}/jjas_100mm_rainfall_mask_0p25.nc"
)
JJAS_DIR = JJAS_MASK_FILE
MONSOON_DIR = PROCESSED_DATA_DIR / "Monsoon" / "2026"

SHAPEFILE_MASK_FILE = _pick_shapefile_dir(
    EXTERNAL_DATA_DIR / "shapefile",
    os.getenv("ROMP_SHAPEFILE_URI")
    or f"{BUCKET_ROOT}/DATA/{region}/shapefile",
)
shapefile = SHAPEFILE_MASK_FILE
def _find_apps_dir(proj_root: Path) -> Path:
    marker = Path("benchmarking") / "ROMP" / "momp" / "driver.py"
    for base in (proj_root, CURRENT_FILE.parent, PKG_DIR):
        if (base / "apps" / marker).exists():
            return base / "apps"
    return proj_root / "apps"


APPS_DIR = _find_apps_dir(PROJ_ROOT)
ROMP_ROOT = APPS_DIR / "benchmarking" / "ROMP"
MOMP_PKG_DIR = ROMP_ROOT / "momp"

ROMP_OUT_DIR = PKG_DIR / "data" / "ROMP_OUTPUT"
ROMP_FIG_DIR = ROMP_OUT_DIR / "figures"

ROMP_OUT_DIR.mkdir(parents=True, exist_ok=True)
ROMP_FIG_DIR.mkdir(parents=True, exist_ok=True)

for path_obj in [PROJ_ROOT, PKG_DIR, ROMP_ROOT, MOMP_PKG_DIR]:
    path_str = str(path_obj.resolve())
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def setup_earthdata_auth() -> None:
    pass


# ==============================================================================
# 5. BENCHMARKING CATALOG (Configured with unit_cvt for tp -> mm conversion)
# ==============================================================================
BENCHMARK_MODEL_CATALOG: Dict[str, Dict[str, Any]] = {
    "AIFS": {
        "label": "Deterministic · AIFS",
        "dir": _model_dir("aifs_single"),
        "model_name": "aifs_single",
        "file_pattern": "{}_adm3.nc",
        "var": "tp",
        "unit_cvt": 1000.0,
        "probabilistic": False,
        "category": "Deterministic",
    },
    "FuXi": {
        "label": "Deterministic · FuXi",
        "dir": _model_dir("fuxi"),
        "model_name": "fuxi",
        "file_pattern": "{}.nc",
        "var": "tp",
        "unit_cvt": 1000.0,
        "probabilistic": False,
        "category": "Deterministic",
    },
    "GraphCast": {
        "label": "Deterministic · GraphCast",
        "dir": _model_dir("graphcast"),
        "model_name": "graphcast",
        "file_pattern": "{}.nc",
        "var": "tp",
        "unit_cvt": 1000.0,
        "probabilistic": False,
        "category": "Deterministic",
    },
    "AIFS_ENS": {
        "label": "Probabilistic · AIFS ENS",
        "dir": _model_dir("aifs_ens"),
        "model_name": "aifs_ens",
        "file_pattern": "{}.nc",
        "var": "tp",
        "unit_cvt": 1000.0,
        "probabilistic": True,
        "category": "Probabilistic",
        "slow": True,
    },
    "GenCast": {
        "label": "Probabilistic · genCast",
        "dir": _model_dir("gencast"),
        "model_name": "gencast",
        "file_pattern": "{}.nc",
        "var": "tp",
        "unit_cvt": 1000.0,
        "probabilistic": True,
        "category": "Probabilistic",
        "slow": True,
    },
}

if _MISSING_ITEMS:
    warnings.warn(
        f"config: {len(_MISSING_ITEMS)} input(s) not found locally and bucket fallback is off "
        f"({BUCKET_NOTE}). Missing: " + "; ".join(_MISSING_ITEMS)
        + ". Run `python ai_weather/config.py` for details.",
        stacklevel=2,
    )


def _catalog_models(category: str, catalog: Optional[Dict[str, Any]] = None):
    catalog = catalog or BENCHMARK_MODEL_CATALOG
    return [m for m in catalog.values() if m["category"] == category]


for _key, _meta in BENCHMARK_MODEL_CATALOG.items():
    _expected_bool = (_meta.get("category") == "Probabilistic")
    if _meta.get("probabilistic") != _expected_bool:
        raise ValueError(
            f"BENCHMARK_MODEL_CATALOG['{_key}'] is inconsistent: "
            f"category='{_meta.get('category')}' but probabilistic={_meta.get('probabilistic')}."
        )
del _key, _meta, _expected_bool

# ==============================================================================
# 6. CORE PARAMETERS & VERIFICATION SETTINGS
# ==============================================================================
project_name = "AI benchmarking with CHIRPS 0.25 data"
work_dir = str(ROMP_ROOT.resolve())
pkg_dir = str(MOMP_PKG_DIR.resolve())
layout = ("model", "verification_window")

_OBS_SETS = {
    "CHIRPS": dict(
        dir=OBS_DIR, 
        pattern=("{}.nc", f"CHIRPS_{region}_{{}}.nc", f"CHIRPS_025_{region}_{{}}.nc"), 
        var="RAINFALL"  # <-- Changed from "precip" to "RAINFALL"
    ),
    "IMERG": dict(
        dir=OBS_URI, 
        pattern=("{}.nc", f"IMERG_{region}_{{}}.nc"), 
        var="RAINFALL"
    ),
}

if obs not in _OBS_SETS:
    raise ValueError(f"ROMP_OBS must be one of {list(_OBS_SETS)}, got '{obs}'")
_obs = _OBS_SETS[obs]

obs_dir = resolve_path(_obs["dir"])
obs_file_pattern = _obs["pattern"]
obs_var = _obs["var"]

ref_model = "climatology"
ref_model_dir = obs_dir
ref_model_file_pattern = _obs["pattern"]
ref_model_var = _obs["var"]
ref_model_unit_cvt = None
nc_mask = resolve_path(JJAS_MASK_FILE)
shpfile_dir = SHAPEFILE_MASK_FILE
polygon = False

wet_init = 1
wet_threshold = 20
wet_spell = 3
dry_threshold = 1
dry_spell = 7
dry_extent = 21
thresh_file = None
thresh_var = None

start_date = (2003, 5, 1)
end_date = (2023, 7, 31)
start_year_clim = 2007
end_year_clim = 2024
init_days = (0, 3)
init_type = "weekly"
date_filter_year = 2024

# Matching 1:1 length tuples
verification_window_list = ((1, 15), (16, 30))
tolerance_days_list = (3, 5)
max_forecast_day = 30
day_bins = ((1, 5), (6, 10), (11, 15), (16, 20), (21, 25), (26, 30))

dir_out = os.getenv("ROMP_OUT_DIR", str(ROMP_OUT_DIR.resolve()))
dir_fig = os.getenv("ROMP_FIG_DIR", str(ROMP_FIG_DIR.resolve()))

save_fig = True
save_nc_spatial_far_mr_mae = True
save_csv_score = True
save_nc_climatology = True

plot_climatology_onset = True
show_plot = False
show_panel = False

parallel = True
debug = False

mok = (5, 15)
years = None
years_clim = None
fallback_date = None
apply_idr_calibration = False


def compute_mode_overrides(mode: str = "det", catalog: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    mode_str = str(mode).lower().strip()
    if mode_str not in ["det", "deterministic", "prob", "probabilistic"]:
        raise ValueError("Invalid mode. Choose 'det'/'deterministic' or 'prob'/'probabilistic'.")

    is_prob = mode_str in ["prob", "probabilistic"]
    target_category = "Probabilistic" if is_prob else "Deterministic"
    catalog = catalog if catalog is not None else BENCHMARK_MODEL_CATALOG

    selected_models = []
    for key, meta in catalog.items():
        meta_is_prob = meta.get("probabilistic")
        meta_category = meta.get("category")
        if meta_is_prob is None or meta_category is None:
            raise ValueError(f"Catalog entry '{key}' is missing 'probabilistic' or 'category'.")
        expected_bool = (meta_category == "Probabilistic")
        if meta_is_prob != expected_bool:
            raise ValueError(
                f"Catalog entry '{key}' inconsistent: category='{meta_category}' "
                f"but probabilistic={meta_is_prob}. Fix BENCHMARK_MODEL_CATALOG."
            )
        if meta_category == target_category:
            selected_models.append(meta)

    if not selected_models:
        raise ValueError(f"No models found in catalog for category '{target_category}'.")

    overrides: Dict[str, Any] = {
        "mode": mode_str,
        "probabilistic": is_prob,
        "model_list": tuple(m["model_name"] for m in selected_models),
        "model_dir_list": tuple(resolve_path(m["dir"]) for m in selected_models),
        "model_var_list": tuple(m.get("var", "tp") for m in selected_models),
        "unit_cvt_list": tuple(m.get("unit_cvt") for m in selected_models),
        "file_pattern_list": tuple(m.get("file_pattern", "{}.nc") for m in selected_models),
        "skill_score": True,
        "verification_window_list": ((1, 15), (16, 30)),
        "tolerance_days_list": (3, 5),
        "max_forecast_day": 30,
        "day_bins": ((1, 5), (6, 10), (11, 15), (16, 20), (21, 25), (26, 30)),
        "onset_percentage_threshold": 0.5,
        "members": "All",
    }

    if is_prob:
        overrides.update(
            BS=True, RPS=True, AUC=True, Reliability=True,
            FAR=False, MAE=False, MR=False,
            plot_heatmap_bss_auc=True,
            plot_reliability=True,
            plot_panel_heatmap_skill=True,
            plot_bar_bss_rpss_auc=True,
            plot_spatial_far_mr_mae=False,
            plot_panel_heatmap_error=False,
        )
    else:
        overrides.update(
            BS=False, RPS=False, AUC=False, Reliability=False,
            FAR=True, MAE=True, MR=True,
            plot_spatial_far_mr_mae=True,
            plot_panel_heatmap_error=True,
            plot_heatmap_bss_auc=False,
            plot_reliability=False,
            plot_panel_heatmap_skill=False,
            plot_bar_bss_rpss_auc=False,
        )

    print(f"[config] mode overrides computed for {'PROBABILISTIC' if is_prob else 'DETERMINISTIC'}: "
          f"models={overrides['model_list']}")
    return overrides


def _explain_error(e: Exception) -> str:
    if isinstance(e, ImportError):
        return (f"{type(e).__name__}: {e}. Install the bucket reader in this Python environment: "
                f"pip install gcsfs h5netcdf h5py")
    return f"{type(e).__name__}: {e}." + (_auth_hint(e) or "")


def check_data() -> int:
    items = [("observations", obs_dir), ("region mask", nc_mask)]
    items += [(f"model {k}", resolve_path(m["dir"])) for k, m in BENCHMARK_MODEL_CATALOG.items()]
    missing = errors = 0
    for label, where in items:
        src = "bucket" if is_uri(where) else "local "
        try:
            if is_uri(where):
                fs = gcs_fs()
                path = _strip(where)
                n = (1 if fs.exists(path) else 0) if where.endswith(".nc") else len(fs.glob(path + "/*.nc"))
            else:
                n = 1 if _has_nc(Path(where), True) else 0
        except Exception as e:
            errors += 1
            print(f"  ERROR   [{src}] {label:<14} {where}")
            print(f"          {_explain_error(e)}")
            continue
        print(f"  {'ok     ' if n else 'MISSING'} [{src}] {label:<14} {where}"
              + (f"   ({n} .nc)" if n and not where.endswith(".nc") else ""))
        if not n:
            missing += 1
            if is_uri(where):
                parent = _strip(where).rstrip("/").rsplit("/", 1)[0]
                try:
                    names = sorted(x.rstrip("/").rsplit("/", 1)[-1] for x in gcs_fs().ls(parent))[:12]
                    print(f"          folders next to it: {', '.join(names) or '(none)'}")
                except Exception:
                    print(f"          gs://{parent} not found")
    print(f"  summary: {len(items) - missing - errors} ok, {missing} missing, {errors} error(s)")
    return 1 if errors else (2 if missing else 0)


if __name__ == "__main__":
    print(f"local     : {EXTERNAL_DATA_DIR}")
    print(f"bucket    : {BUCKET_ROOT if BUCKET_OK else '(off: ' + BUCKET_NOTE + ')'}")
    print(f"region    : {region}")
    print(f"outputs   : {ROMP_OUT_DIR}   (results are written locally)")
    if BUCKET_OK:
        print("browse    :")
        for _demo, _rel in (("demo3", ""),
                            ("demo4", f"MODEL/{region}/{MODEL_FOLDERS['AIFS']}"),
                            ("demo4", f"MODEL/{region}/{MODEL_FOLDERS['AIFS_ENS']}"),
                            ("demo4", f"DATA/{region}/{obs}")):
            print(f"  {console_url(_demo, _rel)}")
    print("check     :")
    sys.exit(check_data())
