import os
import re
import shutil
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

# ==============================================================================
# Dynamic Project Root & Directory Resolution
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
# .env  (ai_weather/.env). Real environment variables (e.g. docker -e / --env-file)
# always win over the file; the first .env found wins over later ones.
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

# ------------------------------------------------------------------------------
# Data folder. Order: DEMO_DATA_DIR / ROMP_DATA_DIR (explicit) -> ~/data -> PROJ_ROOT/data.
# Among the automatic candidates, the first one that already holds an `external/`
# folder wins (so a mounted /app/data/external is found even when ~/data exists).
# ------------------------------------------------------------------------------
def _pick_data_dir() -> Path:
    for name in ("DEMO_DATA_DIR", "ROMP_DATA_DIR"):
        v = os.getenv(name)
        if v:
            return Path(v).expanduser()
    candidates = [Path.home() / "data", PROJ_ROOT / "data"]
    for c in candidates:
        if (c / "external").is_dir():
            return c
    for c in candidates:
        if c.is_dir():
            return c
    return PROJ_ROOT / "data"

DATA_DIR = _pick_data_dir()

EXTERNAL_DATA_DIR = DATA_DIR / "external"   # local data (used first); the Google bucket is the fallback
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ENACTS_DIR = EXTERNAL_DATA_DIR / "ENACTS_regridded_025"

# Region (country) and observation dataset
region = os.getenv("ROMP_REGION", "Ethiopia").strip()
obs = os.getenv("ROMP_OBS", "CHIRPS").upper()           # CHIRPS or IMERG (dataset folder name)


# ==============================================================================
# WHERE THE INPUT DATA COMES FROM
#   1. local files in ~/data/external/...   (used first, exactly as before)
#   2. the Google bucket (DEMO_BUCKET in ai_weather/.env)   fallback for whatever is not local.
#      Nothing is downloaded: the bucket is read in place through gs:// URIs, so use the helpers
#      below (exists, glob, open_dataset) wherever a path may be a gs:// URI.
#   DEMO_DATA_SOURCE=auto (default) | local | bucket   forces one source.
#
#   gs://<DEMO_BUCKET>/demo3/*                             forecasting demo (handled by store.py)
#   gs://<DEMO_BUCKET>/demo4/MODEL/<Country>/aifs_single   AIFS forecasts, e.g. MODEL/Ethiopia/...
#   gs://<DEMO_BUCKET>/demo4/MODEL/<Country>/aifs_ens      AIFS ENS forecasts
#   gs://<DEMO_BUCKET>/demo4/MODEL/<Country>/<fuxi|graphcast|gencast>
#   gs://<DEMO_BUCKET>/demo4/DATA/<Country>/CHIRPS         observations, e.g. DATA/Ethiopia/CHIRPS
# The two templates below are the only place that knows the bucket layout: edit them if it differs.
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

DEMO_BUCKET = _clean_bucket(os.getenv("DEMO_BUCKET", ""))
BUCKET_PREFIX = os.getenv("DEMO_BUCKET_PREFIX", BUCKET_PREFIXES["demo4"]).strip("/")

# GCS bucket names: lowercase letters, digits, "-", "_" and "."  (so DEMO_BUCKET can't be real)
BUCKET_OK = bool(re.fullmatch(r"[a-z0-9][a-z0-9._-]{1,220}[a-z0-9]", DEMO_BUCKET)) and \
    DEMO_BUCKET not in ("your-bucket", "your-bucket-name", "bucket-name")
BUCKET_NOTE = "" if BUCKET_OK else (
    "DEMO_BUCKET is not set" if not DEMO_BUCKET else
    f"DEMO_BUCKET is {DEMO_BUCKET!r}, not a valid Google bucket name")
if DATA_MODE == "bucket" and not BUCKET_OK:
    raise ValueError(f"{BUCKET_NOTE}. Put the real bucket name in ai_weather/.env "
                     f"(DEMO_BUCKET=<bucket-name>, lowercase, no gs://).")
if DATA_MODE == "local":
    BUCKET_OK = False
    BUCKET_NOTE = "DEMO_DATA_SOURCE=local"

BUCKET_ROOT = f"gs://{DEMO_BUCKET}/{BUCKET_PREFIX}"     # gs://<bucket>/demo4

MODEL_FOLDERS = {            # catalog key -> folder under demo4/MODEL/  (edit here if yours differ)
    "AIFS": "aifs_single",
    "AIFS_ENS": "aifs_ens",
    "FuXi": "fuxi",
    "GraphCast": "graphcast",
    "GenCast": "gencast",
    "Aurora 1.5": "aurora15",
    "Atlas CRPS": "atlas_crps",
    "NeuralGCM": "neural_gcm",
    "FGN mini": "fgn_mini",

}

def bucket_uri(demo: str = "demo4", rel: str = "") -> str:
    """gs:// location of a demo's data, e.g. bucket_uri('demo3') -> gs://<DEMO_BUCKET>/demo3"""
    prefix = BUCKET_PREFIXES.get(demo, demo)
    return "gs://" + "/".join(p for p in (DEMO_BUCKET, prefix, rel.strip("/")) if p)

def console_url(demo: str = "demo4", rel: str = "") -> str:
    """Browser link to a folder of the bucket (Google Cloud console)."""
    return bucket_uri(demo, rel).replace("gs://", "https://console.cloud.google.com/storage/browser/", 1)

MODEL_URI_TEMPLATE = "{root}/MODEL/{region}/{folder}"    # gs://<bucket>/demo4/MODEL/Ethiopia/aifs_single
OBS_URI_TEMPLATE = "{root}/DATA/{region}/{obs}"           # gs://<bucket>/demo4/DATA/Ethiopia/CHIRPS

def _model_uri(key: str) -> str:
    return MODEL_URI_TEMPLATE.format(root=BUCKET_ROOT, region=region,
                                     folder=MODEL_FOLDERS.get(key, key.lower()))

OBS_URI = OBS_URI_TEMPLATE.format(root=BUCKET_ROOT, region=region, obs=obs)

# ------------------------------------------------------------------------------
# Reading from the bucket (no local copy). Local paths still work in these helpers.
# ------------------------------------------------------------------------------
def is_uri(p: Any) -> bool:
    return str(p).startswith(("gs://", "gcs://"))

def _strip(uri: str) -> str:
    return str(uri).split("://", 1)[1]

# ── Google login: no key file needed ────────────────────────────────────────
# Order: DEMO_BUCKET_ANON=1 -> anonymous | GOOGLE_APPLICATION_CREDENTIALS key file |
#        `gcloud auth application-default login` | the account you are logged into with
#        `gcloud auth login` (via `gcloud auth print-access-token`, or $DEMO_GCS_ACCESS_TOKEN) |
#        anonymous (works only for a publicly readable bucket).
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
    """The `token=` argument for gcsfs (see the order above)."""
    if os.environ.get("DEMO_BUCKET_ANON", "0").strip().lower() in ("1", "true", "yes"):
        return "anon"
    key = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
    if key:
        if not Path(key).expanduser().is_file():
            raise FileNotFoundError(f"GOOGLE_APPLICATION_CREDENTIALS points to a missing file: {key}")
        return "google_default"                     # reads that file (service account or user login)
    if (Path.home() / ".config" / "gcloud" / "application_default_credentials.json").is_file():
        return "google_default"
    tok = _gcloud_token()
    if tok:
        from google.oauth2.credentials import Credentials
        return Credentials(tok)                     # valid for about an hour
    return "anon"                                   # public bucket only


def _auth_hint(err: Exception) -> str:
    t = str(err)
    if "401" in t or "Anonymous" in t or "403" in t or "credentials" in t.lower():
        return (" No Google login was found and the bucket is not public. Run `gcloud auth login` "
                "(or `gcloud auth application-default login`) on this machine, or set "
                "GOOGLE_APPLICATION_CREDENTIALS.")
    return ""


_fs_cache = [0.0, None]


def gcs_fs():
    """Google Cloud Storage client (re-made every 25 minutes so a gcloud token never goes stale)."""
    if _fs_cache[1] is None or time.time() - _fs_cache[0] > 1500:
        import gcsfs
        _fs_cache[:] = [time.time(), gcsfs.GCSFileSystem(token=_auth_token())]
    return _fs_cache[1]


def join(base: Any, name: str) -> str:
    """os.path.join that keeps 'gs://' intact (pathlib would turn it into 'gs:/')."""
    base = str(base)
    return f"{base.rstrip('/')}/{name}" if is_uri(base) else os.path.join(base, name)

def exists(path: Any) -> bool:
    return gcs_fs().exists(_strip(path)) if is_uri(path) else Path(path).exists()

def glob(pattern: str) -> List[str]:
    """Sorted matches for a pattern such as '<dir>/*.nc', as gs:// URIs (or local paths)."""
    if is_uri(pattern):
        return sorted("gs://" + m for m in gcs_fs().glob(_strip(pattern)))
    import glob as _glob
    return sorted(_glob.glob(str(pattern)))

def open_dataset(path: Any, **kwargs):
    """xarray.open_dataset that also reads straight from gs://. Data is streamed from the bucket."""
    import xarray as xr
    if not is_uri(path):
        return xr.open_dataset(path, **kwargs)
    f = gcs_fs().open(_strip(path), "rb")
    try:
        return xr.open_dataset(f, **{"engine": "h5netcdf", **kwargs})      # NetCDF4 / HDF5
    except Exception:
        f.seek(0)
        return xr.open_dataset(f, **{**kwargs, "engine": "scipy"})         # NetCDF3

# ------------------------------------------------------------------------------
# Picking local vs bucket, item by item
# ------------------------------------------------------------------------------
DATA_SOURCES: Dict[str, str] = {}      # what was picked for each item: "local", "bucket" or "missing"
_MISSING_ITEMS: List[str] = []         # reported together in ONE warning (see "Data check" below)

def _match_case(p: Path) -> Path:
    """Case-insensitive match of the folders below ~/data/external (AIFS vs aifs, ...)."""
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
    if p.suffix == ".nc":                                   # a single file (e.g. the mask)
        return p.is_file()
    if not p.is_dir():
        return False
    return next(p.rglob("*.nc") if recursive else p.glob("*.nc"), None) is not None

def _pick(label: str, local_candidates: List[tuple], remote_uri: str):
    """First local candidate that holds .nc files; otherwise the bucket URI (read in place);
    otherwise the first local candidate (so the usual 'not found' message points at it)."""
    if DATA_MODE != "bucket":
        for cand, recursive in local_candidates:
            c = _match_case(Path(cand))
            if _has_nc(c, recursive):
                DATA_SOURCES[label] = "local"
                return c
    if BUCKET_OK:
        DATA_SOURCES[label] = "bucket"
        return remote_uri
    DATA_SOURCES[label] = "missing"
    first = Path(local_candidates[0][0])
    _MISSING_ITEMS.append(f"{label}: {first}")
    return first
LOCAL_MODEL_DIRS = {             # local mapping, aligned with BENCHMARK_MODEL_CATALOG keys
    "aifs_single": EXTERNAL_DATA_DIR / "aifs_single",
    "fuxi":        EXTERNAL_DATA_DIR / "fuxi",
    "graphcast":   EXTERNAL_DATA_DIR / "graphcast",
    "aifs_ens":    EXTERNAL_DATA_DIR / "aifs_ens",
    "gencast":     EXTERNAL_DATA_DIR / "gencast",
    "aurora15":    EXTERNAL_DATA_DIR / "aurora15",
    "atlas_crps":  EXTERNAL_DATA_DIR / "atlas_crps",
    "neuralgcm":   EXTERNAL_DATA_DIR / "neuralgcm",
    "fgn_mini":    EXTERNAL_DATA_DIR / "fgn_mini",
}
def _model_dir(key: str):
    return _pick(f"model {key}", [(LOCAL_MODEL_DIRS[key], True)], _model_uri(key))

# Observations: ~/data/external/CHIRPS/<Country>, else the flat CHIRPS folder, else the older
# CHIRPS_IMERG folder; else gs://<bucket>/demo4/DATA/<Country>/CHIRPS
OBS_DIR = _pick("observations",
                [(EXTERNAL_DATA_DIR / obs / region, True),
                 (EXTERNAL_DATA_DIR / obs, False),
                 (EXTERNAL_DATA_DIR / "CHIRPS", False)],
                OBS_URI)
CHIRPS_DIR = OBS_DIR                                        # legacy alias

# Region mask: ~/data/external/jjas_...nc, else the bucket (set ROMP_MASK_URI if it lives elsewhere)
JJAS_MASK_FILE = _pick("region mask",
                       [(EXTERNAL_DATA_DIR / "jjas_100mm_rainfall_mask_0p25.nc", False)],
                       os.getenv("ROMP_MASK_URI")
                       or f"{BUCKET_ROOT}/DATA/{region}/jjas_100mm_rainfall_mask_0p25.nc")
JJAS_DIR = JJAS_MASK_FILE
MONSOON_DIR = PROCESSED_DATA_DIR / "Monsoon" / "2026"

def _find_apps_dir(proj_root: Path) -> Path:
    """
    Locate the folder holding apps/benchmarking/ROMP/momp.

    PROJ_ROOT is where data/ lives, but the code (apps/) can sit elsewhere, e.g. next to this
    file when the project is nested (ai_weather/ai_weather/apps with data in ai_weather/data).
    """
    marker = Path("benchmarking") / "ROMP" / "momp" / "driver.py"
    for base in (proj_root, CURRENT_FILE.parent, PKG_DIR):
        if (base / "apps" / marker).exists():
            return base / "apps"
    return proj_root / "apps"          # default (may not exist; same behaviour as before)

APPS_DIR = _find_apps_dir(PROJ_ROOT)
ROMP_ROOT = APPS_DIR / "benchmarking" / "ROMP"
MOMP_PKG_DIR = ROMP_ROOT / "momp"

ROMP_OUT_DIR = DATA_DIR / "ROMP_OUT" / "et" / "output"
ROMP_FIG_DIR = DATA_DIR / "ROMP_OUT" / "et" / "figure"

ROMP_OUT_DIR.mkdir(parents=True, exist_ok=True)
ROMP_FIG_DIR.mkdir(parents=True, exist_ok=True)

for path_obj in [PROJ_ROOT, PKG_DIR, ROMP_ROOT, MOMP_PKG_DIR]:
    path_str = str(path_obj.resolve())
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

def resolve_path(p: Any, root: Optional[Path] = None) -> str:
    if not p:
        return ""
    if is_uri(p):                       # gs:// URIs are used as they are
        return str(p).rstrip("/")
    path_obj = Path(str(p)).expanduser()
    if path_obj.is_absolute():
        return str(path_obj.resolve())

    parts = path_obj.parts
    # If path starts with "data/...", resolve against DATA_DIR (~/data)
    if parts and parts[0] == "data":
        return str(DATA_DIR.joinpath(*parts[1:]).resolve())

    if root is None:
        root = PROJ_ROOT
    return str((Path(root) / path_obj).resolve())

def setup_earthdata_auth() -> None:
    pass

# ==============================================================================
# BENCHMARKING CATALOG — single source of truth
# ==============================================================================
BENCHMARK_MODEL_CATALOG: Dict[str, Dict[str, Any]] = {
    "AIFS": {
        "label": "Deterministic · AIFS",
        "dir": _model_dir("aifs_single"),
        "model_name": "aifs_single",
        "file_pattern": "{}_adm3.nc",
        "var": "tp",
        "probabilistic": False,
        "category": "Deterministic",
    },
    "FuXi": {
        "label": "Deterministic · FuXi",
        "dir": _model_dir("fuxi"),
        "model_name": "fuxi",
        "file_pattern": "{}.nc",
        "var": "tp",
        "probabilistic": False,
        "category": "Deterministic",
    },
    "GraphCast": {
        "label": "Deterministic · GraphCast",
        "dir": _model_dir("graphcast"),
        "model_name": "graphcast",
        "file_pattern": "{}.nc",
        "var": "tp",
        "probabilistic": False,
        "category": "Deterministic",
    },
    "AIFS_ENS": {
        "label": "Probabilistic · AIFS ENS",
        "dir": _model_dir("aifs_ens"),
        "model_name": "aifs_ens",
        "file_pattern": "{}_adm3.nc",
        "var": "tp",
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
        "probabilistic": True,
        "category": "Probabilistic",
        "slow": True,
    },

"Aurora15": {
        "label": "Probabilistic · Aurora15",
        "dir": _model_dir("aurora15"),
        "model_name": "aurora15",
        "file_pattern": "{}.nc",
        "var": "tp",
        "probabilistic": True,
        "category": "Probabilistic",
        "slow": True,
    },


   "Atlas CRPS": {
        "label": "Probabilistic · Atlas CRPS",
        "dir": _model_dir("atlas_crps"),
        "model_name": "atlas_crps",
        "file_pattern": "{}.nc",
        "var": "tp",
        "probabilistic": True,
        "category": "Probabilistic",
        "slow": True,
    },


 "NeuralGCM": {
        "label": "Probabilistic · NeuralGCM",
        "dir": _model_dir("neuralgcm"),
        "model_name": "neuralgcm",
        "file_pattern": "{}.nc",
        "var": "tp",
        "probabilistic": True,
        "category": "Probabilistic",
        "slow": True,
    },

 "FGN mini": {
        "label": "Probabilistic · FGN mini",
        "dir": _model_dir("fgn_mini"),
        "model_name": "fgn_mini",
        "file_pattern": "{}.nc",
        "var": "tp",
        "probabilistic": True,
        "category": "Probabilistic",
        "slow": True,
    },
}

# ==============================================================================
# DATA CHECK: one warning (not one per item) when inputs were found nowhere
# ==============================================================================
if _MISSING_ITEMS:
    warnings.warn(
        f"config: {len(_MISSING_ITEMS)} input(s) not found locally and the bucket fallback is off "
        f"({BUCKET_NOTE}). Missing: " + "; ".join(_MISSING_ITEMS)
        + ". Run `python ai_weather/config.py` for details.",
        stacklevel=2,
    )


def _catalog_models(category: str, catalog: Optional[Dict[str, Any]] = None):
    catalog = catalog or BENCHMARK_MODEL_CATALOG
    return [m for m in catalog.values() if m["category"] == category]

# ==============================================================================
# SANITY CHECKS
# ==============================================================================
for _key, _meta in BENCHMARK_MODEL_CATALOG.items():
    _expected_bool = (_meta.get("category") == "Probabilistic")
    if _meta.get("probabilistic") != _expected_bool:
        raise ValueError(
            f"BENCHMARK_MODEL_CATALOG['{_key}'] is inconsistent: "
            f"category='{_meta.get('category')}' but probabilistic={_meta.get('probabilistic')}."
        )
del _key, _meta, _expected_bool

# ==============================================================================
# 1. CORE STRUCTURE & DIMENSIONS
# ==============================================================================
project_name = "probabilistic benchmarking with CHIRPS data"
work_dir = str(ROMP_ROOT.resolve())
pkg_dir = str(MOMP_PKG_DIR.resolve())
layout = ("model", "verification_window")

# ==============================================================================
# 2. OBSERVATION DATA SETTINGS
# ==============================================================================
_OBS_SETS = {
    "CHIRPS": dict(dir=OBS_DIR, pattern=("{}.nc",), var="precip"),
    "IMERG": dict(dir=OBS_DIR, pattern=("{}.nc",), var="RAINFALL"),
    #"ENACTS": dict(dir=ENACTS_DIR, pattern=("{}_0p25.nc",), var="precip"),
}
if obs not in _OBS_SETS:   # obs defaults to CHIRPS (see top of file)
    raise ValueError(f"ROMP_OBS must be one of {list(_OBS_SETS)}, got '{obs}'")
_obs = _OBS_SETS[obs]

obs_dir = resolve_path(_obs["dir"])
obs_file_pattern = _obs["pattern"]
obs_var = _obs["var"]
obs_unit_cvt = None

# ==============================================================================
# 3. REFERENCE MODEL (BENCHMARK) SETTINGS
# ==============================================================================
ref_model = "climatology"
ref_model_dir = obs_dir
ref_model_file_pattern = _obs["pattern"][0]
ref_model_var = _obs["var"]
ref_model_unit_cvt = None

# ==============================================================================
# 4. REGION DEFINITIONS
# ==============================================================================
# region is set at the top of the file (ROMP_REGION, default Ethiopia)
nc_mask = resolve_path(JJAS_MASK_FILE)
shpfile_dir = EXTERNAL_DATA_DIR / "shapefile"        # only used when polygon=True
polygon = False

# ==============================================================================
# 5. ONSET CRITERIA (Physics Logic)
# ==============================================================================
wet_init = 1
wet_threshold = 20
wet_spell = 3
dry_threshold = 1
dry_spell = 7
dry_extent = 21
thresh_file = None
thresh_var = None

# ==============================================================================
# 6. TEMPORAL SETTINGS
# ==============================================================================
start_date = (2003, 5, 1)
end_date = (2023, 7, 31)
start_year_clim = 2007
end_year_clim = 2024
init_days = (0, 3)
init_type = "weekly"
date_filter_year = 2024

# ==============================================================================
# 7. EVALUATION & VERIFICATION PARAMETERS
# ==============================================================================
verification_window_list = ((1, 15), (16, 30))
tolerance_days_list = (3, 3, 3, 5, 5, 5)
max_forecast_day = 30
day_bins = ((1, 5), (6, 10), (11, 15), (16, 20), (21, 25), (26, 30))

# ==============================================================================
# 8. OUTPUT & GRAPHICS SETTINGS
# ==============================================================================
dir_out = os.getenv("ROMP_OUT_DIR", str(ROMP_OUT_DIR.resolve()))
dir_fig = os.getenv("ROMP_FIG_DIR", str(ROMP_FIG_DIR.resolve()))

save_fig = True
save_nc_spatial_far_mr_mae = True  # Changed to True to match config.in
save_csv_score = True
save_nc_climatology = True

plot_climatology_onset = True

show_plot = False
show_panel = False

# ==============================================================================
# 9. PARALLELIZATION & DEBUG
# ==============================================================================
parallel = True
debug = False

# ==============================================================================
# RESERVED / LEGACY PARAMETERS
# ==============================================================================
mok = (5, 15)
years = None
years_clim = None
fallback_date = None

apply_idr_calibration = False

# ==============================================================================
# MODE-DEPENDENT OVERRIDES
# ==============================================================================
def compute_mode_overrides(mode: str = "det", catalog: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Returns a dict of config overrides for the given mode.
    """
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

    n = len(selected_models)
    overrides: Dict[str, Any] = {
        "mode": mode_str,
        "probabilistic": is_prob,
        "model_list": tuple(m["model_name"] for m in selected_models),
        "model_dir_list": tuple(resolve_path(m["dir"]) for m in selected_models),
        "model_var_list": tuple(m.get("var", "tp") for m in selected_models),
        "unit_cvt_list": (None,) * n,
        "file_pattern_list": tuple(m.get("file_pattern", "{}.nc") for m in selected_models),
        "skill_score": True,
        # Align with config.in values
        "verification_window_list": ((1, 15), (16, 30)),
        "tolerance_days_list": (3, 5),
        "max_forecast_day": 30,
        "day_bins": ((1, 5), (6, 10), (11, 15), (16, 20), (21, 25), (26, 30)),
        "onset_percentage_threshold": 0.5,  # Match config.in
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
    """One line saying what is wrong and what to do about it."""
    if isinstance(e, ImportError):
        return (f"{type(e).__name__}: {e}. Install the bucket reader in this Python environment: "
                f"pip install gcsfs h5netcdf h5py")
    return f"{type(e).__name__}: {e}." + (_auth_hint(e) or "")


def check_data() -> int:
    """Check everything the benchmark reads (local files, or the bucket where it is used).
    Returns 0 if all found, 2 if something is missing, 1 if the bucket could not be read."""
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
        except Exception as e:                       # one bad item must not hide the others
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
    # python ai_weather/config.py    -> show where each input comes from (local or bucket) and check it exists
    print(f".env file : {ENV_FILE}")
    print(f"mode      : {DATA_MODE}  (local first, bucket as fallback)" if DATA_MODE == "auto" else f"mode      : {DATA_MODE}")
    print(f"local     : {EXTERNAL_DATA_DIR}")
    print(f"bucket    : {BUCKET_ROOT if BUCKET_OK else '(off: ' + BUCKET_NOTE + ')'}")
    print(f"region    : {region}")
    print(f"outputs   : {ROMP_OUT_DIR}   (results are written locally)")
    if BUCKET_OK:
        print("browse    :")
        for _demo, _rel in (("demo3", ""), ("demo4", f"MODEL/{region}/{MODEL_FOLDERS['AIFS']}"),
                            ("demo4", f"MODEL/{region}/{MODEL_FOLDERS['AIFS_ENS']}"),
                            ("demo4", f"DATA/{region}/{obs}")):
            print(f"  {console_url(_demo, _rel)}")
    print("check     :")
    sys.exit(check_data())
