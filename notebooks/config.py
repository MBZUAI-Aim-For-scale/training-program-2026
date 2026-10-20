import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Optional
from dotenv import load_dotenv

load_dotenv()

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
_env_root = os.getenv("AI_WEATHER_REPO")
PROJ_ROOT = Path(_env_root).expanduser().resolve() if _env_root else BASE_DIR

_env_data_dir = os.getenv("ROMP_DATA_DIR")
#DATA_DIR = Path(_env_data_dir).expanduser().resolve() if _env_data_dir else (PROJ_ROOT / "data")
DATA_DIR = Path(
    "/Users/mesfind/opt/Research/uchicago/ETAgroMetAIx/HCWF/"
    "ETAgroMetAIx/etagrometaix/data"
)

EXTERNAL_DATA_DIR = DATA_DIR / "external"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ENACTS_DIR = EXTERNAL_DATA_DIR / "ENACTS_regridded_025"
CHIRPS_DIR = EXTERNAL_DATA_DIR / "CHIRPS_IMERG"
JJAS_MASK_FILE = DATA_DIR / "jjas_100mm_rainfall_mask_0p25.nc"
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
    if root is None:
        root = PROJ_ROOT
    path_obj = Path(p).expanduser()
    if not path_obj.is_absolute():
        path_obj = Path(root) / path_obj
    return str(path_obj.resolve())

def setup_earthdata_auth() -> None:
    pass

# ==============================================================================
# BENCHMARKING CATALOG — single source of truth
# ==============================================================================
# IMPORTANT: Directory names MUST match the actual filesystem case exactly.
# Verify with: ls data/external/
BENCHMARK_MODEL_CATALOG: Dict[str, Dict[str, Any]] = {
    "AIFS": {
        "label": "Deterministic · AIFS",
        "dir": EXTERNAL_DATA_DIR / "AIFS" / "0p25",
        "model_name": "AIFS",
        "file_pattern": "{}.nc",
        "var": "tp",
        "probabilistic": False,
        "category": "Deterministic",
    },
    "FuXi": {
        "label": "Deterministic · FuXi",
        "dir": EXTERNAL_DATA_DIR / "fuxi", 
        "model_name": "FuXi",
        "file_pattern": "{}.nc",
        "var": "tp",
        "probabilistic": False,
        "category": "Deterministic",
    },
    "GraphCast": {
        "label": "Deterministic · GraphCast",
        "dir": EXTERNAL_DATA_DIR / "graphcast",
        "model_name": "GraphCast",
        "file_pattern": "{}.nc",
        "var": "tp",
        "probabilistic": False,
        "category": "Deterministic",
    },
    "AIFS_ENS": {
        "label": "Probabilistic · AIFS ENS",
        "dir": EXTERNAL_DATA_DIR / "AIFS_ENS" / "0p25",  
        "model_name": "AIFS_ENS",
        "file_pattern": "{}.nc",
        "var": "tp",
        "probabilistic": True,
        "category": "Probabilistic",
    },
    "gencast": {
        "label": "Probabilistic · GenCast",
        "dir": EXTERNAL_DATA_DIR / "gencast" / "0p25",  
        "model_name": "gencast",
        "file_pattern": "{}.nc",
        "var": "tp",
        "probabilistic": True,
        "category": "Probabilistic",
    },
}

# ==============================================================================
# AUTO-DETECT ACTUAL FOLDER CASE ON DISK
# ==============================================================================
# This prevents "File not found" errors caused by case mismatches
# (e.g., catalog says "AIFS_ENS" but folder is "aifs_ens")
def _autocorrect_catalog_paths(catalog: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    For each catalog entry, check if the declared directory exists.
    If not, try case-insensitive match in the parent directory.
    """
    for key, meta in catalog.items():
        declared_dir = Path(meta["dir"])
        if declared_dir.exists():
            continue

        parent = declared_dir.parent
        target_name = declared_dir.name.lower()

        if not parent.exists():
            continue

        # Try to find a case-insensitive match
        for child in parent.iterdir():
            if child.is_dir() and child.name.lower() == target_name:
                meta["dir"] = child
                print(f"[config] Auto-corrected catalog path for '{key}': "
                      f"{declared_dir} -> {child}")
                break

    return catalog

BENCHMARK_MODEL_CATALOG = _autocorrect_catalog_paths(BENCHMARK_MODEL_CATALOG)

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
project_name = "probabilistic benchmarking with ENACTS data"
work_dir = str(ROMP_ROOT.resolve())
pkg_dir = str(MOMP_PKG_DIR.resolve())
layout = ("model", "verification_window")

# ==============================================================================
# 2. OBSERVATION DATA SETTINGS
# ==============================================================================
# IMPORTANT: Use "CHIRPS" as the key to match config.in expectations.
# The config.in uses obs = "CHIRPS_IMERG" but the variable name is "RAINFALL"
# and the directory is CHIRPS_DIR. We unify this here.
_OBS_SETS = {
    "ENACTS": dict(dir=ENACTS_DIR, pattern=("{}_0p25.nc",), var="precip"),
    "CHIRPS": dict(dir=CHIRPS_DIR, pattern=("{}.nc",), var="RAINFALL"),
    "CHIRPS_IMERG": dict(dir=CHIRPS_DIR, pattern=("{}.nc",), var="RAINFALL"),  # Alias for config.in compatibility
}
obs = os.getenv("ROMP_OBS", "CHIRPS_IMERG").upper()  # Default to CHIRPS to match config.in
if obs not in _OBS_SETS:
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
region = "Ethiopia"
nc_mask = resolve_path(JJAS_MASK_FILE)
shpfile_dir = EXTERNAL_DATA_DIR / "shapefile"
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
start_date = (2007, 5, 1)
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
tolerance_days_list = (3, 5)
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
# SANITY CHECKS (paths)
# ==============================================================================
for _label, _p in (
    ("data directory", DATA_DIR),
    ("observation directory", Path(obs_dir)),
    ("region mask (nc_mask)", Path(nc_mask)),
):
    if not _p.exists():
        warnings.warn(f"config: {_label} not found: {_p} (PROJ_ROOT={PROJ_ROOT})", stacklevel=2)
del _label, _p

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