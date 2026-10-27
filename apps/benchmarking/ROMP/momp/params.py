"""
momp/params.py
Default parameter configuration for MOMP (Rainy Season Onset Metrics Package).
Dynamically imports and falls back to values defined in ai_weather/config.py.
"""

import sys
from pathlib import Path

# ==============================================================================
# Robust config.py Resolution
# ==============================================================================
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

# ==============================================================================
# 1. CORE STRUCTURE & DIMENSIONS
# ==============================================================================
project_name = getattr(config, "project_name", "probabilistic benchmarking with CHIRPS data")
work_dir = getattr(config, "work_dir", str(config.ROMP_ROOT.resolve()))
pkg_dir = getattr(config, "pkg_dir", str(config.MOMP_PKG_DIR.resolve()))
layout = getattr(config, "layout", ("model", "verification_window"))

# ==============================================================================
# 2. OBSERVATION DATA SETTINGS
# ==============================================================================
region = getattr(config, "region", "Ethiopia")
obs = getattr(config, "obs", "CHIRPS")
obs_dir = str(getattr(config, "obs_dir", getattr(config, "OBS_DIR", config.EXTERNAL_DATA_DIR / "CHIRPS")))
obs_file_pattern = getattr(config, "obs_file_pattern", ("{}.nc",))
obs_var = getattr(config, "obs_var", "precip")
obs_unit_cvt = getattr(config, "obs_unit_cvt", None)

# ==============================================================================
# 3. REFERENCE MODEL (BENCHMARK) SETTINGS
# ==============================================================================
ref_model = getattr(config, "ref_model", "climatology")
ref_model_dir = str(getattr(config, "ref_model_dir", obs_dir))
ref_model_file_pattern = getattr(config, "ref_model_file_pattern", "{}.nc")
ref_model_var = getattr(config, "ref_model_var", obs_var)
ref_model_unit_cvt = getattr(config, "ref_model_unit_cvt", None)

# ==============================================================================
# 4. REGION & MASK DEFINITIONS
# ==============================================================================
nc_mask = str(getattr(config, "nc_mask", getattr(config, "JJAS_MASK_FILE", config.EXTERNAL_DATA_DIR / "jjas_100mm_rainfall_mask_0p25.nc")))
shpfile_dir = str(getattr(config, "shpfile_dir", config.EXTERNAL_DATA_DIR / "shapefile"))
polygon = getattr(config, "polygon", False)

# ==============================================================================
# 5. ONSET CRITERIA (PHYSICS LOGIC)
# ==============================================================================
wet_init = getattr(config, "wet_init", 1)
wet_threshold = getattr(config, "wet_threshold", 20)
wet_spell = getattr(config, "wet_spell", 3)
dry_threshold = getattr(config, "dry_threshold", 1)
dry_spell = getattr(config, "dry_spell", 7)
dry_extent = getattr(config, "dry_extent", 21)
thresh_file = getattr(config, "thresh_file", None)
thresh_var = getattr(config, "thresh_var", None)
onset_percentage_threshold = getattr(config, "onset_percentage_threshold", 0.5)

# ==============================================================================
# 6. TEMPORAL SETTINGS
# ==============================================================================
start_date = getattr(config, "start_date", (2003, 5, 1))
end_date = getattr(config, "end_date", (2023, 7, 31))
start_year_clim = getattr(config, "start_year_clim", 2007)
end_year_clim = getattr(config, "end_year_clim", 2024)
init_days = getattr(config, "init_days", (0, 3))
init_type = getattr(config, "init_type", "weekly")
date_filter_year = getattr(config, "date_filter_year", 2024)

# ==============================================================================
# 7. EVALUATION & VERIFICATION PARAMETERS
# ==============================================================================
verification_window_list = getattr(config, "verification_window_list", ((1, 15), (16, 30)))
tolerance_days_list = getattr(config, "tolerance_days_list", (3, 5))
max_forecast_day = getattr(config, "max_forecast_day", 30)
day_bins = getattr(config, "day_bins", ((1, 5), (6, 10), (11, 15), (16, 20), (21, 25), (26, 30)))

# Model Lists & Modes
model_list = getattr(config, "model_list", ("aifs_single",))
model_dir_list = getattr(config, "model_dir_list", (str(config.EXTERNAL_DATA_DIR / "aifs_single"),))
model_var_list = getattr(config, "model_var_list", ("tp",))
unit_cvt_list = getattr(config, "unit_cvt_list", (None,))
file_pattern_list = getattr(config, "file_pattern_list", ("{}.nc",))
probabilistic = getattr(config, "probabilistic", False)
members = getattr(config, "members", "All")

# Diagnostic & Metric Flags
FAR = getattr(config, "FAR", True)
MAE = getattr(config, "MAE", True)
MR = getattr(config, "MR", True)
BS = getattr(config, "BS", False)
RPS = getattr(config, "RPS", False)
AUC = getattr(config, "AUC", False)
Reliability = getattr(config, "Reliability", False)
skill_score = getattr(config, "skill_score", False)

# ==============================================================================
# 8. OUTPUT & GRAPHICS SETTINGS
# ==============================================================================
dir_out = str(getattr(config, "ROMP_OUT_DIR", config.DATA_DIR / "ROMP_OUT" / "et" / "output"))
dir_fig = str(getattr(config, "ROMP_FIG_DIR", config.DATA_DIR / "ROMP_OUT" / "et" / "figure"))

save_fig = getattr(config, "save_fig", True)
save_nc_spatial_far_mr_mae = getattr(config, "save_nc_spatial_far_mr_mae", True)
save_csv_score = getattr(config, "save_csv_score", True)
save_nc_climatology = getattr(config, "save_nc_climatology", True)

plot_climatology_onset = getattr(config, "plot_climatology_onset", True)
plot_spatial_far_mr_mae = getattr(config, "plot_spatial_far_mr_mae", True)
plot_panel_heatmap_error = getattr(config, "plot_panel_heatmap_error", True)
plot_heatmap_bss_auc = getattr(config, "plot_heatmap_bss_auc", False)
plot_reliability = getattr(config, "plot_reliability", False)
plot_panel_heatmap_skill = getattr(config, "plot_panel_heatmap_skill", False)
plot_bar_bss_rpss_auc = getattr(config, "plot_bar_bss_rpss_auc", False)

show_plot = getattr(config, "show_plot", False)
show_panel = getattr(config, "show_panel", False)

# ==============================================================================
# 9. PARALLELIZATION & DEBUG
# ==============================================================================
parallel = getattr(config, "parallel", True)
debug = getattr(config, "debug", False)

# Reserved / Legacy Attributes
mok = getattr(config, "mok", (5, 15))
years = getattr(config, "years", None)
years_clim = getattr(config, "years_clim", None)
fallback_date = getattr(config, "fallback_date", None)
apply_idr_calibration = getattr(config, "apply_idr_calibration", False)
