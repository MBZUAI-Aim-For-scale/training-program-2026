from __future__ import annotations
import os
import re
import sys
import tomllib
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple
import geopandas as gpd
import numpy as np
import pandas as pd
import requests
import streamlit as st


# ==============================================================================
# Suppress Streamlit "No runtime found" warnings when config is imported by CLI scripts
# ==============================================================================
import logging
_streamlit_logger = logging.getLogger("streamlit.runtime.caching.cache_data_api")
_streamlit_logger.addFilter(
    lambda record: "No runtime found" not in record.getMessage()
)

# ==============================================================================
# Core Application Directory Hierarchy
# ==============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
EXTERNAL_DATA_DIR = PROJECT_ROOT / "data" /"external"
ENACTS_REGRIDDED_DIR = EXTERNAL_DATA_DIR / "ENACTS_regridded_025"
ENACTS_DIR = EXTERNAL_DATA_DIR / "ENACTS"

# ==============================================================================
# Onset verification config
# ==============================================================================
CLIM_ONSET_PCTL_EARLY = 25
CLIM_ONSET_PCTL_LATE  = 75
DPI = 300
PLOT_COLORS = {
    "early": "#1b9e77",
    "normal": "#66a61e",
    "late": "#d95f02",
    "no_onset": "#999999"
}
REF_ACCUM_MM = 20.0
REF_DAILY_MM = 0.1
ROLLING_DAYS = 3
plot_obs_days = 30

# ==============================================================================
# Forecast Verification Paths & Configuration
# ==============================================================================
ONSET_INFO_DIR = DATA_DIR / "onset_info"

# CORRECTED: ONSET_DATA_DIR points to where forecast data actually lives
ONSET_DATA_DIR = ONSET_INFO_DIR / "data"  # data/onset_info/data

# ONSET_VERIFICATION_DIR points to Monsoon_Data (reference files, models, results)
ONSET_VERIFICATION_DIR = ONSET_INFO_DIR / "Monsoon_Data"  # data/onset_info/Monsoon_Data

# ONSET_FEATURE_DATA_DIR for dissemination cells
ONSET_FEATURE_DATA_DIR = ONSET_INFO_DIR / "onset_features"

MESSAGE_CSV_PATH = ONSET_VERIFICATION_DIR / "message_progression.csv"
ONSET_PADDING_CSV = ONSET_VERIFICATION_DIR / "onset_padding.csv"

ONSET_RUN_OUTPUT_ROOT = ONSET_INFO_DIR / "operational" / "monsoon-onset" / "IMERG" / "data" / "ethiopia" / "OUTPUT"
ONSET_RUN_INPUT_ROOT = ONSET_INFO_DIR / "operational" / "monsoon-onset" / "IMERG" / "data" / "ethiopia" / "PKL"

ENACTS_CLIM_PKL_PATH = ONSET_RUN_INPUT_ROOT / "imd_clim_mok_date_wide.pkl"

MESSAGE_YEAR = datetime.now().year
YEAR = MESSAGE_YEAR

# Base directories for processed data and reference files
ONSET_PROCESSED_DIR = ONSET_VERIFICATION_DIR / "Processed_Data"
ONSET_REFERENCE_DIR = ONSET_VERIFICATION_DIR / "reference"

# ==============================================================================
# Operational Pipeline Defaults
# ==============================================================================
DEFAULT_WORK_DIR = ONSET_PROCESSED_DIR / str(YEAR)
DEFAULT_COEF_DIR = ONSET_VERIFICATION_DIR / "results" / "dry_spell_aifs_aifs_ens"
DEFAULT_COEF_TAG = "clim_mok_date_2022_year2022"
DEFAULT_BLEND_MODEL = "blended_model"
DEFAULT_REGION = "Ethiopia"
DEFAULT_ISSUE_DATE = f"{YEAR}-06-15"

DEFAULT_MODEL_SINGLE = "aifs"
DEFAULT_MODEL_ENS = "aifs_ens"


# ==============================================================================
# Onset Forecast Shapefiles
# ==============================================================================
SHAPEFILE_DIR = PROJECT_ROOT / "data" "shapefile"



# ==============================================================================
# Onset Forecast Specs (AIFS)
# ==============================================================================
ONSET_FORECAST_AIFS_SPEC = {
    "output": {
        "out_dir": str(ONSET_PROCESSED_DIR / str(YEAR)),
    },
    "type": "rainfall_forecast",
    "input": {
        "nc_folder": str(ONSET_DATA_DIR / "forecast" / "aifs_v2" / "single"),
        "file_regex": rf"^{YEAR}_adm3\.nc$",
        "value_col": "tp",
        "wide_day_dim": "day",
        "wide_prefix": "rain",
        "parallel": True,
        "workers": 9,
    },
    "dimensions": {
        "rename": {
            "time": "time",
            "day": "day",
            "number": "number",
        }
    },
    "options": {
        "window": 3,
        "onset_definition": {
            "wet_day_min_mm": 1.0,
            "follow_days": 21,
            "dry_spell": {
                "mode": "consecutive_dry",
                "min_dry_days": 5,
                "dry_day_min_mm": 1.0,
            },
        },
        "min_day": 1,
        "max_day": 45,
    },
    "thresholds": {
        "file": str(ONSET_REFERENCE_DIR / "thresholds_df.csv"),
        "lat_col": "lat",
        "lon_col": "lon",
        "adm3_col": "adm3_name",
        "thresh_col": "onset_threshold",
    },
    "mok": {
        "file": str(ONSET_REFERENCE_DIR / "MOK_Onset_May.csv"),
        "year_col": "Year",
        "day_col": "MOK",
        "base_date": "05-01",
    },
    "filter": {
        "dissemination_cells_file": str(ONSET_VERIFICATION_DIR / "dissemination_cells.csv"),
    },
}

# ==============================================================================
# Onset Forecast Specs (AIFS ENS)
# ==============================================================================
ONSET_FORECAST_AIFS_ENS_SPEC = {
    "output": {
        "out_dir": str(ONSET_PROCESSED_DIR / str(YEAR)),
    },
    "type": "rainfall_forecast",
    "input": {
        "nc_folder": str(ONSET_DATA_DIR / "forecast" / "aifs_v2" / "ensemble"),
        "file_regex": rf"^{YEAR}_adm3\.nc$",
        "value_col": "tp",
        "wide_day_dim": "day",
        "wide_prefix": "rain",
        "parallel": True,
        "workers": 9,
    },
    "dimensions": {
        "rename": {
            "time": "time",
            "day": "day",
            "number": "number",
        }
    },
    "options": {
        "window": 3,
        "onset_definition": {
            "wet_day_min_mm": 1.0,
            "follow_days": 21,
            "dry_spell": {
                "mode": "consecutive_dry",
                "min_dry_days": 5,
                "dry_day_min_mm": 1.0,
            },
        },
        "min_day": 1,
        "max_day": 45,
    },
    "thresholds": {
        "file": str(ONSET_REFERENCE_DIR / "thresholds_df.csv"),
        "lat_col": "lat",
        "lon_col": "lon",
        "adm3_col": "adm3_name",
        "thresh_col": "onset_threshold",
    },
    "mok": {
        "file": str(ONSET_REFERENCE_DIR / "MOK_Onset_May.csv"),
        "year_col": "Year",
        "day_col": "MOK",
        "base_date": "05-01",
    },
    "filter": {
        "dissemination_cells_file": str(ONSET_VERIFICATION_DIR / "dissemination_cells.csv"),
        "max_number": 30,
    },
}

# ==============================================================================
# Onset Forecast Specs (GenCast)
# ==============================================================================
ONSET_FORECAST_GENCAST_SPEC = {
    "output": {
        "out_dir": str(ONSET_PROCESSED_DIR / str(YEAR)),
    },
    "type": "rainfall_forecast",
    "input": {
        "nc_folder": str(ONSET_DATA_DIR / "forecast" / "gencast"),
        "file_regex": rf"^{YEAR}_.*_adm3\.nc$",
        "value_col": "tp",
        "wide_day_dim": "day",
        "wide_prefix": "rain",
        "parallel": True,
        "workers": 9,
    },
    "dimensions": {
        "rename": {
            "latitude": "lat",
            "longitude": "lon",
            "time": "time",
            "day": "day",
            "number": "number",
        }
    },
    "options": {
        "window": 3,
        "onset_definition": {
            "wet_day_min_mm": 1.0,
            "follow_days": 21,
            "dry_spell": {
                "mode": "consecutive_dry",
                "min_dry_days": 5,
                "dry_day_min_mm": 1.0,
            },
        },
        "min_day": 1,
        "max_day": 45,
    },
    "thresholds": {
        "file": str(ONSET_REFERENCE_DIR / "thresholds_df.csv"),
        "lat_col": "lat",
        "lon_col": "lon",
        "adm3_col": "adm3_name",
        "thresh_col": "onset_threshold",
    },
    "mok": {
        "file": str(ONSET_REFERENCE_DIR / "MOK Onset May.csv"),
        "year_col": "Year",
        "day_col": "MOK",
        "base_date": "05-01",
    },
    "filter": {
        "dissemination_cells_file": str(ONSET_VERIFICATION_DIR / "dissemination_cells.csv"),
        "max_number": 30,
    },
}

# ==============================================================================
# Onset Forecast Specs (IMD Climatology MOK Date)
# ==============================================================================
ONSET_FORECAST_IMD_CLIM_SPEC = {
    "type": "ground_truth_rainfall",
    "input": {
        "nc_folder": str(ONSET_DATA_DIR / "ENACTS"),
        "gt_path": str(ONSET_VERIFICATION_DIR / "Models" / "dry_spell" / "imd_clim_mok_date_wide.pkl"),
        "file_regex": r"^[0-9]{4}_adm3\.nc$",
        "value_col": "precip",
        "parallel": False,
        "workers": 10,
    },
    "dimensions": {
        "rename": {
            "TIME": "time",
            "time": "time",
        }
    },
    "output": {
        "out_dir": str(ONSET_PROCESSED_DIR / str(YEAR)),
    },
    "options": {
        "window": 3,
        "onset_definition": {
            "wet_day_min_mm": 1.0,
            "follow_days": 21,
            "dry_spell": {
                "mode": "consecutive_dry",
                "min_dry_days": 5,
                "dry_day_min_mm": 1.0,
            },
        },
        "cutoff_month_day": "05-01",
        "cell_transform_enabled": False,
    },
    "thresholds": {
        "file": str(ONSET_REFERENCE_DIR / "thresholds_df.csv"),
        "lat_col": "lat",
        "lon_col": "lon",
        "adm3_col": "adm3_name",
        "thresh_col": "onset_threshold",
    },
    "mok": {
        "file": str(ONSET_REFERENCE_DIR / "MOK Onset May.csv"),
        "year_col": "Year",
        "day_col": "MOK",
        "base_date": "05-01",
    },
    "filter": {
        "dissemination_cells_file": str(ONSET_VERIFICATION_DIR / "dissemination_cells.csv"),
    },
    "paths": {
        "climatology_out_dir": str(ONSET_PROCESSED_DIR / str(YEAR)),
        "climatology_out_stem": "imd_clim_mok_date_clim_issue",
    },
    "climatologies": {
        "clim": {
            "out_stem": "imd_clim_mok_date_clim_issue",
            "onset_col": "mr_onset_day",
            "conditional": True,
            "train_year_min": 2000,
            "train_year_max": 2022,
            "test_year_min": YEAR,
            "test_year_max": YEAR,
            "cv_by_year": False,
            "season_start_md": "05-01",
            "issue_end_md": "07-31",
            "forecast_window": 28,
            "min_onset_years": 10,
        },
        "clim_unc": {
            "out_stem": "imd_clim_mok_date_clim_unc_issue",
            "onset_col": "mr_onset_day",
            "conditional": False,
            "train_year_min": 2000,
            "train_year_max": 2022,
            "test_year_min": YEAR,
            "test_year_max": YEAR,
            "cv_by_year": False,
            "season_start_md": "05-01",
            "issue_end_md": "07-31",
            "forecast_window": 28,
            "min_onset_years": 10,
        },
    },
}

# ==============================================================================
# Onset Forecast Specs (NGCM)
# ==============================================================================
ONSET_FORECAST_NGCM_SPEC = {
    "output": {
        "out_dir": str(ONSET_PROCESSED_DIR / str(YEAR)),
    },
    "type": "rainfall_forecast",
    "input": {
        "nc_folder": str(ONSET_DATA_DIR / "forecast" / "gcm"),
        "file_regex": rf"^{YEAR}_.*_adm3\.nc$",
        "value_col": "tp",
        "wide_day_dim": "day",
        "wide_prefix": "rain",
        "parallel": True,
        "workers": 9,
    },
    "dimensions": {
        "rename": {
            "time": "time",
            "day": "day",
            "number": "number",
        }
    },
    "options": {
        "window": 3,
        "onset_definition": {
            "wet_day_min_mm": 1.0,
            "follow_days": 21,
            "dry_spell": {
                "mode": "consecutive_dry",
                "min_dry_days": 5,
                "dry_day_min_mm": 1.0,
            },
        },
        "min_day": 1,
        "max_day": 45,
    },
    "thresholds": {
        "file": str(ONSET_REFERENCE_DIR / "thresholds_df.csv"),
        "lat_col": "lat",
        "lon_col": "lon",
        "adm3_col": "adm3_name",
        "thresh_col": "onset_threshold",
    },
    "mok": {
        "file": str(ONSET_REFERENCE_DIR / "MOK Onset May.csv"),
        "year_col": "Year",
        "day_col": "MOK",
        "base_date": "05-01",
    },
    "filter": {
        "dissemination_cells_file": str(ONSET_VERIFICATION_DIR / "dissemination_cells.csv"),
        "max_number": 30,
    },
}

# ==============================================================================
# Onset Forecast Specs (Combine Forecasts Template - Clim MOK Date)
# ==============================================================================
ONSET_FORECAST_COMBINE_CLIM_SPEC = {
    "type": "combine_forecasts",
    "input": {
        "ground_truth_wide_rds": str(ONSET_VERIFICATION_DIR/ "Models" / "dry_spell" / "imd_clim_mok_date_wide.pkl"),
        "climatologies": {
            "clim": {
                "rds": str(ONSET_PROCESSED_DIR / str(YEAR) / "imd_clim_mok_date_clim_issue.pkl"),
                "out_prefix": "clim_p_onset",
            },
            "clim_unc": {
                "rds": str(ONSET_PROCESSED_DIR / str(YEAR) / "imd_clim_mok_date_clim_unc_issue.pkl"),
                "out_prefix": "clim_unc_p_onset",
            },
        },
    },
    "options": {
        "join": "inner",
        "plus_label_template": "day_{plus}plus",
        "trim_forecasts_after_true_onset": True,
        "default_prob_col": "predicted_prob",
    },
    "forecasts": {
        "aifs_ens": {
            "max_day": 45,
            "sources": [
                {
                    "file": str(ONSET_PROCESSED_DIR / str(YEAR) / f"aifs_ens_{YEAR}_wide.pkl"),
                    "years": f"{YEAR}:{YEAR}",
                }
            ],
            "constants": [
                {"col": "onset_thresh", "out": "onset_thresh"},
                {"col": "mok_date", "out": "mok_date"},
            ],
            "daily": [
                {"col": "predicted_prob", "out": "p_onset", "add_plus": True},
                {"col": "predicted_prob_clim_mok_date", "out": "p_onset_clim_mok_date", "add_plus": True},
                {"col": "predicted_prob_mok", "out": "p_onset_mok", "add_plus": True},
                {"col": "forecast_rain", "out": "rain_mean", "add_plus": False},
            ],
        },
        "aifs": {
            "max_day": 45,
            "sources": [
                {
                    "file": str(ONSET_PROCESSED_DIR / str(YEAR) / f"aifs_{YEAR}_wide.pkl"),
                    "years": f"{YEAR}:{YEAR}",
                    "prob_col": "predicted_prob_clim_mok_date",
                }
            ],
            "daily": [
                {"col": "predicted_prob", "out": "p_onset", "add_plus": True},
                {"col": "predicted_prob_mok", "out": "p_onset_mok", "add_plus": True},
                {"col": "predicted_prob_clim_mok_date", "out": "p_onset_clim_mok_date", "add_plus": True},
                {"col": "forecast_rain", "out": "rain_mean", "add_plus": False},
            ],
        },
    },
    "climatology": {
        "max_day": 28,
        "prob_col": "predicted_prob",
    },
    "output": {
        "out_dir": str(ONSET_PROCESSED_DIR / str(YEAR)),
    },
}

# ==============================================================================
# Onset Forecast Specs (Combine Forecasts Template - NGCM & AIFS)
# ==============================================================================
ONSET_FORECAST_COMBINE_NGCM_AIFS_SPEC = {
    "type": "combine_forecasts",
    "input": {
        "ground_truth_wide_rds": str(ONSET_VERIFICATION_DIR / "Models" / "dry_spell" / "imd_clim_mok_date_wide.pkl"),
        "climatologies": {
            "clim": {
                "rds": str(ONSET_PROCESSED_DIR / str(YEAR) / "imd_clim_mok_date_clim_issue.pkl"),
                "out_prefix": "clim_p_onset",
            },
            "clim_unc": {
                "rds": str(ONSET_PROCESSED_DIR / str(YEAR) / "imd_clim_mok_date_clim_unc_issue.pkl"),
                "out_prefix": "clim_unc_p_onset",
            },
        },
    },
    "options": {
        "join": "inner",
        "plus_label_template": "day_{plus}plus",
        "trim_forecasts_after_true_onset": True,
        "default_prob_col": "predicted_prob",
    },
    "forecasts": {
        "ngcm": {
            "max_day": 45,
            "sources": [
                {
                    "file": str(ONSET_PROCESSED_DIR / str(YEAR) / f"ngcm_{YEAR}_wide.pkl"),
                    "years": f"{YEAR}:{YEAR}",
                }
            ],
            "constants": [
                {"col": "onset_thresh", "out": "onset_thresh"},
                {"col": "mok_date", "out": "mok_date"},
            ],
            "daily": [
                {"col": "predicted_prob", "out": "p_onset", "add_plus": True},
                {"col": "predicted_prob_clim_mok_date", "out": "p_onset_clim_mok_date", "add_plus": True},
                {"col": "predicted_prob_mok", "out": "p_onset_mok", "add_plus": True},
                {"col": "forecast_rain", "out": "rain_mean", "add_plus": False},
            ],
        },
        "aifs": {
            "max_day": 45,
            "sources": [
                {
                    "file": str(ONSET_PROCESSED_DIR / str(YEAR) / f"aifs_{YEAR}_wide.pkl"),
                    "years": f"{YEAR}:{YEAR}",
                    "prob_col": "predicted_prob_clim_mok_date",
                }
            ],
            "daily": [
                {"col": "predicted_prob", "out": "p_onset", "add_plus": True},
                {"col": "predicted_prob_mok", "out": "p_onset_mok", "add_plus": True},
                {"col": "predicted_prob_clim_mok_date", "out": "p_onset_clim_mok_date", "add_plus": True},
                {"col": "forecast_rain", "out": "rain_mean", "add_plus": False},
            ],
        },
    },
    "climatology": {
        "max_day": 28,
        "prob_col": "predicted_prob",
    },
    "output": {
        "out_dir": str(ONSET_PROCESSED_DIR / str(YEAR)),
    },
}

# ==============================================================================
# Onset Forecast Specs (Combine Forecasts Template - GenCast & AIFS)
# ==============================================================================
ONSET_FORECAST_COMBINE_GENCAST_AIFS_SPEC = {
    "type": "combine_forecasts",
    "input": {
        "ground_truth_wide_rds": str(ONSET_VERIFICATION_DIR / "Models" / "dry_spell" / "imd_clim_mok_date_wide.pkl"),
        "climatologies": {
            "clim": {
                "rds": str(ONSET_PROCESSED_DIR / str(YEAR) / "imd_clim_mok_date_clim_issue.pkl"),
                "out_prefix": "clim_p_onset",
            },
            "clim_unc": {
                "rds": str(ONSET_PROCESSED_DIR / str(YEAR) / "imd_clim_mok_date_clim_unc_issue.pkl"),
                "out_prefix": "clim_unc_p_onset",
            },
        },
    },
    "options": {
        "join": "inner",
        "plus_label_template": "day_{plus}plus",
        "trim_forecasts_after_true_onset": True,
        "default_prob_col": "predicted_prob",
    },
    "forecasts": {
        "gencast": {
            "max_day": 45,
            "sources": [
                {
                    "file": str(ONSET_PROCESSED_DIR / str(YEAR) / f"gencast_{YEAR}_wide.pkl"),
                    "years": f"{YEAR}:{YEAR}",
                }
            ],
            "constants": [
                {"col": "onset_thresh", "out": "onset_thresh"},
                {"col": "mok_date", "out": "mok_date"},
            ],
            "daily": [
                {"col": "predicted_prob", "out": "p_onset", "add_plus": True},
                {"col": "predicted_prob_clim_mok_date", "out": "p_onset_clim_mok_date", "add_plus": True},
                {"col": "predicted_prob_mok", "out": "p_onset_mok", "add_plus": True},
                {"col": "forecast_rain", "out": "rain_mean", "add_plus": False},
            ],
        },
        "aifs": {
            "max_day": 45,
            "sources": [
                {
                    "file": str(ONSET_PROCESSED_DIR / str(YEAR) / f"aifs_{YEAR}_wide.pkl"),
                    "years": f"{YEAR}:{YEAR}",
                    "prob_col": "predicted_prob_clim_mok_date",
                }
            ],
            "daily": [
                {"col": "predicted_prob", "out": "p_onset", "add_plus": True},
                {"col": "predicted_prob_mok", "out": "p_onset_mok", "add_plus": True},
                {"col": "predicted_prob_clim_mok_date", "out": "p_onset_clim_mok_date", "add_plus": True},
                {"col": "forecast_rain", "out": "rain_mean", "add_plus": False},
            ],
        },
    },
    "climatology": {
        "max_day": 28,
        "prob_col": "predicted_prob",
    },
    "output": {
        "out_dir": str(ONSET_PROCESSED_DIR / str(YEAR)),
    },
}

# ==============================================================================
# Onset Benchmarking directory structure
# ==============================================================================
APP_DIR = PROJECT_ROOT
BMARK_ROOT = APP_DIR / "apps" / "benchmarking"
ROMP_ROOT = BMARK_ROOT / "ROMP"
MOMP_PKG_DIR = ROMP_ROOT / "momp"  # Points to the folder containing driver.py
ROMP_DEMO_ET_DIR = DATA_DIR / "ROMP_OUT" / "et"
ROMP_DEMO_FIG_DIR = ROMP_DEMO_ET_DIR / "figure"
ROMP_DEMO_OUT_DIR = ROMP_DEMO_ET_DIR / "output"
ROMP_RUNTIME_ROOT = APP_DIR / ".romp_runtime"

# Expose environment variables for MOMP internal fallback mechanisms
os.environ["ROMP_ROOT"] = str(ROMP_ROOT.resolve())
os.environ["WORK_DIR"] = str(ROMP_ROOT.resolve())
os.environ["PKG_DIR"] = str(MOMP_PKG_DIR.resolve())
os.environ["MOMP_DIR"] = str(MOMP_PKG_DIR.resolve())

# ==============================================================================
# Spatial Data & Shapefile Directories
# ==============================================================================
STATION_DATA_PATH = DATA_DIR / "stations.csv"
STATION_CSV_PATH = EXTERNAL_DATA_DIR / "RF_Station_Grid_Format.csv"
DEFAULT_MASK_NC = EXTERNAL_DATA_DIR / "jjas_100mm_rainfall_mask_0p25.nc"
SHAPEFILE_DIR = EXTERNAL_DATA_DIR / "shapefile"
SHAPEFILE_PATH = SHAPEFILE_DIR / "eth_admbnda_admin3_csa_bofedb_2021.geojson"
DEFAULT_MASK_NC = DATA_DIR / "chirps_jjas_seasonal_mask_ethiopia_0p25.nc"
# ETH_GEOJSON_PATH = SHAPEFILE_DIR / "ETH_Zones.geojson"
ETH_GEOJSON_PATH = DATA_DIR / "ethiopia.geojson"
ZONE_WOREDA_SHP = SHAPEFILE_DIR / "ETH_Zones_Woreda.geojson"
WOREDA_SHAPEFILE_PATH = SHAPEFILE_DIR
SHAPEFILE = SHAPEFILE_PATH
STUDYAREA_SHAPEFILE = ZONE_WOREDA_SHP



# ==============================================================================
# CHIRPS & Dataset Local Directories & Derived Caches
# ==============================================================================
CHRIPS_BASE_DIR = DATA_DIR / "CHRIPS"
# Data input directories
CHRIPS_FINAL_DIR = CHRIPS_BASE_DIR / "chirps_v3_daily_final"
CHRIPS_PRELIM_DIR = CHRIPS_BASE_DIR / "chirps_v3_daily_prelim"
CHRIPS_FORECAST_DIR = CHRIPS_BASE_DIR / "chirps_gefs_v3_daily"
# Cache files
CHRIPS_WEIGHTS_CACHE = CHRIPS_BASE_DIR / "chirps_v3_woreda_weights.npz"
CHRIPS_AGG_CACHE = CHRIPS_BASE_DIR / "chirps_v3_woreda_daily.csv"
# Output files & plot directory
CHRIPS_OUT_DIR = CHRIPS_BASE_DIR / "woreda_onset_plots"
CHRIPS_FEATURES_CSV = CHRIPS_BASE_DIR / "woreda_onset_features.csv"
CHRIPS_MAP_PNG = CHRIPS_BASE_DIR / "woreda_onset_map.png"
# Remote data source roots
CHRIPS_FINAL_URL = "https://data.chc.ucsb.edu/products/CHIRPS/v3.0/daily/final/sat"
CHRIPS_PRELIM_URL = "https://data.chc.ucsb.edu/products/CHIRPS/v3.0/daily/prelim/sat"
CHRIPS_GEFS_URL = "https://data.chc.ucsb.edu/products/CHIRPS-GEFS/v3/daily/global"
# File matching regex patterns
CHRIPS_FINAL_RE = re.compile(r"chirps-v3.0.sat.(\d{4}).(\d{2}).(\d{2}).(?:nc|tif)")
CHRIPS_PRELIM_RE = re.compile(r"chirps-v3.0.prelim.(\d{4}).(\d{2}).(\d{2}).(?:nc|tif)")
CHRIPS_GEFS_RE = re.compile(r"c3g_(\d{4}).(\d{2}).(\d{2}).(?:nc|tif)")
CHRIPS_SUBDIR_RE = re.compile(r'href="(\d{2,4})/"')
# Grid resolution parameters
HALF = 0.025  # CHIRPS v3 pixel radius (0.05° resolution)

# ==============================================================================
# Woreda Onset Thresholds & Parameters
# ==============================================================================
DAYS_BACK = 55  # final-sat rolling cache window
PLOT_DAYS = 52  # recent days to render per plot
# Onset logic thresholds
WET_MM = 1.0        # per-day wet threshold (>=)
WET_TOTAL_MM = 20.0 # cumulative threshold across the 3-day wet spell
DRY_MM = 1.0        # per-day dry threshold (<)
WET_RUN = 3         # wet-day streak length
POST_DAYS = 21      # follow-up window after the wet streak
DRY_RUN = 7         # disqualifying dry-day streak length
FILTER_MODE = "features"
# "features" mode settings
FEATURES_LOOKBACK_DAYS = 28      # candidate day-0 span for onset categories
RECENT_ACCUM_LOOKBACK_DAYS = 21  # candidate day-0 span for 3-day accumulation
PARTIAL_POST_DAYS = 14           # no-dry-spell window for "partial onset"
STARTED_POST_DAYS = 7            # no-dry-spell window for "onset started"
FORECAST_PAD_DAYS = 16           # GEFS forecast horizon extension for 21d windows
BUCKET_ORDER = ["onset_met", "partial_onset", "onset_started", "recent_accum", "none"]
BUCKET_COLORS = {
    "onset_met": "#006837",
    "partial_onset": "#31a354",
    "onset_started": "#78c679",
    "recent_accum": "#c2e699",
    "none": "#e0e0e0",
}
BUCKET_LABELS = {
    "onset_met": "Onset met",
    "partial_onset": "Partial onset (no dry spell 14d)",
    "onset_started": "Onset started (no dry spell 7d)",
    "recent_accum": "3-day accumulation (last 3 wks)",
    "none": "None",
}
# MSWEP settings
MSWEP_DATA_DIR = DATA_DIR / "MSWEP"
MSWEP_CSV = DATA_DIR / "MSWEP" / "mswep_rtm" / "woreda" / "mswep_woreda_daily.csv"
MSWEP_MAP_PNG = DATA_DIR / "MSWEP" / "woreda_onset_map_mswep.png"
MSWEP_FEATURE = "mswep_onset_fcst"
MSWEP_WET_LOOKBACK_DAYS = 7
MSWEP_FCST_FWD_DAYS = 14
MSWEP_FEATURE_COLORS = {True: "#08519c", False: "#e0e0e0"}
MSWEP_FEATURE_LABEL = "MSWEP wet spell (last 7d) + no dry spell to +14d forecast"
# "window" mode & priority settings
WINDOW_BACK_DAYS = 12
WINDOW_FWD_DAYS = 12
SOURCE_PRIORITY = {"final": 0, "prelim": 1, "mswep": 2, "forecast": 3}
FEATURE_SPECS = [
    ("onset_7d", 0, 7),
    ("onset_14d", 7, 14),
    ("onset_21d", 14, 21),
]
# Display scale settings
LOG_SCALE = False
LOG_BASE = 2
LOG_FLOOR = 0

# ==============================================================================
# Google Drive & External Credentials
# ==============================================================================
NRT_DRIVE_FOLDER_ID = "1S4eAvK_7gXetc9kXjW91s4HxxkboItBt"
GOOGLE_APPLICATION_CREDENTIALS = os.environ.get(
    "GOOGLE_APPLICATION_CREDENTIALS", ".envs/service_account.json"
)
EARTHDATA_SECRET_KEYS = ("EARTHDATA_USERNAME", "EARTHDATA_PASSWORD")
EARTHDATA_COMMON_TYPO_KEYS = ("ARTHDATA_USERNAME", "ARTHDATA_PASSWORD")

# ==============================================================================
# General Datasets & AWS Configurations
# ==============================================================================
DATASET_FOLDERS: Dict[str, str] = {
    "CHIRPS": str(EXTERNAL_DATA_DIR / "CHIRPS_IMERG"),
    "ENACTS": str(EXTERNAL_DATA_DIR / "ENACTS"),
    "IMERG": str(EXTERNAL_DATA_DIR / "CHIRPS_IMERG"),
    "ERA5": "data/ERA5",
    "WRF": str(EXTERNAL_DATA_DIR / "WRF"),
}
RAIN_VAR_BY_KEY: Dict[str, str] = {
    "CHIRPS": "precip",
    "ENACTS": "precip",
    "IMERG": "precip",
    "ERA5": "precip",
}
RAIN_VAR = "precip"
FILE_GLOB = "*.nc"
CHUNKSIZE = 365
LINEWIDTH_ETH = 1.8
_NEEDS_TIME_FIX: set = {"IMERG", "ERA5", "MSWEP"}
STATION_KEY = "EMI Stations"
DATASET_COLORS: Dict[str, Dict[str, str]] = {
    "CHIRPS": {"rain": "#4A82B8", "wet": "#1E5C99", "onset": "#0D3A6B"},
    "ENACTS": {"rain": "#C2714A", "wet": "#8B4A2E", "onset": "#5C2A14"},
    "IMERG": {"rain": "#4A7A28", "wet": "#2D5016", "onset": "#1A3209"},
    "ERA5": {"rain": "#D4A017", "wet": "#9E7800", "onset": "#6B5000"},
    STATION_KEY: {"rain": "#9467BD", "wet": "#5A3A7A", "onset": "#3A2550"},
}
_ELEV_BANDS = [
    (0, 1000, "green", "#22C55E", " < 1 000 m · Lowland"),
    (1000, 2000, "orange", "#F59E0B", "1 000–2 000 m · Mid-altitude"),
    (2000, 3500, "red", "#DC2626", "2 000–3 500 m · Highland"),
    (3500, 9999, "darkred", "#7F1D1D", " > 3 500 m · Alpine"),
]
REGION_SCORES: Dict[str, float] = {
    "Tigray": 0.70,
    "Afar": 0.61,
    "Amhara": 0.83,
    "Oromia": 0.87,
    "Somali": 0.55,
    "Benishangul-Gumuz": 0.75,
    "South West Ethiopia Peoples' Region": 0.74,
    "Gambela": 0.68,
    "Harari": 0.72,
    "Addis Ababa": 0.80,
    "Dire Dawa": 0.76,
}
MJO_ETH_IMPACT: Dict[int, Tuple[str, str, str]] = {
    1: ("Neutral–Enhancing", "#D97706", "Convection developing over Indian Ocean; modest enhancement possible for south/east Ethiopia."),
    2: ("Enhanced ↑↑", "#15803D", "Active MJO over Indian Ocean. Strong moisture flux into East Africa. High onset likelihood."),
    3: ("Enhanced ↑", "#22C55E", "Active convection approaching Maritime Continent. Positive rainfall signal persists."),
    4: ("Neutral", "#6D28D9", "MJO over Maritime Continent. Neutral impact; monitor for local forcing."),
    5: ("Suppressed ↓", "#F59E0B", "MJO over western Pacific. Suppressed phase begins; drying tendency for east."),
    6: ("Suppressed ↓↓", "#DC2626", "Strong suppressed signal over Indian Ocean. Reduced onset probability."),
    7: ("Neutral–Suppressing", "#C0392B", "MJO over eastern Pacific. Residual suppression possible; onset delays likely."),
    8: ("Transitioning ↑", "#0D7377", "MJO re-entering western Indian Ocean. Onset window may open soon."),
}
# AWS pipeline parameters
DEFAULT_DATA_DIR = "data/gauge/AWS"
DEFAULT_OUTPUT_DIR = "data/gauge/AWS_outputs"
DEFAULT_ACCUM_WINDOW = 3
DEFAULT_ACCUM_THRESH = 20.0
DEFAULT_DRY_SPELL_DAYS = 7
DEFAULT_ANOMALY_SD = 1.5
PRECIP_KEYWORDS = ["precipitation", "precip", "rain", "sum", "aprecp", "aprec", "prcp", "rf", "rr", "rainfall"]
PRECIP_FALLBACK = ["p"]
DATE_KEYS = ["date", "dat", "dt", "datum", "timestamp", "time"]
YEAR_KEYS = ["year", "yr", "yyyy"]
MONTH_KEYS = ["month", "mon", "mm"]
DAY_KEYS = ["day", "dd"]
CHART_COLORS = [
    "#E41A1C", "#377EB8", "#4DAF4A", "#FF7F00", "#984EA3",
    "#F781BF", "#A65628", "#999999", "#8DD3C7", "#FB8072",
]
ALLOWED_EXTENSIONS = {".csv", ".parquet", ".xlsx"}
PREFERRED_STATION_TARGET = "MOTTA"
MAX_WORKERS = 4
# WRF Forecast paths
WRF_DATA_DIR = DATA_DIR / "WRF"
WRF_SHAPEFILE_PATH = SHAPEFILE_DIR / "ETH_Zones.geojson"
WRF_LOGO_PATH = APP_DIR / "figs" / "emi_logo.jpg"
WRF_MASK_PATH = WRF_DATA_DIR / "EthiopiaMaskP1deg.nc"

# ==============================================================================
# Benchmarking Catalogs 
# ==============================================================================
BENCHMARK_MODEL_CATALOG: Dict[str, Dict[str, Any]] = {
    "AIFS": {
        "label": "Deterministic · AIFS",
        "dir": EXTERNAL_DATA_DIR / "AIFS",
        "model_name": "AIFS",
        "probabilistic": False,
        "category": "Deterministic",
        "slow": False,
    },
    "FuXi": {
        "label": "Deterministic · FuXi",
        "dir": EXTERNAL_DATA_DIR / "fuxi",
        "model_name": "FuXi",
        "probabilistic": False,
        "category": "Deterministic",
        "slow": False,
    },
    "GraphCast": {
        "label": "Deterministic · GraphCast",
        "dir": EXTERNAL_DATA_DIR / "graphcast",
        "model_name": "GraphCast",
        "probabilistic": False,
        "category": "Deterministic",
        "slow": False,
    },
    "AIFS_ENS": {
        "label": "Probabilistic · AIFS ENS",
        "dir": EXTERNAL_DATA_DIR / "AIFS_ENS",
        "model_name": "AIFS_ENS",
        "probabilistic": True,
        "category": "Probabilistic",
        "slow": True,
    },
    "GenCast": {
        "label": "Probabilistic · GenCast",
        "dir": EXTERNAL_DATA_DIR / "gencast",
        "model_name": "GenCast",
        "probabilistic": True,
        "category": "Probabilistic",
        "slow": True,
    },
}
BENCHMARK_OBS_CATALOG: Dict[str, Dict[str, Any]] = {
    "ENACTS": {
        "label": "ENACTS gauge-gridded rainfall",
        "dir": EXTERNAL_DATA_DIR / "ENACTS",
        "file_pattern": "{}_0p25.nc",
        "var": "precip",
        "category": "dashboard",
    },
    "CHIRPS": {
        "label": "CHIRPS rainfall",
        "dir": EXTERNAL_DATA_DIR / "CHIRPS_IMERG",
        "file_pattern": "{}.nc",
        "var": "precip",
        "category": "dashboard",
    },
    "IMERG": {
        "label": "IMERG rainfall",
        "dir": EXTERNAL_DATA_DIR / "CHIRPS_IMERG",
        "file_pattern": "{}.nc",
        "var": "precip",
        "category": "dashboard",
    },
}

# ==============================================================================
# ROMP / MOMP WORKFLOW CONFIGURATION
# ==============================================================================
project_name = "deterministic benchmarking with ENACTS data"
# Direct absolute paths for ROMP execution
work_dir = str(ROMP_ROOT.resolve())
pkg_dir = str(MOMP_PKG_DIR.resolve())
# Internal data layout & model list
layout = ("model", "verification_window")
model_list = ("AIFS", "FuXi", "Graphcast")
# Observation settings
obs = "ENACTS"
obs_dir = str((EXTERNAL_DATA_DIR / "ENACTS").resolve())
obs_file_pattern = ("{}_0p25.nc",)
obs_var = "precip"
obs_unit_cvt = None
# Reference Model (Benchmark) Settings
ref_model = "climatology"
ref_model_dir = str((EXTERNAL_DATA_DIR / "ENACTS").resolve())
ref_model_file_pattern = "{}_0p25.nc"
ref_model_var = "precip"
ref_model_unit_cvt = None
# Forecast Model Settings
model_dir_list = (
    str((EXTERNAL_DATA_DIR / "AIFS").resolve()),
    str((EXTERNAL_DATA_DIR / "fuxi").resolve()),
    str((EXTERNAL_DATA_DIR / "graphcast").resolve()),
)
model_var_list = ("tp", "tp", "tp")
unit_cvt_list = (None, None, None)
file_pattern_list = ("{}.nc", "{}.nc", "{}.nc")
# Spatial & Mask Definitions
region = "Ethiopia"
nc_mask = str((EXTERNAL_DATA_DIR / "jjas_100mm_rainfall_mask_0p25.nc").resolve())
shpfile_dir = str(SHAPEFILE_DIR.resolve())
polygon = False
# Physics & Onset Criteria Logic
wet_init = 1
wet_threshold = 20
wet_spell = 3
dry_threshold = 1
dry_spell = 7
dry_extent = 0
thresh_file = None
thresh_var = None
onset_percentage_threshold = 0.5
# Temporal Settings
start_date = (2019, 5, 1)
end_date = (2022, 7, 31)
start_year_clim = 2015
end_year_clim = 2022
init_days = (0, 3)
date_filter_year = 2024
# Evaluation & Verification Parameters
verification_window_list = ((1, 15),)
tolerance_days_list = (3,)
max_forecast_day = 15
day_bins = ((1, 5), (6, 10), (11, 15))
# Metrics Selection
FAR = True
MAE = True
MR = True
probabilistic = False
members = "All"
BS = True
RPS = True
AUC = True
Reliability = True
skill_score = True
# Output & Plotting Settings
dir_out = str(ROMP_DEMO_OUT_DIR.resolve())
dir_fig = str(ROMP_DEMO_FIG_DIR.resolve())
save_fig = True
save_nc_spatial_far_mr_mae = True
save_csv_score = True
save_nc_climatology = True
plot_spatial_far_mr_mae = True
plot_heatmap_bss_auc = True
plot_reliability = True
plot_climatology_onset = True
plot_panel_heatmap_error = True
plot_panel_heatmap_skill = True
plot_bar_bss_rpss_auc = True
show_plot = False
show_panel = False
parallel = True
debug = False
# Reserved / Legacy compatibility
mok = (5, 15)
years = None
years_clim = None
fallback_date = None

# ==============================================================================
# Helper Functions
# ==============================================================================
@st.cache_data
def get_woreda_zone_map() -> pd.DataFrame:
    """Load shapefile to get the mapping of woreda (pcode) to zone."""
    if not ZONE_WOREDA_SHP.exists():
        return pd.DataFrame(columns=["adm3_pcode", "adm2_name"])
    try:
        gdf = gpd.read_file(ZONE_WOREDA_SHP)
    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to read shapefile {ZONE_WOREDA_SHP}: {e}")
        return pd.DataFrame(columns=["adm3_pcode", "adm2_name"])
    return gdf[["adm3_pcode", "adm2_name"]].drop_duplicates()

def _years_from_netcdf_dir(directory: Path) -> List[int]:
    years = set()
    if not directory.exists():
        return []
    for path in directory.glob("*.nc"):
        match = re.search(r"(19|20)\d{2}", path.stem)
        if match:
            years.add(int(match.group(0)))
    return sorted(years)

@st.cache_data(show_spinner=False)
def get_benchmark_model_sources() -> Dict[str, Dict[str, Any]]:
    models = {}
    for key, meta in BENCHMARK_MODEL_CATALOG.items():
        years = _years_from_netcdf_dir(meta["dir"])
        if not years:
            continue
        models[key] = {
            **meta,
            "dir": str(meta["dir"].resolve()),
            "years": years,
            "year_min": years[0],
            "year_max": years[-1],
        }
    return models

def _benchmark_climatology_years(obs_years: List[int]) -> List[int]:
    target_years = set(range(2015, 2023))
    return [year for year in obs_years if year in target_years]

@st.cache_data(show_spinner=False)
def get_benchmark_reference_sources() -> Dict[str, Dict[str, Any]]:
    sources = {}
    for key, meta in BENCHMARK_OBS_CATALOG.items():
        years = _years_from_netcdf_dir(meta["dir"])
        if not years:
            continue
        clim_years = _benchmark_climatology_years(years)
        if clim_years != list(range(2015, 2023)):
            continue
        sources[key] = {
            **meta,
            "dir": str(meta["dir"].resolve()),
            "years": years,
            "year_min": years[0],
            "year_max": years[-1],
        }
    return sources

def current_utc_label() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

def pretty_benchmark_path(path_str: str) -> str:
    path = Path(path_str).expanduser().resolve()
    for base, label in [
        (BMARK_ROOT.resolve(), "apps/benchmarking"),
        (ROMP_ROOT.resolve(), "ROMP"),
        (PROJECT_ROOT.resolve(), "etagrometaix"),
    ]:
        try:
            return f"{label}/{path.relative_to(base).as_posix()}"
        except ValueError:
            continue
    return path.name or str(path)

def _read_toml_mapping(path: Path) -> Dict[str, Any]:
    try:
        with path.open("rb") as fh:
            data = tomllib.load(fh)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}

def earthdata_credentials() -> Dict[str, str]:
    home_secret_path = Path.home() / ".streamlit" / "secrets.toml"
    local_secret_path = APP_DIR / ".streamlit" / "secrets.toml"
    sources = [
        ("home", home_secret_path, _read_toml_mapping(home_secret_path)),
        ("local", local_secret_path, _read_toml_mapping(local_secret_path)),
    ]
    try:
        st_secret_map = {k: st.secrets.get(k, "") for k in EARTHDATA_SECRET_KEYS}
        typo_map = {k: st.secrets.get(k, "") for k in EARTHDATA_COMMON_TYPO_KEYS}
        sources.append(("streamlit", None, {**st_secret_map, **typo_map}))
    except Exception:
        pass
    for source_name, source_path, mapping in sources:
        username = str(mapping.get("EARTHDATA_USERNAME", "")).strip()
        password = str(mapping.get("EARTHDATA_PASSWORD", "")).strip()
        typo_keys = [k for k in EARTHDATA_COMMON_TYPO_KEYS if str(mapping.get(k, "")).strip()]
        if username and password:
            return {
                "ok": "true",
                "username": username,
                "password": password,
                "source": source_name,
                "source_path": str(source_path) if source_path else "Streamlit runtime secrets",
                "message": f"Earthdata credentials loaded from {source_path if source_path else 'Streamlit runtime secrets'}.",
            }
        if typo_keys:
            return {
                "ok": "false",
                "username": "",
                "password": "",
                "source": source_name,
                "source_path": str(source_path) if source_path else "Streamlit runtime secrets",
                "message": (
                    f"Found misnamed Earthdata keys ({', '.join(typo_keys)}) in "
                    f"{source_path if source_path else 'Streamlit runtime secrets'}. "
                    f"Rename them to EARTHDATA_USERNAME and EARTHDATA_PASSWORD."
                ),
            }
    return {
        "ok": "false",
        "username": "",
        "password": "",
        "source": "",
        "source_path": str(home_secret_path),
        "message": (
            "Earthdata credentials are missing. Add EARTHDATA_USERNAME and "
            f"EARTHDATA_PASSWORD to {home_secret_path} "
            "or expose them through Streamlit runtime secrets. Register free at urs.earthdata.nasa.gov"
        ),
    }

def earthdata_auth_status() -> Dict[str, str]:
    creds = earthdata_credentials()
    return {"ok": creds["ok"], "message": creds["message"]}

def setup_earthdata_auth() -> bool:
    creds = earthdata_credentials()
    if creds["ok"] != "true":
        return False
    try:
        netrc_path = Path.home() / ".netrc"
        netrc_content = (
            "machine urs.earthdata.nasa.gov\n"
            f"    login {creds['username']}\n"
            f"    password {creds['password']}\n"
        )
        netrc_path.write_text(netrc_content)
        netrc_path.chmod(0o600)
        os.environ["EARTHDATA_USERNAME"] = creds["username"]
        os.environ["EARTHDATA_PASSWORD"] = creds["password"]
        return True
    except Exception:
        return False

@st.cache_resource
def load_region_bounds_from_shapefile() -> Dict[str, Dict]:
    if not SHAPEFILE_PATH.exists():
        return {}
    try:
        gdf = gpd.read_file(SHAPEFILE_PATH)
    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to read shapefile {SHAPEFILE_PATH}: {e}")
        return {}
    if "ADM1_EN" not in gdf.columns:
        return {}
    # Detailed admin boundary files (this one is admin-3 / woreda level) commonly
    # ship with a handful of unclosed or self-intersecting rings. dissolve() unions
    # geometries internally and raises "IllegalArgumentException: Points of
    # LinearRing do not form a closed linestring" (a GEOS/JTS validity error) on
    # the first bad ring it hits. Repair invalid geometries before dissolving, and
    # degrade to an empty preset list rather than crashing the app if that fails.
    invalid = ~gdf.geometry.is_valid
    if invalid.any():
        logging.getLogger(__name__).warning(
            f"Repairing {int(invalid.sum())} invalid geometries in {SHAPEFILE_PATH} "
            "before dissolve."
        )
        gdf.loc[invalid, "geometry"] = gdf.loc[invalid, "geometry"].buffer(0)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
    if gdf.empty:
        return {}
    try:
        regions = gdf.dissolve(by="ADM1_EN").reset_index()
    except Exception as e:
        logging.getLogger(__name__).error(
            f"dissolve() failed on {SHAPEFILE_PATH} even after geometry repair: {e}"
        )
        return {}
    bounds = {}
    for _, row in regions.iterrows():
        name = row["ADM1_EN"]
        minx, miny, maxx, maxy = row.geometry.bounds
        bounds[name] = {"lat": [miny, maxy], "lon": [minx, maxx]}
    return bounds

def get_region_presets() -> List[Dict[str, Any]]:
    bounds = load_region_bounds_from_shapefile()
    if not bounds:
        return []
    presets = []
    for name, bbox in bounds.items():
        lat_center = (bbox["lat"][0] + bbox["lat"][1]) / 2
        lon_center = (bbox["lon"][0] + bbox["lon"][1]) / 2
        presets.append({
            "name": name,
            "lat": lat_center,
            "lon": lon_center,
            "score": REGION_SCORES.get(name, 0.65),
        })
    return sorted(presets, key=lambda x: x["name"])

def get_operational_season_context(now: datetime | None = None) -> Dict[str, Any]:
    now = now or datetime.utcnow()
    month = now.month
    year = now.year
    if month in [2, 3, 4, 5]:
        return {
            "label": f"Belg {year}",
            "season_code": "MAM",
            "months": [3, 4, 5],
            "brief": "Belg onset monitoring",
            "focus": "Short-rains onset and early season persistence",
        }
    if month in [6, 7, 8, 9]:
        return {
            "label": f"Kiremt {year}",
            "season_code": "JJAS",
            "months": [6, 7, 8, 9],
            "brief": "Kiremt main-rains monitoring",
            "focus": "Main-season rainfall and onset support",
        }
    return {
        "label": f"Transition {year}",
        "season_code": "OND",
        "months": [10, 11, 12],
        "brief": "Transition-season monitoring",
        "focus": "Large-scale drivers and pre-onset readiness",
    }

@st.cache_data(show_spinner=False)
def fetch_mjo_status() -> Dict[str, Any] | None:
    url = "https://www.bom.gov.au/clim_data/IDCKGEM000/rmm.74toRealtime.txt"
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        rows = []
        for raw_line in resp.text.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or not line[0].isdigit():
                continue
            parts = [p for p in re.split(r"[,\s]+", line) if p]
            numeric = []
            for token in parts:
                try:
                    numeric.append(float(token))
                except ValueError:
                    continue
            if len(numeric) < 7:
                continue
            try:
                year, month, day = map(int, numeric[:3])
                rmm1, rmm2 = float(numeric[3]), float(numeric[4])
                phase = int(float(numeric[5]))
                amplitude = float(numeric[6])
            except ValueError:
                continue
            rows.append({
                "date": pd.Timestamp(year=year, month=month, day=day),
                "rmm1": rmm1,
                "rmm2": rmm2,
                "phase": phase,
                "amplitude": amplitude,
            })
        if not rows:
            return None
        latest = rows[-1]
        age_days = (pd.Timestamp.utcnow().tz_localize(None).normalize() - latest["date"].normalize()).days
        latest["strength"] = "Active" if latest["amplitude"] >= 1.0 else "Weak"
        latest["age_days"] = age_days
        latest["stale"] = age_days > 3
        latest["source"] = "BoM RMM realtime"
        latest["source_url"] = url
        return latest
    except Exception:
        return None