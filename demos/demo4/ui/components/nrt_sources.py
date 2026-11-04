# ui/components/nrt_sources.py
from __future__ import annotations

from typing import Dict

NRT_SOURCES: Dict[str, Dict] = {
    # Observed Rainfall
    "CPC RFE · 7-day Observed": {
        "url": "https://www.cpc.ncep.noaa.gov/products/international/africa_rfe/africa_rfe_7day_ea_obs.gif",
        "referer": "https://www.cpc.ncep.noaa.gov/products/international/africa_rfe/",
        "desc": "7-day accumulated rainfall (mm) from NOAA/CPC Rain Estimate algorithm.",
        "update": "Daily ~06Z", "region": "East Africa", "tag": "Rainfall",
    },
    "CPC ARC2 · 7-day Anomaly": {
        "url": "https://www.cpc.ncep.noaa.gov/products/international/africa_arc/africa_arc_7day_ea_anom.gif",
        "referer": "https://www.cpc.ncep.noaa.gov/products/international/africa_arc/",
        "desc": "7-day anomaly (% of clim.) from Africa Rainfall Climatology v2.",
        "update": "Daily ~06Z", "region": "East Africa", "tag": "Anomaly",
    },
    # MJO
    "MJO · RMM Phase Diagram": {
        "url": "https://www.cpc.ncep.noaa.gov/products/precip/CWlink/MJO/ENSO/rmm.gif",
        "referer": "https://www.cpc.ncep.noaa.gov/products/precip/CWlink/MJO/mjo.shtml",
        "desc": "Real-time MJO phase-space (RMM). Phases 2-3 enhance East Africa rainfall.",
        "update": "Daily", "region": "Global / East Africa", "tag": "MJO",
    },
    "MJO · GEFS Ensemble Forecast": {
        "url": "https://www.cpc.ncep.noaa.gov/products/precip/mjo/img/GEFS_BC.png",
        "referer": "https://www.cpc.ncep.noaa.gov/products/precip/mjo/",
        "desc": "GEFS-based MJO ensemble forecast (15-day).",
        "update": "Daily", "region": "Global", "tag": "MJO",
    },
    # OLR / ITCZ
    "OLR Anomaly · CDAS": {
        "url": "https://www.cpc.ncep.noaa.gov/products/precip/CWlink/MJO/ENSO/olr.anom.gif",
        "referer": "https://www.cpc.ncep.noaa.gov/products/precip/CWlink/MJO/mjo.shtml",
        "desc": "Outgoing Longwave Radiation anomaly (W/m²). Negative = enhanced convection.",
        "update": "Daily", "region": "Global", "tag": "OLR",
    },
    "ITCZ · Current Position": {
        "url": "https://www.cpc.ncep.noaa.gov/products/international/itf/itcz.jpg",
        "referer": "https://www.cpc.ncep.noaa.gov/products/international/itf/",
        "desc": "ITCZ latitude anomaly. Northward position enhances rainfall over Ethiopia.",
        "update": "Daily", "region": "Africa", "tag": "Synoptic",
    },
    # CORe 200 hPa
    "CORe · 200 hPa Wind (Avg)": {
        "url": "https://ftp.cpc.ncep.noaa.gov/International/CORe/Africa/fig_dir/avg_wind_200_avg.png",
        "referer": "https://ftp.cpc.ncep.noaa.gov/International/CORe/Africa/",
        "desc": "200 hPa wind — Tropical Easterly Jet (TEJ). Check magnitude and positioning.",
        "update": "Daily", "region": "Africa", "tag": "Wind-200",
    },
    "CORe · 200 hPa Wind Anomaly": {
        "url": "https://ftp.cpc.ncep.noaa.gov/International/CORe/Africa/fig_dir/avg_wind_200_anom.png",
        "referer": "https://ftp.cpc.ncep.noaa.gov/International/CORe/Africa/",
        "desc": "200 hPa wind anomaly. Southward TEJ shift → Kiremt break; northward → rainfall enhancement.",
        "update": "Daily", "region": "Africa", "tag": "Wind-200",
    },
    # CORe 500 hPa
    "CORe · 500 hPa Wind (Avg)": {
        "url": "https://ftp.cpc.ncep.noaa.gov/International/CORe/Africa/fig_dir/avg_wind_500_avg.png",
        "referer": "https://ftp.cpc.ncep.noaa.gov/International/CORe/Africa/",
        "desc": "500 hPa wind — mid-tropospheric monsoon trough and AEJ positioning.",
        "update": "Daily", "region": "Africa", "tag": "Wind-500",
    },
    "CORe · 500 hPa Wind Anomaly": {
        "url": "https://ftp.cpc.ncep.noaa.gov/International/CORe/Africa/fig_dir/avg_wind_500_anom.png",
        "referer": "https://ftp.cpc.ncep.noaa.gov/International/CORe/Africa/",
        "desc": "500 hPa wind anomaly. Trough deepening into East Africa favours convection.",
        "update": "Daily", "region": "Africa", "tag": "Wind-500",
    },
    # CORe 700 hPa
    "CORe · 700 hPa Wind (Avg)": {
        "url": "https://ftp.cpc.ncep.noaa.gov/International/CORe/Africa/fig_dir/avg_wind_700_avg.png",
        "referer": "https://ftp.cpc.ncep.noaa.gov/International/CORe/Africa/",
        "desc": "700 hPa wind — important over Ethiopian highlands for moisture convergence.",
        "update": "Daily", "region": "Africa", "tag": "Wind-700",
    },
    "CORe · 700 hPa Wind Anomaly": {
        "url": "https://ftp.cpc.ncep.noaa.gov/International/CORe/Africa/fig_dir/avg_wind_700_anom.png",
        "referer": "https://ftp.cpc.ncep.noaa.gov/International/CORe/Africa/",
        "desc": "700 hPa wind anomaly over the highlands.",
        "update": "Daily", "region": "Africa", "tag": "Wind-700",
    },
    # CORe 850 hPa
    "CORe · 850 hPa Wind (Avg)": {
        "url": "https://ftp.cpc.ncep.noaa.gov/International/CORe/Africa/fig_dir/avg_wind_850_avg.png",
        "referer": "https://ftp.cpc.ncep.noaa.gov/International/CORe/Africa/",
        "desc": "850 hPa wind vectors — Somali Jet and low-level moisture transport.",
        "update": "Daily", "region": "Africa", "tag": "Wind-850",
    },
    "CORe · 850 hPa Wind Anomaly": {
        "url": "https://ftp.cpc.ncep.noaa.gov/International/CORe/Africa/fig_dir/avg_wind_850_anom.png",
        "referer": "https://ftp.cpc.ncep.noaa.gov/International/CORe/Africa/",
        "desc": "850 hPa wind anomaly. Enhanced westerlies signal greater Kiremt moisture flux.",
        "update": "Daily", "region": "Africa", "tag": "Wind-850",
    },
    # Divergence
    "CORe · Divergence 200 hPa (Avg)": {
        "url": "https://ftp.cpc.ncep.noaa.gov/International/CORe/Africa/fig_dir/avg_div_200_avg.png",
        "referer": "https://ftp.cpc.ncep.noaa.gov/International/CORe/Africa/",
        "desc": "Upper-level divergence (200 hPa). Positive = ascending air → rainfall support.",
        "update": "Daily", "region": "Africa", "tag": "Divergence",
    },
    "CORe · Convergence 925 hPa (Avg)": {
        "url": "https://ftp.cpc.ncep.noaa.gov/International/CORe/Africa/fig_dir/avg_div_925_avg.png",
        "referer": "https://ftp.cpc.ncep.noaa.gov/International/CORe/Africa/",
        "desc": "Low-level convergence (925 hPa). Negative divergence = convergence → uplift.",
        "update": "Daily", "region": "Africa", "tag": "Divergence",
    },
    # GEFS Rainfall Forecasts
    "GEFS · Day 3 Rainfall": {
        "url": "https://www.cpc.ncep.noaa.gov/products/international/cpci/data/00/gfs.t00z.totp.day3.eafrica.gif",
        "referer": "https://www.cpc.ncep.noaa.gov/products/international/cpci/",
        "desc": "GFS total precipitation day 3 forecast over East Africa.",
        "update": "Daily", "region": "East Africa", "tag": "Forecast-Rain",
    },
    "GEFS · Week 1 Rainfall (>25mm)": {
        "url": "https://www.cpc.ncep.noaa.gov/products/international/cpci/data/00/gefs.t00z.wk1_precip_tot_gt_25mm.eafrica.gif",
        "referer": "https://www.cpc.ncep.noaa.gov/products/international/cpci/",
        "desc": "GEFS week-1 probability of accumulated rainfall >25 mm.",
        "update": "Daily", "region": "East Africa", "tag": "Forecast-Rain",
    },
    "GEFS · Week 2 Rainfall (>25mm)": {
        "url": "https://www.cpc.ncep.noaa.gov/products/international/cpci/data/00/gefs.t00z.wk2_precip_tot_gt_25mm.eafrica.gif",
        "referer": "https://www.cpc.ncep.noaa.gov/products/international/cpci/",
        "desc": "GEFS week-2 probability of accumulated rainfall >25 mm.",
        "update": "Daily", "region": "East Africa", "tag": "Forecast-Rain",
    },
    # GEFS Wind Forecasts
    "GEFS · 500 hPa Wind (Wk1)": {
        "url": "https://www.cpc.ncep.noaa.gov/products/international/cpci/data/00/gdas.t00z.500mb_wind.f006.eafrica.gif",
        "referer": "https://www.cpc.ncep.noaa.gov/products/international/cpci/",
        "desc": "GEFS 500 hPa wind forecast — monsoon trough positioning.",
        "update": "Daily", "region": "East Africa", "tag": "Forecast-Wind",
    },
    "GEFS · 700 hPa Anom (Wk1)": {
        "url": "https://www.cpc.ncep.noaa.gov/products/international/gefs/gefs_week1_af_700wind_anom.gif",
        "referer": "https://www.cpc.ncep.noaa.gov/products/international/gefs/",
        "desc": "GEFS week-1 700 hPa wind anomaly — mid-level moisture transport.",
        "update": "Daily", "region": "Africa", "tag": "Forecast-Wind",
    },
    "GEFS · 700 hPa Anom (Wk2)": {
        "url": "https://www.cpc.ncep.noaa.gov/products/international/gefs/gefs_week2_af_700wind_anom.gif",
        "referer": "https://www.cpc.ncep.noaa.gov/products/international/gefs/",
        "desc": "GEFS week-2 700 hPa wind anomaly.",
        "update": "Daily", "region": "Africa", "tag": "Forecast-Wind",
    },
    "GEFS · 200 hPa Anom (Wk1)": {
        "url": "https://www.cpc.ncep.noaa.gov/products/international/gefs/gefs_week1_af_200wind_anom.gif",
        "referer": "https://www.cpc.ncep.noaa.gov/products/international/gefs/",
        "desc": "GEFS week-1 200 hPa wind anomaly — TEJ strength and positioning.",
        "update": "Daily", "region": "Africa", "tag": "Forecast-Wind",
    },
    "GEFS · 200 hPa Anom (Wk2)": {
        "url": "https://www.cpc.ncep.noaa.gov/products/international/gefs/gefs_week2_af_200wind_anom.gif",
        "referer": "https://www.cpc.ncep.noaa.gov/products/international/gefs/",
        "desc": "GEFS week-2 200 hPa wind anomaly.",
        "update": "Daily", "region": "Africa", "tag": "Forecast-Wind",
    },
    # GEFS Divergence
    "GEFS · 200 hPa Divergence Anom (Wk1)": {
        "url": "https://www.cpc.ncep.noaa.gov/products/international/gefs/gefs_week1_af_200divg_anom.gif",
        "referer": "https://www.cpc.ncep.noaa.gov/products/international/gefs/",
        "desc": "GEFS week-1 upper-level divergence anomaly. Positive anomaly supports deep convection.",
        "update": "Daily", "region": "Africa", "tag": "Forecast-Div",
    },
    "GEFS · 200 hPa Divergence Anom (Wk2)": {
        "url": "https://www.cpc.ncep.noaa.gov/products/international/gefs/gefs_week2_af_200divg_anom.gif",
        "referer": "https://www.cpc.ncep.noaa.gov/products/international/gefs/",
        "desc": "GEFS week-2 upper-level divergence anomaly.",
        "update": "Daily", "region": "Africa", "tag": "Forecast-Div",
    },
    "GEFS · 850 hPa Divergence Anom (Wk1)": {
        "url": "https://www.cpc.ncep.noaa.gov/products/international/gefs/gefs_week1_af_850divg_anom.gif",
        "referer": "https://www.cpc.ncep.noaa.gov/products/international/gefs/",
        "desc": "GEFS week-1 low-level divergence anomaly. Negative = convergence → uplift.",
        "update": "Daily", "region": "Africa", "tag": "Forecast-Div",
    },
    "GEFS · 850 hPa Divergence Anom (Wk2)": {
        "url": "https://www.cpc.ncep.noaa.gov/products/international/gefs/gefs_week2_af_850divg_anom.gif",
        "referer": "https://www.cpc.ncep.noaa.gov/products/international/gefs/",
        "desc": "GEFS week-2 low-level divergence anomaly.",
        "update": "Daily", "region": "Africa", "tag": "Forecast-Div",
    },
    # CPC Expert Outlooks
    "CPC · Week-1 Expert Outlook": {
        "url": "https://www.cpc.ncep.noaa.gov/products/international/africa/expert/week1.jpg",
        "referer": "https://www.cpc.ncep.noaa.gov/products/international/africa/",
        "desc": "CPC expert forecast week-1 total rainfall.",
        "update": "Weekly", "region": "Africa", "tag": "Forecast-Rain",
    },
    "CPC · Week-2 Expert Outlook": {
        "url": "https://www.cpc.ncep.noaa.gov/products/international/africa/expert/week2.jpg",
        "referer": "https://www.cpc.ncep.noaa.gov/products/international/africa/",
        "desc": "CPC expert forecast week-2 total rainfall.",
        "update": "Weekly", "region": "Africa", "tag": "Forecast-Rain",
    },
    "CPC · Week-3 Expert Outlook": {
        "url": "https://www.cpc.ncep.noaa.gov/products/international/africa/expert/week3.jpg",
        "referer": "https://www.cpc.ncep.noaa.gov/products/international/africa/",
        "desc": "CPC expert forecast week-3 total rainfall.",
        "update": "Weekly", "region": "Africa", "tag": "Forecast-Rain",
    },
}

_TAG_COLORS: Dict[str, str] = {
    "Rainfall": "#1A6B9A", "Anomaly": "#D97706", "MJO": "#6D28D9",
    "OLR": "#0D7377", "Synoptic": "#2C3E50",
    "Wind-200": "#7C3AED", "Wind-500": "#0D7377", "Wind-700": "#1D4ED8",
    "Wind-850": "#1E5C99",
    "Divergence": "#9467BD",
    "Forecast-Rain": "#15803D", "Forecast-Wind": "#6D28D9", "Forecast-Div": "#9467BD",
}