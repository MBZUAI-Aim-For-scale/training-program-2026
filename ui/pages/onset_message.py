# ui/pages/forecast_dispatch.py
from __future__ import annotations

import os
import json
import requests
import streamlit as st
import pandas as pd
from datetime import datetime
from typing import Optional, List, Dict
from dotenv import load_dotenv

from ai_weather.background_task import BackgroundTask

load_dotenv()

# ==============================================================================
# API Configuration (Passwordless via Environment Variables)
# ==============================================================================
API_BASE_URL = os.environ.get("ONSET_API_URL", "http://localhost:8000")
SHARED_API_KEY = os.environ.get("ONSET_API_KEY", "")


def get_jwt_token(api_key: str) -> str:
    response = requests.post(
        f"{API_BASE_URL}/token", headers={"X-API-Key": api_key}, timeout=10,
    )
    response.raise_for_status()
    return response.json()["access_token"]


def get_districts(token: str) -> List[str]:
    response = requests.get(
        f"{API_BASE_URL}/districts",
        headers={"Authorization": f"Bearer {token}"}, timeout=10,
    )
    response.raise_for_status()
    districts = response.json()["districts"]
    if not districts:
        raise ValueError("No districts available from API. Please check backend connection.")
    return districts


def get_forecast(
    token: str, district: Optional[str], all_districts: bool,
    thresholds: List[float], season: str,
) -> List[Dict]:
    response = requests.post(
        f"{API_BASE_URL}/forecast",
        headers={"Authorization": f"Bearer {token}"},
        json={"district": district, "all_districts": all_districts,
              "thresholds": thresholds, "season": season},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def _connect(token):
    """Network work only; workers never access Streamlit or session state."""
    token = token or get_jwt_token(SHARED_API_KEY)
    return {"token": token, "districts": get_districts(token)}


def _fetch_forecasts(token, districts, all_districts, thresholds, season):
    # Keep API requests ordered, as before: the backend may share model resources.
    results, errors = [], []
    for district in ([None] if all_districts else districts):
        try:
            results.extend(get_forecast(token, district, all_districts, thresholds, season))
        except Exception as exc:
            errors.append(f"{district or 'All districts'}: {type(exc).__name__}: {exc}")
    return {"results": results, "errors": errors}


@st.fragment(run_every=1)
def render_message_job(key, label):
    job = st.session_state.get(key)
    if job is None:
        return
    snapshot = job.snapshot()
    if not snapshot["done"]:
        st.info(f"{label} Task running in the background… You can use other menus.")
        return
    st.session_state.pop(key)
    if key == "onset_connection_job":
        if snapshot["error"]:
            st.session_state["onset_connection_error"] = snapshot["error"]
        else:
            result = snapshot["result"]
            st.session_state["onset_token"] = result["token"]
            st.session_state["onset_districts"] = result["districts"]
    else:
        result = snapshot["result"] or {"results": [], "errors": [snapshot["error"]]}
        st.session_state["forecast_results"] = result["results"]
        st.session_state["onset_forecast_errors"] = result["errors"]
        st.session_state["onset_forecast_finished"] = True
    st.rerun()


def ensure_authenticated():
    if st.session_state.get("onset_token") and st.session_state.get("onset_districts"):
        return True
    if not SHARED_API_KEY and not st.session_state.get("onset_token"):
        st.error("ONSET_API_KEY is not set. Add it to your .env file to enable Forecast API access.")
        return False
    error = st.session_state.get("onset_connection_error")
    if error:
        st.error(f"Forecast API connection failed: {error}")
        if st.button("Retry API connection", key="onset_connection_retry"):
            st.session_state.pop("onset_connection_error")
            st.rerun()
        return False
    if "onset_connection_job" not in st.session_state:
        st.session_state["onset_connection_job"] = BackgroundTask(
            _connect, st.session_state.get("onset_token")
        ).start()
    render_message_job("onset_connection_job", "Connecting to Forecast API.")
    return False


def normalize_chance(record: Dict) -> str:
    """Normalizes classification string from string values or numeric category codes."""
    chance = record.get("chance")
    if chance and isinstance(chance, str):
        return chance.strip().title()

    cat = record.get("category")
    if cat == 1:
        return "High"
    elif cat == 2:
        return "Moderate"
    elif cat == 3:
        return "Uncertain"

    return "Uncertain"


def render_kpi_cards(results: List[Dict]):
    """Calculates summary counts dynamically across all districts and renders HTML cards."""
    high_count = 0
    moderate_count = 0
    uncertain_count = 0

    for r in results:
        status = normalize_chance(r)
        if status == "High":
            high_count += 1
        elif status == "Moderate":
            moderate_count += 1
        else:
            uncertain_count += 1

    total_count = len(results)
    high_pct = (high_count / total_count * 100) if total_count > 0 else 0.0
    moderate_pct = (moderate_count / total_count * 100) if total_count > 0 else 0.0
    uncertain_pct = (uncertain_count / total_count * 100) if total_count > 0 else 0.0

    html_content = f"""<div class="forecast-summary-bar"><div><div class="summary-label">Total Woredas Analyzed</div></div><div class="summary-value">{total_count}</div></div><div class="forecast-kpi-grid"><div class="forecast-kpi-card forecast-kpi-high"><div class="forecast-kpi-accent"></div><div class="forecast-kpi-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/></svg></div><div class="forecast-kpi-content"><div class="forecast-kpi-label">High Chance</div><div class="forecast-kpi-value">{high_count}</div><div class="forecast-kpi-subtitle">{high_pct:.1f}% of districts</div></div></div><div class="forecast-kpi-card forecast-kpi-moderate"><div class="forecast-kpi-accent"></div><div class="forecast-kpi-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/></svg></div><div class="forecast-kpi-content"><div class="forecast-kpi-label">Moderate</div><div class="forecast-kpi-value">{moderate_count}</div><div class="forecast-kpi-subtitle">{moderate_pct:.1f}% of districts</div></div></div><div class="forecast-kpi-card forecast-kpi-uncertain"><div class="forecast-kpi-accent"></div><div class="forecast-kpi-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg></div><div class="forecast-kpi-content"><div class="forecast-kpi-label">Uncertain</div><div class="forecast-kpi-value">{uncertain_count}</div><div class="forecast-kpi-subtitle">{uncertain_pct:.1f}% of districts</div></div></div></div>"""

    st.markdown(html_content, unsafe_allow_html=True)


# ==============================================================================
# Main Page Render
# ==============================================================================
def page():
    st.markdown(
        """
        <div class="sec-hd sky">
            <span class="sec-hd-accent"></span>
            Kiremt Onset Forecast Message Dispatch
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not ensure_authenticated():
        return

    st.markdown(
        '<div class="sec-hd teal"><span class="sec-hd-accent"></span>Configuration</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### Dispatch Settings")

        districts = st.session_state["onset_districts"]
        if not districts:
            st.error("No districts available from API. Please check backend connection.")
            return

        selection_mode = st.radio(
            "Target Scope",
            ["Single District", "Multiple Districts", "All Districts"],
            horizontal=True,
            key="scope_radio",
        )

        selected_districts: List[str] = []

        if selection_mode == "Single District":
            district = st.selectbox("Select District", districts, key="district_select")
            if district:
                selected_districts = [district]

        elif selection_mode == "Multiple Districts":
            default_selection = districts[:3] if len(districts) >= 3 else districts
            selected_districts = st.multiselect(
                "Select Target Woredas",
                districts,
                default=default_selection,
                key="multi_district_select",
            )

        st.markdown("#### Probability Thresholds")
        t1 = st.slider("Week 1 (High)", 0.0, 1.0, 0.40, 0.05, key="t1")
        t2 = st.slider("Week 2-3 (Moderate)", 0.0, 1.0, 0.20, 0.05, key="t2")
        t3 = st.slider("Week 3-4 (Low)", 0.0, 1.0, 0.10, 0.05, key="t3")
        t4 = st.slider("Week 4+ (Uncertain)", 0.0, 1.0, 0.05, 0.05, key="t4")
        thresholds = [t1, t2, t3, t4]

        season = st.selectbox("Season", ["Kiremt Geba"], key="season_select")

        active = "onset_forecast_job" in st.session_state
        if st.button("Generate Forecast", type="primary", width="stretch", disabled=active):
            if selection_mode != "All Districts" and not selected_districts:
                st.warning("Please select at least one district.")
            elif "onset_forecast_job" not in st.session_state:
                st.session_state.pop("forecast_results", None)
                st.session_state.pop("onset_forecast_errors", None)
                st.session_state.pop("onset_forecast_finished", None)
                st.session_state["onset_forecast_job"] = BackgroundTask(
                    _fetch_forecasts, st.session_state.onset_token, selected_districts,
                    selection_mode == "All Districts", thresholds, season,
                ).start()
                st.rerun()
        if "onset_forecast_job" in st.session_state:
            render_message_job("onset_forecast_job", "Generating forecast messages.")
        elif st.session_state.get("onset_forecast_finished"):
            for error in st.session_state.get("onset_forecast_errors", []):
                st.error(f"Forecast request failed: {error}")
            if st.session_state.get("forecast_results"):
                st.success("Forecast task finished.")
            else:
                st.warning("Forecast task finished. No forecast results returned.")

    with col2:
        if "forecast_results" in st.session_state and st.session_state.forecast_results:
            results = st.session_state.forecast_results

            st.markdown("### Forecast Summary")
            render_kpi_cards(results)

            if selection_mode == "Single District" and len(results) == 1:
                res = results[0]
                chance = normalize_chance(res)

                raw_pct = res.get("prob_pct")
                if raw_pct is None or raw_pct == 0:
                    raw_val = res.get("prob_raw", 0.0)
                    raw_pct = round(raw_val * 100, 1) if raw_val < 1.0 else round(raw_val, 1)

                card_class_map = {
                    "High": "card-high",
                    "Moderate": "card-moderate",
                    "Uncertain": "card-uncertain"
                }
                badge_class_map = {
                    "High": "badge-high",
                    "Moderate": "badge-moderate",
                    "Uncertain": "badge-uncertain"
                }

                card_class = card_class_map.get(chance, "card-uncertain")
                badge_class = badge_class_map.get(chance, "badge-uncertain")

                st.markdown("### Selected Woreda Details")
                st.markdown(
                    f"""
                    <div class="woreda-detail-card {card_class}">
                        <div class="detail-accent"></div>
                        <div class="woreda-detail-grid">
                            <div class="woreda-detail-item">
                                <div class="woreda-detail-label">District</div>
                                <div class="woreda-detail-value">{res.get('district', 'Unknown')} <span style="font-weight:400; font-size:0.88rem; color:#64748B;">({res.get('adm3_pcode', 'Unknown')})</span></div>
                            </div>
                            <div class="woreda-detail-item">
                                <div class="woreda-detail-label">Category</div>
                                <div class="badge-chip {badge_class}">
                                    {chance} ({raw_pct}%)
                                </div>
                            </div>
                            <div class="woreda-detail-item">
                                <div class="woreda-detail-label">Forecast Period</div>
                                <div class="woreda-detail-value">{res.get('date_start', 'N/A')} – {res.get('date_end', 'N/A')}</div>
                            </div>
                            <div class="woreda-detail-item">
                                <div class="woreda-detail-label">Climatology Date</div>
                                <div class="woreda-detail-value">{res.get('clim_date', 'Unknown')} <span style="font-weight:400; font-size:0.88rem; color:#64748B;">(DOY: {res.get('clim_doy', 'N/A')})</span></div>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                message = res.get("message", "")
                if message:
                    st.markdown("### Localized Alert Message")
                    st.markdown(
                        f"""
                        <div class="bench-card">
                            <div style="padding: 1rem; line-height: 1.6; font-size: 0.95rem;">
                                {message}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            df = pd.DataFrame(results)
            st.markdown("#### Detailed Results")
            display_cols = [c for c in ["district", "adm3_pcode", "chance", "prob_pct", "date_start", "date_end", "clim_date", "message"] if c in df.columns]
            st.dataframe(
                df[display_cols],
                width="stretch",
                hide_index=True,
            )

            st.markdown("### Export Results")
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                st.download_button(
                    "Download CSV",
                    df.to_csv(index=False),
                    f"forecast_export_{datetime.now().strftime('%Y%m%d')}.csv",
                    "text/csv",
                    width="stretch",
                )
            with col_d2:
                st.download_button(
                    "Download JSON",
                    json.dumps(results, indent=2),
                    f"forecast_export_{datetime.now().strftime('%Y%m%d')}.json",
                    "application/json",
                    width="stretch",
                )
        else:
            st.info(
                "Configure settings in the left panel and click 'Generate Forecast' to view results."
            )

    st.markdown("---")
    if st.button("Reset Session", width="stretch", disabled="onset_forecast_job" in st.session_state):
        for key in ("onset_token", "onset_districts", "onset_connection_error", "forecast_results",
                    "onset_forecast_errors", "onset_forecast_finished"):
            st.session_state.pop(key, None)
        st.rerun()


render = page

if __name__ == "__main__":
    page()