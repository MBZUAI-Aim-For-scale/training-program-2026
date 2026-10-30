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

load_dotenv()

# ==============================================================================
# API Configuration (Passwordless via Environment Variables)
# ==============================================================================
API_BASE_URL = os.environ.get("ONSET_API_URL", "http://localhost:8000")
SHARED_API_KEY = os.environ.get("ONSET_API_KEY", "")


def get_jwt_token(api_key: str) -> Optional[str]:
    if not api_key:
        return None
    try:
        response = requests.post(
            f"{API_BASE_URL}/token",
            headers={"X-API-Key": api_key},
            timeout=10,
        )
        if response.status_code == 200:
            return response.json()["access_token"]
        return None
    except Exception as e:
        st.error(f"Authentication failed: {e}")
        return None


def ensure_authenticated():
    if "onset_token" not in st.session_state or not st.session_state.onset_token:
        if not SHARED_API_KEY:
            st.markdown(
                """
                <div class="ibox rose">
                    <b>Configuration Error</b><br>
                    The <code>ONSET_API_KEY</code> environment variable is not set.
                    Please add it to your <code>.env</code> file to enable passwordless communication.
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.stop()

        with st.spinner("Authenticating with Forecast API..."):
            token = get_jwt_token(SHARED_API_KEY)
            if token:
                st.session_state.onset_token = token
            else:
                st.markdown(
                    """
                    <div class="ibox rose">
                        <b>Authentication Failed</b><br>
                        The provided <code>ONSET_API_KEY</code> is invalid, or its hash does not match the backend.
                        Please verify your environment variables.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.stop()


def get_districts(token: str) -> List[str]:
    try:
        response = requests.get(
            f"{API_BASE_URL}/districts",
            headers={"Authorization": f"Bearer {token}"},
            timeout=10,
        )
        if response.status_code == 200:
            return response.json()["districts"]
        return []
    except Exception as e:
        st.error(f"Failed to fetch districts: {e}")
        return []


def get_forecast(
    token: str,
    district: Optional[str],
    all_districts: bool,
    thresholds: List[float],
    season: str,
) -> List[Dict]:
    try:
        response = requests.post(
            f"{API_BASE_URL}/forecast",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "district": district,
                "all_districts": all_districts,
                "thresholds": thresholds,
                "season": season,
            },
            timeout=60,
        )
        if response.status_code == 200:
            return response.json()
        st.error(f"Forecast API Error [{response.status_code}]: {response.text}")
        return []
    except Exception as e:
        st.error(f"Forecast request failed: {e}")
        return []


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

    ensure_authenticated()

    st.markdown(
        '<div class="sec-hd teal"><span class="sec-hd-accent"></span>Configuration</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### Dispatch Settings")

        districts = get_districts(st.session_state.onset_token)
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

        if st.button("Generate Forecast", type="primary", width="stretch"):
            if "forecast_results" in st.session_state:
                del st.session_state["forecast_results"]

            if selection_mode == "All Districts":
                with st.spinner("Fetching forecast data for all districts..."):
                    results = get_forecast(
                        st.session_state.onset_token,
                        district=None,
                        all_districts=True,
                        thresholds=thresholds,
                        season=season,
                    )
            else:
                if not selected_districts:
                    st.warning("Please select at least one district.")
                    results = []
                else:
                    with st.spinner(
                        f"Fetching forecast data for {len(selected_districts)} district(s)..."
                    ):
                        results = []
                        for dist in selected_districts:
                            res = get_forecast(
                                st.session_state.onset_token,
                                district=dist,
                                all_districts=False,
                                thresholds=thresholds,
                                season=season,
                            )
                            if res:
                                results.extend(res)

            if results:
                st.session_state.forecast_results = results
            else:
                st.warning("No forecast results returned.")

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
    if st.button("Reset Session", width="stretch"):
        st.session_state.onset_token = None
        if "forecast_results" in st.session_state:
            del st.session_state["forecast_results"]
        st.rerun()


render = page

if __name__ == "__main__":
    page()