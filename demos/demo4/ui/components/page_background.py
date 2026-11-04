# Authors: The africlimate AI team
# SPDX-License-Identifier: MIT
from __future__ import annotations

import streamlit as st


def set_background(image_url: str = "app/static/isobar_white.png") -> None:
    """Inject isobar pattern as page background via Streamlit's static file server."""
    st.markdown(
        f"""<style>
        /* Base layer — the actual image */
        body, html {{
            background-image: url("{image_url}") !important;
            background-size: cover !important;
            background-position: center center !important;
            background-repeat: no-repeat !important;
            background-attachment: fixed !important;
        }}
        /* Make every Streamlit wrapper transparent so the body image shows through */
        #root,
        .stApp,
        [data-testid="stApp"],
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"],
        [data-testid="stHeader"],
        [data-testid="stBottom"],
        [data-testid="stBottomBlockContainer"] {{
            background: transparent !important;
            background-color: transparent !important;
        }}
        </style>""",
        unsafe_allow_html=True,
    )


def set_sidebar_background(image_url: str) -> None:
    st.markdown(
        f"""<style>
        [data-testid="stSidebar"] > div:first-child {{
            background-image: url("{image_url}") !important;
            background-size: cover !important;
            background-position: center !important;
            background-repeat: no-repeat !important;
            background-attachment: local !important;
        }}
        </style>""",
        unsafe_allow_html=True,
    )
