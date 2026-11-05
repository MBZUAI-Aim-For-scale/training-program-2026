# ui/components/styles/main.py
"""
Styles loader module for ETAgroMetAIx UI components.
Compiles and injects decoupled CSS stylesheets into Streamlit app DOM.
"""
from __future__ import annotations
import os
from pathlib import Path
import streamlit as st

STYLES_DIR = Path(__file__).parent.resolve()

DECOUPLED_SHEETS = [
    "main.css",
    "nav.css",
    "control_center.css",
    "tabs.css",
    "filters.css",
    "footer.css",
    "verification.css",
    "loader.css",
]


def load_all_styles() -> str:
    """Compile and return all decoupled CSS stylesheets as a single bundle string."""
    bundle = []
    for sheet in DECOUPLED_SHEETS:
        file_path = STYLES_DIR / sheet
        if file_path.exists():
            with open(file_path, "r", encoding="utf-8") as f:
                bundle.append(f"/* --- {sheet} --- */\n{f.read()}")
    return "\n\n".join(bundle)


def inject_styles():
    """Inject all compiled component CSS stylesheets into the Streamlit app session."""
    css_content = load_all_styles()
    if css_content:
        st.markdown(f"<style>\n{css_content}\n</style>", unsafe_allow_html=True)


def inject_sheet(sheet_name: str):
    """Inject a specific stylesheet by name."""
    file_path = STYLES_DIR / sheet_name
    if file_path.exists():
        with open(file_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>\n{f.read()}\n</style>", unsafe_allow_html=True)
