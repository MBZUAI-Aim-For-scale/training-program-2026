# utils/helpers.py
import streamlit as st
from pathlib import Path
from config import APP_DIR

STYLES_DIR = APP_DIR / "styles"

def load_css(css_file: str = "main.css") -> None:
    css_path = STYLES_DIR / css_file
    if css_path.exists():
        with open(css_path, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def initialize_session_state(keys: dict) -> None:
    for key, default in keys.items():
        if key not in st.session_state:
            st.session_state[key] = default

def get_page_title(page: str) -> str:
    return f"⚡ {page.replace('_', ' ').title()}"