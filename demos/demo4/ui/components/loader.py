# ui/components/loader.py
"""
Eye-Catching Loading Components & Animated Skeletons for ETAgroMetAIx UI.
"""
from __future__ import annotations
from contextlib import contextmanager
from pathlib import Path
import streamlit as st

STYLES_DIR = Path(__file__).parent / "styles"
LOADER_CSS_PATH = STYLES_DIR / "loader.css"


def inject_loader_css():
    """Inject loading animation CSS into Streamlit DOM."""
    if LOADER_CSS_PATH.exists():
        with open(LOADER_CSS_PATH, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def show_loader(
    title: str = "Processing Weather & Forecast Pipeline...",
    subtext: str = "Synthesizing ensemble models and spatial data layers"
):
    """Render an eye-catching glassmorphic animated loading card."""
    inject_loader_css()
    html = f"""
    <div class="agy-loader-card">
        <div class="agy-spinner-wrapper">
            <div class="agy-spinner-outer"></div>
            <div class="agy-spinner-inner"></div>
        </div>
        <div class="agy-loader-title">{title}</div>
        <div class="agy-loader-subtext">{subtext}</div>
        <div class="agy-progress-track">
            <div class="agy-progress-bar"></div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def show_skeleton(height: int = 180, count: int = 1):
    """Render shimmer skeleton placeholder cards."""
    inject_loader_css()
    cards_html = "".join(
        f'<div class="agy-skeleton-box" style="height: {height}px;"></div>'
        for _ in range(count)
    )
    st.markdown(cards_html, unsafe_allow_html=True)


def show_metric_skeletons(count: int = 4):
    """Render grid of shimmer metric card placeholders."""
    inject_loader_css()
    boxes = "".join(
        '<div class="agy-skeleton-box" style="height: 90px; margin-bottom: 0;"></div>'
        for _ in range(count)
    )
    html = f'<div class="agy-skeleton-card-grid">{boxes}</div>'
    st.markdown(html, unsafe_allow_html=True)


@contextmanager
def loading_context(
    title: str = "Loading Forecast Data...",
    subtext: str = "Please wait while calculations complete"
):
    """Context manager to display an eye-catching animated loader during execution."""
    placeholder = st.empty()
    with placeholder.container():
        show_loader(title=title, subtext=subtext)
    try:
        yield
    finally:
        placeholder.empty()
