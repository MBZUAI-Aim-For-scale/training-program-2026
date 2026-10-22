"""
Landing page – platform introduction
"""
import os
import sys
import logging
from pathlib import Path
import streamlit as st
from config import APP_DIR
from ui.components.cards import pillar_card, feature_card

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from config import APP_DIR
logger = logging.getLogger(__name__)


def load_css():
    """
    Compiles and structures all decoupled architectural stylesheets dynamically,
    maintaining consistency with the NRT module's CSS delivery system.
    """
    possible_dirs = [
        Path(os.path.dirname(__file__)) / "components" / "styles" / "page_styles",
        Path(os.path.dirname(__file__)) / "styles" / "page_styles",
        Path(os.path.dirname(__file__)).parent / "styles" / "page_styles",
        Path(APP_DIR) / "styles" / "page_styles"
    ]
    
    decoupled_sheets = [
        "main.css", 
        "nav.css", 
        "control_center.css", 
        "tabs.css", 
        "filters.css", 
        "footer.css"
    ]
    
    css_content = []
    styles_dir = None
    
    for p_dir in possible_dirs:
        if p_dir.exists():
            styles_dir = p_dir
            break
            
    if styles_dir:
        for sheet in decoupled_sheets:
            sheet_path = styles_dir / sheet
            if sheet_path.exists():
                try:
                    with open(sheet_path, "r", encoding="utf-8") as f:
                        css_content.append(f.read())
                except Exception as e:
                    logger.error(f"Error reading CSS sheet {sheet}: {e}")
                    
    if css_content:
        st.markdown(f"<style>{chr(10).join(css_content)}</style>", unsafe_allow_html=True)

def page():
    # 1. Compile stylesheet DOM injection
    load_css()
    
    # 2. Hero Section Header
    st.markdown("""
        <div style='max-width:800px;margin:0 auto 40px;text-align:center;'>
          <p style='font-family:Inter,sans-serif;font-size:1.8rem;font-weight:500;
              letter-spacing:0.2em;text-transform:uppercase;color:#0a9938;'> AI-Powered Weather Forecasting </p>
          <h1 style='font-family:Inter,sans-serif;font-size:2.4rem;font-weight:600;
              color:#171d1a;line-height:1.2;margin:0 0 12px;'>
            <span style='color:#213b7d;'> A  Forecasting and Benchmarking Platform </span>
          </h1>
          <p style='font-family:Inter,sans-serif;font-size:1.1rem;color:#4B5563;
              max-width:600px;margin:0 auto 32px;'>
            Short and  medium-range  forecasting for onset
            plus calibrated onset blending and alert dispatch.
          </p>
        </div>
    """, unsafe_allow_html=True)
    
    # 3. Platform Pillars (3-Column Layout)
    col1, col2, col3 = st.columns(3, gap="large")
    
    with col1:
        pillar_card(
            icon="⚙️",
            title="Benchmarking",
            subtitle="Model improvement",
            description="Bias correction, skill scores, and multi‑model intercomparison."
        )
        
    with col2:
        pillar_card(
            icon="📈",
            title="AI Models Blending",
            subtitle="To get best performance",
            description="Lead‑time error curves, spatial patterns, monthly seasonality."
        )
    with col3:
        pillar_card(
            icon="📈",
            title="Short/Medium Forecasting",
            subtitle="1 week – 4 weeks",
            description="Lead‑time error curves, spatial patterns, monthly seasonality."
        )
        

        
    st.markdown("---")
    
    # 4. Operational Tools Section
    st.subheader("Operational Tools")
    col_a, col_b = st.columns(2, gap="medium")
    
    with col_a:
        feature_card(
            icon="🌾",
            title="AI Dashboard",
            desc="Monitor near‑real‑time onset likelihood using observations, MJO, and guardrails."
        )
        
    with col_b:
        feature_card(
            icon="📨",
            title="Forecast Dispatch",
            desc="Blend AI models, generate official onset forecasts, and manage SMS alerts."
        )
        
    st.caption("Onset Forecasting Platform ")