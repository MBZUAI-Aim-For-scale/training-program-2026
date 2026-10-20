# ui/utils/css.py
import os
from pathlib import Path

def load_css(css_file):
    css_path = Path(__file__).parent.parent / "components" / "styles" / css_file
    if css_path.exists():
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        # Silently ignore missing files
        pass

