#!/usr/bin/env python3
import os
import sys
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import importlib
from pathlib import Path
import sys
import base64
import re
import traceback
import streamlit as st

# ─── PROJECT ROOT & IMPORT PATHS ────────────────────────────────────────────
# The pages import `config` and `run_momp`, which live in the project root. That folder
# is not necessarily the parent of this file, so walk upwards until both files are found.
APP_DIR = Path(__file__).resolve().parent


def _find_project_root() -> Path:
    for candidate in (APP_DIR, *APP_DIR.parents):
        if (candidate / "config.py").exists() and (candidate / "run_momp.py").exists():
            return candidate
    return APP_DIR.parent  # previous behaviour


PROJECT_ROOT = _find_project_root()

for _p in (APP_DIR.parent, APP_DIR, PROJECT_ROOT):  # last insert ends up first on sys.path
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
current_year = datetime.now().year
current_month = datetime.now().month
current_day = datetime.now().day

# ─── FORCE LIGHT THEME CONFIGURATION ────────────────────────────────────────
def ensure_light_theme_config():
    """
    Make sure .streamlit/config.toml (in the current working directory, where Streamlit
    reads it) contains `base = "light"` under [theme]. Takes effect on the next restart.
    """
    try:
        config_dir = Path(".streamlit")
        config_dir.mkdir(exist_ok=True)
        config_file = config_dir / "config.toml"
        lines = config_file.read_text(encoding="utf-8").splitlines() if config_file.exists() else []
        wanted = 'base = "light"'

        start = next((i for i, l in enumerate(lines) if l.strip() == "[theme]"), None)
        new = list(lines)
        if start is None:
            new += ([""] if lines else []) + ["[theme]", wanted]
        else:
            # only look inside the [theme] table (appending at end of file could land in another table)
            end = next((i for i in range(start + 1, len(lines)) if lines[i].strip().startswith("[")), len(lines))
            base_idx = next(
                (i for i in range(start + 1, end) if lines[i].split("#")[0].split("=")[0].strip() == "base"),
                None,
            )
            if base_idx is None:
                new.insert(start + 1, wanted)
            else:
                value = lines[base_idx].split("=", 1)[1].split("#")[0].strip().strip("\"'")
                if value != "light":
                    new[base_idx] = wanted

        if new != lines:
            config_file.write_text("\n".join(new) + "\n", encoding="utf-8")
    except OSError as exc:  # read-only filesystem etc. must never stop the app
        print(f"[theme] could not update .streamlit/config.toml: {exc}")

ensure_light_theme_config()

# ─── EXTENDED SYSTEM MODULE CONFIGURATION MATRIX ───────────────────────────
PAGE_MODULE_MAPPING = {
    "Home": "ui.pages.home",
    # "AI_Setup": "ui.pages.ai_setup", 
    # "AI Forecast": "ui.pages.ai_forecasting", 
    "SM_Benchmarking": "ui.pages.short_medium_benchmarking",
}

PRIMARY_MENU_CONFIG = {
    "Home": {"target": "Home", "label": "Home"},
    # "AI Lab Setup": {"target": "AI_Setup", "label": "Demo1"},
    # "AI Forecast": {"target": "AI Forecast", "label": "Demo3"},
    "AI Benchmarking": {"target": "SM_Benchmarking", "label": "Benchmarking"},
}

COMING_SOON = {
    "NRT_Verification": False,
    "Seasonal_Calibration": True,
    "Seasonal_Benchmarking": True,
}

MAX_WORKERS = min(4, os.cpu_count() or 1)

@st.cache_resource
def pre_import_modules():
    """Load page modules sequentially to avoid concurrent native-library imports."""
    for module_path in PAGE_MODULE_MAPPING.values():
        try:
            importlib.import_module(module_path)
        except Exception as exc:
            print(
                f"[WARMUP ERROR] Failed to pre-import {module_path}: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            # Continue loading other pages. The active page can report
            # its own import error when Streamlit attempts to render it.
    return True

@st.cache_data(ttl=30, show_spinner=False)
def collect_config_problems():
    """Cheap, read-only sanity check of what config.py / run_momp.py will need."""
    try:
        import config
    except Exception as exc:  # noqa: BLE001
        return [f"config.py could not be imported: {type(exc).__name__}: {exc}"]

    problems = []
    if not (PROJECT_ROOT / "run_momp.py").exists():
        problems.append(f"run_momp.py not found in {PROJECT_ROOT}")
    try:
        data_dir = Path(config.DATA_DIR)
        if not data_dir.is_dir():
            problems.append(f"Data directory not found: {data_dir}")
        driver = Path(config.MOMP_PKG_DIR) / "driver.py"
        if not driver.exists():
            problems.append(f"ROMP driver not found: {driver}")
        for label, path in (("Observation folder", config.obs_dir), ("Region mask", config.nc_mask)):
            if path and not config.exists(path):          # works for local paths and gs://
                problems.append(f"{label} not found: {path}")
        for key, meta in config.BENCHMARK_MODEL_CATALOG.items():
            folder = str(meta["dir"])                      # not Path(): it would turn gs:// into gs:/
            if not config.exists(folder):
                problems.append(f"{key}: model folder not found: {folder}")
            elif not config.glob(folder.rstrip("/") + "/*.nc"):
                problems.append(f"{key}: no .nc files in {folder}")
    except Exception as exc:  # noqa: BLE001
        problems.append(f"config check failed: {type(exc).__name__}: {exc}")
    return problems


def load_css():
    """
    Inject the global stylesheet bundle into the Streamlit app.
    """
    styles_dir = Path(os.path.dirname(__file__)) / "components" / "styles"

    # Order matters: tokens → shared widgets → shell chrome → pages
    decoupled_sheets = [
        # Design system (MUST come first — everything else uses these variables)
        "tokens.css",
        "components.css",
        # Shell chrome
        "main.css",
        "nav.css",
        "header.css",
        "control_center.css",
        "tabs.css",
        "filters.css",
        "footer.css",
        "loader.css",
        "forecast.css",
        # Page stylesheets (warm-up injection; pages also self-load)
        "page_styles/home.css",
        "page_styles/benchmarking.css",
        "page_styles/ai_setup.css",
    ]

    compiled_bundle = ""
    seen = set()
    for sheet in decoupled_sheets:
        if sheet in seen:
            continue
        seen.add(sheet)

        full_path = styles_dir / sheet
        if full_path.exists():
            try:
                compiled_bundle += f"\n/* --- {sheet} --- */\n" + full_path.read_text(encoding="utf-8")
            except OSError as exc:
                compiled_bundle += f"\n/* [MISSING] {sheet}: {exc} */\n"

    # Streamlit-specific theme overrides (these don't belong in a shared stylesheet
    # because they target Streamlit's internal data-testid attributes)
    compiled_bundle += """
    /* ── Force Light Theme Variables & Backgrounds ───────────────────── */
    [data-testid="stApp"], .stApp, [data-theme="dark"] {
        --primary-color: #0a9938 !important;
        --background-color: #ffffff !important;
        --secondary-background-color: #f0f2f6 !important;
        --text-color: #262730 !important;
        --secondary-text-color: #555555 !important;
        --font: 'Inter', sans-serif !important;
        background-color: #ffffff !important;
        color: #262730 !important;
    }
    [data-testid="stSidebar"] {
        background-color: #f0f2f6 !important;
        color: #262730 !important;
    }

    [data-testid="stApp"] .stButton > button {
    width: 100% !important;
    max-width: 100% !important;
    }

    /* ── Navbar button stretch ───────────────────────────────────────── */
    [data-testid="stApp"] .stButton > button {
        width: 100% !important;
        max-width: 100% !important;
    }

    /* ── Remove horizontal white space on left and right ─────────────── */
    /* Streamlit adds default padding (~1rem on mobile, ~3rem on desktop)
       around the main content block. Reset it so content reaches the edges. */
    section.main .block-container,
    [data-testid="stAppViewBlockContainer"],
    .main .block-container {
        max-width: 100% !important;
        padding-left: 0.5rem !important;
        padding-right: 0.5rem !important;
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
    }

    /* Remove the gap between columns so navbar buttons touch seamlessly */
    [data-testid="column"] {
        padding: 0 !important;
    }
    [data-testid="stHorizontalBlock"] {
        gap: 0 !important;
        align-items: stretch !important;
    }

    /* Make the top navbar banner stretch edge-to-edge */
    .st-key-top_navbar {
        margin-left: -0.5rem !important;
        margin-right: -0.5rem !important;
        width: calc(100% + 1rem) !important;
    }

    /* Footer should also span full width */
    .green-footer {
        margin-left: -0.5rem !important;
        margin-right: -0.5rem !important;
        width: calc(100% + 1rem) !important;
    }
    """
    st.markdown(f"<style>{compiled_bundle}</style>", unsafe_allow_html=True)
def _inject_heartbeat():
    """Inject heartbeat script using st.iframe (replacing deprecated components.iframe)."""
    html_content = """
<script>
(function () {
    let last_ok = Date.now();
    const origin = (function () {
        try { return window.parent.location.origin; }
        catch (_) { return window.location.origin; }
    })();
    setInterval(async () => {
        try {
            const r = await fetch(origin + "/_stcore/health", { cache: "no-store" });
            if (r.ok) { last_ok = Date.now(); return; }
        } catch (_) {}
        if (Date.now() - last_ok > 45000) {
            console.warn("[heartbeat] backend unreachable for 45s, reloading");
            try { window.parent.location.reload(); }
            catch (_) { window.location.reload(); }
        }
    }, 10000);
})();
</script>
"""
    b64_html = base64.b64encode(html_content.encode("utf-8")).decode("utf-8")
    # FIX: st.iframe requires a positive integer for height. Using 1 keeps it visually invisible.
    st.iframe(f"data:text/html;base64,{b64_html}", height=1)

def get_inline_svg(filename: str) -> str:
    here = Path(__file__).resolve().parent
    candidates = [
        here / "static" / filename,
        here.parent / "static" / filename,
        here.parent.parent / "static" / filename,
        PROJECT_ROOT / "static" / filename,
        Path.cwd() / "static" / filename,
    ]
    svg_path = next((p for p in candidates if p.exists()), None)
    if svg_path is None:
        return ""
    try:
        svg_text = svg_path.read_text(encoding="utf-8")
    except Exception:
        return ""
    # Strip <?xml ...?> or DOCTYPE headers if present so it embeds cleanly in HTML
    svg_text = re.sub(r'<\?xml[^>]*\?>', '', svg_text)
    svg_text = re.sub(r'<!DOCTYPE[^>]*>', '', svg_text)
    return svg_text.strip()

# Fetch raw SVGs
UCHICAGO_SVG_RAW = get_inline_svg("UChicago.svg")
AICE_SVG_RAW = get_inline_svg("AICE.svg")
AIMFORSCALE_SVG_RAW = get_inline_svg("aimforscale.svg")
MBZUAI_SVG_RAW = get_inline_svg("MBZUSIAIMForScale.svg") 
AI4AG_SVG_RAW = get_inline_svg("AI4AG.svg")

# Wrap them in containers

UCHICAGO_LOGO_HTML = (
    f'<div class="footer-logo-wrapper uchicago-wrapper">{UCHICAGO_SVG_RAW}</div>'
    if UCHICAGO_SVG_RAW else '<span style="color:#fff;font-weight:bold;">UChicago</span>'
)

AICE_LOGO_HTML = (
    f'<div class="footer-logo-wrapper uchicago-wrapper">{AICE_SVG_RAW}</div>'
    if AICE_SVG_RAW else '<span style="color:#fff;font-weight:bold;">AICE</span>'
)

AIMFORSCALE_LOGO_HTML = (
    f'<div class="footer-logo-wrapper logo-wrapper">{AIMFORSCALE_SVG_RAW}</div>'
    if AIMFORSCALE_SVG_RAW else '<span style="color:#fff;font-weight:bold;">AIMFORSCALE</span>'
)


MBZUAI_LOGO_HTML = (
    f'<div class="footer-logo-wrapper logo-wrapper">{MBZUAI_SVG_RAW}</div>'
    if MBZUAI_SVG_RAW else '<span style="color:#fff;font-weight:bold;">MBZUAI</span>'
)


AI4AG_LOGO_HTML = (
    f'<div class="footer-logo-wrapper logo-wrapper">{AI4AG_SVG_RAW}</div>'
    if AI4AG_SVG_RAW else '<span style="color:#fff;font-weight:bold;">AI4AG</span>'
)


def main():
    st.set_page_config(
        page_title="Benchmarking",
        page_icon="🌍",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={"Get Help": None, "Report a bug": None, "About": None},
    )
    load_css()
    pre_import_modules()

    if "uploaded_df" not in st.session_state:
        st.session_state["uploaded_df"] = None
    if "metadata" not in st.session_state:
        st.session_state["metadata"] = {}

    query_page = st.query_params.get("page", "Home")
    if query_page not in PAGE_MODULE_MAPPING:
        query_page = "Home"

    nav_key_mapping = {v["target"]: k for k, v in PRIMARY_MENU_CONFIG.items()}
    nav_key_mapping.update({
        "SM_Verification": "Home",
        "SM_Threshold": "Home",
        "Onset Climatology": "Onset Dashboard",
    })
    active_primary = nav_key_mapping.get(query_page, "Home")

    # ─── TOP NAVBAR RENDERING ────────────────────────────────────────────────────
    with st.container(key="top_navbar"):
        brand_col, menu_col, badge_col = st.columns([1.2, 5.6, 1.2], gap="small")
        
        with brand_col:
            st.markdown(
                '<div class="brand-text">Onset Forecast Platform <span>| AI</span></div>',
                unsafe_allow_html=True,
            )
            
        with menu_col:
            # Equal width columns; CSS forces them to stretch and fill remaining space
            nav_btn_cols = st.columns(len(PRIMARY_MENU_CONFIG), gap="small")
            for nav_col, (k, v) in zip(nav_btn_cols, PRIMARY_MENU_CONFIG.items()):
                with nav_col:
                    if st.button(
                        v["label"],
                        key=f"primary_nav_{v['target']}",
                        type="primary" if active_primary == k else "secondary",
                        width="stretch",
                    ):
                        st.query_params["page"] = v["target"]
                        st.rerun()
                        
        with badge_col:
            st.markdown(
                f'<div class="weather-badge"><span class="blink-dot"></span>'
                f'Live {current_month}/{current_day}/{current_year}</div>',
                unsafe_allow_html=True,
            )

    # ─── SUB-NAVIGATION LOGIC ────────────────────────────────────────────────
    if active_primary == "AI Lab Setup":
        sub_cols = st.columns([1, 3])
        setup_actions = {
            "AI_Setup": "Demo Status",
        }
        for s_idx, (m_route, label) in enumerate(setup_actions.items()):
            with sub_cols[s_idx]:
                if st.button(
                    label,
                    type="primary" if query_page == m_route else "secondary",
                    width="stretch",
                    key=f"s_btn_{m_route}",
                ):
                    st.query_params["page"] = m_route
                    st.rerun()               
                    
    elif active_primary == "Onset Benchmarking":
        sub_cols = st.columns([1, 3])
        bm_actions = {
            "SM_Benchmarking": "Performance Benchmarks",
        }
        for s_idx, (m_route, label) in enumerate(bm_actions.items()):
            with sub_cols[s_idx]:
                if st.button(
                    label,
                    type="primary" if query_page == m_route else "secondary",
                    width="stretch",
                    key=f"s_btn_{m_route}",
                ):
                    st.query_params["page"] = m_route
                    st.rerun()
    if active_primary == "Verification":
        sub_cols = st.columns([1, 1, 2])
        sm_actions = {
            "SM_Verification": "Verification Metrics",
            "SM_Threshold": "Threshold Engine",
        }
        for s_idx, (m_route, label) in enumerate(sm_actions.items()):
            with sub_cols[s_idx]:
                if st.button(
                    label,
                    type="primary" if query_page == m_route else "secondary",
                    width="stretch",
                    key=f"s_btn_{m_route}",
                ):
                    st.query_params["page"] = m_route
                    st.rerun()
                    
    elif query_page.startswith("Seasonal_"):
        sub_cols = st.columns([1, 1, 1, 2])
        seasonal_actions = {
            "Seasonal_Verification": "Seasonal Verification",
            "Seasonal_Calibration": "Seasonal Calibration",
            "Seasonal_Benchmarking": "Seasonal Benchmarking",
        }
        for s_idx, (m_route, label) in enumerate(seasonal_actions.items()):
            with sub_cols[s_idx]:
                is_soon = COMING_SOON.get(m_route, False)
                if st.button(
                    f"{label} (Soon)" if is_soon else label,
                    type="primary" if query_page == m_route else "secondary",
                    width="stretch",
                    disabled=is_soon,
                    key=f"s_btn_{m_route}",
                ):
                    st.query_params["page"] = m_route
                    st.rerun()
    
    st.markdown("<br>", unsafe_allow_html=True)

    config_problems = collect_config_problems()
    if config_problems:
        with st.expander(f"Configuration issues ({len(config_problems)})"):
            for item in config_problems:
                st.write(f"- {item}")

    # ─── PAGE RENDERING ──────────────────────────────────────────────────────
    active_module_path = PAGE_MODULE_MAPPING[query_page]
    try:
        target_module = importlib.import_module(active_module_path)
        if hasattr(target_module, "render"):
            target_module.render()
        elif hasattr(target_module, "page"):
            target_module.page()
        else:
            st.error(f"Module '{active_module_path}' has no entrypoint (render or page).")
    except Exception as e:
        traceback.print_exc()  # full traceback in the terminal
        st.error(f"Workspace mounting failure at route cluster: {type(e).__name__}: {e}")
        with st.expander("Technical details"):
            st.exception(e)
        if st.button("Retry Connection", width="stretch"):
            st.rerun()

    # ─── FOOTER ──────────────────────────────────────────────────────────────
    st.markdown(
        f"""
<div class="green-footer">
    <div class="footer-content">
        <div class="footer-row">
            <div class="footer-left">{AIMFORSCALE_LOGO_HTML}</div>
            <div class="footer-left">{MBZUAI_LOGO_HTML}</div>
            <div class="footer-center">
                <div class="brand-group">
                    <span class="brand">AIM FOR SCALE</span>
                    <span class="sep">|</span>
                    <span class="brand">Human-Centered Weather Forecasts Initiative</span>
                    <span class="sep">|</span>
                    <span class="brand">University of Chicago</span>
                     <span class="sep">|</span>
                    <span class="brand">AICE</span>
                </div>
            </div>
            <div class="footer-right">{UCHICAGO_LOGO_HTML}</div>
            <div class="footer-right">{AICE_LOGO_HTML}</div>
        </div>
        <div class="meta">
            &copy; 2026 AI4AG &amp; HCWF </br>
            <span> All rights reserved.</span>
        </div>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )
    _inject_heartbeat()

if __name__ == "__main__":
    main()