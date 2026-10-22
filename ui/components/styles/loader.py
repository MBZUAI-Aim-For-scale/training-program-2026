"""
Central CSS loader for the platform.

Usage
-----
    app.py       -> load_css()               # globals + tokens, ONCE per run
    ui/pages/*   -> load_page_css("home")    # that page's sheet only
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import streamlit as st

logger = logging.getLogger(__name__)

STYLES_ROOT = Path(__file__).resolve().parent
PAGE_STYLES_DIR = STYLES_ROOT / "page_styles"

GLOBAL_SHEETS: tuple[str, ...] = (
    "main.css",
    "nav.css",
    "header.css",
    "control_center.css",
    "tabs.css",
    "filters.css",
    "footer.css",
    "loader.css",
    "forecast.css",
    "page_styles/home.css",
    "page_styles/benchmarking.css",
    "page_styles/ai_setup.css",
)

_MISSING: set[str] = set()


def _normalise(name: str) -> str:
    name = str(name).strip()
    if not name:
        raise ValueError("CSS sheet name cannot be empty")
    if "/" in name or "\\" in name or name in {".", ".."}:
        raise ValueError(f"Use a bare sheet name, not a path: {name!r}")
    return name if name.endswith(".css") else f"{name}.css"


def resolve_sheet(name: str, *, page_first: bool = False) -> Path | None:
    """Resolve a bare sheet name against page_styles/ and the styles root."""
    filename = _normalise(name)
    order = (PAGE_STYLES_DIR, STYLES_ROOT) if page_first else (STYLES_ROOT, PAGE_STYLES_DIR)
    for base in order:
        candidate = base / filename
        if candidate.is_file():
            return candidate
    return None


@st.cache_data(show_spinner=False)
def _read_cached(path_str: str, mtime_ns: int, size: int) -> str:
    """Keyed on mtime+size so editing a .css file takes effect on rerun."""
    return Path(path_str).read_text(encoding="utf-8")


def _read(path: Path) -> str:
    stat = path.stat()
    return _read_cached(str(path), stat.st_mtime_ns, stat.st_size)


def _bundle(names: Sequence[str], *, page_first: bool) -> str:
    chunks: list[str] = []
    seen: set[str] = set()

    for raw in names:
        path = resolve_sheet(raw, page_first=page_first)
        if path is None:
            if raw not in _MISSING:
                _MISSING.add(raw)
                logger.warning("CSS sheet not found: %s (looked in %s)", raw, STYLES_ROOT)
            continue

        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        chunks.append(f"/* {path.relative_to(STYLES_ROOT)} */\n{_read(path)}")

    return "\n".join(chunks)


def _emit(css: str) -> None:
    if css.strip():
        st.markdown(f'<style data-romp-css="1">\n{css}\n</style>', unsafe_allow_html=True)


def load_css(names: Sequence[str] = GLOBAL_SHEETS) -> None:
    """Inject global sheets. Call exactly once, from app.main()."""
    _emit(_bundle(names, page_first=False))


def load_page_css(*names: str, tokens: bool = False) -> None:
    """
    Inject one or more page sheets.

    tokens=True is only needed when running a page file directly
    (streamlit run ui/pages/home.py), where app.py never ran.
    """
    if not names:
        raise ValueError(
            "load_page_css() requires at least one sheet name, "
            "e.g. load_page_css('home')"
        )
    sheets = (["tokens.css"] if tokens else []) + list(names)
    _emit(_bundle(sheets, page_first=True))


def page_css_path(name: str) -> Path:
    """Convenience for tests / CLI tooling."""
    return PAGE_STYLES_DIR / _normalise(name)