# Authors: The africlimate AI team
# SPDX-License-Identifier: MIT
"""Shared, cached access to discovery — the dashboard's data entry point."""
from __future__ import annotations

from pathlib import Path

import streamlit as st
from ui.services.menu import MenuIndex


def ui_results_root() -> Path:
    """The single results root the menu and loader both read from.

    ``VERIFICATION_DATA_DIR`` (or ``~/.africlimateai/data``) + the deployment domain
    (``config.domain()`` — the country lowercased, else ``africa``). Defined once so
    discovery and loading can never drift to different roots, and so a national deployment
    reads exactly the domain its own ``verification.run`` wrote.
    """
    from apps.config.data_path import DATA_DIR
    from apps.verification import config
    return DATA_DIR / config.domain()


@st.cache_resource(show_spinner=False)
def get_menu_index() -> MenuIndex:
    """The discovery-backed menu source, built once per process.

    Scans :func:`ui_results_root` — the same root the loader reads — so the
    menu can only offer runs the loader can actually open. New runs surface
    on the next app restart / cache clear.
    """
    return MenuIndex.from_discovery(root=ui_results_root())
