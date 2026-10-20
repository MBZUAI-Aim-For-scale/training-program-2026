# ui/pages/data_readiness.py

from __future__ import annotations

from pathlib import Path

import streamlit as st

from apps.benchmarking.inventory import (
    build_model_inventory,
    build_obs_inventory,
    filter_rows_by_category,
    inventory_to_dataframe,
    readiness_summary,
)


# ======================================================================
# UI helpers
# ======================================================================
def _inject_theme_css() -> None:
    css_path = Path("ui/components/styles/page_styles/benchmarking.css")

    if css_path.exists():
        core_css = css_path.read_text()
    else:
        core_css = "/* Warning: benchmarking.css not found. */"

    st.markdown(f"<style>{core_css}</style>", unsafe_allow_html=True)


def info_box(text: str, kind: str = "") -> None:
    st.markdown(f'<div class="ibox {kind}">{text}</div>', unsafe_allow_html=True)


def section_header(text: str, accent: str = "sky") -> None:
    st.markdown(
        f"""
        <div class="sec-hd {accent}">
            <span class="sec-hd-accent"></span>
            {text}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ======================================================================
# Cached inventory loaders
# ======================================================================
@st.cache_data(show_spinner=False, ttl=300)
def cached_model_inventory():
    return build_model_inventory()


@st.cache_data(show_spinner=False, ttl=300)
def cached_obs_inventory():
    return build_obs_inventory()


# ======================================================================
# Page
# ======================================================================
def page() -> None:
    _inject_theme_css()

    st.markdown(
        '<div class="page-title">Data Inventory & Readiness</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="page-subtitle">'
        "Inspect model and observation data availability before running benchmarks."
        "</div>",
        unsafe_allow_html=True,
    )

    header_cols = st.columns([3, 1])

    with header_cols[1]:
        if st.button(
            "Refresh inventory",
            width="stretch",
            key="btn_refresh_inventory",
        ):
            cached_model_inventory.clear()
            cached_obs_inventory.clear()
            st.rerun()

    model_rows = cached_model_inventory()
    obs_rows = cached_obs_inventory()

    all_rows = model_rows + obs_rows
    summary = readiness_summary(all_rows)

    kpi_cols = st.columns(4)

    kpi_cols[0].metric("Ready", summary.get("Ready", 0))
    kpi_cols[1].metric("Warning", summary.get("Warning", 0))
    kpi_cols[2].metric("No data", summary.get("No data", 0))
    kpi_cols[3].metric("Error", summary.get("Error", 0))

    # ==================================================================
    # Model inventory
    # ==================================================================
    section_header("Model data readiness", "teal")

    category_filter = st.radio(
        "Model category",
        ["All", "Deterministic", "Probabilistic"],
        horizontal=True,
        key="inventory_model_category_filter",
    )

    filtered_model_rows = filter_rows_by_category(model_rows, category_filter)

    model_df = inventory_to_dataframe(filtered_model_rows)

    if model_df.empty:
        info_box("No model sources detected.", "amber")
    else:
        st.dataframe(
            model_df,
            width="stretch",
            hide_index=True,
        )

        with st.expander("Model issues", expanded=False):
            has_issue = False

            for row in filtered_model_rows:
                issues = row.get("issues", [])

                if issues:
                    has_issue = True

                    st.markdown(f"**{row.get('label', row.get('key'))}**")

                    for issue in issues:
                        st.markdown(f"- {issue}")

            if not has_issue:
                st.markdown("No issues detected.")

        with st.expander("Detected variables and coordinates", expanded=False):
            for row in filtered_model_rows:
                st.markdown(f"**{row.get('label', row.get('key'))}**")

                variables = row.get("variables", [])
                dimensions = row.get("dimensions", [])
                coordinates = row.get("coordinates", [])

                if variables:
                    st.write("Variables:", ", ".join(variables))
                else:
                    st.write("Variables: not detected")

                if dimensions:
                    st.write("Dimensions:", ", ".join(dimensions))
                else:
                    st.write("Dimensions: not detected")

                if coordinates:
                    st.write("Coordinates:", ", ".join(coordinates))
                else:
                    st.write("Coordinates: not detected")

                if row.get("lat_range"):
                    st.write(
                        f"Latitude range: {row['lat_range'][0]:.3f} to {row['lat_range'][1]:.3f}"
                    )

                if row.get("lon_range"):
                    st.write(
                        f"Longitude range: {row['lon_range'][0]:.3f} to {row['lon_range'][1]:.3f}"
                    )

                st.divider()

    # ==================================================================
    # Observation inventory
    # ==================================================================
    section_header("Observation / ground-truth readiness", "violet")

    obs_df = inventory_to_dataframe(obs_rows)

    if obs_df.empty:
        info_box("No observation sources detected.", "amber")
    else:
        st.dataframe(
            obs_df,
            width="stretch",
            hide_index=True,
        )

        with st.expander("Observation issues", expanded=False):
            has_issue = False

            for row in obs_rows:
                issues = row.get("issues", [])

                if issues:
                    has_issue = True

                    st.markdown(f"**{row.get('label', row.get('key'))}**")

                    for issue in issues:
                        st.markdown(f"- {issue}")

            if not has_issue:
                st.markdown("No issues detected.")


if __name__ == "__main__":
    page()