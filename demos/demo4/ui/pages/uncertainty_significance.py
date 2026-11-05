# ui/pages/uncertainty_significance.py

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from config import ROMP_DEMO_OUT_DIR

from apps.benchmarking.uncertainty import (
    compute_metric_uncertainty,
    compute_significance_vs_baseline,
)

from apps.benchmarking.verification import METRIC_REGISTRY


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


def _fair_windows_from_session() -> tuple[str, ...] | None:
    fair_plan = st.session_state.get("fair_comparison_plan")

    if isinstance(fair_plan, dict):
        windows = fair_plan.get("forecast_windows")

        if windows:
            return tuple(windows)

    return None


# ======================================================================
# Cached Phase 5 computation
# ======================================================================
@st.cache_data(show_spinner=False, ttl=600)
def cached_phase5_results(
    n_bootstrap: int,
    ci_level: float,
    seed: int,
    fair_windows: tuple[str, ...] | None,
    baseline_model: str,
):
    uncertainty_df = compute_metric_uncertainty(
        output_dir=ROMP_DEMO_OUT_DIR,
        n_bootstrap=n_bootstrap,
        ci_level=ci_level,
        seed=seed,
        fair_windows=fair_windows,
    )

    significance_df = compute_significance_vs_baseline(
        output_dir=ROMP_DEMO_OUT_DIR,
        baseline_model=baseline_model,
        n_bootstrap=n_bootstrap,
        ci_level=ci_level,
        seed=seed,
        fair_windows=fair_windows,
    )

    return uncertainty_df, significance_df


# ======================================================================
# Display helpers
# ======================================================================
def display_uncertainty_table(uncertainty_df: pd.DataFrame) -> None:
    if uncertainty_df.empty:
        info_box("No uncertainty results available.", "amber")
        return

    display_df = uncertainty_df.copy()

    for col in ["value", "lower_ci", "upper_ci"]:
        display_df[col] = display_df[col].round(4)

    display_df["ci"] = display_df.apply(
        lambda row: (
            f"[{row['lower_ci']:.4f}, {row['upper_ci']:.4f}]"
            if np.isfinite(row["lower_ci"]) and np.isfinite(row["upper_ci"])
            else "—"
        ),
        axis=1,
    )

    st.dataframe(
        display_df[
            [
                "model",
                "category",
                "lead_window",
                "metric_label",
                "value",
                "ci",
                "n_samples",
            ]
        ],
        width="stretch",
        hide_index=True,
    )


def display_significance_table(significance_df: pd.DataFrame) -> None:
    if significance_df.empty:
        info_box(
            "No significance results available. A climatology or baseline output "
            "is required.",
            "amber",
        )
        return

    display_df = significance_df.copy()

    for col in [
        "model_value",
        "baseline_value",
        "difference",
        "lower_ci",
        "upper_ci",
        "prob_better",
        "p_value",
        "p_adjusted",
    ]:
        display_df[col] = display_df[col].round(4)

    display_df["ci"] = display_df.apply(
        lambda row: (
            f"[{row['lower_ci']:.4f}, {row['upper_ci']:.4f}]"
            if np.isfinite(row["lower_ci"]) and np.isfinite(row["upper_ci"])
            else "—"
        ),
        axis=1,
    )

    display_df["verdict_display"] = display_df["verdict"].map(
        {
            "Better": "✅ Better",
            "Worse": "❌ Worse",
            "Not significant": "⚪ Not significant",
            "Unknown": "❔ Unknown",
        }
    )

    st.dataframe(
        display_df[
            [
                "model",
                "category",
                "lead_window",
                "metric_label",
                "model_value",
                "baseline_value",
                "difference",
                "ci",
                "prob_better",
                "p_adjusted",
                "verdict_display",
            ]
        ],
        width="stretch",
        hide_index=True,
    )


def plot_uncertainty_bars(
    uncertainty_df: pd.DataFrame,
    metric_key: str,
    lead_window: str,
) -> None:
    plot_df = uncertainty_df[
        (uncertainty_df["metric"] == metric_key)
        & (uncertainty_df["lead_window"] == lead_window)
    ].copy()

    if plot_df.empty:
        info_box("No data available for the selected metric and lead window.", "amber")
        return

    plot_df = plot_df.sort_values("value", ascending=False).reset_index(drop=True)

    metric_def = METRIC_REGISTRY.get(metric_key)

    unit = metric_def.unit if metric_def else ""
    label = metric_def.label if metric_def else metric_key

    fig, ax = plt.subplots(figsize=(8, max(3, 0.55 * len(plot_df))))

    y = np.arange(len(plot_df))

    values = plot_df["value"].values

    lower_err = values - plot_df["lower_ci"].values
    upper_err = plot_df["upper_ci"].values - values

    lower_err = np.where(np.isfinite(lower_err), lower_err, 0.0)
    upper_err = np.where(np.isfinite(upper_err), upper_err, 0.0)

    ax.barh(
        y,
        values,
        xerr=[lower_err, upper_err],
        capsize=4,
        color="#3B82F6",
        alpha=0.85,
    )

    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["model"].values)
    ax.invert_yaxis()

    title = f"{label} ({unit})" if unit else label

    ax.set_title(f"{title} — {lead_window}")
    ax.set_xlabel(title)

    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)


# ======================================================================
# Main page
# ======================================================================
def page() -> None:
    _inject_theme_css()

    st.markdown(
        '<div class="page-title">Uncertainty & Statistical Significance</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="page-subtitle">'
        "Bootstrap confidence intervals and significance testing for benchmark metrics."
        "</div>",
        unsafe_allow_html=True,
    )

    section_header("Settings", "teal")

    control_cols = st.columns(4)

    n_bootstrap = control_cols[0].number_input(
        "Bootstrap samples",
        min_value=100,
        max_value=5000,
        value=1000,
        step=100,
        key="phase5_n_bootstrap",
    )

    ci_level = control_cols[1].select_slider(
        "Confidence level",
        options=[0.80, 0.90, 0.95, 0.99],
        value=0.95,
        key="phase5_ci_level",
    )

    seed = control_cols[2].number_input(
        "Random seed",
        min_value=0,
        max_value=999999,
        value=42,
        step=1,
        key="phase5_seed",
    )

    baseline_model = control_cols[3].text_input(
        "Baseline model",
        value="Climatology",
        key="phase5_baseline_model",
    )

    fair_windows = _fair_windows_from_session()

    if fair_windows:
        st.caption(
            "Fair comparison windows from Phase 3 are active: "
            + ", ".join(fair_windows)
        )
    else:
        st.caption("No active fair-comparison window filter.")

    st.divider()

    run_cols = st.columns([1, 3])

    compute_clicked = run_cols[0].button(
        "Compute uncertainty",
        type="primary",
        width="stretch",
        key="btn_compute_phase5",
    )

    if compute_clicked:
        st.session_state["phase5_compute_requested"] = True

    if not st.session_state.get("phase5_compute_requested", False):
        info_box(
            "Click **Compute uncertainty** to generate bootstrap confidence "
            "intervals and significance diagnostics.",
            "sky",
        )
        return

    with st.spinner("Computing bootstrap uncertainty and significance..."):
        uncertainty_df, significance_df = cached_phase5_results(
            n_bootstrap=int(n_bootstrap),
            ci_level=float(ci_level),
            seed=int(seed),
            fair_windows=fair_windows,
            baseline_model=baseline_model,
        )

    if uncertainty_df.empty and significance_df.empty:
        info_box(
            "No uncertainty results could be computed. Ensure MOMP outputs contain "
            "MAE / FAR / Miss Rate variables.",
            "amber",
        )
        return

    tabs = st.tabs(
        [
            "Metric uncertainty",
            "Significance vs baseline",
            "Uncertainty plot",
            "Raw tables",
        ]
    )

    # ------------------------------------------------------------------
    # Tab 1: Metric uncertainty
    # ------------------------------------------------------------------
    with tabs[0]:
        section_header("Metric uncertainty", "sky")

        st.caption(
            "Bootstrap confidence intervals are currently computed by resampling "
            "finite grid cells from spatial metric outputs. For full temporal "
            "uncertainty, per-year score outputs are required."
        )

        display_uncertainty_table(uncertainty_df)

    # ------------------------------------------------------------------
    # Tab 2: Significance vs baseline
    # ------------------------------------------------------------------
    with tabs[1]:
        section_header(f"Significance versus {baseline_model}", "green")

        st.caption(
            "Positive difference means model value minus baseline value. "
            "For lower-is-better metrics, negative differences indicate improvement."
        )

        display_significance_table(significance_df)

    # ------------------------------------------------------------------
    # Tab 3: Uncertainty plot
    # ------------------------------------------------------------------
    with tabs[2]:
        section_header("Uncertainty plot", "violet")

        if uncertainty_df.empty:
            info_box("No uncertainty results available for plotting.", "amber")
        else:
            plot_control_cols = st.columns(2)

            metric_options = sorted(uncertainty_df["metric"].unique().tolist())

            metric_key = plot_control_cols[0].selectbox(
                "Metric",
                metric_options,
                format_func=lambda key: (
                    METRIC_REGISTRY[key].label
                    if key in METRIC_REGISTRY
                    else key
                ),
                key="phase5_plot_metric",
            )

            lead_options = sorted(uncertainty_df["lead_window"].unique().tolist())

            lead_window = plot_control_cols[1].selectbox(
                "Lead window",
                lead_options,
                key="phase5_plot_lead_window",
            )

            plot_uncertainty_bars(
                uncertainty_df=uncertainty_df,
                metric_key=metric_key,
                lead_window=lead_window,
            )

    # ------------------------------------------------------------------
    # Tab 4: Raw tables
    # ------------------------------------------------------------------
    with tabs[3]:
        section_header("Raw uncertainty table", "amber")
        st.dataframe(uncertainty_df, width="stretch", hide_index=True)

        section_header("Raw significance table", "amber")
        st.dataframe(significance_df, width="stretch", hide_index=True)


if __name__ == "__main__":
    page()