# ui/pages/benchmark_console.py

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from apps.benchmarking.benchmarking import summarize_benchmark_dataset

from apps.benchmarking.dashboard import (
    ROMP_DEMO_FIG_DIR,
    ROMP_DEMO_OUT_DIR,
    build_fair_plan_safe,
    build_provenance,
    get_model_catalog,
    get_obs_catalog,
    get_output_summary,
    get_phase_status,
    load_inventory_summary,
    load_verification_scores,
    run_momp_model,
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

    extra_css = """
    <style>
    .dash-chip-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin: 0.75rem 0;
    }
    .dash-muted {
        color: var(--color-slate-500);
        font-size: 0.85rem;
    }
    </style>
    """

    st.markdown(f"<style>{core_css}{extra_css}</style>", unsafe_allow_html=True)


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


def status_chip(label: str, ok: bool) -> str:
    kind = "success" if ok else "idle"
    return f'<span class="status-chip {kind}">{label}</span>'


def render_phase_chips(phase_status: dict) -> None:
    chips = []

    for label, ok in phase_status.items():
        chips.append(status_chip(label, ok))

    st.markdown(
        f'<div class="dash-chip-row">{"".join(chips)}</div>',
        unsafe_allow_html=True,
    )


# ======================================================================
# Cached loaders
# ======================================================================
@st.cache_data(show_spinner=False, ttl=300)
def cached_output_summary():
    return get_output_summary()


@st.cache_data(show_spinner=False, ttl=300)
def cached_inventory_summary():
    return load_inventory_summary()


@st.cache_data(show_spinner=False, ttl=300)
def cached_verification_scores(fair_windows: tuple[str, ...] | None):
    return load_verification_scores(fair_windows)


# ======================================================================
# Page sections
# ======================================================================
def render_overview_tab() -> None:
    phase_status = get_phase_status()
    output_summary = cached_output_summary()
    catalog = get_model_catalog()

    render_phase_chips(phase_status)

    kpi_cols = st.columns(4)

    kpi_cols[0].metric("Configured models", len(catalog))
    kpi_cols[1].metric("Figures", output_summary.get("figures", 0))
    kpi_cols[2].metric("NetCDF outputs", output_summary.get("netcdf", 0))
    kpi_cols[3].metric(
        "Latest output",
        output_summary.get("latest_file") or "None",
    )

    st.divider()

    section_header("Quick status", "teal")

    inventory = cached_inventory_summary()

    if inventory.get("available"):
        summary = inventory.get("summary", {})

        readiness_cols = st.columns(4)

        readiness_cols[0].metric("Ready", summary.get("Ready", 0))
        readiness_cols[1].metric("Warning", summary.get("Warning", 0))
        readiness_cols[2].metric("No data", summary.get("No data", 0))
        readiness_cols[3].metric("Error", summary.get("Error", 0))

    else:
        info_box(
            "Phase 2 inventory module is not available. "
            "Add apps/benchmarking/inventory.py to enable data readiness reporting.",
            "amber",
        )

    fair_plan = st.session_state.get("fair_comparison_plan")

    if fair_plan:
        st.success("A fair-comparison plan is active in this session.")

        if fair_plan.get("evaluation_years"):
            st.write(
                f"Evaluation years: {len(fair_plan['evaluation_years'])} "
                f"({fair_plan['evaluation_years'][0]}–{fair_plan['evaluation_years'][-1]})"
            )

    verification = cached_verification_scores(None)

    if verification.get("available") and not verification["scores"].empty:
        scores = verification["scores"]

        st.success("Verification scores are available.")

        st.write(
            f"Models with scores: {scores['model'].nunique()} | "
            f"Metrics: {scores['metric'].nunique()} | "
            f"Rows: {len(scores)}"
        )

    else:
        info_box(
            "No verification scores are currently available. "
            "Run a benchmark or install Phase 4 to enable score reporting.",
            "amber",
        )


def render_run_center_tab() -> None:
    section_header("Run Center", "green")

    catalog = get_model_catalog()

    if not catalog:
        info_box("No models found in config.BENCHMARK_MODEL_CATALOG.", "amber")
        return

    model_options = {
        key: meta.get("label", key)
        for key, meta in catalog.items()
    }

    selected_models = st.multiselect(
        "Models to run",
        list(model_options.keys()),
        format_func=lambda key: model_options[key],
        key="dash_selected_models",
    )

    if selected_models:
        selected_meta = [catalog[key] for key in selected_models]

        categories = sorted({meta.get("category", "Other") for meta in selected_meta})

        st.markdown(
            f'<div class="dash-muted">Selected categories: {", ".join(categories)}</div>',
            unsafe_allow_html=True,
        )

    spec_path = st.text_input(
        "Optional benchmark spec path",
        value="",
        help="Optional path to benchmark_spec.json. If blank, run_momp defaults are used.",
        key="dash_spec_path",
    )

    run_clicked = st.button(
        "Run selected models",
        type="primary",
        width="stretch",
        key="dash_run_button",
    )

    if run_clicked:
        if not selected_models:
            st.warning("Select at least one model to run.")
        else:
            results = []

            with st.spinner("Running selected benchmark models..."):
                for model_key in selected_models:
                    meta = catalog[model_key]

                    mode = "prob" if meta.get("probabilistic", False) else "det"

                    result = run_momp_model(
                        mode=mode,
                        model_key=model_key,
                        spec_path=spec_path.strip() or None,
                    )

                    results.append(result)

            st.session_state["dashboard_run_results"] = results
            st.rerun()

    results = st.session_state.get("dashboard_run_results", [])

    if results:
        st.divider()

        section_header("Last run results", "amber")

        for result in results:
            model_label = result.get("model", "model")
            mode_label = result.get("mode", "")

            if result.get("ok"):
                st.markdown(
                    f'<span class="status-chip success">{model_label} ({mode_label}): Completed</span>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<span class="status-chip error">{model_label} ({mode_label}): Failed</span>',
                    unsafe_allow_html=True,
                )

            with st.expander(f"Log: {model_label}", expanded=False):
                st.code(result.get("stderr", "") or result.get("stdout", ""))


def render_data_readiness_tab() -> None:
    section_header("Data Readiness", "teal")

    inventory = cached_inventory_summary()

    if not inventory.get("available"):
        info_box(
            "Phase 2 inventory module is not available. "
            "Add apps/benchmarking/inventory.py to enable this tab.",
            "amber",
        )
        return

    summary = inventory.get("summary", {})

    readiness_cols = st.columns(4)

    readiness_cols[0].metric("Ready", summary.get("Ready", 0))
    readiness_cols[1].metric("Warning", summary.get("Warning", 0))
    readiness_cols[2].metric("No data", summary.get("No data", 0))
    readiness_cols[3].metric("Error", summary.get("Error", 0))

    st.divider()

    readiness_tabs = st.tabs(["Model inventory", "Observation inventory"])

    with readiness_tabs[0]:
        model_df = inventory.get("model_df", pd.DataFrame())

        if model_df.empty:
            info_box("No model inventory rows found.", "amber")
        else:
            st.dataframe(
                model_df,
                width="stretch",
                hide_index=True,
            )

    with readiness_tabs[1]:
        obs_df = inventory.get("obs_df", pd.DataFrame())

        if obs_df.empty:
            info_box("No observation inventory rows found.", "amber")
        else:
            st.dataframe(
                obs_df,
                width="stretch",
                hide_index=True,
            )


def render_fair_comparison_tab() -> None:
    section_header("Fair Comparison", "violet")

    phase_status = get_phase_status()

    if not phase_status.get("Phase 3: Fair Comparison", False):
        info_box(
            "Phase 3 fairness module is not available. "
            "Add apps/benchmarking/fairness.py to enable this tab.",
            "amber",
        )
        return

    selected_models = st.session_state.get("dash_selected_models", [])

    if not selected_models:
        info_box(
            "No models selected. Select models in the Run Center tab first.",
            "amber",
        )
        return

    obs_catalog = get_obs_catalog()

    obs_options = list(obs_catalog.keys()) or ["ENACTS", "CHIRPS"]

    control_cols = st.columns(3)

    ground_truth = control_cols[0].selectbox(
        "Ground truth",
        obs_options,
        key="dash_fair_ground_truth",
    )

    forecast_window = control_cols[1].selectbox(
        "Forecast window",
        ["1-15 days", "16-30 days", "Both"],
        key="dash_fair_forecast_window",
    )

    benchmark_coverage = control_cols[2].text_input(
        "Benchmark coverage",
        value="Ethiopia (Default)",
        key="dash_fair_coverage",
    )

    build_clicked = st.button(
        "Build fair comparison plan",
        type="primary",
        width="stretch",
        key="dash_build_fair_plan",
    )

    if build_clicked:
        with st.spinner("Building fair comparison plan..."):
            result = build_fair_plan_safe(
                selected_model_keys=selected_models,
                ground_truth=ground_truth,
                forecast_window=forecast_window,
                benchmark_coverage=benchmark_coverage,
            )

        if not result.get("available"):
            st.error(f"Fair comparison engine error: {result.get('error')}")
            return

        plan = result.get("plan")

        st.session_state["fair_comparison_plan"] = result.get("plan_dict", {})
        st.session_state["fair_comparison_df"] = result.get("plan_df", pd.DataFrame())

        if plan is not None:
            if plan.valid:
                st.success("Fair comparison plan is valid.")
            else:
                st.error("Fair comparison plan has blocking issues.")

            for issue in plan.issues:
                st.error(issue)

            for warning in plan.warnings:
                st.warning(warning)

    plan_dict = st.session_state.get("fair_comparison_plan")
    plan_df = st.session_state.get("fair_comparison_df")

    if plan_dict:
        st.divider()

        section_header("Active fair plan", "green")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Evaluation years",
            len(plan_dict.get("evaluation_years", [])),
        )

        if plan_dict.get("common_year_min") and plan_dict.get("common_year_max"):
            common_window = (
                f"{plan_dict['common_year_min']}–{plan_dict['common_year_max']}"
            )
        else:
            common_window = "None"

        c2.metric("Common window", common_window)
        c3.metric("Climatology policy", plan_dict.get("climatology_policy", ""))
        c4.metric("Ground truth", plan_dict.get("ground_truth", ""))

        if isinstance(plan_df, pd.DataFrame) and not plan_df.empty:
            st.dataframe(
                plan_df,
                width="stretch",
                hide_index=True,
            )


def render_verification_tab() -> None:
    section_header("Verification Engine", "sky")

    phase_status = get_phase_status()

    fair_plan = st.session_state.get("fair_comparison_plan")

    fair_windows = None

    if isinstance(fair_plan, dict):
        windows = fair_plan.get("forecast_windows")

        if windows:
            fair_windows = tuple(windows)

    verification = cached_verification_scores(fair_windows)

    if verification.get("available"):
        scores = verification.get("scores", pd.DataFrame())

        if scores.empty:
            info_box("No verification scores found.", "amber")
            return

        filter_cols = st.columns(3)

        category_options = ["All"] + sorted(scores["category"].unique().tolist())

        category_filter = filter_cols[0].selectbox(
            "Category",
            category_options,
            key="dash_verification_category",
        )

        lead_options = ["All"] + sorted(scores["lead_window"].unique().tolist())

        lead_filter = filter_cols[1].selectbox(
            "Lead window",
            lead_options,
            key="dash_verification_lead",
        )

        model_options = ["All"] + sorted(scores["model"].unique().tolist())

        model_filter = filter_cols[2].selectbox(
            "Model",
            model_options,
            key="dash_verification_model",
        )

        filtered_scores = scores.copy()

        if category_filter != "All":
            filtered_scores = filtered_scores[
                filtered_scores["category"] == category_filter
            ]

        if lead_filter != "All":
            filtered_scores = filtered_scores[
                filtered_scores["lead_window"] == lead_filter
            ]

        if model_filter != "All":
            filtered_scores = filtered_scores[
                filtered_scores["model"] == model_filter
            ]

        st.dataframe(
            filtered_scores,
            width="stretch",
            hide_index=True,
        )

        st.download_button(
            "Download verification scores CSV",
            data=filtered_scores.to_csv(index=False).encode(),
            file_name="verification_scores.csv",
            mime="text/csv",
            key="dash_download_verification_scores",
        )

        return

    info_box(
        "Phase 4 verification module is not available. "
        "Falling back to legacy scoreboard if available.",
        "amber",
    )

    try:
        from apps.benchmarking.benchmarking import runtime_benchmark_scoreboard

        legacy_scores = runtime_benchmark_scoreboard()

        if legacy_scores.empty:
            info_box("No legacy scoreboard results found.", "amber")
        else:
            st.dataframe(
                legacy_scores,
                width="stretch",
                hide_index=True,
            )

    except Exception as exc:
        info_box(f"Legacy scoreboard unavailable: {exc}", "rose")


def render_uncertainty_tab() -> None:
    section_header("Uncertainty & Statistical Significance", "amber")

    phase_status = get_phase_status()

    if not phase_status.get("Phase 5: Uncertainty", False):
        info_box(
            "Phase 5 uncertainty module is not available. "
            "Add apps/benchmarking/uncertainty.py to enable this tab.",
            "amber",
        )
        return

    try:
        from apps.benchmarking.uncertainty import (
            compute_metric_uncertainty,
            compute_significance_vs_baseline,
        )

    except Exception as exc:
        info_box(f"Could not import uncertainty module: {exc}", "rose")
        return

    control_cols = st.columns(4)

    n_bootstrap = control_cols[0].number_input(
        "Bootstrap samples",
        min_value=100,
        max_value=5000,
        value=1000,
        step=100,
        key="dash_uncertainty_n_bootstrap",
    )

    ci_level = control_cols[1].select_slider(
        "Confidence level",
        options=[0.80, 0.90, 0.95, 0.99],
        value=0.95,
        key="dash_uncertainty_ci_level",
    )

    seed = control_cols[2].number_input(
        "Random seed",
        min_value=0,
        max_value=999999,
        value=42,
        step=1,
        key="dash_uncertainty_seed",
    )

    baseline_model = control_cols[3].text_input(
        "Baseline model",
        value="Climatology",
        key="dash_uncertainty_baseline",
    )

    compute_clicked = st.button(
        "Compute uncertainty",
        type="primary",
        width="stretch",
        key="dash_compute_uncertainty",
    )

    fair_plan = st.session_state.get("fair_comparison_plan")

    fair_windows = None

    if isinstance(fair_plan, dict):
        windows = fair_plan.get("forecast_windows")

        if windows:
            fair_windows = tuple(windows)

    if compute_clicked:
        with st.spinner("Computing bootstrap uncertainty and significance..."):
            uncertainty_df = compute_metric_uncertainty(
                output_dir=ROMP_DEMO_OUT_DIR,
                n_bootstrap=int(n_bootstrap),
                ci_level=float(ci_level),
                seed=int(seed),
                fair_windows=fair_windows,
            )

            significance_df = compute_significance_vs_baseline(
                output_dir=ROMP_DEMO_OUT_DIR,
                baseline_model=baseline_model,
                n_bootstrap=int(n_bootstrap),
                ci_level=float(ci_level),
                seed=int(seed),
                fair_windows=fair_windows,
            )

        st.session_state["dash_uncertainty_results"] = {
            "uncertainty": uncertainty_df,
            "significance": significance_df,
            "params": {
                "n_bootstrap": int(n_bootstrap),
                "ci_level": float(ci_level),
                "seed": int(seed),
                "baseline_model": baseline_model,
            },
        }

        st.rerun()

    results = st.session_state.get("dash_uncertainty_results")

    if not results:
        info_box(
            "Click **Compute uncertainty** to generate bootstrap confidence "
            "intervals and significance diagnostics.",
            "sky",
        )
        return

    uncertainty_df = results.get("uncertainty", pd.DataFrame())
    significance_df = results.get("significance", pd.DataFrame())

    uncertainty_tabs = st.tabs(
        [
            "Metric uncertainty",
            "Significance vs baseline",
            "Raw tables",
        ]
    )

    with uncertainty_tabs[0]:
        if uncertainty_df.empty:
            info_box("No uncertainty results available.", "amber")
        else:
            st.dataframe(
                uncertainty_df,
                width="stretch",
                hide_index=True,
            )

    with uncertainty_tabs[1]:
        if significance_df.empty:
            info_box("No significance results available.", "amber")
        else:
            st.dataframe(
                significance_df,
                width="stretch",
                hide_index=True,
            )

    with uncertainty_tabs[2]:
        st.markdown("**Uncertainty table**")
        st.dataframe(uncertainty_df, width="stretch", hide_index=True)

        st.markdown("**Significance table**")
        st.dataframe(significance_df, width="stretch", hide_index=True)


def render_outputs_reference_tab() -> None:
    section_header("Outputs & Reference Library", "teal")

    output_summary = cached_output_summary()

    output_kpis = st.columns(3)

    output_kpis[0].metric("Figures", output_summary.get("figures", 0))
    output_kpis[1].metric("NetCDF outputs", output_summary.get("netcdf", 0))
    output_kpis[2].metric("Latest file", output_summary.get("latest_file") or "None")

    st.divider()

    output_tabs = st.tabs(["Figures", "Datasets"])

    with output_tabs[0]:
        figure_paths = output_summary.get("figure_paths", [])

        if not figure_paths:
            info_box("No PNG figures found.", "amber")
        else:
            figure_path = st.selectbox(
                "Figure",
                figure_paths,
                format_func=lambda p: Path(p).name,
                key="dash_figure_select",
            )

            st.image(str(figure_path), width="stretch", caption=Path(figure_path).name)

    with output_tabs[1]:
        netcdf_paths = output_summary.get("netcdf_paths", [])

        if not netcdf_paths:
            info_box("No NetCDF outputs found.", "amber")
        else:
            netcdf_path = st.selectbox(
                "Dataset",
                netcdf_paths,
                format_func=lambda p: Path(p).name,
                key="dash_netcdf_select",
            )

            try:
                summary = summarize_benchmark_dataset(netcdf_path)

                variables = summary.get("variables", [])
                means = summary.get("means", {})

                st.write("Variables:", ", ".join(variables) if variables else "None")

                if means:
                    means_df = pd.DataFrame(
                        list(means.items()),
                        columns=["Variable", "Mean"],
                    )

                    st.dataframe(
                        means_df,
                        width="stretch",
                        hide_index=True,
                    )

            except Exception as exc:
                info_box(f"Could not open dataset: {exc}", "rose")


def render_provenance_export_tab() -> None:
    section_header("Provenance & Export", "violet")

    provenance = build_provenance()

    st.json(provenance)

    st.download_button(
        "Download provenance JSON",
        data=json.dumps(provenance, indent=2).encode(),
        file_name="benchmark_provenance.json",
        mime="application/json",
        key="dash_download_provenance",
    )

    st.divider()

    section_header("Export available tables", "green")

    inventory = cached_inventory_summary()

    if inventory.get("available"):
        model_df = inventory.get("model_df", pd.DataFrame())
        obs_df = inventory.get("obs_df", pd.DataFrame())

        if not model_df.empty:
            st.download_button(
                "Download model inventory CSV",
                data=model_df.to_csv(index=False).encode(),
                file_name="model_inventory.csv",
                mime="text/csv",
                key="dash_download_model_inventory",
            )

        if not obs_df.empty:
            st.download_button(
                "Download observation inventory CSV",
                data=obs_df.to_csv(index=False).encode(),
                file_name="observation_inventory.csv",
                mime="text/csv",
                key="dash_download_obs_inventory",
            )

    verification = cached_verification_scores(None)

    if verification.get("available"):
        scores = verification.get("scores", pd.DataFrame())

        if not scores.empty:
            st.download_button(
                "Download verification scores CSV",
                data=scores.to_csv(index=False).encode(),
                file_name="verification_scores.csv",
                mime="text/csv",
                key="dash_download_verification_scores_provenance",
            )

    fair_plan = st.session_state.get("fair_comparison_plan")

    if fair_plan:
        st.download_button(
            "Download fair comparison plan JSON",
            data=json.dumps(fair_plan, indent=2, default=str).encode(),
            file_name="fair_comparison_plan.json",
            mime="application/json",
            key="dash_download_fair_plan",
        )


# ======================================================================
# Main page
# ======================================================================
def page() -> None:
    _inject_theme_css()

    st.markdown(
        '<div class="page-title">Benchmark Console</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="page-subtitle">'
        "Professional consolidated dashboard for data readiness, fair comparison, "
        "verification, uncertainty, execution, and provenance."
        "</div>",
        unsafe_allow_html=True,
    )

    tabs = st.tabs(
        [
            "Overview",
            "Run Center",
            "Data Readiness",
            "Fair Comparison",
            "Verification",
            "Uncertainty",
            "Outputs & Reference",
            "Provenance & Export",
        ]
    )

    with tabs[0]:
        render_overview_tab()

    with tabs[1]:
        render_run_center_tab()

    with tabs[2]:
        render_data_readiness_tab()

    with tabs[3]:
        render_fair_comparison_tab()

    with tabs[4]:
        render_verification_tab()

    with tabs[5]:
        render_uncertainty_tab()

    with tabs[6]:
        render_outputs_reference_tab()

    with tabs[7]:
        render_provenance_export_tab()


if __name__ == "__main__":
    page()