"""Onset blending controls and results using the operational pipeline."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

from ai_weather import config
from ai_weather.background_task import BackgroundTask
from ai_weather.benchmark_job import BenchmarkJob
from ai_weather.blending import blending_defaults, blending_outputs, run_blending
from ai_weather.blending_config import runtime_spec_defaults, validate_runtime_overrides
from ai_weather.blending_inputs import input_check_matches, inspect_blending_inputs
from ui.components.cards import section_header
from ui.components.styles.loader import load_page_css


@st.fragment(run_every=1)
def render_blending_job():
    job = st.session_state.get("blending_active_job")
    if job is None:
        return
    snapshot = job.snapshot()
    if snapshot["done"]:
        st.session_state["blending_last_run"] = {
            **snapshot,
            "signature": job.signature,
            "settings": st.session_state.pop("blending_run_settings"),
        }
        st.session_state.pop("blending_active_job")
        st.rerun()
        return
    st.info("Stopping onset blending…" if snapshot["stopping"] else "Onset blending running.")
    steps = [
        int(match.group(1))
        for line in snapshot["logs"]
        if (match := re.search(r"Step (\d+)/8:", line))
    ]
    st.progress((max(steps, default=1) - 1) / 8, text="Preparing and blending forecasts")
    if st.button("Stop Blending", disabled=snapshot["stopping"], key="blending_stop"):
        job.cancel()
    if snapshot["logs"]:
        st.code("\n".join(snapshot["logs"][-20:]), language="text")


def render_results(settings):
    outputs = blending_outputs(settings)
    st.caption(f"Saved outputs for {settings['issue_date']} · {outputs['root']}")
    if st.button("Refresh saved results", key="blending_refresh"):
        st.rerun()
    probabilities, maps, downloads = st.tabs(
        ["Onset Probabilities", "Forecast Maps", "Data & Export"],
        key="blending_result_tabs", on_change="rerun",
    )
    with probabilities:
        if probabilities.open:
            summary = outputs["summary"]
            if summary.is_file():
                try:
                    table = pd.read_csv(summary)
                    st.dataframe(table, hide_index=True, width="stretch")
                    bins = ["week1", "week2", "week3", "week4", "later"]
                    if not table.empty and set(bins).issubset(table.columns):
                        st.caption(
                            "Mean probability across districts. Week 1 begins the day after issue date."
                        )
                        st.bar_chart(table[bins].mean())
                    st.caption("Columns named prob_clim_mr_* are model inputs in logit units.")
                except (OSError, ValueError) as exc:
                    st.error(f"Could not load onset probabilities: {exc}")
            else:
                st.info("Run onset blending to generate probabilities for this issue date.")
    with maps:
        if maps.open:
            available = [path for path in outputs["maps"] if path.is_file()]
            if not available:
                st.info("No forecast maps are available for this issue date.")
            for path in available:
                st.image(str(path), caption=path.stem.replace("_", " ").title(), width="stretch")
    with downloads:
        if downloads.open:
            for name in ("summary", "predictions"):
                path = outputs[name]
                if path.is_file():
                    st.download_button(
                        f"Download {name} CSV",
                        path.read_bytes(),
                        file_name=path.name,
                        mime="text/csv",
                        key=f"blending_{name}_csv",
                    )
            for path in outputs["maps"]:
                if path.is_file():
                    st.download_button(
                        f"Download {path.stem}",
                        path.read_bytes(),
                        file_name=path.name,
                        mime="image/png",
                        key=f"blending_{path.stem}",
                    )


PARAMETER_LABELS = {
    "file_regex": "Filename regular expression",
    "value_col": "Rainfall variable",
    "wide_day_dim": "Lead-day dimension",
    "wide_prefix": "Rainfall column prefix",
    "parallel": "Parallel input processing",
    "workers": "Input workers",
    "rename": "Dimension mappings",
    "window": "Onset accumulation window (days)",
    "wet_day_min_mm": "Wet-day minimum rainfall (mm)",
    "follow_days": "Follow-up period (days)",
    "mode": "Dry-spell definition",
    "min_dry_days": "Consecutive dry days",
    "dry_day_min_mm": "Dry-day threshold (mm)",
    "sum_window": "Dry-spell sum window (days)",
    "sum_min_mm": "Minimum dry-spell accumulation (mm)",
    "min_day": "Minimum lead day",
    "max_day": "Maximum lead day",
    "file": "Reference file",
    "lat_col": "Latitude column",
    "lon_col": "Longitude column",
    "adm3_col": "District identifier column",
    "thresh_col": "Onset threshold column",
    "year_col": "Year column",
    "day_col": "MOK day column",
    "base_date": "MOK base date (MM-DD)",
    "dissemination_cells_file": "District selection CSV",
    "max_number": "Maximum ensemble member number",
    "train_year_min": "First climatology training year",
    "train_year_max": "Last climatology training year",
    "season_start_md": "Season start (MM-DD)",
    "issue_end_md": "Last issue day (MM-DD)",
    "forecast_window": "Climatology forecast window (days)",
    "min_onset_years": "Minimum onset observations",
    "conditional": "Condition on onset not yet occurring",
    "cv_by_year": "Leave-year-out climatology",
    "onset_col": "Historical onset column",
    "out_stem": "Climatology output stem",
    "join": "Dataset join",
    "trim_forecasts_after_true_onset": "Exclude forecasts after observed onset",
    "default_prob_col": "Default forecast probability column",
    "prob_col": "Probability column",
    "plus_label_template": "Later-bin label template",
    "constants": "Constant field mappings",
    "daily": "Daily field mappings",
    "country_boundary": "Country boundary file",
    "district_boundary": "District boundary file",
    "zoom_to_data": "Zoom maps to district coverage",
    "onset_file": "Observed-onset district CSV (optional)",
}


def _render_parameter_tree(values, prefix, errors, path=(), section=None):
    """Render config-derived leaves while keeping complex mappings editable as YAML."""
    result = {}
    for name, value in values.items():
        location = (*path, name)
        key = f"{prefix}_{'_'.join(location)}"
        label = PARAMETER_LABELS.get(name, name.replace("_", " ").capitalize())
        reserved_help = None
        if path == ("input",) and name in ("parallel", "workers"):
            reserved_help = (
                "Reserved setting; the current raw reader processes files sequentially."
            )
        if section == "combine" and (
            (path == ("options",) and name in ("default_prob_col", "plus_label_template"))
            or (path == ("climatology",) and name in ("max_day", "prob_col"))
        ):
            reserved_help = "Legacy setting; combination uses explicit daily mappings and infers later-bin labels."
        control = {"disabled": reserved_help is not None, "help": reserved_help}
        if isinstance(value, dict) and name != "rename":
            st.markdown(f"**{label}**")
            nested = copy.deepcopy(value)
            if name == "dry_spell":
                # Both modes are supported by the existing onset detector.
                nested.setdefault("sum_window", 10)
                nested.setdefault("sum_min_mm", 5.0)
            result[name] = _render_parameter_tree(nested, prefix, errors, location, section)
        elif isinstance(value, (dict, list)) or value is None:
            if key not in st.session_state:
                st.session_state[key] = yaml.safe_dump(value, sort_keys=False)
            text = st.text_area(
                label,
                key=key,
                help="Edit the configured field mappings as YAML.",
            )
            try:
                parsed = yaml.safe_load(text)
                if value is not None and not isinstance(parsed, type(value)):
                    raise ValueError(f"Expected a {type(value).__name__}")
                result[name] = parsed
            except (yaml.YAMLError, ValueError) as exc:
                errors.append(f"{'.'.join(location)}: {exc}")
                result[name] = value
        elif type(value) is bool:
            if key not in st.session_state:
                st.session_state[key] = value
            result[name] = st.checkbox(label, key=key, **control)
        elif type(value) is int:
            if key not in st.session_state:
                st.session_state[key] = value
            result[name] = st.number_input(label, step=1, key=key, **control)
        elif isinstance(value, float):
            if key not in st.session_state:
                st.session_state[key] = value
            result[name] = st.number_input(label, step=0.1, key=key, **control)
        elif name == "mode" and "dry_spell" in path:
            options = ["consecutive_dry", "window_sum"]
            if key not in st.session_state:
                st.session_state[key] = value
            result[name] = st.selectbox(label, options, key=key)
        elif name == "join":
            options = ["inner", "full"]
            if key not in st.session_state:
                st.session_state[key] = value
            result[name] = st.selectbox(label, options, key=key)
        else:
            if key not in st.session_state:
                st.session_state[key] = str(value)
            result[name] = st.text_input(label, key=key, **control)
    return result


def _selection_inputs(context, defaults):
    """Remember dependent inputs per model/year without resetting unrelated settings."""
    previous = st.session_state.get("blending_input_context")
    saved = st.session_state.setdefault("blending_context_inputs", {})
    if previous != context:
        if previous is not None:
            saved[previous] = {
                key: st.session_state.get(key, value) for key, value in defaults.items()
            }
        for key, value in saved.get(context, defaults).items():
            st.session_state[key] = value
        st.session_state["blending_input_context"] = context


def render_configuration(defaults, settings, context, errors):
    sections = copy.deepcopy(defaults)
    # Folder overrides have one control each, shared with the CLI inputs above.
    sections["single"]["input"].pop("nc_folder", None)
    sections["ensemble"]["input"].pop("nc_folder", None)
    titles = {
        "single": "AI model: NetCDF, onset definition and references",
        "ensemble": "Companion model: NetCDF, onset definition and references",
        "climatology": "Climatology: training periods and forecast options",
        "combine": "Dataset combination and field mappings",
        "maps": "Maps: boundaries and observed onset",
    }
    rendered = {}
    for name, values in sections.items():
        with st.expander(titles[name]):
            if name == "climatology":
                st.caption(
                    "Climatology forecast years follow the selected forecast year. Historical ground truth is set above."
                )
            if name == "combine":
                st.caption(
                    "Input filenames, years and output locations are connected automatically between stages."
                )
            rendered[name] = _render_parameter_tree(
                values, f"blending_spec_{context}_{name}", errors, section=name
            )
    rendered["single"]["input"]["nc_folder"] = settings["aifs_nc_folder"]
    rendered["ensemble"]["input"]["nc_folder"] = settings["aifs_ens_nc_folder"]
    try:
        validate_runtime_overrides(rendered)
    except ValueError as exc:
        errors.append(str(exc))
    return rendered


def _use_available_date(settings):
    """Apply an inspected date to dependent UI state without changing input data."""
    selected = date.fromisoformat(st.session_state["blending_available_date"])
    old_year = int(settings["year"])
    new_context = f"{settings['model_single']}_{settings['model_ens']}_{selected.year}"
    previous_output = str(blending_outputs({**settings, "work_dir": None})["root"])
    current_output = str(Path(settings["work_dir"]).expanduser().resolve())
    work_dir = settings["work_dir"]
    if current_output == previous_output:
        work_dir = str(
            blending_outputs({**settings, "year": selected.year, "work_dir": None})["root"]
        )
    st.session_state.setdefault("blending_context_inputs", {})[new_context] = {
        "blending_work_dir": work_dir,
        "blending_single_folder": settings["aifs_nc_folder"],
        "blending_ens_folder": settings["aifs_ens_nc_folder"],
    }
    old_context = f"{settings['model_single']}_{settings['model_ens']}_{old_year}"
    old_prefix, new_prefix = f"blending_spec_{old_context}_", f"blending_spec_{new_context}_"
    for key, value in list(st.session_state.items()):
        if key.startswith(old_prefix):
            st.session_state[new_prefix + key[len(old_prefix) :]] = copy.deepcopy(value)
    for section in ("single", "ensemble"):
        pattern = settings["runtime_overrides"][section]["input"]["file_regex"]
        key = f"blending_spec_{new_context}_{section}_input_file_regex"
        st.session_state[key] = pattern.replace(str(old_year), str(selected.year))
    st.session_state["blending_year"] = selected.year
    st.session_state["blending_date"] = selected
    st.session_state["blending_previous_year"] = selected.year
    st.session_state["blending_input_check"] = None
    st.session_state["blending_input_date_applied"] = selected.isoformat()


@st.fragment(run_every=1)
def render_input_inspection_job():
    job = st.session_state.get("blending_input_job")
    if job is None:
        return
    snapshot = job.snapshot()
    if not snapshot["done"]:
        st.info("Task running in background: reading forecast input calendars…")
        return
    if snapshot["error"]:
        st.session_state["blending_input_check"] = {
            "ok": False,
            "errors": [f"Could not inspect forecast inputs: {snapshot['error']}"],
            "records": [], "common_dates": [], "signature": "", "fingerprints": [],
        }
    else:
        st.session_state["blending_input_check"] = snapshot["result"]
    st.session_state.pop("blending_input_job")
    st.rerun()


def render_input_inspection(settings, invalid=False, active=False):
    """Inspect only on request, and require a current successful raw-input check."""
    section_header("Forecast input readiness", accent="amber")
    required = not settings["dry_run"] and any(
        settings["skip_to"] <= stage <= settings["stop_at"] for stage in (2, 3)
    )
    applied_date = st.session_state.get("blending_input_date_applied")
    if applied_date:
        st.info(
            f"Selected {applied_date}. Source files were unchanged. Inspect forecast inputs again before running preparation."
        )
    inspecting = st.session_state.get("blending_input_job") is not None
    if st.button(
        "Inspect forecast inputs", key="blending_inspect_inputs",
        disabled=invalid or active or inspecting,
    ):
        st.session_state["blending_input_job"] = BackgroundTask(
            inspect_blending_inputs, settings
        ).start()
        st.session_state.pop("blending_input_date_applied", None)
        st.rerun()
    if inspecting:
        render_input_inspection_job()
        return False
    report = st.session_state.get("blending_input_check")
    if report is None:
        if required:
            st.info("Inspect forecast inputs before running stages that process raw NetCDF files.")
        else:
            st.caption(
                "Input inspection is optional for command previews or runs that skip raw forecast processing."
            )
        return not required
    current = input_check_matches(settings, report)
    if not current:
        st.warning(
            "The input check is stale. Inspect again after changing source settings, forecast year, issue date, or input files."
        )
    elif report.get("ok"):
        st.success("Forecast inputs match the selected year and issue date.")
    for message in report.get("errors", []):
        st.error(message if current else f"Previous input check: {message}")
    if report.get("records"):
        columns = [
            "source",
            "file",
            "filename_year",
            "actual_years",
            "first_date",
            "last_date",
            "valid",
            "error",
        ]
        with st.expander("Forecast file inspection", expanded=not report.get("ok")):
            st.dataframe(
                pd.DataFrame(report["records"]).reindex(columns=columns),
                hide_index=True,
                width="stretch",
            )
    available = report.get("common_dates", [])
    if available:
        st.selectbox(
            "Available initialization date",
            available,
            key="blending_available_date",
            help="Dates present in both forecast sources. Selecting a date changes dashboard settings only.",
        )
        st.button(
            "Use available date",
            key="blending_use_available_date",
            disabled=not current or active,
            on_click=_use_available_date,
            args=(copy.deepcopy(settings),),
        )
        st.caption(
            "After applying an available date, inspect again to validate the corresponding year and filenames."
        )
    return not required or (current and bool(report.get("ok")))


def page():
    load_page_css("benchmarking")
    defaults = blending_defaults()
    st.markdown('<div class="page-title">Onset Blending</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">Blend AI rainfall forecasts with climatology '
        "to predict rainy season onset across Ethiopian districts.</div>",
        unsafe_allow_html=True,
    )
    last = st.session_state.get("blending_last_run")
    active = st.session_state.get("blending_active_job")
    status = "Running" if active else last["status"].title() if last else "Idle"
    cols = st.columns(3)
    cols[0].metric("Status", status)
    cols[1].metric("Forecast bins", "4 weeks + later")
    cols[2].metric("Pipeline stages", "8")

    section_header("Blending Inputs", accent="violet")
    left, right = st.columns(2)
    if "blending_year" not in st.session_state:
        st.session_state["blending_year"] = int(defaults["year"])
    year = left.number_input(
        "Forecast year",
        min_value=1900,
        max_value=2200,
        key="blending_year",
    )
    previous_year = st.session_state.get("blending_previous_year")
    if previous_year is not None and previous_year != year and "blending_date" in st.session_state:
        previous_date = st.session_state["blending_date"]
        if previous_date.year == previous_year:
            try:
                st.session_state["blending_date"] = previous_date.replace(year=year)
            except ValueError:
                st.session_state["blending_date"] = previous_date.replace(year=year, day=28)
    st.session_state["blending_previous_year"] = year
    if "blending_date" not in st.session_state:
        st.session_state["blending_date"] = date.fromisoformat(defaults["issue_date"]).replace(
            year=year
        )
    issue_date = right.date_input("Issue date", value=None, key="blending_date")
    model = left.selectbox(
        "AI model",
        ["aifs", "gencast", "ngcm"],
        index=["aifs", "gencast", "ngcm"].index(defaults["model_single"]),
        key="blending_model",
    )
    companions = [config.DEFAULT_MODEL_ENS] if model == "aifs" else ["aifs"]
    companion = right.selectbox("Companion model", companions, key="blending_companion")
    settings = {
        **defaults,
        "year": year,
        "issue_date": issue_date.isoformat() if issue_date else "",
        "model_single": model,
        "model_ens": companion,
    }
    stage_defaults = runtime_spec_defaults(model, year, companion)
    context = f"{model}_{companion}_{year}"
    _selection_inputs(
        context,
        {
            "blending_work_dir": str(blending_outputs({**settings, "work_dir": None})["root"]),
            "blending_single_folder": stage_defaults["single"]["input"]["nc_folder"],
            "blending_ens_folder": stage_defaults["ensemble"]["input"]["nc_folder"],
        },
    )
    with st.expander("Shared settings and input paths", expanded=True):
        left, right = st.columns(2)
        settings["region"] = left.text_input(
            "Region label", defaults["region"], key="blending_region"
        )
        settings["work_dir"] = right.text_input("Output directory", key="blending_work_dir")
        settings["coef_dir"] = left.text_input(
            "Trained coefficient directory", defaults["coef_dir"], key="blending_coef_dir"
        )
        settings["coef_tag"] = right.text_input(
            "Coefficient tag", defaults["coef_tag"], key="blending_coef_tag"
        )
        settings["blend_model"] = left.text_input(
            "Blend model", defaults["blend_model"], key="blending_formula"
        )
        settings["blend_spec"] = right.text_input(
            "Trained model formula YAML",
            key="blending_spec",
            help="Leave blank for the configured model-specific formula. Use the formula used to fit the saved coefficients.",
        )
        settings["aifs_nc_folder"] = left.text_input(
            "AI model NetCDF folder", key="blending_single_folder"
        )
        settings["aifs_ens_nc_folder"] = right.text_input(
            "Companion model NetCDF folder", key="blending_ens_folder"
        )
        settings["gt_path"] = left.text_input(
            "Historical ground truth pickle",
            str(config.ONSET_FORECAST_IMD_CLIM_SPEC["input"]["gt_path"]),
            key="blending_gt",
        )
        settings["blend_input"] = right.text_input(
            "Prepared blending input pickle (optional)",
            key="blending_input",
            help="Override the weekly-bin input path, including when resuming at model application.",
        )
    configuration_errors = []
    settings["runtime_overrides"] = render_configuration(
        stage_defaults, settings, context, configuration_errors
    )
    with st.expander("Advanced run controls"):
        left, right = st.columns(2)
        settings["skip_to"] = left.number_input(
            "Start at stage", min_value=1, max_value=8, value=1, key="blending_start"
        )
        settings["stop_at"] = right.number_input(
            "Stop after stage", min_value=1, max_value=8, value=8, key="blending_end"
        )
        settings["dry_run"] = st.checkbox("Preview commands only", key="blending_dry_run")
        st.caption("Resuming requires outputs from earlier stages in the output directory.")

    invalid = settings["stop_at"] < settings["skip_to"] or any(
        not str(settings[key]).strip()
        for key in (
            "work_dir",
            "coef_dir",
            "coef_tag",
            "blend_model",
            "region",
            "aifs_nc_folder",
            "aifs_ens_nc_folder",
            "gt_path",
        )
    )
    if invalid:
        st.error("Provide the required run and input paths, and choose a valid stage range.")
    if issue_date is None or issue_date.year != year:
        configuration_errors.append("Issue date must be within the selected forecast year.")
    for problem in configuration_errors:
        st.error(problem)
    invalid = invalid or bool(configuration_errors)
    inputs_ready = render_input_inspection(settings, invalid=invalid, active=active is not None)
    invalid = invalid or not inputs_ready
    with st.expander("Review run configuration"):
        st.json(settings)
    signature = hashlib.sha256(
        json.dumps(settings, sort_keys=True, default=str).encode()
    ).hexdigest()
    if last and last["signature"] != signature:
        st.warning(
            "Settings changed since the latest run. Saved outputs may belong to earlier settings."
        )
    if st.button(
        "Run Onset Blending",
        type="primary",
        disabled=invalid or active is not None,
        key="blending_run",
    ):
        plan = {
            "mode": "blending",
            "model_key": model,
            "label": "Onset Blending",
            "config": settings,
        }
        st.session_state["blending_run_settings"] = settings.copy()
        st.session_state["blending_active_job"] = BenchmarkJob(
            run_blending, [plan], signature,
            resource_key=str(blending_outputs(settings)["root"].resolve()),
        ).start()
        st.rerun()
    if active is not None:
        render_blending_job()
    if last:
        if last["status"] == "completed":
            preview = last["settings"].get("dry_run")
            st.success(
                "Command preview completed. No pipeline outputs were generated."
                if preview
                else "Onset blending completed successfully."
            )
        elif last["status"] == "cancelled":
            st.warning(
                "Onset blending cancelled. Outputs from the interrupted run may be incomplete."
            )
        else:
            st.error("Onset blending failed. See the execution log for the cause.")
        with st.expander("Latest blending execution log", expanded=last["status"] == "failed"):
            for result in last["runs"]:
                if result.get("command"):
                    st.code(result["command"], language="bash")
                st.code(
                    result.get("stdout") or result.get("stderr") or "No output.", language="text"
                )
    section_header("Results", accent="sky")
    render_results(settings)


if __name__ == "__main__":
    page()
