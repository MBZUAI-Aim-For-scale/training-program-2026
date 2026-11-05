"""Dashboard adapter for the same onset blending CLI used at the command line."""

from __future__ import annotations

import os
import shlex
import tempfile
from pathlib import Path

import yaml

from ai_weather import config
from ai_weather.blending_config import validate_runtime_overrides
from ai_weather.process_runner import spawn_process
from ai_weather.runtime import select_runtime_python

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OPTIONS = (
    "year",
    "issue_date",
    "model_single",
    "model_ens",
    "work_dir",
    "coef_dir",
    "coef_tag",
    "blend_model",
    "region",
    "aifs_nc_folder",
    "aifs_ens_nc_folder",
    "gt_path",
    "blend_input",
    "blend_spec",
    "skip_to",
    "stop_at",
    "runtime_config",
)


def blending_defaults():
    return {
        "year": config.YEAR,
        "issue_date": config.DEFAULT_ISSUE_DATE,
        "model_single": config.DEFAULT_MODEL_SINGLE,
        "work_dir": str(config.DEFAULT_WORK_DIR),
        "coef_dir": str(config.DEFAULT_COEF_DIR),
        "coef_tag": config.DEFAULT_COEF_TAG,
        "blend_model": config.DEFAULT_BLEND_MODEL,
        "region": config.DEFAULT_REGION,
    }


def build_blending_command(settings):
    unknown = set(settings) - set(OPTIONS) - {"dry_run", "runtime_overrides"}
    if unknown:
        raise ValueError(f"Unknown blending options: {sorted(unknown)}")
    command = [select_runtime_python(PROJECT_ROOT), "-u", str(PROJECT_ROOT / "run_blending.py")]
    for key in OPTIONS:
        value = settings.get(key)
        if value is not None and str(value).strip():
            command.extend([f"--{key}", str(value)])
    if settings.get("dry_run"):
        command.append("--dry_run")
    return command


def blending_outputs(settings):
    """Return only artifacts for the selected issue date and forecast year."""
    year = int(settings["year"])
    root = (
        Path(settings.get("work_dir") or Path(config.DEFAULT_WORK_DIR).parent / str(year))
        .expanduser()
        .resolve()
    )
    stamp = str(settings["issue_date"]).replace("-", "")
    return {
        "root": root,
        "summary": root / f"blend_output_summary_{stamp}.csv",
        "predictions": root
        / f"{settings.get('blend_model', config.DEFAULT_BLEND_MODEL)}_global_year{year}_preds.csv",
        "maps": [
            root / "maps" / f"prob_weeks1-4_{stamp}.png",
            root / "maps" / f"map_max_period_{stamp}.png",
        ],
    }


def run_blending(mode, model_key, settings, *, progress=None, cancel_event=None):
    """BenchmarkJob-compatible runner; worker threads never invoke Streamlit."""
    output = []
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")

    def report(line):
        output.append(line)
        if progress:
            progress(line)

    with tempfile.TemporaryDirectory(prefix="onset_blending_") as directory:
        command_settings = dict(settings)
        if "runtime_overrides" in settings:
            overrides = validate_runtime_overrides(settings["runtime_overrides"])
            snapshot = Path(directory) / "runtime.yml"
            snapshot.write_text(yaml.safe_dump(overrides), encoding="utf-8")
            command_settings["runtime_config"] = str(snapshot)
        command = build_blending_command(command_settings)
        code, timed_out = spawn_process(
            command, env, str(PROJECT_ROOT), 7200, report, cancel_event
        )
    return {
        "ok": code == 0 and not timed_out,
        "returncode": code,
        "timeout": timed_out,
        "command": shlex.join(command),
        "stdout": "\n".join(output),
        "runtime_dir": str(blending_outputs(settings)["root"]),
        "dry_run": bool(settings.get("dry_run")),
    }
