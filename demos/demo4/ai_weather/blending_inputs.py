"""Read-only forecast calendar inspection for consistent blending selections."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pandas as pd
from xarray.backends.netCDF4_ import NETCDF4_PYTHON_LOCK

from ai_weather import config
from ai_weather.blending_config import merge_spec, runtime_spec_defaults
from apps.onset_blending.python.pipelines._shared.misc import validate_time_year
from apps.onset_blending.python.prepare_data.nc_utils import (
    list_nc_files_with_year,
)
from apps.onset_blending.python.prepare_data.nc_utils import (
    read_forecast_dates as _read_forecast_dates,
)


def read_forecast_dates(nc_path):
    # The raw netCDF4 reader bypasses xarray. Share its native-library lock so
    # background inspection cannot race another session's dataset/map reads.
    with NETCDF4_PYTHON_LOCK:
        return _read_forecast_dates(nc_path)


def _selected_specs(settings):
    year = int(settings["year"])
    defaults = runtime_spec_defaults(
        settings.get("model_single", config.DEFAULT_MODEL_SINGLE), year, settings.get("model_ens")
    )
    specs = {}
    for section, key in (("single", "aifs_nc_folder"), ("ensemble", "aifs_ens_nc_folder")):
        spec = merge_spec(
            defaults[section], settings.get("runtime_overrides", {}).get(section, {})
        )
        if settings.get(key):
            spec["input"]["nc_folder"] = settings[key]
        spec["input"]["nc_folder"] = str(Path(spec["input"]["nc_folder"]).expanduser())
        spec["input"]["file_regex"] = spec["input"]["file_regex"].replace(
            str(config.YEAR), str(year)
        )
        specs[section] = spec
    return specs


def _climatology_runs(settings):
    return merge_spec(
        {"climatologies": config.ONSET_FORECAST_IMD_CLIM_SPEC["climatologies"]},
        settings.get("runtime_overrides", {}).get("climatology", {}),
    )["climatologies"]


def _within_climatology(settings, issue):
    month_day = issue[5:]
    return all(
        run["season_start_md"] <= month_day <= run["issue_end_md"]
        for run in _climatology_runs(settings).values()
    )


def _signature(settings):
    relevant = {
        "year": settings["year"],
        "issue_date": settings["issue_date"],
        "sources": _selected_specs(settings),
        "climatology": _climatology_runs(settings),
    }
    return hashlib.sha256(json.dumps(relevant, sort_keys=True, default=str).encode()).hexdigest()


def input_check_matches(settings, report):
    """Invalidate a check when selected parameters or inspected files change."""
    if not report or report.get("signature") != _signature(settings):
        return False
    try:
        return all(
            (Path(path).stat().st_mtime_ns, Path(path).stat().st_size) == (mtime, size)
            for path, mtime, size in report.get("fingerprints", [])
        )
    except OSError:
        return False


def inspect_blending_inputs(settings):
    """List usable shared issue dates and diagnose the current exact selection."""
    records, errors, fingerprints = [], [], []
    shared_dates = []
    year = int(settings["year"])
    issue = settings["issue_date"]
    for section, selected in _selected_specs(settings).items():
        inventory_spec = merge_spec(
            selected,
            {
                "input": {
                    "file_regex": selected["input"]["file_regex"].replace(str(year), r"\d{4}")
                }
            },
        )
        try:
            files = list_nc_files_with_year(inventory_spec)
        except (OSError, ValueError) as exc:
            errors.append(f"{section}: {exc}")
            shared_dates.append(set())
            continue
        usable_dates, selected_dates, selected_files = set(), set(), []
        for row in files.itertuples(index=False):
            record = {
                "source": section,
                "file": row.nc_path,
                "filename_year": int(row.year),
                "actual_years": "",
                "first_date": "",
                "last_date": "",
                "valid": False,
                "error": "",
            }
            try:
                stat = Path(row.nc_path).stat()
                fingerprints.append((row.nc_path, stat.st_mtime_ns, stat.st_size))
                dates = read_forecast_dates(row.nc_path)
                actual_years = sorted(set(dates.year.tolist()))
                record["actual_years"] = ", ".join(map(str, actual_years))
                if dates.empty:
                    raise ValueError("Forecast file has no initialization dates")
                record["first_date"] = str(dates.min().date())
                record["last_date"] = str(dates.max().date())
                validate_time_year(
                    pd.DataFrame({"time": dates, "year": dates.year}),
                    row.nc_path,
                    expected_year=int(row.year),
                )
                record["valid"] = True
                date_strings = {value.isoformat() for value in dates.date}
                usable_dates.update(date_strings)
            except (OSError, ValueError, TypeError, KeyError) as exc:
                record["error"] = str(exc)
                date_strings = set()
            records.append(record)
            if re.search(selected["input"]["file_regex"], Path(row.nc_path).name):
                selected_files.append(record)
                if not record["valid"]:
                    errors.append(f"{section}: {record['error']}")
                selected_dates.update(date_strings)
        if not selected_files:
            errors.append(
                f"{section}: no files match {selected['input']['file_regex']} in {selected['input']['nc_folder']}"
            )
        elif (
            not any(not record["valid"] for record in selected_files)
            and issue not in selected_dates
        ):
            errors.append(
                f"{section}: selected issue date {issue} is not an available initialization"
            )
        shared_dates.append(usable_dates)
    common = sorted(set.intersection(*shared_dates)) if len(shared_dates) == 2 else []
    common = [value for value in common if _within_climatology(settings, value)]
    if issue and not _within_climatology(settings, issue):
        errors.append(f"Issue date {issue} is outside the configured climatology issue season")
    if not str(issue).startswith(f"{year}-"):
        errors.append("Issue date must be in the selected forecast year")
    return {
        "ok": not errors,
        "errors": errors,
        "records": records,
        "common_dates": common,
        "signature": _signature(settings),
        "fingerprints": fingerprints,
    }
