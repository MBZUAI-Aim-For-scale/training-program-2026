"""Per-run onset settings shared by the dashboard and operational runner."""

from __future__ import annotations

# Configuration type errors use ValueError so the UI can report them uniformly.
# ruff: noqa: TRY004
import copy
import re
from datetime import date

from ai_weather import config

SECTIONS = ("single", "ensemble", "climatology", "combine", "maps")


def merge_spec(base, overrides):
    """Merge nested settings without changing the process-global configuration."""
    merged = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_spec(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def validate_runtime_overrides(overrides):
    if not isinstance(overrides, dict):
        raise ValueError("Runtime configuration must be a mapping of pipeline sections")
    unknown = set(overrides) - set(SECTIONS)
    if unknown:
        raise ValueError(f"Unknown runtime configuration sections: {sorted(unknown)}")
    for section, values in overrides.items():
        if not isinstance(values, dict):
            raise ValueError(f"Runtime configuration section {section} must be a mapping")
        dictionary_keys = {
            "single": ("input", "dimensions", "options", "thresholds", "mok", "filter"),
            "ensemble": ("input", "dimensions", "options", "thresholds", "mok", "filter"),
            "climatology": ("climatologies",),
            "combine": ("options", "climatology", "forecasts"),
            "maps": (),
        }
        for key in dictionary_keys[section]:
            if key in values and not isinstance(values[key], dict):
                raise ValueError(f"{section}.{key} must be a mapping")
        if section == "maps":
            unknown = set(values) - {
                "country_boundary",
                "district_boundary",
                "zoom_to_data",
                "onset_file",
            }
            if unknown:
                raise ValueError(f"Unknown map settings: {sorted(unknown)}")
        if section in ("single", "ensemble"):
            options = values.get("options", {})
            window = options.get("window", 1)
            if type(window) is not int or window < 1:
                raise ValueError(f"{section}: onset window must be a positive integer")
            if (
                "min_day" in options
                and "max_day" in options
                and options["min_day"] > options["max_day"]
            ):
                raise ValueError(f"{section}: minimum lead day exceeds maximum lead day")
            definition = options.get("onset_definition", {})
            if not isinstance(definition, dict):
                raise ValueError(f"{section}: onset_definition must be a mapping")
            if definition.get("follow_days", 0) < 0:
                raise ValueError(f"{section}: follow-up days cannot be negative")
            dry = definition.get("dry_spell", {})
            if not isinstance(dry, dict):
                raise ValueError(f"{section}: dry_spell must be a mapping")
            if dry.get("mode", "consecutive_dry") not in ("consecutive_dry", "window_sum"):
                raise ValueError(f"{section}: unknown dry-spell mode")
            for key in ("min_dry_days", "sum_window"):
                if key in dry and (type(dry[key]) is not int or dry[key] < 1):
                    raise ValueError(f"{section}: {key} must be a positive integer")
            rename = values.get("dimensions", {}).get("rename", {})
            if not isinstance(rename, dict) or any(
                not isinstance(key, str) or not isinstance(value, str)
                for key, value in rename.items()
            ):
                raise ValueError(f"{section}: dimension mappings must contain string names")
            pattern = values.get("input", {}).get("file_regex")
            if pattern is not None:
                try:
                    re.compile(pattern)
                except (TypeError, re.error) as exc:
                    raise ValueError(
                        f"{section}: invalid filename regular expression: {exc}"
                    ) from exc
        if section == "climatology":
            for name, run in values.get("climatologies", {}).items():
                if not isinstance(run, dict):
                    raise ValueError(f"Climatology {name} must be a mapping")
                for key in ("season_start_md", "issue_end_md"):
                    if key in run:
                        try:
                            date.fromisoformat(f"2000-{run[key]}")
                        except (TypeError, ValueError) as exc:
                            raise ValueError(f"{name}: {key} must be a valid MM-DD date") from exc
                if run.get("train_year_min", 0) > run.get("train_year_max", 9999):
                    raise ValueError(f"{name}: climatology training years are reversed")
                if run.get("min_onset_years", 1) < 1:
                    raise ValueError(f"{name}: minimum onset years must be positive")
        if section == "combine":
            for name, forecast in values.get("forecasts", {}).items():
                if not isinstance(forecast, dict):
                    raise ValueError(f"Forecast {name} must be a mapping")
                for key in ("daily", "constants"):
                    if key not in forecast:
                        continue
                    items = forecast[key]
                    if not isinstance(items, list) or any(
                        not isinstance(item, dict) or not {"col", "out"} <= set(item)
                        for item in items
                    ):
                        raise ValueError(
                            f"{name}.{key} must be a list of mappings with col and out"
                        )
        if section == "combine" and values.get("options", {}).get("join", "inner") not in (
            "inner",
            "full",
        ):
            raise ValueError("Combine join must be inner or full")
    return overrides


def runtime_spec_defaults(model_single, year, model_ens=None):
    """Expose configurable stage inputs; output wiring stays owned by the runner."""
    registry = {
        "aifs": (config.ONSET_FORECAST_AIFS_SPEC, config.ONSET_FORECAST_COMBINE_CLIM_SPEC),
        "gencast": (
            config.ONSET_FORECAST_GENCAST_SPEC,
            config.ONSET_FORECAST_COMBINE_GENCAST_AIFS_SPEC,
        ),
        "ngcm": (config.ONSET_FORECAST_NGCM_SPEC, config.ONSET_FORECAST_COMBINE_NGCM_AIFS_SPEC),
    }
    single, combine = registry[model_single]
    model_ens = model_ens or (config.DEFAULT_MODEL_ENS if model_single == "aifs" else "aifs")
    ensemble = (
        config.ONSET_FORECAST_AIFS_ENS_SPEC
        if model_ens == "aifs_ens"
        else config.ONSET_FORECAST_AIFS_SPEC
    )
    fields = ("input", "dimensions", "options", "thresholds", "mok", "filter")
    result = {
        name: {key: copy.deepcopy(spec[key]) for key in fields if key in spec}
        for name, spec in (("single", single), ("ensemble", ensemble))
    }
    for name in ("single", "ensemble"):
        result[name]["input"]["file_regex"] = result[name]["input"]["file_regex"].replace(
            str(config.YEAR), str(year)
        )
    runs = copy.deepcopy(config.ONSET_FORECAST_IMD_CLIM_SPEC["climatologies"])
    for run in runs.values():
        run.pop("test_year_min", None)
        run.pop("test_year_max", None)
    result["climatology"] = {"climatologies": runs}
    result["combine"] = {
        key: copy.deepcopy(combine[key]) for key in ("options", "climatology", "forecasts")
    }
    for forecast in result["combine"]["forecasts"].values():
        forecast.pop("sources", None)
    result["maps"] = {
        "country_boundary": str(config.ETH_GEOJSON_PATH),
        "district_boundary": str(config.ZONE_WOREDA_SHP),
        "zoom_to_data": False,
        "onset_file": "",
    }
    return result
