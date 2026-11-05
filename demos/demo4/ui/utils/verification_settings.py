"""Validate editable dashboard verification windows before launching a run."""

import ast
from calendar import monthrange
from datetime import date


def temporal_settings_errors(settings, observation_years):
    """Return actionable preflight errors without importing weather processing."""
    errors = []
    start, end = settings["eval_start"], settings["eval_end"]
    if start >= end:
        errors.append("Evaluation End must be after Evaluation Start.")
    available = set(observation_years)
    missing = sorted(set(range(start.year, end.year + 1)) - available)
    if missing:
        errors.append(f"Observations are missing for evaluation years {missing}.")
    first, last = settings["baseline_start_year"], settings["baseline_end_year"]
    if first > last:
        errors.append("Baseline End Year must be at least Baseline Start Year.")
    elif not set(range(first, last + 1)).issubset(available):
        errors.append("Baseline years must all have observation files available.")
    elif settings.get("probabilistic") and last - first < 1:
        errors.append("Probabilistic fair scores need at least two baseline observation years.")
    mode, days = settings["init_type"], settings["initialization_days"]
    reference_year = settings.get("date_filter_year")
    if mode == "weekly" and (type(reference_year) is not int or not 1 <= reference_year <= 9999):
        errors.append("Enter a valid reference calendar year for weekly initialization.")
    if mode not in {"weekly", "daily", "monthly"}:
        errors.append("Choose a weekly, daily or monthly initialization schedule.")
    elif mode != "daily":
        low, high = (0, 6) if mode == "weekly" else (1, 31)
        if not days or any(type(day) is not int or not low <= day <= high for day in days):
            errors.append("Select at least one valid initialization weekday." if mode == "weekly"
                          else "Select at least one initialization day of month (1–31).")
        elif mode == "monthly":
            first_md, last_md = (start.month, start.day), (end.month, end.day)
            def in_season(md):
                return first_md <= md <= last_md if first_md <= last_md else md >= first_md or md <= last_md
            if not any(in_season((month, day)) for year in range(start.year, end.year + 1)
                       for month in range(1, 13) for day in days if day <= monthrange(year, month)[1]):
                errors.append("Selected days of month do not occur within the evaluation season.")
    return errors


def initialization_matches_schedule(parts, settings):
    year, month, day = parts
    start, end = settings["eval_start"], settings["eval_end"]
    if not start.year <= year <= end.year:
        return False
    first, last, md = (start.month, start.day), (end.month, end.day), (month, day)
    if not (first <= md <= last if first <= last else md >= first or md <= last):
        return False
    mode = settings["init_type"]
    if mode == "daily":
        return True
    if mode == "monthly":
        return day in settings["initialization_days"]
    try:
        weekday = date(settings["date_filter_year"], month, day).weekday()
    except ValueError:
        return False
    return weekday in settings["initialization_days"]


def parse_verification_settings(window_text, tolerance_text=None, day_bins=None):
    try:
        windows = ast.literal_eval(window_text)
        if isinstance(windows, (tuple, list)) and len(windows) == 2 and all(type(v) is int for v in windows):
            windows = (windows,)
        if not isinstance(windows, (tuple, list)) or not windows:
            raise ValueError
        windows = tuple(tuple(window) for window in windows)
        if any(len(w) != 2 or any(type(v) is not int for v in w)
               or not 1 <= w[0] <= w[1] <= 60 for w in windows):
            raise ValueError
    except (ValueError, TypeError, SyntaxError):
        raise ValueError("Verification Window List must contain day pairs, for example ((1, 15), (16, 30)).") from None
    if len(set(windows)) != len(windows):
        raise ValueError("Verification windows must not be duplicated.")
    if day_bins is not None:
        for start, end in windows:
            covered = [day for low, high in day_bins if low >= start and high <= end
                       for day in range(low, high + 1)]
            if sorted(covered) != list(range(start, end + 1)):
                raise ValueError(f"Configured day bins do not cover verification window {(start, end)}.")
    if tolerance_text is None:
        tolerances = tuple(3 if end <= 15 else 5 for _, end in windows)
    else:
        try:
            tolerances = ast.literal_eval(tolerance_text)
            if type(tolerances) is int:
                tolerances = (tolerances,)
            if not isinstance(tolerances, (tuple, list)) or len(tolerances) != len(windows):
                raise ValueError
            tolerances = tuple(tolerances)
            if any(type(v) is not int or v < 0 for v in tolerances):
                raise ValueError
        except (ValueError, TypeError, SyntaxError):
            raise ValueError("Tolerance Days List must contain one nonnegative integer per verification window, for example (3, 5).") from None
    return windows, tolerances, max(end for _, end in windows)
