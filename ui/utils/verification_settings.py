"""Validate editable dashboard verification windows before launching a run."""

import ast


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
