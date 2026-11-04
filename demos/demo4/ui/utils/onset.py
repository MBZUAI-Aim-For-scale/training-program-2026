# ui/utils/onset.py
"""
Onset detection utilities.
"""
from typing import Any, Dict, Tuple, Optional
import numpy as np
import pandas as pd


def detect_onset(
    time: Any,
    rain: np.ndarray,
    wet_day_mm: float,
    accum_days: int,
    accum_mm: float,
    dry_spell_days: int,
    lookahead_days: int,
    start_month: int,
    start_day: int,
    end_month: int,
    end_day: int,
) -> Tuple[Optional[pd.Timestamp], int]:
    """
    Detect onset date based on local rainfall and dry-spell criteria.

    Returns:
        (onset_datetime, onset_index) if found, otherwise (None, -1).
    """
    # Defensive check: Ensure time is parsed as a DatetimeIndex or a Datetime series
    if not isinstance(time, pd.DatetimeIndex):
        # Convert Series/Index to DatetimeIndex for reliable element retrieval
        time_seq = pd.to_datetime(time)
        if isinstance(time_seq, pd.Series):
            time_seq = pd.DatetimeIndex(time_seq)
    else:
        time_seq = time

    if len(time_seq) == 0 or len(rain) == 0:
        return None, -1

    # Find the start index (search from start_month/day)
    try:
        start_date = pd.Timestamp(year=time_seq.min().year, month=start_month, day=start_day)
    except ValueError:
        # Fallback if the year calculation fails or is out of bounds
        start_date = time_seq.min()

    # If start_date is before time sequence starts, use time start index (0)
    if start_date < time_seq.min():
        start_idx = 0
    else:
        # np.searchsorted expects a 1D sorted array/Index
        start_idx = int(np.searchsorted(time_seq, start_date))

    # Loop over possible onset positions
    n = len(rain)
    for i in range(start_idx, n - accum_days + 1):
        window = rain[i : i + accum_days]
        
        # Check wet spell: accum_days of >= wet_day_mm and total >= accum_mm
        if np.all(window >= wet_day_mm) and np.sum(window) >= accum_mm:
            # Check if dry spell length exceeds dry_spell_days in the lookahead period
            look_end = min(i + accum_days + lookahead_days, n)
            dry_streak = 0
            
            for j in range(i + accum_days, look_end):
                if rain[j] < wet_day_mm:  # using wet_day_mm as threshold for dry
                    dry_streak += 1
                    if dry_streak >= dry_spell_days:
                        break
                else:
                    dry_streak = 0
            else:
                # No long dry spell found; Ensure onset is within the seasonal boundary
                season_end = pd.Timestamp(year=time_seq[i].year, month=end_month, day=end_day)
                if time_seq[i] <= season_end:
                    return time_seq[i], i

    return None, -1


def find_onset(series: pd.Series, params: Dict[str, Any]) -> Tuple[Optional[pd.Timestamp], int]:
    """
    Wrapper to call detect_onset with a parameters dictionary.
    
    Args:
        series: A pandas Series with a DatetimeIndex and rainfall values.
        params: A dictionary containing the exact threshold keys.
    """
    return detect_onset(
        time=series.index,
        rain=series.values,
        wet_day_mm=params["wet_day_mm"],
        accum_days=params["accum_days"],
        accum_mm=params["accum_mm"],
        dry_spell_days=params["dry_spell_days"],
        lookahead_days=params["lookahead_days"],
        start_month=params["start_month"],
        start_day=params["start_day"],
        end_month=params["end_month"],
        end_day=params["end_day"],
    )