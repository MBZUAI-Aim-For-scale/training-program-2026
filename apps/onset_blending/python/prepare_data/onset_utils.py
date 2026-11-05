# ==============================================================================
# File: apps/onset_blending/python/prepare_data/onset_utils.py
# ==============================================================================
import os
import sys
import numpy as np
import pandas as pd
from datetime import date
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# Import PROJECT_ROOT from config.py for consistent path resolution
from ai_weather.config import PROJECT_ROOT


def read_mok_dates(spec):
    """
    Read Monsoon Onset Kerala (MOK) dates from the file specified in spec.

    Returns a DataFrame with columns: year (int), mok_date (datetime.date).
    Returns None if spec["mok"]["file"] is not set.
    """
    if spec.get("mok") is None or spec["mok"].get("file") is None:
        return None

    f = spec["mok"]["file"]
    
    # Resolve relative paths against PROJECT_ROOT
    if not os.path.isabs(f):
        f = str(PROJECT_ROOT / f)

    if not os.path.exists(f):
        raise FileNotFoundError(f"MOK file not found: {f}")

    mok = pd.read_csv(f)
    ycol = spec["mok"]["year_col"]
    dcol = spec["mok"]["day_col"]
    if ycol not in mok.columns or dcol not in mok.columns:
        raise ValueError(f"MOK file must contain columns '{ycol}' and '{dcol}'")

    base_md = spec["mok"]["base_date"]
    mok["mok_date"] = mok.apply(
        lambda row: pd.to_datetime(f"{int(row[ycol])}-{base_md}") + pd.Timedelta(days=int(row[dcol])),
        axis=1
    ).dt.date
    return mok[[ycol, "mok_date"]].rename(columns={ycol: "year"}).assign(year=lambda d: d["year"].astype(int))


def read_thresholds(spec):
    """
    Read per-grid-cell onset thresholds.

    Returns a DataFrame with columns: lat, lon, onset_thresh.
    Returns None if spec["thresholds"]["file"] is not set.
    """
    if spec.get("thresholds") is None or spec["thresholds"].get("file") is None:
        return None

    f = spec["thresholds"]["file"]

    if isinstance(f, (int, float)):
        return f

    try:
        val = float(f)
        return val
    except (TypeError, ValueError):
        pass

    # Resolve relative paths against PROJECT_ROOT
    if not os.path.isabs(f):
        f = str(PROJECT_ROOT / f)

    if not os.path.exists(f):
        raise FileNotFoundError(f"Threshold file not found: {f}")

    ext = os.path.splitext(f)[1].lower()

    if ext in (".nc", ".nc4", ".netcdf"):
        import netCDF4 as nc4
        ds = nc4.Dataset(f)
        lat_name = spec["thresholds"].get("lat_var") or spec["thresholds"].get("lat_col") or "lat"
        lon_name = spec["thresholds"].get("lon_var") or spec["thresholds"].get("lon_col") or "lon"
        thr_name = spec["thresholds"].get("thresh_var") or spec["thresholds"].get("thresh_col") or "MWmean"
        lats = np.array(ds[lat_name][:]).flatten()
        lons = np.array(ds[lon_name][:]).flatten()
        thresh = np.array(ds[thr_name][:]).flatten()
        ds.close()
        return pd.DataFrame({"lat": lats, "lon": lons, "onset_thresh": thresh}).drop_duplicates()

    if ext == ".mat":
        try:
            import scipy.io as sio
        except ImportError:
            raise ImportError("scipy is required to read .mat threshold files.")
        lat_name = spec["thresholds"].get("mat_lat_name", "lat")
        lon_name = spec["thresholds"].get("mat_lon_name", "lon")
        thr_name = spec["thresholds"].get("mat_thresh_name", "onset.day.thres")
        md = sio.loadmat(f)
        lats = np.array(md[lat_name]).flatten()
        lons = np.array(md[lon_name]).flatten()
        thresh = np.array(md[thr_name])
        grid = pd.DataFrame(
            [(la, lo) for la in lats for lo in lons],
            columns=["lat", "lon"]
        )
        grid["onset_thresh"] = thresh.flatten()
        return grid.drop_duplicates()

    th = pd.read_csv(f)
    tc = spec["thresholds"].get("thresh_col", "onset_thresh")

    adm3_col = spec["thresholds"].get("adm3_col", None)
    if adm3_col and adm3_col in th.columns:
        th = th.rename(columns={adm3_col: "id", tc: "onset_thresh"})
        return th[["id", "onset_thresh"]].drop_duplicates()
    if "adm3_name" in th.columns:
        th = th.rename(columns={"adm3_name": "id", tc: "onset_thresh"})
        return th[["id", "onset_thresh"]].drop_duplicates()
    if "id" in th.columns:
        th = th.rename(columns={tc: "onset_thresh"})
        return th[["id", "onset_thresh"]].drop_duplicates()

    lc = spec["thresholds"].get("lat_col", "lat")
    oc = spec["thresholds"].get("lon_col", "lon")
    th = th.rename(columns={lc: "lat", oc: "lon", tc: "onset_thresh"})
    return th[["lat", "lon", "onset_thresh"]].drop_duplicates()


def roll_sum_na_rm_left(x, k):
    x = np.asarray(x, dtype=float)
    n = len(x)
    if k <= 0 or n < k:
        return np.array([], dtype=float)
    x0 = np.where(np.isnan(x), 0.0, x)
    cs = np.concatenate([[0.0], np.cumsum(x0)])
    return cs[k:] - cs[:n - k + 1]


def roll_sum_na_propagate_left(x, k):
    x = np.asarray(x, dtype=float)
    n = len(x)
    if k <= 0 or n < k:
        return np.array([], dtype=float)
    na = np.isnan(x).astype(float)
    x0 = np.where(np.isnan(x), 0.0, x)
    cs = np.concatenate([[0.0], np.cumsum(x0)])
    cna = np.concatenate([[0.0], np.cumsum(na)])
    s = cs[k:] - cs[:n - k + 1]
    na_ct = cna[k:] - cna[:n - k + 1]
    s[na_ct > 0] = np.nan
    return s


def read_onset_params(spec):
    import collections
    OnsetParams = collections.namedtuple("OnsetParams", [
        "win", "wet_day_min_mm", "follow_days", "mode",
        "min_dry_days", "dry_day_min_mm", "sum_window", "sum_min_mm",
    ])

    opts = spec.get("options", {})
    win = int(opts.get("window", 5))

    od = opts.get("onset_definition") or {}
    wet_day_min_mm = float(od.get("wet_day_min_mm", 1.0))
    follow_days = int(od.get("follow_days", 21))

    ds = od.get("dry_spell") or {}
    mode = str(ds.get("mode", "consecutive_dry"))
    min_dry_days = int(ds.get("min_dry_days", 7))
    dry_day_min_mm = float(ds.get("dry_day_min_mm", wet_day_min_mm))
    sum_window = int(ds.get("sum_window", 10))
    sum_min_mm = float(ds.get("sum_min_mm", 5.0))

    if mode not in ("consecutive_dry", "window_sum"):
        raise ValueError(
            f"onset_definition.dry_spell.mode must be 'consecutive_dry' or "
            f"'window_sum', got '{mode}'"
        )

    return OnsetParams(
        win=win, wet_day_min_mm=wet_day_min_mm, follow_days=follow_days,
        mode=mode, min_dry_days=min_dry_days, dry_day_min_mm=dry_day_min_mm,
        sum_window=sum_window, sum_min_mm=sum_min_mm,
    )


def _precompute_onset(series, params):
    series = np.asarray(series, dtype=float)
    n = len(series)
    win = params.win

    wsum = roll_sum_na_rm_left(series, win)

    if params.mode == "consecutive_dry":
        dry = (~np.isnan(series)) & (series < params.dry_day_min_mm)
        dry_starts = np.zeros(n, dtype=int)
        k = params.min_dry_days
        if n >= k:
            run = np.zeros(n, dtype=int)
            run[0] = int(dry[0])
            for i in range(1, n):
                run[i] = (run[i - 1] + 1) * int(dry[i])
            for e in np.where(run == k)[0]:
                dry_starts[e - k + 1] = 1
        pre_dry = np.concatenate([[0], np.cumsum(dry_starts)])
        return wsum, pre_dry, None

    else:
        sw = params.sum_window
        if n >= sw:
            sum_w = roll_sum_na_propagate_left(series, sw)
            bad = (~np.isnan(sum_w)) & (sum_w < params.sum_min_mm)
            pre_bad = np.concatenate([[0], np.cumsum(bad.astype(int))])
            last_win_start = n - sw
        else:
            pre_bad = np.array([0], dtype=int)
            last_win_start = 0
        return wsum, pre_bad, last_win_start


def find_onset(series, win=None, thresh=None, reject_if_short_followup=False,
               start_day=0, params=None):
    if params is None:
        import collections
        OnsetParams = collections.namedtuple("OnsetParams", [
            "win", "wet_day_min_mm", "follow_days", "mode",
            "min_dry_days", "dry_day_min_mm", "sum_window", "sum_min_mm",
        ])
        params = OnsetParams(
            win=int(win) if win is not None else 5,
            wet_day_min_mm=1.0, follow_days=21,
            mode="consecutive_dry", min_dry_days=7, dry_day_min_mm=1.0,
            sum_window=10, sum_min_mm=5.0,
        )

    series = np.asarray(series, dtype=float)
    n = len(series)
    if n < params.win or thresh is None or np.isnan(thresh):
        return None

    wsum, aux1, aux2 = _precompute_onset(series, params)
    return _find_onset_core(series, n, wsum, aux1, aux2, thresh,
                            params, start_day, reject_if_short_followup)


def find_onset_precomp(series, win, thresh, wsum_all, pre_bad, last10start,
                       reject_if_short_followup=False, start_day=0, params=None):
    if params is None:
        import collections
        OnsetParams = collections.namedtuple("OnsetParams", [
            "win", "wet_day_min_mm", "follow_days", "mode",
            "min_dry_days", "dry_day_min_mm", "sum_window", "sum_min_mm",
        ])
        params = OnsetParams(
            win=int(win) if win is not None else 5,
            wet_day_min_mm=1.0, follow_days=21,
            mode="consecutive_dry", min_dry_days=7, dry_day_min_mm=1.0,
            sum_window=10, sum_min_mm=5.0,
        )

    series = np.asarray(series, dtype=float)
    n = len(series)
    if n < params.win or thresh is None or np.isnan(thresh):
        return None

    wsum, aux1, aux2 = _precompute_onset(series, params)
    return _find_onset_core(series, n, wsum, aux1, aux2, thresh,
                            params, start_day, reject_if_short_followup)


def _find_onset_core(series, n, wsum, aux1, aux2, thresh,
                     params, start_day, reject_if_short_followup):
    win = params.win
    wmm = params.wet_day_min_mm
    follow = params.follow_days

    max_candidate = len(wsum)
    if max_candidate < 1:
        return None

    min_i = max(1, int(np.ceil(start_day)))
    if min_i > max_candidate:
        return None

    idx = np.arange(min_i, max_candidate + 1)

    wet_mat = np.stack(
        [series[idx - 1 + k] >= wmm for k in range(win)],
        axis=1
    )
    all_wet = wet_mat.all(axis=1)
    acc_ok = wsum[idx - 1] > thresh
    base_ok = all_wet & acc_ok
    base_ok = np.where(np.isnan(series[idx - 1]), False, base_ok)

    cand = idx[base_ok]
    if len(cand) == 0:
        return None

    full_end = cand + win - 1 + follow
    if reject_if_short_followup:
        cand = cand[full_end <= n]
        if len(cand) == 0:
            return None
        full_end = full_end[full_end <= n]
    else:
        full_end = np.minimum(n, full_end)

    if params.mode == "consecutive_dry":
        pre_dry = aux1
        lower = cand - 1 + win
        upper = np.minimum(n, cand - 1 + win + follow)
        has_dry_spell = np.where(
            lower < upper,
            (pre_dry[upper] - pre_dry[lower]) > 0,
            False
        )

    else:
        pre_bad = aux1
        last_win_start = aux2
        sw = params.sum_window
        c_lower = cand - 1 + win
        c_upper = np.minimum(last_win_start + 1,
                             np.maximum(0, full_end - sw + 1))
        max_idx = len(pre_bad) - 1
        c_lower = np.minimum(c_lower, max_idx)
        c_upper = np.minimum(c_upper, max_idx)
        has_dry_spell = np.where(
            c_lower < c_upper,
            (pre_bad[c_upper] - pre_bad[c_lower]) > 0,
            False
        )

    ok = ~has_dry_spell
    return int(cand[np.where(ok)[0][0]]) if np.any(ok) else None