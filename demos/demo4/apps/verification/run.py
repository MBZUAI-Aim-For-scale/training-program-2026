# Authors: The africlimate AI team
# SPDX-License-Identifier: MIT
"""CLI to compute, store, and plot verification diagnostics.

Covers the reliability diagram, ROC contingency table, cost-loss value, the scalar CRPS
summary, and the deterministic metrics (mae / bias / correlation). Lead time is given in
days and converted to filename hours by x24. Cost-loss reuses — and always stores — the
ROC contingency table it is derived from.

The **model token drives the source**: a plain token (``graphcast``) scores the raw
forecast; a ``-idr`` token (``graphcast-idr``) scores the IDR-calibrated forecast. Each is
written under exactly the token passed, so ``--model graphcast graphcast-idr`` produces both
side by side. crps and the threshold curves exist only for ``-idr`` tokens (they need the
calibration's output); the deterministic metrics (mae/bias/correlation) are the mirror image —
raw tokens only, since the calibrated forecast is scored by CRPS, not q50 point/rank error.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from verification import diagnostics as d
from verification.regions import default_region, regions_for_domain
from verification.seasons import DEFAULT_SEASON_SLUG, SEASON_SLUGS, season_months

_METRICS = ["roc", "cost_loss", "reliability", "crps", "mae",
             "spearmanr", "pearsonr", "bias"]
_THRESHOLD_METRICS = {"roc", "cost_loss", "reliability"}
_DETERMINISTIC_METRICS = ["mae", "spearmanr", "pearsonr", "bias"]
_CALIBRATED_ONLY = _THRESHOLD_METRICS | {"crps"}   # need the IDR output, so '-idr' tokens only


def select_deterministic_metrics(requested, *, calibrated, is_station):
    """Which of the requested deterministic metrics to compute for one model token.

    Deterministic metrics (mae/bias/correlation) score a single-valued forecast, so they
    apply to the **raw** forecast only — the calibrated ('-idr') forecast is scored by CRPS
    (whole distribution) plus the threshold curves, not q50 point/rank error — hence a
    calibrated token gets none. Stations further drop correlation, since the gridded corr
    reducer needs two spatial dims and a gauge series has one. Order follows ``requested``.
    """
    if calibrated:
        return []
    det = [m for m in requested if m in _DETERMINISTIC_METRICS]
    if is_station:
        det = [m for m in det if m in ("mae", "bias")]
    return det


def _available_leads(fc_root):
    """Lead times in days for which a forecast file exists under ``fc_root``."""
    hours = sorted(int(m.group(1)) for f in Path(fc_root).glob("*h.nc")
                   if (m := re.match(r"(\d+)h\.nc$", f.name)))
    return [h // 24 for h in hours]


def _emit_plots(args, ds, plot_fn, metric, lead_hours, model):
    """Save the requested plots for one diagnostic Dataset (filenames namespaced by model)."""
    prefix = f"{model}_{metric}"
    if args.per_altitude:
        for png in d.plot_by_altitude(ds, "plots", plot_fn, prefix):
            print(f"  plotted {png}")
    elif args.plot:
        band = str(ds["altitude"].values[0])
        print(f"  plotted {plot_fn(ds, f'plots/{prefix}_{lead_hours}h.png', band=band)}")


def main():
    """Parse arguments, then for each model × lead compute, store, and plot the requested metrics."""
    ap = argparse.ArgumentParser(description="Forecast verification diagnostics.")
    _model_choices = list(d.MODELS) + [f"{m}-idr" for m in d.MODELS]
    ap.add_argument("--model", nargs="+", choices=_model_choices, default=["graphcast-idr"],
                    help="model token(s) to evaluate. A plain token (graphcast) scores the raw "
                         "forecast; a '-idr' token (graphcast-idr) scores the IDR-calibrated "
                         "forecast. Output is keyed by exactly the token passed.")
    ap.add_argument("--metric", nargs="+", choices=_METRICS, default=_METRICS)
    ap.add_argument("--lead-day", nargs="+", type=int, default=None,
                    help="lead times in days (x24 -> hours); default: all available")
    ap.add_argument("--threshold", nargs="+", type=int, choices=d.THRESHOLDS_MM_DAY,
                    default=d.THRESHOLDS_MM_DAY, help="exceedance thresholds in mm/day")
    ap.add_argument("--altitude", nargs="+", choices=list(d.ALTITUDE_BANDS),
                    default=list(d.ALTITUDE_BANDS), help="altitude bands")
    ap.add_argument("--region", nargs="+", choices=regions_for_domain(),
                    default=[default_region()],
                    help="region(s) for threshold curves. Default = the whole country "
                         "(written region-less, unchanged). An admin-1 subregion subsets the "
                         "IDR in memory before computing and writes under a region path "
                         "segment (.../{year}/{region}/{lead}h.nc). Regions run sequentially.")
    ap.add_argument("--season", nargs="+", choices=sorted(SEASON_SLUGS),
                    default=[DEFAULT_SEASON_SLUG],
                    help="season(s) for threshold curves. Default = 'yearly' (whole year, "
                         "written season-less, unchanged). A season subsets the calibrated "
                         "forecast to its calendar months before pooling and writes under a "
                         "season path segment (.../{year}/[{region}/]{season}/{lead}h.nc). "
                         "Composes with --region; seasons run sequentially.")
    ap.add_argument("--n-bootstraps", type=int, default=d.RELIABILITY_N_BOOTSTRAPS)
    ap.add_argument("--n-bins", type=int, default=d.RELIABILITY_N_BINS)
    ap.add_argument("--is-station", action="store_true",
                    help="verify against EMA gauge stations instead of ENACTS gridded")
    ap.add_argument("--year", type=int, default=d.DEFAULT_YEAR,
                    help="test year to verify (must be calibrated first); default 2022")
    ap.add_argument("--variable", default=d.DEFAULT_VARIABLE,
                    choices=["precipitation", "temperature_max"],
                    help="calibrated variable to verify (default precipitation). "
                         "temperature_max supports crps only (no exceedance metrics).")
    ap.add_argument("--outdir", default="data/diagnostics")
    ap.add_argument("--fc-root", default=None, help="override the forecast root (single model only)")
    ap.add_argument("--obs", default=None)
    ap.add_argument("--dem", default=None)
    ap.add_argument("--plot", action="store_true", help="save a preview PNG per metric")
    ap.add_argument("--per-altitude", action="store_true",
                    help="save one plot per altitude band instead of the first band")
    args = ap.parse_args()

    # The two verification targets differ only in their on-disk shape (gridded vs
    # gauge points); the diagnostics themselves are identical. Pick the front-end
    # (sample builder + forecast root) and the output segment from --is-station.
    target = "stations" if args.is_station else "enacts"
    year, variable = args.year, args.variable
    obs_dem = {k: v for k, v in (("obs_path", args.obs), ("dem_path", args.dem)) if v is not None}
    # Exceedance metrics (roc/cost_loss/reliability) need p_exceed, which only the
    # precipitation calibration emits; temperature is CRPS-only for now.
    want_threshold = bool(_THRESHOLD_METRICS.intersection(args.metric)) and variable == "precipitation"
    if _THRESHOLD_METRICS.intersection(args.metric) and variable != "precipitation":
        print(f"note: {sorted(_THRESHOLD_METRICS.intersection(args.metric))} are precipitation-only; "
              f"running crps only for {variable}")

    for token in args.model:
        # The token a user types selects the source and is also the token written out.
        base, calibrated = d.parse_model_token(token)
        if calibrated:
            fc_root = args.fc_root or str(
                d.station_fc_root(base, year) if args.is_station
                else d.model_fc_root(base, year, variable))
        else:
            fc_root = args.fc_root or str(
                d.station_raw_fc_root(base, year) if args.is_station
                else d.model_fc_root(base, year, variable, raw=True))
            calib_only = sorted(_CALIBRATED_ONLY.intersection(args.metric))
            if calib_only:
                print(f"note: {token}: {calib_only} need the calibrated forecast; use {base}-idr")
        source = "calibrated" if calibrated else "raw"

        # Deterministic metrics apply to the raw forecast only; a calibrated token is scored
        # by CRPS + threshold curves. Stations further drop correlation (one spatial dim).
        det_metrics = select_deterministic_metrics(args.metric, calibrated=calibrated,
                                                   is_station=args.is_station)
        dropped_det = [m for m in args.metric if m in _DETERMINISTIC_METRICS
                       and m not in det_metrics]
        if dropped_det:
            reason = ("raw-only (the calibrated forecast is scored by CRPS)" if calibrated
                      else "not supported at stations; deterministic there is mae/bias only")
            print(f"note: {token}: {dropped_det} {reason}; skipped")

        lead_days = args.lead_day or _available_leads(fc_root)
        for day in lead_days:
            hours = day * 24
            print(f"{token} | lead {day}d ({hours}h) | {variable} | {target} {year}")

            if calibrated and "crps" in args.metric:
                try:
                    crps = d.compute_crps(hours, model=base, year=year, variable=variable, fc_root=fc_root)
                except FileNotFoundError:
                    print(f"  skip crps {hours}h: no forecast file")
                else:
                    print(f"  wrote {d.write_dataset(crps, 'crps', hours, args.outdir, year=year, subdir='annual_monthly_averages', model=token, target=target, native_var=variable)}")
                    if args.plot:
                        print(f"  plotted {d.plot_crps(crps, f'plots/{token}_{variable}_crps_{hours}h.png')}")

            # Deterministic metrics: the raw forecast only (det_metrics is empty for a
            # calibrated token), written under the token as-is.
            for metric in det_metrics:
                try:
                    if args.is_station:
                        dm = d.compute_deterministic_station(hours, model=base, year=year,
                                                             metric=metric, source=source)
                    else:
                        dm = d.compute_deterministic(hours, model=base, year=year, variable=variable,
                                                     metric=metric, source=source)
                except (FileNotFoundError, KeyError):
                    print(f"  skip {metric} {hours}h: no forecast file")
                    continue
                print(f"  wrote {d.write_dataset(dm, metric, hours, args.outdir, year=year, subdir='annual_monthly_averages', model=token, target=target, native_var=variable)}")
                if args.plot:
                    print(f"  plotted {d.plot_spatial(dm, f'plots/{token}_{metric}_{hours}h.png', metric=metric.upper())}")

            if not (calibrated and want_threshold):
                continue

            # Per region × season: the whole country (default_region) writes region-less and
            # 'yearly' writes season-less (unchanged legacy layout); an admin-1 subregion and/or
            # a season subset the IDR in memory (crop+mask, then month filter) before pooling and
            # write under region/season path segments. Both loops run sequentially so memory stays
            # bounded. PNG previews are only emitted for the whole country + yearly (other combos
            # would collide on the flat plots/ filenames).
            whole = default_region()
            fc_missing = False
            for region in args.region:
                if fc_missing:
                    break
                region_arg = None if region == whole else region
                for season in args.season:
                    season_arg = None if season == DEFAULT_SEASON_SLUG else season
                    months = season_months(season)
                    try:
                        if args.is_station:
                            # --obs/--dem don't apply: the event truth is the gauge archive and
                            # each station carries its own elevation, so there's no DEM to interp.
                            samples = d.build_samples_station(
                                hours, model=base, year=year, thresholds=args.threshold,
                                bands=args.altitude, fc_root=args.fc_root, region=region_arg,
                                months=months)
                        else:
                            samples = d.build_samples(
                                hours, model=base, year=year, thresholds=args.threshold,
                                bands=args.altitude, fc_root=args.fc_root, region=region_arg,
                                months=months, **obs_dem)
                    except FileNotFoundError:
                        print(f"  skip lead {day}d ({hours}h): no forecast file")
                        fc_missing = True  # same forecast file across regions/seasons — missing for all
                        break
                    print(f"  region: {region} | season: {season}")
                    emit = region_arg is None and season_arg is None

                    roc = None
                    if "roc" in args.metric or "cost_loss" in args.metric:
                        roc = d.compute_roc(samples)
                        print(f"    wrote {d.write_dataset(roc, 'roc', hours, args.outdir, year=year, model=token, target=target, region=region_arg, season=season_arg)}")
                        if "roc" in args.metric and emit:
                            _emit_plots(args, roc, d.plot_roc, "roc", hours, token)
                    if "cost_loss" in args.metric:
                        cl = d.compute_cost_loss(roc)
                        print(f"    wrote {d.write_dataset(cl, 'cost_loss', hours, args.outdir, year=year, model=token, target=target, region=region_arg, season=season_arg)}")
                        if emit:
                            _emit_plots(args, cl, d.plot_cost_loss, "cost_loss", hours, token)
                    if "reliability" in args.metric:
                        rel = d.compute_reliability(samples, n_bins=args.n_bins,
                                                    n_bootstraps=args.n_bootstraps)
                        print(f"    wrote {d.write_dataset(rel, 'reliability', hours, args.outdir, year=year, model=token, target=target, region=region_arg, season=season_arg)}")
                        if emit:
                            _emit_plots(args, rel, d.plot_reliability, "reliability", hours, token)

if __name__ == "__main__":
    main()
