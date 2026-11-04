# ui/pages/ai_forecasting.py
"""
AI Forecast Lab — Running Your First AI Weather Forecast.
Integrated into Onset Forecast Platform.

Choose up to 4 models, the start date, lead time, region and use case; load saved
runs or run the models (one after another); then explore the results: how each
forecast was made, event movies (or onset) for the models you pick, a time-series
comparison of all of them, and forecast maps.
"""
from __future__ import annotations

import base64
import shutil
from html import escape
import subprocess
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st
import xarray as xr

# ─── Ensure the 'apps' package is discoverable ──────────────────────────────
try:
    _here = Path(__file__).resolve()
    _root = next((p for p in _here.parents if (p / "apps").is_dir()), _here.parent.parent)
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

    from apps.forecasting import catalog, compare, jobs, movie, onset_view, results, runs, store, theme
    from apps.forecasting.regions import COUNTRIES, GROUPS, Box, country_box
except ImportError as e:
    st.error(f"Failed to import forecasting modules: {e}")
    st.info("Please ensure the 'apps/forecasting' package exists in your project root "
            "and contains catalog, compare, jobs, movie, onset_view, results, runs and store.")
    st.stop()

ss = st.session_state
MAX_MODELS = 4


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════
def _section(title: str, color: str = "sky") -> None:
    st.markdown(f'<div class="sec-hd sec-hd-{color}"><span class="sec-hd-accent"></span>{title}</div>',
                unsafe_allow_html=True)


@st.cache_data(ttl=60)
def gpu_name() -> str | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip().splitlines()[0] if out.returncode == 0 and out.stdout.strip() else None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


@st.cache_resource(max_entries=6)
def open_output(path: str, mtime: float) -> xr.Dataset:
    return xr.open_dataset(path).load()


def _open(path: str) -> xr.Dataset:
    return open_output(path, Path(path).stat().st_mtime)


def gif_html(path: Path, alt: str) -> None:
    # st.image shows only a GIF's first frame; embed it so it animates
    st.html(f'<img src="data:image/gif;base64,{base64.b64encode(path.read_bytes()).decode()}" '
            f'style="width:100%" alt="{alt}">')


def obs_window(init: pd.Timestamp, lead_days: int, run_days: int) -> tuple[int, str | None]:
    """Days of observations (ERA5) available to compare with, and a note if fewer than asked."""
    days = min(lead_days, run_days)
    obs_end = pd.Timestamp(catalog.era5_dates()[1])  # last day of (preliminary) ERA5
    if init >= obs_end:
        return 0, (f"Observations (ERA5) currently end on {obs_end:%-d %b %Y}, so there is nothing "
                   f"yet to compare a {init:%-d %b %Y} forecast with.")
    if init + pd.Timedelta(days=days) > obs_end:
        days = max(1, (obs_end - init).days)
        return days, (f"Observations (ERA5) currently end on {obs_end:%-d %b %Y}, so the movies cover "
                      f"the first {days} day{'s' if days > 1 else ''} of the forecast.")
    return days, None


# ═════════════════════════════════════════════════════════════════════════════
# Event movies (heat / precipitation)
# ═════════════════════════════════════════════════════════════════════════════
def threshold_input(where, kind: str, key: str) -> float | None:
    """Event threshold: automatic, one of the demo thresholds, or one the participant types."""
    opts = catalog.HEAT_THRESHOLDS if kind == "heat" else catalog.RAIN_THRESHOLDS
    unit = "°C" if kind == "heat" else "mm/day"
    auto = "auto (p95)" if kind == "heat" else "auto (p97.5)"
    pick = where.selectbox(f"Threshold ({unit})", [auto] + opts, key=key, accept_new_options=True,
                           help="Days above this count as the event. Pick one of the demo thresholds "
                                "or type your own; auto uses a high percentile of the observations.")
    if pick == auto:
        return None
    try:
        return float(pick)
    except (TypeError, ValueError):
        where.error(f"“{pick}” isn't a number; using auto.")
        return None


def event_movies(paths: dict, keys: list, init, lead_days, box, region_name, use_case) -> None:
    """Obs-vs-forecast event movie and track map, for the models the user picks."""
    kind, var = ("heat", "t2m") if use_case == "Heat" else ("precip", "tp")
    have = {k: _open(paths[k]) for k in keys if k in paths}
    usable = [k for k, ds in have.items() if var not in catalog.missing_in_run(ds)]
    lacking = [catalog.BY_KEY[k].name for k in have if k not in usable]
    if lacking:
        st.caption(f"ⓘ No {use_case.lower()} movie for {', '.join(lacking)}: "
                   f"{'it does' if len(lacking) == 1 else 'they do'} not forecast "
                   f"{catalog.VAR_LABELS[var]}.")
    if not usable:
        return
    run_days = min(int(have[k].lead_time.max()) // 24 for k in usable)
    days, note = obs_window(init, lead_days, run_days)
    if note:
        st.info(note)
    if days == 0:
        return
    c1, c2, c3 = st.columns([3, 2, 2])
    picked = c1.multiselect("Make event movies for", usable, default=usable[:1], key="forecast_movie_models",
                            format_func=lambda k: catalog.BY_KEY[k].name)
    reduce = c2.selectbox("Regional series", ["mean", "max", "p95", "p99"], key="forecast_movie_reduce",
                          index=1 if kind == "heat" else 0,
                          format_func={"mean": "area mean", "max": "area maximum",
                                       "p95": "95th percentile", "p99": "99th percentile"}.get)
    threshold = threshold_input(c3, kind, key="forecast_movie_thr_" + kind)
    if not picked:
        return
    reqs = {k: movie.request_for(paths[k], catalog.BY_KEY[k].name, kind, init, days, box,
                                 region_name, reduce, threshold) for k in picked}
    todo = [k for k in picked if movie.status(reqs[k])["state"] in ("none", "failed")]
    queue = ss.setdefault("forecast_movie_queue", [])
    if todo and not any(k in queue for k in todo):
        label = (f"Make the event movie for {catalog.BY_KEY[todo[0]].name}" if len(todo) == 1
                 else f"Make {len(todo)} event movies (one after another)")
        if st.button(label, type="primary", key="forecast_movie_go",
                     help="Movies are made one at a time. The first for new dates fetches ERA5 "
                          "observations from the cloud (a minute or two); later ones reuse them."):
            queue.extend(k for k in todo if k not in queue)
            st.rerun()
    busy = bool(queue) or any(movie.status(r)["state"] == "running" for r in reqs.values())
    st.fragment(_movie_panel, run_every=3 if busy else None)(picked, reqs)


def _movie_panel(picked, reqs):
    queue = ss.setdefault("forecast_movie_queue", [])
    states = {k: movie.status(r)["state"] for k, r in reqs.items()}
    # one movie at a time: start the next queued one when none is running
    queue[:] = [k for k in queue if k in reqs and states[k] in ("none", "failed", "running")]
    if not any(s == "running" for s in states.values()) and queue:
        k = queue.pop(0)
        movie.launch(reqs[k])
        states[k] = "running"
    tabs = st.tabs([catalog.BY_KEY[k].name for k in picked]) if len(picked) > 1 else [st.container()]
    for tab, k in zip(tabs, picked):
        with tab:
            req, s = reqs[k], movie.status(reqs[k])
            if s["state"] == "running":
                frac, what = movie.progress(req)
                st.progress(frac, text=what)
                with st.expander("Log", key=f"forecast_movie_log_{k}"):
                    st.code(s["log"] or "…")
            elif k in queue:
                st.caption("Waiting its turn.")
            elif s["state"] == "failed":
                st.error(f"Movie failed: {s.get('error')}")
                with st.expander("Log", key=f"forecast_movie_faillog_{k}"):
                    st.code(s["log"])
            elif s["state"] == "none":
                st.caption("Not made yet.")
            else:
                if s.get("gif"):
                    gif_html(Path(s["gif"]), f"{catalog.BY_KEY[k].name}: observations vs forecast")
                    st.download_button("Download movie (GIF)", Path(s["gif"]).read_bytes(),
                                       file_name=Path(s["gif"]).name, mime="image/gif",
                                       key=f"forecast_movie_dl_{k}")
                if s.get("png"):
                    st.image(s["png"], caption="Event footprint and track")
                    st.download_button("Download track map (PNG)", Path(s["png"]).read_bytes(),
                                       file_name=Path(s["png"]).name, mime="image/png",
                                       key=f"forecast_track_dl_{k}")
    was_busy = bool(ss.get("forecast_movies_busy"))
    ss["forecast_movies_busy"] = bool(queue) or any(movie.status(r)["state"] == "running" for r in reqs.values())
    if was_busy and not ss["forecast_movies_busy"]:
        st.rerun()  # all done: a full rerun stops the timer


# ═════════════════════════════════════════════════════════════════════════════
# Onset
# ═════════════════════════════════════════════════════════════════════════════
def onset_results(paths: dict, keys: list, init, lead_days, box, region_name):
    defn, search_from = onset_view.controls(region_name, init)
    loaded = [k for k in keys if k in paths]
    usable = [k for k in loaded if not onset_view.usable(_open(paths[k]), catalog.BY_KEY[k], box)]
    for k in loaded:
        if k not in usable:
            st.caption(f"ⓘ {onset_view.usable(_open(paths[k]), catalog.BY_KEY[k], box)}")
    if not usable:
        return defn, search_from
    show = st.multiselect("Show onset maps for", usable, default=usable[:1], key="forecast_onset_models",
                          format_func=lambda k: catalog.BY_KEY[k].name)
    tabs = st.tabs([catalog.BY_KEY[k].name for k in show]) if len(show) > 1 else [st.container()] * len(show)
    for tab, k in zip(tabs, show):
        with tab:
            onset_view.view(paths[k], _open(paths[k]), catalog.BY_KEY[k], init, box, region_name,
                            lead_days, defn, search_from)

    st.markdown("**Onset movies**", help="One frame per forecast day: where onset has happened so far "
                "in the forecast and in the observations. Made only for the models you choose.")
    picked = st.multiselect("Make onset movies for", usable, default=usable[:1],
                            key="forecast_onset_movie_models",
                            format_func=lambda k: catalog.BY_KEY[k].name)
    gifs = {k: onset_view.movie_path(paths[k], box, defn, init, lead_days) for k in picked}
    todo = [k for k in picked if not gifs[k].exists()]
    if todo and st.button(f"Make {len(todo)} onset movie{'s' if len(todo) > 1 else ''}", type="primary",
                          key="forecast_onset_movie_go"):
        failed = False
        for k in todo:  # one after another
            bar = st.progress(0.0, text=f"{catalog.BY_KEY[k].name}: drawing frames…")
            try:
                onset_view.make_movie(paths[k], catalog.BY_KEY[k], box, region_name, init, lead_days, defn,
                                      search_from, lambda i, n, name=catalog.BY_KEY[k].name:
                                      bar.progress(i / n, text=f"{name}: frame {i} of {n}"))
            except OSError as e:
                failed = True
                st.error(f"Couldn't make onset movie for {catalog.BY_KEY[k].name}: {e}")
            finally:
                bar.empty()
        if not failed:
            st.rerun()
    done = [k for k in picked if gifs[k].exists()]
    tabs = st.tabs([catalog.BY_KEY[k].name for k in done]) if len(done) > 1 else [st.container()] * len(done)
    for tab, k in zip(tabs, done):
        with tab:
            gif_html(gifs[k], f"{catalog.BY_KEY[k].name}: onset by day, forecast vs observed")
            st.download_button("Download onset movie (GIF)", gifs[k].read_bytes(), file_name=gifs[k].name,
                               mime="image/gif", key=f"forecast_onset_movie_dl_{k}")
    return defn, search_from


# ═════════════════════════════════════════════════════════════════════════════
# Run panel (auto-refreshing while models are running)
# ═════════════════════════════════════════════════════════════════════════════
def _run_panel(sig, plan):
    s = runs.state(sig)
    before = len(s["paths"])
    was_busy = runs.busy(s)
    runs.tick(s)
    st.dataframe(runs.status_table(s, plan), hide_index=True, width="stretch")
    if s.get("running"):
        with st.expander(f"Log: {catalog.BY_KEY[s['running']].name}", key="forecast_run_log"):
            st.code(jobs.log_tail(s["job"]) or "…")
    if len(s["paths"]) != before or (was_busy and not runs.busy(s)):
        st.rerun()  # a model finished: refresh the results below


def _model_card_html(m, selected: bool, timings) -> str:
    accent = "#8B5CF6" if m.kind == "ensemble" else "#0F766E"
    accent_bg = "#F5F3FF" if m.kind == "ensemble" else "#F0FDF4"
    border = accent if selected else "#E2E8F0"
    shadow = ("0 4px 6px -1px rgba(0,0,0,0.06), 0 0 0 3px rgba(59,130,246,0.12)"
              if selected else "0 1px 3px rgba(0,0,0,0.05)")
    tip = escape(f"{catalog.timing_label(m, timings)}\nStarts from {m.init_source}"
                 + "".join(f"\nNo {catalog.VAR_LABELS[v]}" for v in m.missing)
                 + (f"\n\n{m.notes}" if m.notes else "")).replace("\n", "&#10;")
    dim = not m.live
    badge = f"{'ENS' if m.kind == 'ensemble' else 'DET'} · {'LIVE' if m.live else 'SAVED ONLY'}"
    return f'''
    <div class="fc-card" style="cursor:help;background:#ffffff;border:1px solid {border};border-top:4px solid {accent};
                border-radius:12px;padding:1.1rem 1.25rem;box-shadow:{shadow};height:100%;
                box-sizing:border-box;opacity:{0.55 if dim else 1};margin-bottom:0.75rem;">
        <div style="font-size:1rem;font-weight:700;color:#0F172A;margin-bottom:0.15rem;">{m.name}</div>
        <div style="font-size:0.8rem;color:#64748B;line-height:1.4;margin-bottom:0.4rem;">{m.org}</div>
        <div style="display:inline-block;padding:0.15rem 0.55rem;background:{accent_bg};color:{accent};
                    border-radius:999px;font-size:0.7rem;font-weight:600;letter-spacing:0.03em;
                    text-transform:uppercase;">{badge}</div>
        <div class="fc-tip">{tip}</div>
    </div>'''


# ═════════════════════════════════════════════════════════════════════════════
# Page
# ═════════════════════════════════════════════════════════════════════════════
def render():
    """Main entry point called by app.py"""
    timings = catalog.load_timings()
    saved = store.list_saved()
    gpu = gpu_name()

    # ── Header & stats ───────────────────────────────────────────────────
    st.title("Running Your First AI forecast | Lab")
    st.caption("Pick models, a date and a region. Run them. Compare with what happened.")

    ready = [m for m in catalog.MODELS if m.live]
    free_gb = shutil.disk_usage(store.OUTPUT_DIR if store.OUTPUT_DIR.exists() else "/").free / 1e9
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Models", len(catalog.MODELS))
    col2.metric("Live on Spark", f"{len(ready)} / {len(catalog.MODELS)}")
    col3.metric("Saved Runs", len(saved))
    col4.metric("Free Disk", f"{free_gb:,.0f} GB")
    st.caption(f"GPU: {gpu}" if gpu else "No GPU detected")

    # ── Developer options (main dashboard) ───────────────────────────────
    with st.expander("Developer options", expanded=False, key="forecast_dev_options"):
        st.caption(f"Output folder: `{store.OUTPUT_DIR}`")
        if store.BUCKET:
            st.caption(f"Shared bucket: `{store.BUCKET}`")
        if not saved.empty:
            st.markdown("**Saved runs**")
            st.dataframe(saved.drop(columns="path"), hide_index=True, width="stretch")

    # ── Step 2a: model(s) ────────────────────────────────────────────────
    st.subheader("Step 2 · Model(s)")
    # Store model keys, not Model objects: a session that outlives a code update would
    # otherwise hold objects of the old class.
    model_keys = st.multiselect(
        f"Models (up to {MAX_MODELS})", [m.key for m in catalog.MODELS], default=["graphcast"],
        max_selections=MAX_MODELS, key="forecast_model_keys",
        format_func=lambda k: catalog.BY_KEY[k].name,
        help="Hover over a card below for its details.")
    models = [catalog.BY_KEY[k] for k in model_keys]

    # card details pop up on hover (browsers' own title tooltips are slow or missing in some)
    st.markdown("<style>.fc-card{position:relative}.fc-card:hover{z-index:20}"
                ".fc-tip{display:none;position:absolute;left:10px;right:10px;top:calc(100% + 6px);"
                "background:#0F172A;color:#fff;font-size:.78rem;line-height:1.45;padding:10px 12px;"
                "border-radius:8px;white-space:pre-line;box-shadow:0 6px 18px rgba(0,0,0,.18)}"
                ".fc-card:hover .fc-tip{display:block}</style>", unsafe_allow_html=True)
    per_row = 3
    for i in range(0, len(catalog.MODELS), per_row):
        row = catalog.MODELS[i:i + per_row]
        for col, m in zip(st.columns(per_row), row):
            with col:
                st.markdown(_model_card_html(m, m.key in model_keys, timings),
                            unsafe_allow_html=True)

    if not models:
        st.info("Pick at least one model above.")
        return
    with st.expander("Model details: what " + ("it needs" if len(models) == 1 else "they need")
                     + " on the Spark", key="forecast_model_details"):
        req_tabs = st.tabs([m.name for m in models]) if len(models) > 1 else [st.container()]
        for tab, m in zip(req_tabs, models):
            with tab:
                st.caption(f"{catalog.timing_label(m, timings)} · starts from {m.init_source}"
                           + "".join(f" · no {catalog.VAR_LABELS[v]}" for v in m.missing))
                if m.notes:
                    st.caption(m.notes)
                if m.slow:
                    st.caption("Slow on the Spark. In class, load a saved run where possible.")
                results.requirements(m, timings, store.OUTPUT_DIR)

    # ── Step 2b: forecast setup ──────────────────────────────────────────
    _section("Forecast setup", "sky")
    ranges = {m.key: m.date_range() for m in models}
    init_lo, init_hi = max(lo for lo, _ in ranges.values()), min(hi for _, hi in ranges.values())
    no_common = init_lo > init_hi
    if no_common:  # no date suits every model: offer all; the ones that can't start are left out
        init_lo, init_hi = min(lo for lo, _ in ranges.values()), max(hi for _, hi in ranges.values())
    c1, c2, c3, c4 = st.columns([2, 1, 2, 2])
    default_init = min(max(date(2025, 7, 1), init_lo), init_hi)
    if len(models) == 1:
        date_help = f"{init_lo:%-d %b %Y} – {init_hi:%-d %b %Y}. {models[0].date_reason()}"
    else:
        date_help = "  \n".join(f"{catalog.BY_KEY[k].name}: {lo:%-d %b %Y} – {hi:%-d %b %Y}"
                                for k, (lo, hi) in ranges.items())
    init_day = c1.date_input("Start date (UTC)", value=default_init, min_value=init_lo, max_value=init_hi,
                             key=f"forecast_init_{'_'.join(sorted(model_keys))}", help=date_help)
    init_hour = c2.selectbox("Hour", [0, 12], format_func=lambda h: f"{h:02d}Z", key="forecast_hour")
    max_lead = min(m.max_lead_days for m in models)
    lead_days = c3.slider("Lead time (days)", 1, max_lead, min(10, max_lead), key="forecast_lead_days",
                          help=f"Up to {max_lead} days" + (", the most every selected model can do"
                                                           if max_lead < max(m.max_lead_days for m in models)
                                                           else ""))
    ens_models = [m for m in models if m.kind == "ensemble"]
    if ens_models:
        members = int(c4.number_input("Ensemble members", 1, max(m.max_members for m in ens_models),
                                      max(m.default_members for m in ens_models), key="forecast_members",
                                      help="For the ensemble models, each capped at its own limit on the "
                                           "Spark. Deterministic models give one forecast."))
    else:
        members = 1
        c4.text_input("Ensemble members", "1 (deterministic)", disabled=True, key="forecast_members_text")
    init = pd.Timestamp(init_day) + pd.Timedelta(hours=init_hour)
    if no_common:
        st.warning("No start date works for all the selected models; the ones that can't use the "
                   "chosen date are left out.")

    # ── Step 2c: region ──────────────────────────────────────────────────
    _section("Region", "violet")
    r1, r2 = st.columns([1, 3])
    region_mode = r1.radio("Region", ["Country", "Custom box"], label_visibility="collapsed",
                           key="forecast_region_mode")
    if region_mode == "Country":
        group_of = {c: g for g, cs in GROUPS.items() for c in cs}
        country = r2.selectbox("Country", COUNTRIES, format_func=lambda c: f"{c}  ·  {group_of[c]}",
                               key="forecast_country", help="Maps cover the country plus a margin.")
        box = country_box(country)
        region_name = country
    else:
        b1, b2, b3, b4 = r2.columns(4)
        box = Box(b1.number_input("Lat min", -90.0, 90.0, 3.0, 0.25, key="forecast_lat_min"),
                  b2.number_input("Lat max", -90.0, 90.0, 15.0, 0.25, key="forecast_lat_max"),
                  b3.number_input("Lon min", -180.0, 180.0, 33.0, 0.25, key="forecast_lon_min"),
                  b4.number_input("Lon max", -180.0, 180.0, 48.0, 0.25, key="forecast_lon_max"))
        region_name = "Custom region"
        if not box.is_valid():
            r2.error("Min must be below max; latitude within ±90°, longitude within ±180°.")

    # ── Step 2d: use case ────────────────────────────────────────────────
    _section("What do you want to look at?", "amber")
    use_case = st.segmented_control(
        "Use case", ["Heat", "Precipitation", "Onset"], default="Precipitation", key="forecast_use_case",
        help="**Heat:** daily maximum 2 m temperature (°C).  \n**Precipitation:** daily rainfall (mm).  \n"
             "For both: a movie of the forecast next to the observations, and a map of where the event "
             "went.  \n**Onset:** when the rainy season starts, by a definition you choose (ICPAC, "
             "Moron–Robertson), from observed rain so far plus the forecast.") or "Precipitation"

    for m in models:
        for level, msg in catalog.check(m, init, lead_days, runs.members_for(m, members), box,
                                        region_name, use_case):
            prefix = f"**{m.name}:** " if len(models) > 1 else ""
            suffix = " It is left out of this run." if level == "error" and len(models) > 1 else ""
            (st.error if level == "error" else st.warning)(prefix + msg + suffix)

    # ── Step 3: run ──────────────────────────────────────────────────────
    _section("Step 3 · Run the forecasts" if len(models) > 1 else "Step 3 · Run the forecast", "green")
    tags = [("Model" + ("s" if len(models) > 1 else ""), ", ".join(m.name for m in models)),
            ("Start", f"{init:%Y-%m-%d %HZ}"), ("Lead", f"{lead_days} days"),
            ("Members", str(members) if ens_models else "1"), ("Region", region_name),
            ("Use case", use_case)]
    st.markdown('<div class="kpi-row" style="margin-top:0.5rem;">' +
                "".join(f'<span class="metric-pill"><strong>{k}:</strong> {v}</span>' for k, v in tags) +
                '</div>', unsafe_allow_html=True)

    plan = runs.plan(model_keys, init, lead_days, members, box, region_name, use_case, synthetic=False)
    sig = runs.signature(model_keys, init, lead_days, members)
    s = runs.state(sig)
    actionable = any(w != runs.SKIP for w, _, _ in plan.values())
    if not s["started"]:
        if st.button(runs.button_label(plan), type="primary",
                     disabled=not (actionable and box.is_valid()), key="forecast_run_go"):
            runs.start(s, plan)
            st.rerun()
    st.fragment(_run_panel, run_every=3 if runs.busy(s) else None)(sig, plan)

    # ── Results ──────────────────────────────────────────────────────────
    paths = {k: p for k, p in s["paths"].items() if Path(p).exists()}
    loaded = [k for k in model_keys if k in paths]
    if not loaded:
        return

    _section("Results", "teal")

    with st.expander("How " + ("this forecast was" if len(loaded) == 1 else "these forecasts were") + " made",
                     key="forecast_pipeline_expander"):
        k = (st.selectbox("Model", loaded, key="forecast_pipeline_model",
                          format_func=lambda k: catalog.BY_KEY[k].name)
             if len(loaded) > 1 else loaded[0])
        m, full = catalog.BY_KEY[k], _open(paths[k])
        st.markdown(f"**{m.name}**, started {init:%-d %b %Y %HZ} from "
                    f"{full.attrs.get('init_source', m.init_source)}")
        results.pipeline(paths[k], full, m, timings)

    defn = search_from = None
    if use_case == "Onset":
        st.markdown("#### Onset (vs observations)")
        defn, search_from = onset_results(paths, loaded, init, lead_days, box, region_name)
    else:
        st.markdown("#### Event movies (vs observations)",
                    help="An animated map of the observations (ERA5) next to the forecast, with the "
                         "regional series building up day by day, then a map of the event's footprint "
                         "and track.  \nAIFS and several other models train on ERA5, so this is not a "
                         "fully independent check.")
        event_movies(paths, loaded, init, lead_days, box, region_name, use_case)

    st.markdown("#### Time-series comparison",
                help=f"Every loaded model for {region_name}, from {init:%-d %b %Y %HZ}, as one series.")
    var = "t2m" if use_case == "Heat" else "tp"
    with_var = [k for k in loaded if var not in catalog.missing_in_run(_open(paths[k]))]
    if with_var:
        compare.chart(paths, with_var, init, lead_days, box, region_name, var)
        if use_case == "Onset" and defn is not None:
            compare.onset_table(paths, with_var, init, lead_days, box, region_name, defn, search_from)
    else:
        st.caption(f"None of the loaded models forecasts {catalog.VAR_LABELS[var]}.")

    st.markdown("#### Forecast maps")
    k = (st.selectbox("Model", loaded, key="forecast_maps_model",
                      format_func=lambda k: catalog.BY_KEY[k].name)
         if len(loaded) > 1 else loaded[0])
    m, p = catalog.BY_KEY[k], Path(paths[k])
    full = _open(paths[k])
    ds = results.crop(full, box, lead_days * 24)
    if ds.sizes["lat"] < 2 or ds.sizes["lon"] < 2:
        st.error(f"{region_name} holds fewer than 2 grid points of {m.name}'s "
                 f"{m.grid_deg:g}° grid; choose a larger region.")
    else:
        results.field_maps(ds, m, box, region_name)
    data = results.netcdf_bytes(paths[k], p.stat().st_mtime,
                                (box.lat_min, box.lat_max, box.lon_min, box.lon_max), lead_days * 24)
    if store.BUCKET and store.in_bucket(p):
        theme.hint("✓ In the shared bucket", f"Other Sparks can load this {m.name} run.")
    st.download_button(f"Download this {m.name} forecast for {region_name} (NetCDF, "
                       f"{len(data) / 1e6:.1f} MB)", data, mime="application/x-netcdf",
                       file_name=f"{m.key}_{init:%Y%m%dT%H}_{lead_days}d_"
                                 f"{region_name.replace(' ', '_').lower()}.nc", key=f"forecast_nc_{k}",
                       help=f"Variables tp (mm per 6 h), t2m (K), z500 (m² s⁻²) by member and lead "
                            f"time. The full global file is on the Spark at "
                            f"`{p.relative_to(store.OUTPUT_DIR.parent)}` ({p.stat().st_size / 1e6:.0f} MB).")
