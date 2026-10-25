"""Get forecasts for the selected models: load saved runs, download shared ones,
and run the rest on the Spark one after another (never two at once, so they
don't compete for memory). State lives in st.session_state["runs"].
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from . import catalog, jobs, store

READY, DOWNLOAD, RUN, SKIP = "ready", "download", "run", "skip"


def members_for(m: catalog.Model, members: int) -> int:
    return min(members, m.max_members) if m.kind == "ensemble" else 1


def plan(keys, init, lead_days, members, box, region_name, use_case) -> dict:
    """Per model: (what, info, RunRequest). what: ready | download | run | skip."""
    out = {}
    for k in keys:
        m = catalog.BY_KEY[k]
        req = store.RunRequest(k, init, lead_days * 24, members_for(m, members))
        errs = [msg for lvl, msg in catalog.check(m, init, lead_days, req.members, box, region_name,
                                                   use_case) if lvl == "error"]
        saved = store.find_saved(req) or store.find_any(k, init, lead_days * 24)
        remote = None if saved else store.find_remote(req)
        if saved:
            out[k] = (READY, str(saved), req)
        elif errs:
            out[k] = (SKIP, errs[0].split(". ")[0], req)
        elif remote:
            out[k] = (DOWNLOAD, remote, req)
        elif m.live:
            out[k] = (RUN, "", req)
        else:
            out[k] = (SKIP, f"no saved run, and {m.name} can't run live on this Spark", req)
    return out


def signature(keys, init, lead_days, members) -> tuple:
    return (tuple(keys), str(init), lead_days, members)


def state(sig) -> dict:
    """The run state for these settings (a fresh one when the settings change)."""
    ss = st.session_state
    if ss.get("runs") is None or ss["runs"]["sig"] != sig:
        ss["runs"] = dict(sig=sig, started=False, paths={}, queue=[], running=None, job=None,
                          failed={}, plan={})
    return ss["runs"]


def start(s: dict, p: dict) -> None:
    s.update(started=True, plan=p, failed={}, running=None, job=None,
             paths={k: info for k, (w, info, _) in p.items() if w == READY},
             queue=[k for k, (w, _, _) in p.items() if w in (DOWNLOAD, RUN)])


def busy(s: dict) -> bool:
    return bool(s.get("running") or s.get("queue"))


def tick(s: dict) -> None:
    """Advance: collect a finished run, then start the next one (downloads happen right here)."""
    if s["running"]:
        js = jobs.status(s["job"])
        if js["state"] == "done":
            s["paths"][s["running"]] = js["output"]
            s["running"] = s["job"] = None
        elif js["state"] not in ("queued", "running", "unknown"):
            s["failed"][s["running"]] = js.get("message", "see the job log")
            s["running"] = s["job"] = None
    while not s["running"] and s["queue"]:
        k = s["queue"].pop(0)
        what, info, req = s["plan"][k]
        if what == DOWNLOAD:
            try:
                with st.spinner(f"Downloading the saved {catalog.BY_KEY[k].name} run from the shared bucket…"):
                    s["paths"][k] = str(store.download(k, info[0]))
            except Exception as e:  # network or permission trouble
                s["failed"][k] = f"download failed: {e}"
        else:
            s["running"], s["job"] = k, jobs.launch(req)


def status_table(s: dict, p: dict) -> pd.DataFrame:
    rows = []
    for k, (what, info, req) in p.items():
        m = catalog.BY_KEY[k]
        members = f"{req.members} member{'s' if req.members > 1 else ''}"
        if k in s["paths"]:
            status = "loaded"
        elif what == SKIP:
            status = f"not included: {info}"
        elif s.get("running") == k:
            js = jobs.status(s["job"])
            status = (f"running on the Spark: {js.get('message', '')} "
                      f"({100 * js.get('progress', 0):.0f}%, {js.get('elapsed_s', 0):.0f} s)")
        elif k in s["failed"]:
            status = f"failed: {s['failed'][k]}"
        elif s["started"]:
            status = "waiting its turn"
        elif what == READY:
            status = "saved run (loads instantly)"
        elif what == DOWNLOAD:
            status = f"saved run in the shared bucket ({info[1] / 1e6:,.0f} MB)"
        else:
            status = f"will run on the Spark {catalog.timing_label(m)}"
        rows.append((m.name, members, status))
    return pd.DataFrame(rows, columns=["Model", "Members", "Status"])


def button_label(p: dict) -> str:
    n = {w: sum(1 for x, _, _ in p.values() if x == w) for w in (READY, DOWNLOAD, RUN)}
    parts = []
    if n[READY] or n[DOWNLOAD]:
        parts.append(f"load {n[READY] + n[DOWNLOAD]} saved run{'s' if n[READY] + n[DOWNLOAD] > 1 else ''}")
    if n[RUN]:
        parts.append(f"run {n[RUN]} model{'s' if n[RUN] > 1 else ''}"
                     + (" one after another" if n[RUN] > 1 else ""))
    return (" and ".join(parts) or "nothing to run").capitalize()
