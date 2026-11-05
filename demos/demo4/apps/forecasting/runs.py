"""Get forecasts for the selected models: load saved runs, download shared ones,
and run the rest on the Spark one after another (never two at once, so they
don't compete for memory). State lives in st.session_state["runs"].
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import threading

import pandas as pd
import streamlit as st

from . import catalog, jobs, store

READY, DOWNLOAD, RUN, SKIP = "ready", "download", "run", "skip"

# Downloads use only file/network APIs. Session state is updated by tick on the UI thread.
_download_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="forecast-download")
_download_lock = threading.Lock()
_downloads = {}


def _download(model, name):
    # Two sessions requesting the same destination must not write the same .part file.
    with _download_lock:
        key = (model, name)
        pending = _downloads.get(key)
        if pending is None or pending.done():
            for old in [key for key, future in _downloads.items() if future.done()]:
                del _downloads[old]
            pending = _download_pool.submit(store.download, model, name)
            _downloads[key] = pending
        return pending


def members_for(m: catalog.Model, members: int) -> int:
    return min(members, m.max_members) if m.kind == "ensemble" else 1


def plan(keys, init, lead_days, members, box, region_name, use_case, synthetic=False) -> dict:
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
    if ss.get("runs") is None or (ss["runs"]["sig"] != sig and not busy(ss["runs"])):
        ss["runs"] = dict(sig=sig, started=False, paths={}, queue=[], running=None, job=None,
                          failed={}, plan={}, download=None)
    return ss["runs"]


def start(s: dict, p: dict) -> None:
    if busy(s):
        return
    s.update(started=True, plan=p, failed={}, running=None, job=None, download=None,
             paths={k: info for k, (w, info, _) in p.items() if w == READY},
             queue=[k for k, (w, _, _) in p.items() if w in (DOWNLOAD, RUN)])


def busy(s: dict) -> bool:
    return bool(s.get("running") or s.get("queue"))


def tick(s: dict) -> None:
    """Poll without waiting; keep GPU runs sequential and download off the UI thread."""
    download = s.get("download")
    if download is not None:
        if not download.done():
            return
        try:
            s["paths"][s["running"]] = str(download.result())
        except Exception as exc:
            s["failed"][s["running"]] = f"download failed: {exc}"
        s["running"] = s["download"] = None
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
            s["download"] = _download(k, info[0])
            s["running"] = k
        else:
            try:
                s["job"] = jobs.launch(req)
                s["running"] = k
            except Exception as exc:
                s["failed"][k] = f"could not start: {exc}"



def status_table(s: dict, p: dict) -> pd.DataFrame:
    rows = []
    for k, (what, info, req) in p.items():
        m = catalog.BY_KEY[k]
        members = f"{req.members} member{'s' if req.members > 1 else ''}"
        if k in s["paths"]:
            status = "loaded"
        elif what == SKIP:
            status = f"not included: {info}"
        elif s.get("running") == k and s.get("download") is not None:
            status = "Task running: downloading saved run in the background…"
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


def launch_onset_movies(requests):
    """Render movies in an isolated process; shared Matplotlib state stays out of the UI."""
    import os
    import pickle
    import subprocess
    import sys
    import tempfile
    from pathlib import Path

    folder = tempfile.TemporaryDirectory(prefix="forecast-onset-")
    payload = Path(folder.name) / "request.pkl"
    payload.write_bytes(pickle.dumps(requests))
    log = (Path(folder.name) / "worker.log").open("w")
    command = """
import fcntl
import pickle
import sys
from apps.forecasting import onset_view

requests = pickle.load(open(sys.argv[1], 'rb'))
onset_view.MOVIES.mkdir(parents=True, exist_ok=True)
# Multiple sessions can request the same GIF; serialize writers across processes.
with (onset_view.MOVIES / '.onset-render.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    for args in requests:
        onset_view.make_movie(*args, progress=lambda i, n: print(f'Frame {i}/{n}', flush=True))
"""
    env = dict(os.environ, MPLBACKEND="Agg")
    try:
        process = subprocess.Popen([sys.executable, "-c", command, str(payload)],
                                   cwd=Path(__file__).resolve().parents[2], env=env,
                                   stdout=log, stderr=subprocess.STDOUT)
    except Exception:
        folder.cleanup()
        raise
    finally:
        log.close()
    return {"process": process, "folder": folder}
