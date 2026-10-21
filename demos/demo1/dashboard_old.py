"""Spark setup dashboard: progress of demo1's setup run, and live health of every demo container.

Started by `python3 spark_setup.py`, which does the setup work; this page only reports.
Reads state/status.json written by spark_setup.py, and asks Docker directly for each
container's state, so it flags crashed or unhealthy demos even between setup runs.
"""
from __future__ import annotations

import json
import os
import shutil
import re
import subprocess
import time

import streamlit as st

import spark_setup as setup

st.set_page_config(page_title="Spark setup", page_icon="🛰️", layout="wide")

CMD = "python3 demos/demo1-setup/spark_setup.py"
ICON = {"pending": "⚪", "running": "🔵", "ok": "🟢", "warn": "🟡", "failed": "🔴", "skipped": "⚫"}
# Lines in a container's log that are worth flagging (Streamlit prints tracebacks on page errors)
LOG_ERROR = re.compile(r"Traceback|^\s*\w*(Error|Exception)\b|CUDA error|out of memory|Killed|FATAL", re.I)


# --- Data ---------------------------------------------------------------------------------

def load_run() -> dict | None:
    try:
        run = json.loads(setup.STATUS.read_text())
    except (OSError, ValueError):
        return None
    if run["state"] == "running" and not setup.pid_alive(run["pid"]):
        run["state"] = "interrupted"
    return run


@st.cache_data(ttl=3, show_spinner=False)
def live_demo(demo: dict) -> dict:
    """What Docker says about a demo right now."""
    c = setup.inspect("container", demo["container"])
    image = setup.inspect("image", demo["image"])
    info = {"image": bool(image), "exists": bool(c), "running": False, "status": "not created",
            "ports": [], "healthy": False, "error": "", "exit_code": None, "started": ""}
    if c:
        s = c["State"]
        info.update(running=s["Running"], status=s["Status"], error=s.get("Error", ""),
                    exit_code=s.get("ExitCode"), started=s.get("StartedAt", ""), ports=setup.host_ports(c),
                    restarting=s.get("Restarting", False))
        port = demo["port"] if demo["port"] in info["ports"] or not info["ports"] else info["ports"][0]
        info["healthy"] = s["Running"] and setup.healthy(port, timeout=1.5)
    return info


@st.cache_data(ttl=15, show_spinner=False)
def log_errors(container: str) -> tuple[list[str], str]:
    """Error-looking lines in the container's recent log, and the log tail."""
    text = setup.container_logs(container, 400)
    return [l for l in text.splitlines() if LOG_ERROR.search(l)], "\n".join(text.splitlines()[-80:])


@st.cache_data(ttl=10, show_spinner=False)
def machine() -> dict:
    out = {}
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,utilization.gpu,temperature.gpu",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
        name, util, temp = [x.strip() for x in r.stdout.strip().split(",")]
        out["gpu"] = (name, util, temp)
    except Exception:
        out["gpu"] = None
    try:
        mem = dict(l.split(":", 1) for l in open("/proc/meminfo"))
        total, avail = (int(mem[k].split()[0]) / 1e6 for k in ("MemTotal", "MemAvailable"))
        out["mem"] = (total - avail, total)
    except Exception:
        out["mem"] = None
    du = shutil.disk_usage(os.path.expanduser("~"))
    out["disk"] = (du.free / 1e9, du.total / 1e9)
    try:
        out["docker"] = setup.docker("info", "--format", "{{.ServerVersion}}", timeout=5).returncode == 0
    except Exception:
        out["docker"] = False
    return out


# --- Issues -------------------------------------------------------------------------------

def collect_issues(cfg: dict, run: dict | None, live: dict) -> list[dict]:
    """Everything that needs attention: failed/warned steps plus live container problems."""
    issues = []
    if run:
        if run["state"] == "interrupted":
            issues.append({"level": "warn", "where": "setup", "what": "The last setup run stopped before finishing.",
                           "hint": f"Run it again: `{CMD}`"})
        for s in run["steps"]:
            if s["status"] in ("failed", "warn"):
                issues.append({"level": "error" if s["status"] == "failed" else "warn", "where": s["group"],
                               "what": f"{s['title']}: {s['detail']}", "hint": s["hint"], "log": s.get("log")})

    gpu_ok = any(s["id"] == "host.cdi" and s["status"] == "ok" for s in (run or {}).get("steps", []))
    ports: dict[int, str] = {}
    for d in cfg["demo"]:
        if d["port"] in ports:
            issues.append({"level": "error", "where": d["id"],
                           "what": f"Port {d['port']} is used by both {ports[d['port']]} and {d['id']} in demos.toml.",
                           "hint": "Give each demo its own port, or only one of them can run."})
        ports[d["port"]] = d["id"]

        info = live[d["id"]]
        if not info["exists"]:
            continue  # not installed yet: shown in its card, not an error
        if info.get("restarting"):
            issues.append({"level": "error", "where": d["id"], "what": f"`{d['container']}` keeps restarting.",
                           "hint": "Check the container log below."})
        elif not info["running"] and (info["exit_code"] or info["error"]):
            what = info["error"] or f"exited with code {info['exit_code']}"
            errs, tail = log_errors(d["container"])
            retry = f"`{CMD} --only {d['id']}`"
            hint = setup.hint_for(what + "\n" + tail) or f"Check the container log, then {retry}."
            if "CDI" in what and gpu_ok:
                hint = f"Containers can reach the GPU again (host check passed), so {retry} should bring it back."
            issues.append({"level": "error", "where": d["id"], "what": f"`{d['container']}` is down: {what}", "hint": hint})
        elif info["running"] and not info["healthy"]:
            issues.append({"level": "error", "where": d["id"], "what": f"`{d['container']}` is running but its page isn't answering.",
                           "hint": "It may still be starting; otherwise check the container log."})
        if info["ports"] and d["port"] not in info["ports"]:
            issues.append({"level": "warn", "where": d["id"],
                           "what": f"`{d['container']}` publishes :{', :'.join(map(str, info['ports']))}, demos.toml says :{d['port']}.",
                           "hint": ("Stop it, then " if info["running"] else "")
                                   + f"`{CMD} --only {d['id']}` recreates it on the right port."})
        if info["running"]:
            errs, _ = log_errors(d["container"])
            if errs:
                issues.append({"level": "warn", "where": d["id"],
                               "what": f"{len(errs)} error line(s) in `{d['container']}`'s recent log; last: `{errs[-1].strip()[:160]}`",
                               "hint": "Open **Container log** on its card."})
    return issues


# --- Rendering ----------------------------------------------------------------------------

def ago(t: float | None) -> str:
    if not t:
        return ""
    s = int(time.time() - t)
    return f"{s}s ago" if s < 60 else f"{s // 60} min ago" if s < 3600 else f"{s // 3600} h ago"


def duration(step: dict) -> str:
    if not step.get("started"):
        return ""
    s = int((step.get("finished") or time.time()) - step["started"])
    return f"{s}s" if s < 120 else f"{s // 60} min"


def render_steps(steps: list[dict]):
    for s in steps:
        cols = st.columns([0.06, 0.94])
        cols[0].markdown(ICON[s["status"]])
        with cols[1]:
            line = f"**{s['title']}**"
            if s["detail"]:
                line += f" — {s['detail']}"
            if duration(s):
                line += f"  ·  _{duration(s)}_"
            st.markdown(line)
            if s["status"] == "running" and s.get("progress") is not None:
                st.progress(min(max(s["progress"], 0.0), 1.0))
            if s["hint"]:
                st.caption(f"Fix: {s['hint']}")
            if s.get("log") and s["status"] != "pending":
                with st.expander("Step log"):
                    st.code(setup.tail(setup.LOGS / s["log"], 60) or "(empty)", language="text")


def render_demo(d: dict, run: dict | None, info: dict):
    with st.container(border=True):
        head, badge = st.columns([0.7, 0.3])
        head.subheader(d["title"])
        if info["healthy"]:
            badge.success(f"Up · [localhost:{d['port']}](http://localhost:{d['port']})")
        elif info["running"]:
            badge.warning("Running, page not answering")
        elif info["exists"]:
            badge.error(f"Down ({info['status']})")
        else:
            badge.info("Not installed" if not info["image"] else "Image built, not started")
        st.caption(f"image `{d['image']}` {'✓' if info['image'] else '✗'}  ·  container `{d['container']}`"
                   + (f"  ·  started {info['started'][:16].replace('T', ' ')} UTC" if info["running"] else ""))

        steps = [s for s in (run or {}).get("steps", []) if s["group"] == d["id"]]
        if steps:
            render_steps(steps)
        else:
            st.caption("Not part of the last setup run.")

        if info["exists"]:
            errs, tail = log_errors(d["container"])
            with st.expander(f"Container log{f' · {len(errs)} error line(s)' if errs else ''}"):
                st.code(tail or "(empty)", language="text")


@st.fragment(run_every="3s")
def page():
    cfg = setup.load_config()
    run = load_run()
    busy = bool(run and run["state"] == "running")
    live = {d["id"]: live_demo(d) for d in cfg["demo"]}
    issues = collect_issues(cfg, run, live)

    # Header
    st.title("Spark setup")
    if run:
        label = {"running": "🔵 running", "done": "🟢 finished", "failed": "🔴 finished with errors",
                 "interrupted": "🟡 interrupted"}[run["state"]]
        st.caption(f"Setup run: {label} · started {ago(run['started'])} · "
                   f"`python3 spark_setup.py {' '.join(run.get('args') or [])}`")
    else:
        st.caption(f"No setup run yet. Start one in a terminal: `{CMD}`")

    # Summary
    steps = (run or {}).get("steps", [])
    finished = [s for s in steps if s["status"] in ("ok", "warn", "failed", "skipped")]
    up = sum(i["healthy"] for i in live.values())
    m = st.columns(4)
    m[0].metric("Setup steps done", f"{len(finished)}/{len(steps)}" if steps else "–")
    m[1].metric("Errors", sum(i["level"] == "error" for i in issues))
    m[2].metric("Warnings", sum(i["level"] == "warn" for i in issues))
    m[3].metric("Demos up", f"{up}/{len(live)}")
    if busy and steps:
        current = next((s for s in steps if s["status"] == "running"), None)
        st.progress(len(finished) / len(steps), text=f"Now: {current['title']}" if current else "Starting…")

    # Issues
    if issues:
        st.subheader("Needs attention")
        for i in issues:
            box = st.error if i["level"] == "error" else st.warning
            box(f"**{i['where']}** · {i['what']}" + (f"\n\n**Fix:** {i['hint']}" if i.get("hint") else ""),
                icon="🚨" if i["level"] == "error" else "⚠️")
    elif run and run["state"] == "done":
        st.success("Everything is set up and every demo is answering.", icon="✅")

    # Demo 1: the Spark itself
    left, right = st.columns([0.55, 0.45], gap="large")
    with left:
        with st.container(border=True):
            st.subheader("Demo 1 · Spark host")
            host_steps = [s for s in steps if s["group"] == "demo1"]
            if host_steps:
                render_steps(host_steps)
            else:
                st.caption("Host checks haven't run yet.")
    with right:
        with st.container(border=True):
            st.subheader("Machine now")
            mc = machine()
            if mc["gpu"]:
                name, util, temp = mc["gpu"]
                st.markdown(f"**{name}** · {util}% busy · {temp} °C")
            else:
                st.markdown("🔴 `nvidia-smi` not answering")
            if mc["mem"]:
                used, total = mc["mem"]
                st.progress(used / total, text=f"Memory (shared CPU/GPU): {used:.0f} / {total:.0f} GB")
            free, total = mc["disk"]
            st.progress(1 - free / total, text=f"Disk: {free:,.0f} GB free of {total:,.0f} GB")
            st.markdown("🟢 Docker running" if mc["docker"] else "🔴 Docker not answering")

    # One card per demo container
    cols = st.columns(2, gap="large")
    for n, d in enumerate(cfg["demo"]):
        with cols[n % 2]:
            render_demo(d, run, live[d["id"]])


page()
