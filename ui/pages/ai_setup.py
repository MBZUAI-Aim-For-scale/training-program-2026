"""
ui/pages/ai_setup.py
Demo 1 - AI Weather Lab Setup Dashboard.
Shows the status of all training demos and their containers.
"""
from __future__ import annotations

import json
import subprocess
import sys
from html import escape
from pathlib import Path
from typing import Any, Dict, Tuple

import streamlit as st
from ui.components.styles.loader import load_page_css

# ======================================================================
# Safe import of spark_setup — lives at apps/setup/spark_setup.py
# ======================================================================
SPARK_SETUP_AVAILABLE = False
setup = None

try:
    from apps.setup import spark_setup as setup  # type: ignore
    SPARK_SETUP_AVAILABLE = True
except Exception:
    try:
        import spark_setup as setup  # type: ignore  # fallback if apps/setup is on sys.path
        SPARK_SETUP_AVAILABLE = True
    except Exception:
        setup = None  # type: ignore



# ======================================================================
# Demo registry
# ======================================================================
DEMOS = [
    {"n": 1, "title": "Setting up your AI weather forecasting lab",
     "about": "Check the Spark and get a working environment before diving into the models.",
     "setup": "demo1"},
    {"n": 2, "title": "Observational data readiness",
     "about": "How ENACTS is built by combining rain gauges with satellite products."},
    {"n": 3, "title": "Running your first AI weather forecast",
     "about": "Run AI weather models locally, end to end, for your use case.",
     "container": "demo3", "setup": "demo3"},
    {"n": 4, "title": "The AI weather model scorecard",
     "about": "Compare models across evaluation metrics and explore the AI Almanac for your use case.",
     "container": "ai_weather", "setup": "demo4"},
    {"n": 5, "title": "Putting your forecast to the test",
     "about": "Evaluate the forecasts you ran in Demo 3 against different metrics."},
]

# ======================================================================
# Status helpers
# ======================================================================
# kind -> tag class (accent is now driven by CSS :has(.ai-demo[data-status=...]))
KINDS = {
    "running": "ok",
    "ready":   "ready",
    "stopped": "warn",
    "none":    "off",
    "unavail": "unavail",
}


@st.cache_data(ttl=60)
def gpu_name() -> str | None:
    """Detect GPU via nvidia-smi. Returns None if unavailable."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip().splitlines()[0]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def _demo_status(demo: Dict[str, Any]) -> Tuple[str, str, str | None]:
    """Return (kind, label, link) for a demo."""
    if demo["n"] == 1:
        return "running", "This dashboard", None

    if not SPARK_SETUP_AVAILABLE:
        return "unavail", "Spark setup unavailable", None

    name = demo.get("container")
    if not name:
        return "none", "Coming soon", None

    c = setup.inspect("container", name)
    if c and c["State"]["Running"]:
        ports = setup.host_ports(c)
        link = f"http://localhost:{ports[0]}" if ports else None
        return "running", "Running", link

    if c:
        return "stopped", f"Stopped ({c['State']['Status']})", None

    if setup.inspect("image", name):
        return "ready", "Ready to launch", None

    return "none", "Image not built", None


def _last_run() -> Dict[str, Any] | None:
    if not SPARK_SETUP_AVAILABLE:
        return None
    try:
        run = json.loads(setup.STATUS.read_text())
    except (OSError, ValueError):
        return None
    if run["state"] == "running" and not setup.pid_alive(run["pid"]):
        run["state"] = "interrupted"
    return run


def _progress(steps: list, run_state: str) -> str:
    if not SPARK_SETUP_AVAILABLE:
        return ""
    done = sum(
        1 if s["status"] in ("ok", "warn", "failed", "skipped")
        else (s.get("progress") or 0) if s["status"] == "running" else 0
        for s in steps
    )
    pct = round(100 * done / len(steps)) if steps else 0
    failed = next((s for s in steps if s["status"] == "failed"), None)
    skipped = next((s for s in steps if s["status"] == "skipped"), None)
    current = next((s for s in steps if s["status"] == "running"), None)

    if failed:
        tone, text = "error", (
            f"<b>Failed</b> · {escape(failed['title'].replace('`', ''))}: "
            f"{escape(failed['detail'])}"
        )
    elif skipped:
        tone, text = "warn", f"<b>Not started</b> · {escape(skipped['detail'])}"
    elif current and run_state == "running":
        tone = "ok"
        text = (
            f"<b>{pct}%</b> · {escape(current['title'].replace('`', ''))} "
            + (f": {escape(current['detail'])} " if current["detail"] else " ")
        )
    elif current:
        tone, text = "warn", f"<b>Stopped at {pct}%</b> · the setup run was interrupted"
    elif pct < 100:
        tone, text = "ok", f"<b>{pct}%</b> · waiting for earlier steps"
    else:
        tone, text = "ok", "<b>Done</b> · " + escape(steps[-1]["detail"])

    fix = (
        f'<div class="fix">Fix: {escape(failed["hint"])}</div>'
        if failed and failed.get("hint") else ""
    )
    return (
        f'<div class="ai-prog" data-tone="{tone}">'
        f'<div class="bar"><span style="width:{pct}%"></span></div>'
        f'<div class="txt">{text}</div>{fix}</div>'
    )


def _output(steps: list) -> str:
    if not SPARK_SETUP_AVAILABLE:
        return "spark_setup module not available."
    parts = []
    for s in steps:
        if s.get("log"):
            parts.append(
                f"──── {s['title'].replace('`', '')} [{s['status']}]\n"
                + setup.tail(setup.LOGS / s["log"], 200)
            )
    return "\n\n".join(parts)


def _setup_running() -> bool:
    if not SPARK_SETUP_AVAILABLE:
        return False
    return setup.LOCK.exists() and setup.pid_alive(setup.LOCK.read_text().strip())


def _launch(demo: Dict[str, Any]) -> None:
    if not SPARK_SETUP_AVAILABLE:
        st.error("spark_setup module is not available in this environment.")
        return
    args = (
        ["--host-only"] if demo["setup"] == "demo1"
        else ["--only", demo["setup"]]
    )
    with open(setup.STATE / "launch.log", "a") as log:
        subprocess.Popen(
            [sys.executable, str(setup.HERE / "spark_setup.py"), *args, "--no-dashboard"],
            cwd=setup.HERE,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )


# ======================================================================
# UI Components
# ======================================================================
def _launch_button(demo: Dict[str, Any], kind: str, busy: bool) -> None:
    if demo["n"] == 1:
        text, why = "Run checks", "Re-run the Spark host checks"
    else:
        text, why = "Launch", "Build the image if needed, start the container and wait for its page"

    if not demo.get("setup"):
        why = "This demo doesn't have a container yet"
    elif busy:
        why = "Wait for the current setup run to finish"
    elif kind == "running" and demo["n"] != 1:
        why = "Already running"
    elif kind == "unavail":
        why = "Spark setup module is not available"

    disabled = (
        not demo.get("setup")
        or busy
        or (kind == "running" and demo["n"] != 1)
        or kind == "unavail"
    )

    if st.button(
        text,
        key=f"launch-{demo['n']}",
        type="primary",
        disabled=disabled,
        help=why,
        width="stretch",
    ):
        _launch(demo)


def _demo_row(demo: Dict[str, Any], kind: str, label: str, link: str | None) -> str:
    tag = KINDS[kind]
    open_link = (
        f'<a href="{escape(link)}" target="_blank">Open {escape(link.split("//")[1])} ↗</a>'
        if link else ""
    )
    # data-status drives the card's left-border colour via CSS :has()
    return (
        f'<div class="ai-demo" data-status="{escape(kind)}">'
        f'<div class="num">{demo["n"]}</div>'
        f'<div class="body">'
        f'<div class="n">Demo {demo["n"]} · {escape(demo["title"])}</div>'
        f'<div class="o">{escape(demo["about"])}</div>'
        f'</div>'
        f'<div class="side">'
        f'<span class="ai-tag {tag}">{escape(label)}</span>'
        f'{open_link}'
        f'</div>'
        f'</div>'
    )


def _demo_card(
    demo: Dict[str, Any],
    kind: str,
    label: str,
    link: str | None,
    run: Dict[str, Any] | None,
    busy: bool,
) -> None:
    with st.container(key=f"aicard-{demo['n']}"):
        steps = (
            [s for s in run["steps"] if s["group"] == demo.get("setup")]
            if run and demo.get("setup") else []
        )
        info, button = st.columns([0.84, 0.16])
        # No more per-card <style> injection — CSS :has() handles the accent.
        info.markdown(_demo_row(demo, kind, label, link), unsafe_allow_html=True)
        with button:
            _launch_button(demo, kind, busy)

        if not steps:
            return

        if run["state"] != "done" or any(s["status"] in ("failed", "skipped") for s in steps):
            st.markdown(_progress(steps, run["state"]), unsafe_allow_html=True)
            with st.expander("Command-line output"):
                st.code(_output(steps) or "no output yet", language=None, height=320)


# ======================================================================
# Live refresh fragment
# ======================================================================
@st.fragment(run_every=3)
def _live_status():
    """Re-reads Docker and setup progress every 3 seconds."""
    statuses = [_demo_status(d) for d in DEMOS]
    run = _last_run()
    stats = [
        ("Demos", str(len(DEMOS)), "blue"),
        ("Running", str(sum(k == "running" for k, _, _ in statuses[1:])), "green"),
        ("Ready to launch", str(sum(k == "ready" for k, _, _ in statuses)), "amber"),
    ]
    st.markdown(
        '<div class="ai-stats">'
        + "".join(
            f'<div class="ai-stat" data-tone="{tone}">'
            f'<div class="k">{k}</div>'
            f'<div class="v">{v}</div>'
            f'</div>'
            for k, v, tone in stats
        )
        + '</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="ai-section">Demos</div><div class="ai-rule"></div>',
        unsafe_allow_html=True,
    )
    busy = _setup_running()
    for d, s in zip(DEMOS, statuses):
        _demo_card(d, *s, run, busy)


# ======================================================================
# Page entrypoint (matches home.py / short_medium_benchmarking.py pattern)
# ======================================================================
def page() -> None:
    # Inject page CSS ONCE per run (outside the fragment, so it's not re-sent every 3s).
    load_page_css("ai_setup")
  
    gpu = gpu_name()
    dot = "dot" if gpu else "dot off"

    # Header
    st.markdown(
        f'<div class="ai-header">'
        f'<div class="title">AI Weather Lab <small>| Demo 1: Spark Setup</small></div>'
        f'<div class="ai-live">'
        f'<span class="{dot}"></span>'
        f'{escape("GPU: " + gpu if gpu else "No GPU detected")}'
        f'</div>'
        f'</div>'
        f'<div class="ai-page-title">Setting Up Your AI Weather Forecasting Lab</div>'
        f'<div class="ai-page-sub">Every demo in the training program, and whether it is running on this Spark.</div>',
        unsafe_allow_html=True,
    )

    # Banner when spark_setup is missing
    if not SPARK_SETUP_AVAILABLE:
        st.markdown(
            '<div class="ai-unavail-banner">'
            '<b>⚠️ Spark setup module not available.</b><br>'
            'This dashboard requires the <code>spark_setup</code> module at '
            '<code>apps/setup/spark_setup.py</code>. On your local machine, demo containers '
            'cannot be launched from here — use the Docker Compose workflow instead '
            '(see <code>docker-start.sh</code>).'
            '</div>',
            unsafe_allow_html=True,
        )

    _live_status()


if __name__ == "__main__":
    page()