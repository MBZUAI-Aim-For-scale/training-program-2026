"""
ui/pages/ai_setup.py
Demo 1 - AI Weather Lab Setup Dashboard
Shows status of all training demos, their containers, and setup progress.
Includes selective/individual container stopping and global shutdown.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from html import escape
from pathlib import Path
from typing import Any, Dict, List, Tuple

import streamlit as st

# Safe import of spark_setup from apps.setup
try:
    from apps.setup import spark_setup as setup  # type: ignore
    SPARK_SETUP_AVAILABLE = True
except ImportError:
    try:
        import spark_setup as setup  # type: ignore
        SPARK_SETUP_AVAILABLE = True
    except ImportError:
        SPARK_SETUP_AVAILABLE = False
        setup = None  # type: ignore

REPO = Path(__file__).resolve().parents[2]

# Demo registry aligned with demos.toml and Dockerfile paths
DEMOS = [
    {
        "n": 1,
        "title": "Setting up your AI weather forecasting lab",
        "about": "Check the Spark host and verify GPU environment.",
        "setup": "demo1",
    },
    {
        "n": 2,
        "title": "Observational data readiness",
        "about": "How ENACTS is built by combining rain gauges with satellite products.",
    },
    {
        "n": 3,
        "title": "Running your first AI weather forecast",
        "about": "Run AI weather models locally, end to end, for your use case.",
        "container": "demo3",
        "setup": "demo3",
        "dockerfile": "demos/demo3/docker/Dockerfile",
    },
    {
        "n": 4,
        "title": "The AI weather model scorecard",
        "about": "Compare models across evaluation metrics and explore the AI Almanac.",
        "container": "ai_weather",
        "setup": "demo4",
        "dockerfile": "docker/Dockerfile",
    },
    {
        "n": 5,
        "title": "Putting your forecast to the test",
        "about": "Evaluate the forecasts you ran in Demo 3 against different metrics.",
    },
]

TAG_CLASSES = {
    "running": "ok",
    "ready":   "ready",
    "stopped": "warn",
    "none":    "off",
    "unavail": "unavail",
}


def load_css() -> None:
    css_locations = [
        Path(__file__).parent / "ai_setup.css",
        Path(__file__).parent.parent / "styles" / "ai_setup.css",
        REPO / "apps" / "setup" / "ai_setup.css",
    ]
    for css_path in css_locations:
        if css_path.exists():
            st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)
            return


@st.cache_data(ttl=60)
def gpu_name() -> str | None:
    try:
        cmd = shutil.which("nvidia-smi") or "/usr/bin/nvidia-smi"
        out = subprocess.run(
            [cmd, "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip().splitlines()[0]
    except Exception:
        pass
    return None


def _demo_status(demo: Dict[str, Any]) -> Tuple[str, str, str | None]:
    if demo["n"] == 1:
        return "running", "This dashboard", None

    name = demo.get("container")
    if not name:
        return "none", "Coming soon", None

    if SPARK_SETUP_AVAILABLE and setup:
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
    if not SPARK_SETUP_AVAILABLE or not setup or not setup.STATUS.exists():
        return None
    try:
        run = json.loads(setup.STATUS.read_text())
    except (OSError, ValueError):
        return None
    if run.get("state") == "running" and not setup.pid_alive(run.get("pid")):
        run["state"] = "interrupted"
    return run


def _progress(steps: list, run_state: str) -> str:
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
        tone, text = "blue", (
            f"<b>{pct}%</b> · {escape(current['title'].replace('`', ''))} "
            + (f": {escape(current['detail'])} " if current["detail"] else " ")
        )
    elif current:
        tone, text = "warn", f"<b>Stopped at {pct}%</b> · setup run was interrupted"
    elif pct < 100:
        tone, text = "blue", f"<b>{pct}%</b> · waiting for earlier steps"
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
    if not SPARK_SETUP_AVAILABLE or not setup:
        return "Log output unavailable."
    parts = []
    for s in steps:
        if s.get("log"):
            parts.append(
                f"──── {s['title'].replace('`', '')} [{s['status']}]\n"
                + setup.tail(setup.LOGS / s["log"], 200)
            )
    return "\n\n".join(parts)


def _setup_running() -> bool:
    if not SPARK_SETUP_AVAILABLE or not setup:
        return False
    return setup.LOCK.exists() and setup.pid_alive(setup.LOCK.read_text().strip())


# ======================================================================
# Container Management Executors
# ======================================================================

def _launch(demo: Dict[str, Any]) -> None:
    """Launch container via apps/setup/launch_dockerfiles.sh (or fallback to spark_setup.py)."""
    script_sh = REPO / "apps" / "setup" / "launch_dockerfiles.sh"
    dockerfile = demo.get("dockerfile")

    if script_sh.exists() and dockerfile:
        log_dir = REPO / "apps" / "setup" / "state"
        log_dir.mkdir(parents=True, exist_ok=True)
        with open(log_dir / "launch.log", "a") as log:
            subprocess.Popen(
                ["bash", str(script_sh), dockerfile],
                cwd=REPO,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        st.toast(f"Launching {demo['title']} via launch_dockerfiles.sh…")
        return

    if SPARK_SETUP_AVAILABLE and setup:
        args = (
            ["--host-only"] if demo["setup"] == "demo1"
            else ["--only", demo["setup"]]
        )
        script = str(setup.HERE / "spark_setup.py")
        with open(setup.STATE / "launch.log", "a") as log:
            subprocess.Popen(
                [sys.executable, script, *args, "--no-dashboard"],
                cwd=REPO,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        st.toast(f"Running setup for {demo['title']} via spark_setup.py…")


def _stop_containers(container_names: List[str]) -> None:
    """Stop specific container(s) by name asynchronously."""
    if not container_names:
        return
    for name in container_names:
        subprocess.Popen(
            ["docker", "stop", name],
            cwd=REPO,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    targets = ", ".join(container_names)
    st.toast(f"Stopping container(s): {targets}…")


def _shutdown_all() -> None:
    """Stop setup runs and ALL containers via apps/setup/shutdown.sh."""
    script_sh = REPO / "apps" / "setup" / "shutdown.sh"
    if script_sh.exists():
        log_dir = REPO / "apps" / "setup" / "state"
        log_dir.mkdir(parents=True, exist_ok=True)
        with open(log_dir / "shutdown.log", "a") as log:
            subprocess.Popen(
                ["bash", str(script_sh)],
                cwd=REPO,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        st.toast("Stopping all containers and active setup runs via shutdown.sh…")
    else:
        st.error("apps/setup/shutdown.sh script not found.")


# ======================================================================
# UI Components
# ======================================================================

def _action_button(demo: Dict[str, Any], kind: str, busy: bool) -> None:
    """Render a Launch button when idle, or a Stop button when running."""
    container_name = demo.get("container")

    # Running container action: Stop
    if kind == "running" and demo["n"] != 1 and container_name:
        if st.button(
            "Stop",
            key=f"aistop-{demo['n']}",
            type="secondary",
            help=f"Stop the running '{container_name}' container",
            use_container_width=True,
        ):
            _stop_containers([container_name])
        return

    # Idle container action: Launch / Run checks
    if demo["n"] == 1:
        text, why = "Run checks", "Re-run the Spark host checks"
    else:
        text, why = "Launch", "Build and start container via launch_dockerfiles.sh"

    if not demo.get("setup"):
        why = "This demo doesn't have a container yet"
    elif busy:
        why = "Wait for current setup run to finish"

    disabled = not demo.get("setup") or busy

    if st.button(
        text,
        key=f"ailaunch-{demo['n']}",
        type="primary",
        disabled=disabled,
        help=why,
        use_container_width=True,
    ):
        _launch(demo)


def _demo_row(demo: Dict[str, Any], kind: str, label: str, link: str | None) -> str:
    tag = TAG_CLASSES.get(kind, "off")
    open_link = (
        f'<a href="{escape(link)}" target="_blank" style="text-decoration:none; font-weight:600;">Open App ↗</a>'
        if link else ""
    )
    return (
        f'<div class="ai-demo" data-status="{escape(kind)}">'
        f'<div class="num">{demo["n"]}</div>'
        f'<div class="body">'
        f'<div class="n">Demo {demo["n"]} · {escape(demo["title"])}</div>'
        f'<div class="o">{escape(demo["about"])}</div>'
        f'</div>'
        f'<div class="side">'
        f'<span class="ai-tag {tag}">{escape(label)}</span>'
        f'<div style="margin-top:4px;">{open_link}</div>'
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
    with st.container(border=True):
        steps = (
            [s for s in run["steps"] if s["group"] == demo.get("setup")]
            if run and demo.get("setup") else []
        )
        info, button = st.columns([0.82, 0.18], vertical_alignment="center")
        with info:
            st.markdown(_demo_row(demo, kind, label, link), unsafe_allow_html=True)
        with button:
            _action_button(demo, kind, busy)

        if not steps:
            return

        if run and (run.get("state") != "done" or any(s["status"] in ("failed", "skipped") for s in steps)):
            st.markdown(_progress(steps, run["state"]), unsafe_allow_html=True)
            with st.expander("Command-line output", expanded=False):
                st.code(_output(steps) or "no output yet", language=None, height=280)


@st.fragment(run_every=3)
def _live_status():
    statuses = [_demo_status(d) for d in DEMOS]
    run = _last_run()
    
    # Collect currently running demo containers
    running_demos = [
        d for d, (kind, _, _) in zip(DEMOS, statuses)
        if kind == "running" and d.get("container")
    ]

    # Clean Stat Metrics Header
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Demos", len(DEMOS))
    col2.metric("Running Containers", sum(k == "running" for k, _, _ in statuses[1:]))
    col3.metric("Ready to Launch", sum(k == "ready" for k, _, _ in statuses))

    st.markdown("---")

    # Section Header with Bulk Container Management
    c_title, c_mgmt = st.columns([0.65, 0.35], vertical_alignment="center")
    with c_title:
        st.subheader("Environment & Demo Workspaces")
    with c_mgmt:
        if running_demos:
            with st.popover("🛑 Bulk Stop Controls", use_container_width=True):
                st.markdown("**Stop Running Containers**")
                c_options = {f"Demo {d['n']}: {d['container']}": d["container"] for d in running_demos}
                selected_labels = st.multiselect(
                    "Select containers:",
                    options=list(c_options.keys()),
                    default=list(c_options.keys()),
                )
                if st.button("Stop Selected", key="stop-selected-btn", type="primary", use_container_width=True):
                    selected_containers = [c_options[lbl] for lbl in selected_labels]
                    _stop_containers(selected_containers)
                st.divider()
                if st.button("Shutdown All (shutdown.sh)", key="stop-all-btn", type="secondary", use_container_width=True):
                    _shutdown_all()

    busy = _setup_running()
    for d, s in zip(DEMOS, statuses):
        _demo_card(d, *s, run, busy)


def page() -> None:
    load_css()

    gpu = gpu_name()
    dot_status = "🟢" if gpu else "🟠"

    # Header section
    st.title("AI Weather Lab Setup Dashboard")
    st.caption(f"{dot_status} **Hardware Environment:** {gpu if gpu else 'No NVIDIA GPU detected on host'}")
    st.markdown("Monitor and control training demos, containers, and GPU node setup directly from this control panel.")
    st.markdown("---")

    _live_status()


if __name__ == "__main__":
    page()