"""
ui/pages/ai_setup.py
Demo 1 - AI Weather Lab Setup Dashboard
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

# ======================================================================
# Safe import of spark_setup (only available in Spark-hosted deployments)
# ======================================================================
try:
    import spark_setup as setup  # type: ignore
    SPARK_SETUP_AVAILABLE = True
except Exception:
    SPARK_SETUP_AVAILABLE = False
    setup = None  # type: ignore

# ======================================================================
# Demo registry
# ======================================================================
DEMOS = [
    {
        "n": 1,
        "title": "Setting up your AI weather forecasting lab",
        "about": "Check the Spark and get a working environment before diving into the models.",
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
    },
    {
        "n": 4,
        "title": "The AI weather model scorecard",
        "about": "Compare models across evaluation metrics and explore the AI Almanac for your use case.",
        "container": "ai_weather",
        "setup": "demo4",
    },
    {
        "n": 5,
        "title": "Putting your forecast to the test",
        "about": "Evaluate the forecasts you ran in Demo 3 against different metrics.",
    },
]

# ======================================================================
# Theme colors (aligned with the main app's green palette)
# ======================================================================
GREEN_PRIMARY = "#0a9938"
GREEN_DARK = "#077a2c"
TEAL = "#3f8f7a"
BLUE = "#213b7d"
BLUE_LIGHT = "#4b6fd6"
ORANGE = "#d4913a"
RED = "#c2410c"

INK = "#111827"
MUTED = "#6b7280"
LINE = "#e5e7eb"
PAGE = "#f6f7f9"

# ======================================================================
# Page-specific CSS
# ======================================================================
CSS = f"""
<style>
.ai-header {{
    background: linear-gradient(90deg, {GREEN_DARK}, {GREEN_PRIMARY});
    color: #fff;
    border-radius: 0 0 10px 10px;
    padding: 18px 28px;
    margin: 0 -1rem 1.4rem -1rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    flex-wrap: wrap;
    border-bottom: 3px solid {BLUE};
}}
.ai-header .title {{ font-size: 1.35rem; font-weight: 700; }}
.ai-header .title small {{ font-weight: 400; opacity: .75; font-size: .8rem; margin-left: 8px; }}
.ai-live {{
    background: rgba(255,255,255,.12);
    border: 1px solid rgba(255,255,255,.25);
    border-radius: 999px;
    padding: 5px 14px;
    font-family: ui-monospace, monospace;
    font-size: .8rem;
}}
.ai-live .dot {{
    display: inline-block; width: 8px; height: 8px; border-radius: 50%;
    margin-right: 8px; background: {GREEN_PRIMARY};
}}
.ai-live .dot.off {{ background: {ORANGE}; }}

.ai-page-title {{ font-size: 1.9rem; font-weight: 800; color: {INK}; margin: .2rem 0 .1rem; }}
.ai-page-sub {{ color: {MUTED}; margin-bottom: 1.2rem; }}

.ai-stats {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
    margin-bottom: 1.6rem;
}}
@media (max-width: 700px) {{ .ai-stats {{ grid-template-columns: 1fr; }} }}
.ai-stat {{
    background: #fff;
    border: 1px solid {LINE};
    border-left: 3px solid var(--c);
    border-radius: 8px;
    padding: 14px 18px;
}}
.ai-stat .k {{
    font-size: .72rem; letter-spacing: .08em; color: {MUTED};
    text-transform: uppercase;
}}
.ai-stat .v {{
    font-size: 1.35rem; font-weight: 700; color: {INK}; margin-top: 4px;
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}}

.ai-section {{
    font-size: 1.2rem; font-weight: 700; color: {INK};
    border-left: 4px solid {GREEN_PRIMARY};
    padding-left: 10px;
    margin: 1.6rem 0 .8rem;
}}
.ai-rule {{ border-bottom: 1px solid {LINE}; margin: -.3rem 0 1rem; }}

div[class*="st-key-aicard-"] {{
    background: #fff;
    border: 1px solid {LINE};
    border-left: 4px solid {LINE};
    border-radius: 8px;
    padding: 20px 18px;
    gap: 1rem;
}}
.ai-demo {{
    display: flex; align-items: center; gap: 16px; margin-bottom: 1rem;
}}
.ai-demo .num {{
    flex: none; width: 2.2rem; height: 2.2rem; border-radius: 50%;
    background: {PAGE}; color: {GREEN_DARK}; font-weight: 800;
    display: flex; align-items: center; justify-content: center;
}}
.ai-demo .body {{ flex: 1; min-width: 0; }}
.ai-demo .n {{ font-weight: 700; color: {INK}; font-size: 1.02rem; }}
.ai-demo .o {{ color: {MUTED}; font-size: .88rem; margin-top: 2px; }}
.ai-demo .side {{ flex: none; text-align: right; }}
.ai-demo .side a {{ display: block; font-size: .82rem; margin-top: 6px; color: {BLUE}; }}
@media (max-width: 600px) {{
    .ai-demo {{ flex-wrap: wrap; }}
    .ai-demo .side {{ text-align: left; }}
}}

.ai-tag {{
    display: inline-block; font-size: .68rem; font-weight: 700;
    letter-spacing: .06em; border-radius: 6px; padding: 3px 8px;
    text-transform: uppercase; white-space: nowrap;
}}
.ai-tag.ok {{ background: #e7f3ee; color: #2f6f57; }}
.ai-tag.ready {{ background: #e8eefb; color: #34509e; }}
.ai-tag.warn {{ background: #fdf1e2; color: #9a5b12; }}
.ai-tag.off {{ background: #f1f2f4; color: {MUTED}; }}
.ai-tag.unavail {{ background: #fff4e6; color: #a0522d; }}

div[class*="st-key-aicard-"] div[data-testid="stHorizontalBlock"] {{
    align-items: center;
}}
div[class*="st-key-aicard-"] div[data-testid="stExpander"] {{
    margin-left: calc(2.2rem + 16px);
    width: calc(100% - 2.2rem - 16px);
}}
div[class*="st-key-aicard-"] div[data-testid="stExpander"] summary p {{
    font-size: .85rem;
}}
div[class*="st-key-aicard-"] div[data-testid="stCode"] pre,
div[class*="st-key-aicard-"] div[data-testid="stCode"] code {{
    font-size: .72rem !important; line-height: 1.45;
}}

.ai-prog {{ margin-left: calc(2.2rem + 16px); margin-bottom: 1rem; }}
.ai-prog .bar {{
    height: 8px; background: {PAGE}; border: 1px solid {LINE};
    border-radius: 999px; overflow: hidden;
}}
.ai-prog .bar span {{
    display: block; height: 100%; background: var(--p);
    border-radius: 999px; transition: width .6s ease;
}}
.ai-prog .txt {{
    color: {MUTED}; font-size: .82rem; margin-top: 6px;
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}}
.ai-prog .txt b {{ color: {INK}; }}
.ai-prog .fix {{ color: #9a5b12; font-size: .82rem; margin-top: 4px; }}

.ai-unavail-banner {{
    background: #fff4e6;
    border-left: 4px solid {ORANGE};
    padding: 14px 18px;
    border-radius: 6px;
    margin-bottom: 1.2rem;
    color: #7a3e0f;
    font-size: .92rem;
}}
</style>
"""

# ======================================================================
# Status helpers
# ======================================================================
# kind -> (accent color, tag class)
KINDS = {
    "running": (GREEN_PRIMARY, "ok"),
    "ready":   (BLUE_LIGHT,   "ready"),
    "stopped": (ORANGE,       "warn"),
    "none":    (LINE,         "off"),
    "unavail": (ORANGE,       "unavail"),
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
    """
    Return (kind, label, link) for a demo.
    Falls back to 'unavail' if spark_setup is not installed.
    """
    if not SPARK_SETUP_AVAILABLE:
        return "unavail", "Spark setup unavailable", None

    if demo["n"] == 1:
        return "running", "This dashboard", None

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
        color, text = RED, (
            f"<b>Failed</b> · {escape(failed['title'].replace('`', ''))}: "
            f"{escape(failed['detail'])}"
        )
    elif skipped:
        color, text = ORANGE, f"<b>Not started</b> · {escape(skipped['detail'])}"
    elif current and run_state == "running":
        color, text = BLUE_LIGHT, (
            f"<b>{pct}%</b> · {escape(current['title'].replace('`', ''))} "
            + (f": {escape(current['detail'])} " if current["detail"] else " ")
        )
    elif current:
        color, text = ORANGE, f"<b>Stopped at {pct}%</b> · the setup run was interrupted"
    elif pct < 100:
        color, text = BLUE_LIGHT, f"<b>{pct}%</b> · waiting for earlier steps"
    else:
        color, text = GREEN_PRIMARY, "<b>Done</b> · " + escape(steps[-1]["detail"])

    fix = (
        f'<div class="fix">Fix: {escape(failed["hint"])}</div>'
        if failed and failed.get("hint") else ""
    )
    return (
        f'<div class="ai-prog" style="--p:{color}">'
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
        key=f"ailaunch-{demo['n']}",
        type="primary",
        disabled=disabled,
        help=why,
        width="stretch",
    ):
        _launch(demo)


def _demo_row(demo: Dict[str, Any], kind: str, label: str, link: str | None) -> str:
    _, tag = KINDS[kind]
    open_link = (
        f'<a href="{escape(link)}" target="_blank">Open {escape(link.split("//")[1])} ↗</a>'
        if link else ""
    )
    return (
        f'<div class="ai-demo">'
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
        info.markdown(
            f'<style>.st-key-aicard-{demo["n"]} {{ border-left-color: {KINDS[kind][0]} !important; }}</style>'
            + _demo_row(demo, kind, label, link),
            unsafe_allow_html=True,
        )
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
        ("Demos", str(len(DEMOS)), BLUE),
        ("Running", str(sum(k == "running" for k, _, _ in statuses[1:])), GREEN_PRIMARY),
        ("Ready to launch", str(sum(k == "ready" for k, _, _ in statuses)), ORANGE),
    ]
    st.markdown(
        '<div class="ai-stats">'
        + "".join(
            f'<div class="ai-stat" style="--c:{c}">'
            f'<div class="k">{k}</div>'
            f'<div class="v">{v}</div>'
            f'</div>'
            for k, v, c in stats
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
# Page entrypoint (matches the pattern used by home.py and short_medium_benchmarking.py)
# ======================================================================
def page() -> None:
    st.markdown(CSS, unsafe_allow_html=True)

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
            '<b>Spark setup module not available.</b><br>'
            'This dashboard requires the <code>spark_setup</code> module, which is only '
            'installed on Spark-hosted deployments. On your local machine, demo containers '
            'cannot be launched from here — use the Docker Compose workflow instead '
            '(see <code>docker-start.sh</code>).'
            '</div>',
            unsafe_allow_html=True,
        )

    _live_status()


if __name__ == "__main__":
    page()