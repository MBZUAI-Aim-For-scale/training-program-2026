"""Demo 1 dashboard. Started by `python3 spark_setup.py dashboard` on http://localhost:8500.

Styled like the Demo 3 app (demos/demo3/demo3/theme.py): navy header, light page, white cards.
"""
from __future__ import annotations

import json
import subprocess
import sys
from html import escape

import streamlit as st

import spark_setup as setup

# Every demo in the training program, in order. `container` is set once a demo has a Docker image;
# `setup` is its id in demos.toml, whose steps spark_setup.py records in state/status.json.
DEMOS = [
    {"n": 1, "title": "Setting up your AI weather forecasting lab",
     "about": "Check the Spark and get a working environment before diving into the models.", "setup": "demo1"},
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
    # {"n": 6, "title": "Ground truth challenge",
    #  "about": "Compare model performance against different observational datasets."},
    # {"n": 7, "title": "Feasibility assessment",
    #  "about": "Decide which models, data and thresholds to take forward into a pilot."},
]

# Same palette as Demo 3
NAVY, NAVY_DARK, BLUE, TEAL = "#1f2f6b", "#172457", "#4b6fd6", "#3f8f7a"
GREEN, ORANGE, RED = "#5fb58a", "#d4913a", "#c2410c"
INK, MUTED, LINE, PAGE = "#111827", "#6b7280", "#e5e7eb", "#f6f7f9"

CSS = f"""
<style>
#MainMenu, footer, header[data-testid="stHeader"] {{ visibility: hidden; height: 0; }}
.stApp {{ background: {PAGE}; }}
.block-container {{ padding-top: 0.5rem; max-width: 1000px; }}

.d1-header {{ background: linear-gradient(90deg, {NAVY_DARK}, {NAVY}); color: #fff;
  border-radius: 0 0 10px 10px; padding: 18px 28px; margin: 0 -1rem 1.4rem -1rem;
  display: flex; align-items: center; justify-content: space-between; gap: 12px; flex-wrap: wrap;
  border-bottom: 3px solid {BLUE}; }}
.d1-header .title {{ font-size: 1.35rem; font-weight: 700; }}
.d1-header .title small {{ font-weight: 400; opacity: .75; font-size: .8rem; margin-left: 8px; }}
.d1-live {{ background: rgba(255,255,255,.12); border: 1px solid rgba(255,255,255,.25);
  border-radius: 999px; padding: 5px 14px; font-family: ui-monospace, monospace; font-size: .8rem; }}
.d1-live .dot {{ display: inline-block; width: 8px; height: 8px; border-radius: 50%;
  margin-right: 8px; background: {GREEN}; }}
.d1-live .dot.off {{ background: {ORANGE}; }}

.d1-page-title {{ font-size: 1.9rem; font-weight: 800; color: {INK}; margin: .2rem 0 .1rem; }}
.d1-page-sub {{ color: {MUTED}; margin-bottom: 1.2rem; }}

.d1-stats {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-bottom: 1.6rem; }}
@media (max-width: 700px) {{ .d1-stats {{ grid-template-columns: 1fr; }} }}
.d1-stat {{ background: #fff; border: 1px solid {LINE}; border-left: 3px solid var(--c);
  border-radius: 8px; padding: 14px 18px; }}
.d1-stat .k {{ font-size: .72rem; letter-spacing: .08em; color: {MUTED}; text-transform: uppercase; }}
.d1-stat .v {{ font-size: 1.35rem; font-weight: 700; color: {INK}; margin-top: 4px;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}

.d1-section {{ font-size: 1.2rem; font-weight: 700; color: {INK}; border-left: 4px solid {TEAL};
  padding-left: 10px; margin: 1.6rem 0 .8rem; }}
.d1-rule {{ border-bottom: 1px solid {LINE}; margin: -.3rem 0 1rem; }}

div[class*="st-key-d1card-"] {{ background: #fff; border: 1px solid {LINE}; border-left: 4px solid {LINE};
  border-radius: 8px; padding: 20px 18px; gap: 1rem; }}
/* the extra bottom margin makes up for Streamlit's -1rem under markdown, which let the dropdown overlap */
.d1-demo {{ display: flex; align-items: center; gap: 16px; margin-bottom: 1rem; }}
.d1-demo .num {{ flex: none; width: 2.2rem; height: 2.2rem; border-radius: 50%; background: {PAGE};
  color: {NAVY}; font-weight: 800; display: flex; align-items: center; justify-content: center; }}
.d1-demo .body {{ flex: 1; min-width: 0; }}
.d1-demo .n {{ font-weight: 700; color: {INK}; font-size: 1.02rem; }}
.d1-demo .o {{ color: {MUTED}; font-size: .88rem; margin-top: 2px; }}
.d1-demo .side {{ flex: none; text-align: right; }}
.d1-demo .side a {{ display: block; font-size: .82rem; margin-top: 6px; color: {BLUE}; }}
@media (max-width: 600px) {{ .d1-demo {{ flex-wrap: wrap; }} .d1-demo .side {{ text-align: left; }} }}

.d1-tag {{ display: inline-block; font-size: .68rem; font-weight: 700; letter-spacing: .06em;
  border-radius: 6px; padding: 3px 8px; text-transform: uppercase; white-space: nowrap; }}
.d1-tag.ok {{ background: #e7f3ee; color: #2f6f57; }}
.d1-tag.ready {{ background: #e8eefb; color: #34509e; }}
.d1-tag.warn {{ background: #fdf1e2; color: #9a5b12; }}
.d1-tag.off {{ background: #f1f2f4; color: {MUTED}; }}

div[class*="st-key-d1card-"] div[data-testid="stHorizontalBlock"] {{ align-items: center; }}
div[class*="st-key-launch-"] button {{ border-radius: 8px; }}
div[class*="st-key-launch-"] button[kind="primary"]:not(:disabled) {{ background: {NAVY}; border-color: {NAVY}; }}
div[class*="st-key-launch-"] button[kind="primary"]:not(:disabled):hover {{ background: {NAVY_DARK}; border-color: {NAVY_DARK}; }}
div[class*="st-key-launch-"] button:disabled {{ background: {PAGE}; border-color: {LINE}; color: {MUTED}; }}

.d1-prog {{ margin-left: calc(2.2rem + 16px); margin-bottom: 1rem; }}
.d1-prog .bar {{ height: 8px; background: {PAGE}; border: 1px solid {LINE}; border-radius: 999px; overflow: hidden; }}
.d1-prog .bar span {{ display: block; height: 100%; background: var(--p); border-radius: 999px;
  transition: width .6s ease; }}
.d1-prog .txt {{ color: {MUTED}; font-size: .82rem; margin-top: 6px;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
.d1-prog .txt b {{ color: {INK}; }}
.d1-prog .fix {{ color: #9a5b12; font-size: .82rem; margin-top: 4px; }}
div[class*="st-key-d1card-"] div[data-testid="stExpander"] {{ margin-left: calc(2.2rem + 16px);
  width: calc(100% - 2.2rem - 16px); }}
div[class*="st-key-d1card-"] div[data-testid="stExpander"] summary p {{ font-size: .85rem; }}
div[class*="st-key-d1card-"] div[data-testid="stCode"] pre,
div[class*="st-key-d1card-"] div[data-testid="stCode"] code {{ font-size: .72rem !important; line-height: 1.45; }}
</style>
"""

# status kind -> (accent colour on the card, tag style)
KINDS = {"running": (GREEN, "ok"), "ready": (BLUE, "ready"), "stopped": (ORANGE, "warn"), "none": (LINE, "off")}


@st.cache_data(ttl=60)
def gpu_name() -> str | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip().splitlines()[0] if out.returncode == 0 and out.stdout.strip() else None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def status(demo: dict) -> tuple[str, str, str | None]:
    """(kind, label, link) for a demo, from its Docker container (if it has one)."""
    if demo["n"] == 1:
        return "running", "This dashboard", None
    name = demo.get("container")
    if not name:
        return "none", "Coming soon", None
    c = setup.inspect("container", name)
    if c and c["State"]["Running"]:
        ports = setup.host_ports(c)
        return "running", "Running", f"http://localhost:{ports[0]}" if ports else None
    if c:
        return "stopped", f"Stopped ({c['State']['Status']})", None
    if setup.inspect("image", name):
        return "ready", "Ready to launch", None
    return "none", "Image not built", None


def last_run() -> dict | None:
    """The latest spark_setup.py run from state/status.json, with `state` corrected if it died."""
    try:
        run = json.loads(setup.STATUS.read_text())
    except (OSError, ValueError):
        return None
    if run["state"] == "running" and not setup.pid_alive(run["pid"]):
        run["state"] = "interrupted"
    return run


def progress(steps: list[dict], run_state: str) -> str:
    """Progress bar over a demo's setup steps: finished steps count fully, the running one by its fraction."""
    done = sum(1 if s["status"] in ("ok", "warn", "failed", "skipped") else
               (s.get("progress") or 0) if s["status"] == "running" else 0 for s in steps)
    pct = round(100 * done / len(steps))
    failed = next((s for s in steps if s["status"] == "failed"), None)
    skipped = next((s for s in steps if s["status"] == "skipped"), None)
    current = next((s for s in steps if s["status"] == "running"), None)
    if failed:
        color, text = RED, f"<b>Failed</b> · {escape(failed['title'].replace('`', ''))}: {escape(failed['detail'])}"
    elif skipped:
        color, text = ORANGE, f"<b>Not started</b> · {escape(skipped['detail'])}"
    elif current and run_state == "running":
        color, text = BLUE, (f"<b>{pct}%</b> · {escape(current['title'].replace('`', ''))}"
                             + (f": {escape(current['detail'])}" if current["detail"] else ""))
    elif current:
        color, text = ORANGE, f"<b>Stopped at {pct}%</b> · the setup run was interrupted"
    elif pct < 100:
        color, text = BLUE, f"<b>{pct}%</b> · waiting for earlier steps"
    else:
        color, text = GREEN, "<b>Done</b> · " + escape(steps[-1]["detail"])
    fix = f'<div class="fix">Fix: {escape(failed["hint"])}</div>' if failed and failed.get("hint") else ""
    return (f'<div class="d1-prog" style="--p:{color}"><div class="bar"><span style="width:{pct}%"></span></div>'
            f'<div class="txt">{text}</div>{fix}</div>')


def output(steps: list[dict]) -> str:
    """Command-line output of a demo's setup steps, from state/logs/."""
    parts = []
    for s in steps:
        if s.get("log"):
            parts.append(f"──── {s['title'].replace('`', '')} [{s['status']}]\n"
                         + setup.tail(setup.LOGS / s["log"], 200))
    return "\n\n".join(parts)


def setup_running() -> bool:
    """spark_setup.py holds this lock while it runs; only one run at a time."""
    return setup.LOCK.exists() and setup.pid_alive(setup.LOCK.read_text().strip())


def launch(demo: dict):
    """Run spark_setup.py for one demo in the background; its progress lands in state/status.json."""
    args = ["--host-only"] if demo["setup"] == "demo1" else ["--only", demo["setup"]]
    with open(setup.STATE / "launch.log", "a") as log:
        subprocess.Popen([sys.executable, str(setup.HERE / "spark_setup.py"), *args, "--no-dashboard"],
                         cwd=setup.HERE, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)


def launch_button(demo: dict, kind: str, busy: bool):
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
    disabled = not demo.get("setup") or busy or (kind == "running" and demo["n"] != 1)
    if st.button(text, key=f"launch-{demo['n']}", type="primary", disabled=disabled, help=why, width="stretch"):
        launch(demo)


def demo_row(demo: dict, kind: str, label: str, link: str | None) -> str:
    _, tag = KINDS[kind]
    open_link = f'<a href="{escape(link)}" target="_blank">Open {escape(link.split("//")[1])} ↗</a>' if link else ""
    return (f'<div class="d1-demo"><div class="num">{demo["n"]}</div>'
            f'<div class="body"><div class="n">Demo {demo["n"]} · {escape(demo["title"])}</div>'
            f'<div class="o">{escape(demo["about"])}</div></div>'
            f'<div class="side"><span class="d1-tag {tag}">{escape(label)}</span>{open_link}</div></div>')


def demo_card(demo: dict, kind: str, label: str, link: str | None, run: dict | None, busy: bool):
    with st.container(key=f"d1card-{demo['n']}"):
        steps = [s for s in run["steps"] if s["group"] == demo.get("setup")] if run and demo.get("setup") else []
        info, button = st.columns([0.84, 0.16])
        info.markdown(f'<style>.st-key-d1card-{demo["n"]} {{ border-left-color: {KINDS[kind][0]} !important; }}</style>'
                      + demo_row(demo, kind, label, link), unsafe_allow_html=True)
        with button:
            launch_button(demo, kind, busy)
        if not steps:
            return
        # The bar shows while a run is going, or if it didn't finish cleanly; the output stays available.
        if run["state"] != "done" or any(s["status"] in ("failed", "skipped") for s in steps):
            st.markdown(progress(steps, run["state"]), unsafe_allow_html=True)
        with st.expander("Command-line output"):
            st.code(output(steps) or "no output yet", language=None, height=320)


st.set_page_config(page_title="Demo 1 · Spark setup", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)

gpu = gpu_name()
dot = "dot" if gpu else "dot off"
st.markdown(
    f'<div class="d1-header"><div class="title">AI Weather Lab<small>| Demo 1: Spark Setup</small></div>'
    f'<div class="d1-live"><span class="{dot}"></span>{escape("GPU: " + gpu if gpu else "No GPU detected")}'
    f'</div></div>'
    f'<div class="d1-page-title">Setting Up Your AI Weather Forecasting Lab</div>'
    f'<div class="d1-page-sub">Every demo in the training program, and whether it is running on this Spark.</div>',
    unsafe_allow_html=True)


@st.fragment(run_every=3)  # re-reads Docker and the setup progress every few seconds
def live():
    statuses = [status(d) for d in DEMOS]
    run = last_run()
    stats = [("Demos", str(len(DEMOS)), BLUE),
             ("Running", str(sum(k == "running" for k, _, _ in statuses[1:])), GREEN),
             ("Ready to launch", str(sum(k == "ready" for k, _, _ in statuses)), ORANGE)]
    st.markdown('<div class="d1-stats">' + "".join(
        f'<div class="d1-stat" style="--c:{c}"><div class="k">{k}</div><div class="v">{v}</div></div>'
        for k, v, c in stats) + "</div>", unsafe_allow_html=True)

    st.markdown('<div class="d1-section">Demos</div><div class="d1-rule"></div>', unsafe_allow_html=True)
    busy = setup_running()
    for d, s in zip(DEMOS, statuses):
        demo_card(d, *s, run, busy)


live()
