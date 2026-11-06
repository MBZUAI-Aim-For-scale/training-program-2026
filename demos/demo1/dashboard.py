"""Demo 1 dashboard and demo launcher. Started by `python3 spark_setup.py dashboard` on http://localhost:8500.

The top bar is laid out like the shared site's (ui/app.py): brand on the left, one button per demo,
a live badge on the right. Demo 1 is the Spark setup page. Every other demo gets a short page (summary
and learning goals from its lesson in docs/_episodes/) with a Launch button that opens a new tab: that
tab starts the demo's container through spark_setup.py, shows the progress, then moves on to the demo.

Cards and colours are styled like the Demo 3 app (demos/demo3/demo3/theme.py): light page, white cards.
"""
from __future__ import annotations

from html import escape
import json
from pathlib import Path
import re
import subprocess
import sys
import time
from urllib.parse import quote

import country_data
import forecastcheck_data
import spark_setup as setup
import streamlit as st

# Every demo in the training program, in order. `setup` is its id in demos.toml (once it has a Docker
# image), whose steps spark_setup.py records in state/status.json; its `container` and `image` names
# come from there too (below), so the two can't disagree.
# AI Almanac (`n` "almanac", with its own `label`) is an app next to the demos, not a numbered demo
# with a lesson: its code is its own repository in the home folder, and its link carries its access
# token (`token` in demos.toml).
# `path` is added to the container's address, to open a page inside a bigger site; `health` is the page
# that says it's up, for containers that aren't Streamlit apps. `tools` are extra containers a demo's page
# launches with their own button (named by `label`); they count towards the one-demo-at-a-time rule too.
DEMOS = [
    {"n": 1, "title": "Setting up your AI weather forecasting lab",
     "about": "Check the Spark and get a working environment before diving into the models.", "setup": "demo1"},
    {"n": 2, "title": "Observational data readiness",
     "about": "How ENACTS is built by combining rain gauges with satellite products.",
     "tools": [{"n": 2, "tool": "cdt", "label": "CDT", "title": "Climate Data Tools",
                "about": "The CDT desktop app, in the browser over VNC.", "setup": "cdt",
                "path": "/vnc.html?autoconnect=1&reconnect=1&resize=remote", "health": "/vnc.html"}]},
    {"n": 3, "title": "Running your first AI weather forecast",
     "about": "Run AI weather models locally, end to end, for your use case.",
     "setup": "demo3"},
    {"n": 4, "title": "The AI weather model scorecard",
     "about": "Compare models across evaluation metrics and explore the AI Almanac for your use case.",
     "setup": "demo4", "path": "/?page=SM_Benchmarking"},
    {"n": 5, "title": "Putting your forecast to the test",
     "about": "Evaluate the forecasts you ran in Demo 3 against different metrics.",
     "setup": "demo5"},
    {"n": "almanac", "label": "AI Almanac", "badge": "AI",
     "title": "Assess, blend and forecast with AI weather models",
     "about": "Pick a region, compare how well AI and conventional models predicted the rainy season "
              "onset there, blend the best of them, and run the blend for the current season. Its "
              "assistant can set up each step from a plain-language request.",
     "setup": "almanac"},
    # {"n": 6, "title": "Ground truth challenge",
    #  "about": "Compare model performance against different observational datasets."},
    # {"n": 7, "title": "Feasibility assessment",
    #  "about": "Decide which models, data and thresholds to take forward into a pilot."},
]
BY_N = {str(d["n"]): d for d in DEMOS}
# Each demo's (and tool's) container and image, as demos.toml names them. A `setup` that isn't
# in demos.toml (or Demo 1 itself) gets no container: its page says it has none yet.
_IN_TOML = {d["id"]: d for d in setup.load_config()["demo"]}
for _d in DEMOS + [t for d in DEMOS for t in d.get("tools", [])]:
    if _d.get("setup") in _IN_TOML:
        _d["container"] = _IN_TOML[_d["setup"]]["container"]
        _d["image"] = _IN_TOML[_d["setup"]]["image"]
        for _k in ("checkout", "get", "token", "health"):  # (a `health` set here wins)
            if _k in _IN_TOML[_d["setup"]] and _k not in _d:
                _d[_k] = _IN_TOML[_d["setup"]][_k]
LESSONS = setup.REPO / "docs" / "_episodes"
DOCS_SITE = "https://mbzuai-aim-for-scale.github.io/training-program-2026"
LINK_MAX_AGE = 120  # seconds a Launch link can start a demo (the pages renew theirs every few seconds)

# Same palette as Demo 3; the top bar uses the shared site's navy
NAVY, NAVY_DARK, BLUE, TEAL = "#1f2f6b", "#172457", "#4b6fd6", "#3f8f7a"
GREEN, ORANGE, RED = "#5fb58a", "#d4913a", "#c2410c"
INK, MUTED, LINE, PAGE = "#111827", "#6b7280", "#e5e7eb", "#f6f7f9"
NAV_BLUE, NAV_BORDER, NAV_GREEN = "#0A2F6C", "#3B82F6", "#10B981"

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=DM+Mono:wght@400;500&display=swap');
#MainMenu, footer, header[data-testid="stHeader"] {{ visibility: hidden; height: 0; }}
.stApp {{ background: {PAGE}; font-family: 'Inter', system-ui, sans-serif; }}
.block-container {{ padding-top: 6rem; max-width: 1000px; }}

/* Top bar, as in the shared site (ui/components/styles/main.css and nav.css) */
.st-key-top_navbar {{ position: fixed; top: 0; left: 0; right: 0; z-index: 1001; background: {NAV_BLUE};
  border-bottom: 3px solid {NAV_BORDER}; box-shadow: 0 2px 8px rgba(0,0,0,.15); padding: .75rem 1.5rem; }}
.st-key-top_navbar [data-testid="stHorizontalBlock"] {{ align-items: center; }}
.st-key-top_navbar .stButton button {{ background: transparent !important; color: #fff !important;
  border: 1px solid rgba(255,255,255,.3) !important; border-radius: 6px !important; white-space: nowrap;
  box-shadow: none !important; }}
.st-key-top_navbar .stButton button p {{ font-weight: 600; font-size: .9rem; }}
.st-key-top_navbar .stButton button:hover {{ background: rgba(255,255,255,.1) !important;
  border-color: rgba(255,255,255,.5) !important; }}
.st-key-top_navbar .stButton button[kind="primary"] {{ background: rgba(255,255,255,.2) !important;
  border-color: rgba(255,255,255,.6) !important; }}
.st-key-top_navbar .stButton button[kind="primary"] p {{ font-weight: 700; }}
.st-key-top_navbar .stButton button[kind="primary"]:hover {{ background: rgba(255,255,255,.25) !important; }}
.brand-text {{ color: #fff; font-weight: 700; font-size: 1.1rem; white-space: nowrap; }}
.brand-text span {{ color: rgba(255,255,255,.7); font-weight: 400; font-size: .75rem; margin-left: 6px; }}
.weather-badge {{ display: inline-flex; align-items: center; float: right; color: #fff; font-size: .8rem;
  font-family: 'DM Mono', monospace; background: rgba(255,255,255,.1); border: 1px solid rgba(255,255,255,.2);
  border-radius: 999px; padding: .35rem 1rem; white-space: nowrap; }}
.blink-dot {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 8px;
  background: {NAV_GREEN}; box-shadow: 0 0 4px {NAV_GREEN}; animation: blink 1.2s infinite; }}
.blink-dot.off {{ background: {ORANGE}; box-shadow: none; animation: none; }}
@keyframes blink {{ 0%, 100% {{ opacity: 1; }} 50% {{ opacity: .2; }} }}
@media (prefers-reduced-motion: reduce) {{ .blink-dot {{ animation: none; }} }}
@media (max-width: 768px) {{ .brand-text span, .weather-badge {{ display: none; }} }}
/* Phones stack the bar's columns; keep it in the page there instead of pinned over the content */
@media (max-width: 640px) {{ .st-key-top_navbar {{ position: relative; margin-bottom: 1rem; }}
  .block-container {{ padding-top: 1rem; }} }}

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

div[class*="st-key-d1card-"], div[class*="st-key-d1launch"] {{ background: #fff; border: 1px solid {LINE};
  border-left: 4px solid {LINE}; border-radius: 8px; padding: 20px 18px; gap: 1rem; }}
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
div[class*="st-key-launch-"] button, div[class*="st-key-launch-"] a {{ border-radius: 8px; }}
div[class*="st-key-launch-"] button[kind="primary"]:not(:disabled),
div[class*="st-key-launch-"] a[data-testid="stBaseLinkButton-primary"] {{ background: {NAVY}; border-color: {NAVY};
  color: #fff; }}
div[class*="st-key-launch-"] button[kind="primary"]:not(:disabled):hover,
div[class*="st-key-launch-"] a[data-testid="stBaseLinkButton-primary"]:hover {{ background: {NAVY_DARK};
  border-color: {NAVY_DARK}; }}
div[class*="st-key-launch-"] button:disabled {{ background: {PAGE}; border-color: {LINE}; color: {MUTED}; }}
/* Status buttons on the Demo 1 cards: the Launch button's shape, coloured like the status tags */
div[class*="st-key-status-"] button {{ border-radius: 8px !important; border: 1px solid var(--b) !important;
  background: var(--bg) !important; color: var(--fg) !important; box-shadow: none !important; }}
div[class*="st-key-status-"] button p {{ font-size: .78rem; font-weight: 700; letter-spacing: .06em;
  text-transform: uppercase; }}
div[class*="st-key-status-"] button:hover {{ filter: brightness(.97); }}
div[class*="st-key-status-ok-"] {{ --bg: #e7f3ee; --fg: #2f6f57; --b: #c9e4d7; }}
div[class*="st-key-status-ready-"] {{ --bg: #e8eefb; --fg: #34509e; --b: #cbd7f3; }}
div[class*="st-key-status-warn-"] {{ --bg: #fdf1e2; --fg: #9a5b12; --b: #f3d9b5; }}
div[class*="st-key-status-off-"] {{ --bg: #f1f2f4; --fg: {MUTED}; --b: {LINE}; }}
div[class*="st-key-status-failed-"] {{ --bg: #fdecea; --fg: {RED}; --b: #f5c9c0; }}

.d1-prog {{ margin-left: calc(2.2rem + 16px); margin-bottom: 1rem; }}
div[class*="st-key-d1launch"] .d1-prog {{ margin-left: 0; }}
.d1-prog .bar {{ height: 8px; background: {PAGE}; border: 1px solid {LINE}; border-radius: 999px; overflow: hidden; }}
.d1-prog .bar span {{ display: block; height: 100%; background: var(--p); border-radius: 999px;
  transition: width .6s ease; }}
.d1-prog .txt {{ color: {MUTED}; font-size: .82rem; margin-top: 6px;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
.d1-prog .txt b {{ color: {INK}; }}
.d1-prog .fix {{ color: #9a5b12; font-size: .82rem; margin-top: 4px; }}
div[class*="st-key-d1card-"] div[data-testid="stExpander"] {{ margin-left: calc(2.2rem + 16px);
  width: calc(100% - 2.2rem - 16px); }}
div[class*="st-key-d1card-"] div[data-testid="stExpander"] summary p,
div[class*="st-key-d1launch"] div[data-testid="stExpander"] summary p {{ font-size: .85rem; }}
div[class*="st-key-d1card-"] div[data-testid="stCode"] pre,
div[class*="st-key-d1card-"] div[data-testid="stCode"] code,
div[class*="st-key-d1launch"] div[data-testid="stCode"] pre,
div[class*="st-key-d1launch"] div[data-testid="stCode"] code {{ font-size: .72rem !important; line-height: 1.45; }}

/* Demo pages: the lesson blurb on the left, the launch card on the right */
.d1-blurb {{ color: {INK}; font-size: 1.02rem; line-height: 1.6; margin-bottom: 1.2rem; }}
.d1-list {{ margin: 0 0 1rem; padding-left: 1.2rem; color: {INK}; }}
.d1-list li {{ margin-bottom: .45rem; line-height: 1.5; }}
.d1-list.q li {{ color: {MUTED}; }}
.d1-launch-k {{ font-size: .72rem; letter-spacing: .08em; color: {MUTED}; text-transform: uppercase;
  margin-bottom: 6px; }}
/* Streamlit pulls the next element up under markdown (-1rem); keep the tag clear of the button below */
.d1-launch-status {{ margin-bottom: 1.4rem; }}
.d1-launch-note {{ color: {MUTED}; font-size: .8rem; line-height: 1.5; }}
.d1-launch-note code {{ font-size: .75rem; }}

/* The launcher tab showing a demo: the demo fills the window, over everything else */
.st-key-demoframe iframe {{ position: fixed; inset: 0; width: 100vw !important; height: 100vh !important;
  border: 0; z-index: 2000; background: #fff; }}
</style>
"""

# status kind -> (accent colour on the card, tag style)
KINDS = {"running": (GREEN, "ok"), "ready": (BLUE, "ready"), "stopped": (ORANGE, "warn"), "none": (LINE, "off")}


@st.cache_data(ttl=60)
def gpu_name() -> str | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10, check=False)
        return out.stdout.strip().splitlines()[0] if out.returncode == 0 and out.stdout.strip() else None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


@st.cache_data(ttl=300)
def lesson(n) -> dict:
    """The front matter of a demo's lesson page (docs/_episodes/0<n>-*.md): teaching, objectives, ...

    Only the shapes those pages use: `key: value` lines, and `key:` followed by `- "item"` lines.
    (AI Almanac has none: it isn't a numbered demo.)
    """
    files = sorted(LESSONS.glob(f"{n:02d}-*.md")) if isinstance(n, int) else []
    if not files:
        return {}
    m = re.match(r"---\n(.*?)\n---", files[0].read_text(encoding="utf-8"), re.DOTALL)
    info, key = {}, None
    for line in (m[1] if m else "").splitlines():
        if re.match(r"\w+:", line):
            key, _, value = line.partition(":")
            info[key] = value.strip().strip('"') or []
        elif line.lstrip().startswith("- ") and isinstance(info.get(key), list):
            info[key].append(line.strip()[2:].strip().strip('"'))
    return info


def lesson_url(n) -> str | None:
    """A demo's lesson on the docs site. Its address is the lesson's file name, which Jekyll lowercases."""
    files = sorted(LESSONS.glob(f"{n:02d}-*.md")) if isinstance(n, int) else []
    return f"{DOCS_SITE}/{files[0].stem.lower()}/" if files else None


def page_header(n: int, title: str, sub: str = ""):
    """A demo tab's title, with a button to its lesson on the docs site."""
    text, button = st.columns([0.76, 0.24], vertical_alignment="center")
    text.markdown(f'<div class="d1-page-title">{escape(title)}</div><div class="d1-page-sub">{escape(sub)}</div>',
                  unsafe_allow_html=True)
    if url := lesson_url(n):
        button.link_button("Additional information ↗", url, width="stretch",
                           help=f"Opens the Demo {n} lesson in a new tab")


def access_token(demo: dict) -> str:
    """An app's access token for its link (`token` in demos.toml: a KEY=value line in a file), or ""."""
    tok = demo.get("token")
    if not tok:
        return ""
    try:
        for line in Path(tok["file"]).expanduser().read_text().splitlines():
            key, _, value = line.partition("=")
            if key.strip() == tok["key"]:
                return value.strip()
    except OSError:
        pass
    return ""


def stopped_cleanly(state: dict) -> bool:
    """A container that ended normally: it exited by itself (code 0), or was stopped (docker stop: SIGTERM,
    143, or SIGKILL after the grace period, 137, unless the kernel killed it for memory). Not a crash."""
    if state.get("Status") != "exited":
        return False
    code = state.get("ExitCode")
    return code in (0, 143) or (code == 137 and not state.get("OOMKilled"))


def status(demo: dict) -> tuple[str, str, str | None]:
    """(kind, label, link) for a demo, from its Docker container (if it has one)."""
    if demo["n"] == 1:
        return "running", "This dashboard", None
    name = demo.get("container")
    if not name:
        return "none", "Coming soon", None
    if checkout_missing(demo):
        return "none", "Not on this Spark", None
    c = setup.inspect("container", name)
    if c and c["State"]["Running"]:
        ports = setup.host_ports(c)
        if not ports:
            return "running", "Running", None
        link = f"http://localhost:{ports[0]}{demo.get('path', '')}"
        if token := access_token(demo):
            link += ("&" if "?" in link else "/?") + "token=" + quote(token)
        return "running", "Running", link
    if c:
        if stopped_cleanly(c["State"]):  # (a Stop, or AI Almanac's own setup: ready, like a built image)
            return "ready", "Ready to launch", None
        code = c["State"].get("ExitCode")
        return "stopped", f"Stopped ({c['State']['Status']}" + (f", exit code {code}" if code else "") + ")", None
    if setup.inspect("image", demo.get("image", name)):
        return "ready", "Ready to launch", None
    return "none", "Not set up yet" if demo.get("checkout") else "Image not built", None


def checkout_missing(demo: dict) -> bool:
    """An app in its own checkout (AI Almanac: ~/ai-almanac) that isn't on this Spark: it can't start."""
    return bool(demo.get("checkout")) and not (Path(demo["checkout"]).expanduser() / ".git").exists()


def tracked(demo: dict) -> dict:
    """What a demo's card reports on: its own container or, for a demo that only has a tool (Demo 2), that tool's."""
    tools = demo.get("tools", [])
    return tools[0] if tools and not demo.get("container") else demo


def label(demo: dict) -> str:
    return demo.get("label") or f"Demo {demo['n']}"


def uid(demo: dict) -> str:
    """Tells a demo's tools apart from the demo itself, in widget and session-state keys."""
    return demo.get("tool") or str(demo["n"])


def launch_href(demo: dict, **extra: str) -> str:
    """The link a Launch or Open button follows: a new tab that starts the demo (or tool), then shows it.
    It says when it was made, so that it stops working once it's old (see link_works)."""
    return "?" + "&".join(f"{k}={v}" for k, v in {"demo": demo["n"], "tool": demo.get("tool"), "launch": 1,
                                                  "at": int(time.time()), **extra}.items() if v)


def link_works(demo: dict) -> bool:
    """True if this tab's link may show the Starting page. An old link, such as a tab the browser restored,
    must not start a demo or stop another one. It can still show a demo that's already running.
    A tab whose link was new when it opened keeps working, however long the demo takes to build."""
    key = f"link-{uid(demo)}"
    try:
        new = time.time() - int(st.query_params.get("at", 0)) < LINK_MAX_AGE
    except ValueError:
        new = False
    if new:
        st.session_state[key] = True
    return st.session_state.get(key, False) or status(demo)[0] == "running"


def page_up(demo: dict, link: str | None) -> bool:
    """True once the page behind a status() link answers."""
    m = re.match(r"http://localhost:(\d+)", link or "")
    return bool(m) and setup.healthy(int(m[1]), path=demo.get("health"))


def last_run() -> dict | None:
    """The latest spark_setup.py run from state/status.json, with `state` corrected if it died."""
    try:
        run = json.loads(setup.STATUS.read_text())
    except (OSError, ValueError):
        return None
    if run["state"] == "running" and not setup.pid_alive(run["pid"]):
        run["state"] = "interrupted"
    return run


def steps_for(demo: dict, run: dict | None) -> list[dict]:
    return [s for s in run["steps"] if s["group"] == demo.get("setup")] if run and demo.get("setup") else []


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
    # (--no-pull: a Launch starts what's here; the setup runs update other repositories' code.)
    args = ["--host-only"] if demo["setup"] == "demo1" else ["--only", demo["setup"], "--no-pull"]
    setup.STATE.mkdir(parents=True, exist_ok=True)  # (a dashboard started by itself, before any setup run)
    with open(setup.STATE / "launch.log", "a") as log:
        subprocess.Popen([sys.executable, str(setup.HERE / "spark_setup.py"), *args, "--no-dashboard"],
                         cwd=setup.HERE, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)


def running_others(demo: dict) -> list[dict]:
    """Other demos and tools whose container is running. Only one runs at a time on a Spark; Demo 1 (this
    dashboard and its host checks) is left out of that rule, so it is never counted or stopped."""
    everything = [part for d in DEMOS if d["n"] != 1 for part in (d, *d.get("tools", []))]
    return [p for p in everything if p.get("container") and p["container"] != demo.get("container")
            and (setup.inspect("container", p["container"]) or {}).get("State", {}).get("Running")]


def names(demos: list[dict]) -> str:
    return " and ".join(label(d) for d in demos)


def stop(demos: list[dict]):
    """Stop these demos' containers (kept, not removed: the next setup run recreates them)."""
    for d in demos:
        setup.docker("stop", d["container"], timeout=120)


# --- Top bar ------------------------------------------------------------------------------

def navbar(active: int):
    """Brand, one button per demo, live badge: the layout of the shared site's top bar."""
    gpu = gpu_name()
    with st.container(key="top_navbar"):
        brand, menu, badge = st.columns([1.3, 5.2, 1.5], gap="small", vertical_alignment="center")
        brand.markdown('<div class="brand-text">AI Weather Lab<span>| Demo launcher</span></div>',
                       unsafe_allow_html=True)
        with menu:
            for col, d in zip(st.columns(len(DEMOS), gap="small"), DEMOS):
                if col.button(label(d), key=f"nav-{d['n']}", help=d["title"], width="stretch",
                              type="primary" if d["n"] == active else "secondary"):
                    st.query_params.clear()
                    st.query_params["demo"] = str(d["n"])
                    st.rerun()
        badge.markdown(f'<div class="weather-badge"><span class="blink-dot{"" if gpu else " off"}"></span>'
                       f'{escape("GPU: " + gpu if gpu else "No GPU detected")}</div>', unsafe_allow_html=True)


# --- Demo 1: the Spark setup page -----------------------------------------------------------

def launch_button(demo: dict, kind: str, busy: bool):
    """Demo 1's Run checks button (the other demos launch from their own pages)."""
    why = "Wait for the current setup run to finish" if busy else "Re-run the Spark host checks"
    if st.button("Run checks", key=f"launch-{demo['n']}", type="primary", disabled=busy, help=why, width="stretch"):
        launch(demo)


def status_button(demo: dict, kind: str, text: str, run: dict | None, busy: bool):
    """The demo's status in the Launch button's place; clicking it opens the demo's page (where Launch is)."""
    steps = steps_for(tracked(demo), run)
    tag = KINDS[kind][1]
    if busy and run and run["state"] == "running" and steps:
        tag, text = "ready", "Starting…"
    elif steps and run["state"] != "running" and any(s["status"] == "failed" for s in steps):
        tag, text = "failed", "Setup failed"
    if st.button(text, key=f"status-{tag}-{demo['n']}", width="stretch",
                 help=f"Open the {label(demo)} page" + (" to launch it" if tracked(demo).get("container") else "")):
        st.query_params.clear()
        st.query_params["demo"] = str(demo["n"])
        st.rerun()


def demo_row(demo: dict, kind: str, text: str, link: str | None) -> str:
    _, tag = KINDS[kind]
    # Demo 1 keeps its tag next to its Run checks button; the other demos show their status in the button
    side = f'<span class="d1-tag {tag}">{escape(text)}</span>' if demo["n"] == 1 else ""
    if link:
        side += f'<a href="{escape(launch_href(tracked(demo)))}" target="_blank">Open {escape(label(demo))} ↗</a>'
    return (f'<div class="d1-demo"><div class="num">{escape(str(demo.get("badge", demo["n"])))}</div>'
            f'<div class="body"><div class="n">{escape(label(demo))} · {escape(demo["title"])}</div>'
            f'<div class="o">{escape(demo["about"])}</div></div>'
            f'<div class="side">{side}</div></div>')


def demo_card(demo: dict, kind: str, label: str, link: str | None, run: dict | None, busy: bool):
    with st.container(key=f"d1card-{demo['n']}"):
        steps = steps_for(tracked(demo), run)
        info, button = st.columns([0.84, 0.16])
        info.markdown(f'<style>.st-key-d1card-{demo["n"]} {{ border-left-color: {KINDS[kind][0]} !important; }}</style>'
                      + demo_row(demo, kind, label, link), unsafe_allow_html=True)
        with button:
            if demo["n"] == 1:
                launch_button(demo, kind, busy)
            else:
                status_button(demo, kind, label, run, busy)
        # The bar shows while a run is going, or if it didn't finish cleanly. The output box is always
        # there, but only holds the latest setup run's output, and only if that run included this demo.
        if steps and (run["state"] != "done" or any(s["status"] in ("failed", "skipped") for s in steps)):
            st.markdown(progress(steps, run["state"]), unsafe_allow_html=True)
        with st.expander("Command-line output"):
            st.code(output(steps) or ("no output yet: the latest setup run didn't include this demo"
                                      if tracked(demo).get("setup") else "no output: this demo doesn't have a container yet"),
                    language=None, height=320 if steps else "content")


def setup_page():
    page_header(1, "Setting Up Your AI Weather Forecasting Lab",
                "Every demo in the training program, and whether it is running on this Spark.")

    @st.fragment(run_every=3)  # re-reads Docker and the setup progress every few seconds
    def live():
        statuses = [status(tracked(d)) for d in DEMOS]
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
    country_data_card()


# --- Demo 1: download this Spark's country data from the bucket -------------------------------

@st.cache_data(ttl=300, show_spinner="Checking the bucket …")
def country_plan(country: str) -> tuple[int, int] | None:
    """(files, bytes) of a country's data, or None if the bucket can't be read right now."""
    try:
        files = country_data.plan(country)
    except (OSError, ValueError, KeyError, country_data.bucket_access.BucketError):
        return None
    return len(files), sum(int(o["size"]) for o in files)


def start_country_download():
    """Run country_data.py in the background; it outlives this page, like a setup run."""
    country_data.LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(country_data.LOG, "a") as log:
        subprocess.Popen([sys.executable, str(setup.HERE / "country_data.py")], cwd=setup.HERE,
                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    time.sleep(0.5)  # lets it write its first status, so this page's next look already shows it running


def country_progress(job: dict) -> str:
    state, size = job["state"], country_data.size_text
    pct = round(100 * job["done_bytes"] / job["total_bytes"]) if job["total_bytes"] else 0
    done = f"{size(job['done_bytes'])} of {size(job['total_bytes'])}, {job['done_files']} of {job['total_files']} files"
    if state == "running":
        color, text = BLUE, f"<b>{pct}%</b> · {escape(done if job['total_files'] else job['detail'])}"
    elif state == "interrupted":
        color, text = ORANGE, f"<b>Stopped at {pct}%</b> · {escape(done)}. Finished files are kept."
    elif state == "failed":
        color, text = RED, f"<b>Failed</b> · {escape(job['detail'])}"
    else:
        color, text = GREEN, f"<b>Done</b> · {escape(job['detail'])}"
    return (f'<div class="d1-prog" style="--p:{color}"><div class="bar"><span style="width:{pct}%"></span></div>'
            f'<div class="txt">{text}</div></div>')


@st.fragment(run_every=3)
def country_data_card():
    host = country_data.hostname()
    country = country_data.country_from_hostname(host)
    st.markdown('<div class="d1-section">Country data</div><div class="d1-rule"></div>', unsafe_allow_html=True)
    with st.container(key="d1launch-data"):
        if not country:
            st.button("Download country data", key="launch-data-none", disabled=True)
            st.markdown(f'<div class="d1-launch-note">No single country found in this Spark\'s name '
                        f'(<code>{escape(host)}</code>). Known countries: '
                        f'{escape(", ".join(c.replace("_", " ") for c in country_data.COUNTRIES))}.</div>',
                        unsafe_allow_html=True)
            return
        job = country_data.read_status()
        state = job["state"] if job else None
        tag, text, go = {"running": ("ready", "Downloading", None),
                         "done": ("ok", "Downloaded", "Check for updates"),
                         "failed": ("warn", "Not finished", "Try again"),
                         "interrupted": ("warn", "Stopped", "Resume download"),
                         }.get(state, ("off", "Not downloaded", f"Download {country.replace('_', ' ')} data"))
        plan = country_plan(country)
        about = (f"{plan[0]:,} files, {country_data.size_text(plan[1])}" if plan
                 else "Size unknown: can't read the bucket right now")
        info, button = st.columns([0.72, 0.28])
        info.markdown(f'<div class="d1-launch-k">{escape(host)} · {escape(country.replace("_", " "))}</div>'
                      f'<div class="d1-launch-status"><span class="d1-tag {tag}">{text}</span></div>'
                      f'<div class="d1-launch-note">{escape(about)}, saved to '
                      f'<code>{escape(str(country_data.data_dir(country)))}</code>.</div>', unsafe_allow_html=True)
        with button:
            if state == "running":
                st.button("Stop download", key="stop-data", width="stretch", on_click=country_data.stop,
                          help="Finished files are kept; the next download skips them")
            else:
                st.button(go, key="launch-data", type="primary", width="stretch", on_click=start_country_download,
                          help="Keeps going if you close this page. Files already here are skipped.")
        if job:
            st.markdown(country_progress(job), unsafe_allow_html=True)
            with st.expander("Command-line output"):
                st.code(setup.tail(country_data.LOG, 200) or "no output yet", language=None, height=320)


# --- Demos 2+: a short page about the demo, and a Launch button that opens it in a new tab -----

def bullets(items: list[str], cls: str = "") -> str:
    return f'<ul class="d1-list {cls}">' + "".join(f"<li>{escape(i)}</li>" for i in items) + "</ul>"


def demo_page(demo: dict):
    info = lesson(demo["n"])
    page_header(demo["n"], f'{label(demo)} · {demo["title"]}')

    blurb, card = st.columns([0.6, 0.4], gap="large")
    with blurb:
        html = f'<div class="d1-blurb">{escape(demo["about"])}</div>'
        if info.get("objectives"):
            html += ('<div class="d1-section">What you will learn</div>' + bullets(info["objectives"]))
        if info.get("questions"):
            html += ('<div class="d1-section">Questions it answers</div>' + bullets(info["questions"], "q"))
        st.markdown(html, unsafe_allow_html=True)
    with card:
        if tracked(demo) is demo:  # (a demo that only has a tool gets that tool's card, not a "Coming soon" one)
            launch_card(demo)
        for tool in demo.get("tools", []):
            launch_card(tool)
        if demo.get("setup") == "demo5":
            forecastcheck_data_card()


# --- Demo 5: download this Spark's country data for ForecastCheck ------------------------------

def start_forecastcheck_download():
    """Run forecastcheck_data.py in the background; it outlives this page, like a setup run."""
    forecastcheck_data.LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(forecastcheck_data.LOG, "a") as log:
        subprocess.Popen([sys.executable, str(setup.HERE / "forecastcheck_data.py")], cwd=setup.HERE,
                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    time.sleep(0.5)  # lets it write its first status, so this page's next look already shows it running


def choose_forecastcheck_country():
    """The picker's on_change: Demo 5's country, as `fcstcheck country <name>` sets it."""
    name = st.session_state.get("fcdata-country")
    if name:
        ok, msg = forecastcheck_data.choose(name)
        st.session_state["fcdata-country-msg"] = None if ok else msg or "couldn't set the country"


@st.fragment(run_every=3)
def forecastcheck_data_card():
    """This Spark's country, if ForecastCheck has it (Kenya, Rwanda, Senegal, Sierra Leone); on any
    other Spark, a picker for the country Demo 5 uses (the same as `fcstcheck country <name>`)."""
    host, countries = country_data.hostname(), forecastcheck_data.supported()
    spark = forecastcheck_data.spark_country()
    country = spark or forecastcheck_data.chosen_country()
    names = ", ".join(forecastcheck_data.title(c) for c in countries)
    with st.container(key="d1launch-fcdata"):
        st.markdown('<div class="d1-launch-k">Country data for ForecastCheck</div>', unsafe_allow_html=True)
        if not countries:
            st.button("Download country data", key="fcdata-none", disabled=True, width="stretch")
            st.markdown("<div class='d1-launch-note'>Launch Demo 5 first: its code isn't on this Spark yet.</div>",
                        unsafe_allow_html=True)
            return
        job = forecastcheck_data.read_status()
        if not spark:
            # ForecastCheck doesn't have this Spark's country: Demo 5 works with the one picked here
            st.selectbox("Country for Demo 5", countries, key="fcdata-country",
                         index=countries.index(country) if country in countries else None,
                         placeholder="Choose a country", format_func=forecastcheck_data.title,
                         on_change=choose_forecastcheck_country,
                         disabled=bool(job and job.get("state") == "running"),
                         help="The same as fcstcheck country <name>. A running Demo 5 restarts with it.")
            st.markdown(f'<div class="d1-launch-note">ForecastCheck has data for {escape(names)}; this Spark '
                        f'(<code>{escape(host)}</code>) isn\'t one of them, so Demo 5 uses the country you '
                        f'pick.</div>', unsafe_allow_html=True)
            if msg := st.session_state.get("fcdata-country-msg"):
                st.error(f"The country wasn't changed: {msg}")
        if not country:
            st.button("Download country data", key="fcdata-none", disabled=True, width="stretch")
            return
        state = job["state"] if job and job.get("country") == country else None
        tag, text, go = {"running": ("ready", "Downloading", None),
                         "done": ("ok", "Downloaded", "Check for new data"),
                         "failed": ("warn", "Not finished", "Try again"),
                         "interrupted": ("warn", "Stopped", "Resume download"),
                         }.get(state, ("off", "Not downloaded", f"Download {forecastcheck_data.title(country)} data"))
        st.markdown(f'<div class="d1-launch-status"><span class="d1-tag {tag}">{text}</span></div>',
                    unsafe_allow_html=True)
        if state == "running":
            st.button("Stop download", key="fcdata-stop", width="stretch", on_click=forecastcheck_data.stop,
                      help="What's fetched stays; the next download fetches only what's missing")
        elif not forecastcheck_data.image_ready():
            st.button(go, key="fcdata-go", type="primary", width="stretch", disabled=True)
            st.markdown('<div class="d1-launch-note">Launch Demo 5 first: the download runs in its image.</div>',
                        unsafe_allow_html=True)
        else:
            st.button(go, key="fcdata-go", type="primary", width="stretch", on_click=start_forecastcheck_download,
                      help=f"ForecastCheck's own fetch, for {forecastcheck_data.title(country)}. "
                           "Keeps going if you close this page.")
        st.markdown(f'<div class="d1-launch-note">{escape(forecastcheck_data.title(country))}\'s forecasts and '
                    'observations, from the program\'s bucket into Demo 5\'s data folder '
                    '(<code>fcstcheck where</code>), the same as <code>fcstcheck fetch --apply</code>.</div>',
                    unsafe_allow_html=True)
        if job and state:
            color = {"running": BLUE, "done": GREEN}.get(state, ORANGE if state == "interrupted" else RED)
            st.markdown(f'<div class="d1-prog" style="--p:{color}"><div class="txt">'
                        f'{escape(job.get("detail") or "")}</div></div>', unsafe_allow_html=True)
            with st.expander("Command-line output"):
                st.code(setup.tail(forecastcheck_data.LOG, 200) or "no output yet", language=None, height=320)


@st.fragment(run_every=3)
def launch_card(demo: dict):
    kind, text, link = status(demo)
    run = last_run()
    steps = steps_for(demo, run)
    mine = setup_running() and run and run["state"] == "running" and bool(steps)  # this demo is starting
    key, name = uid(demo), label(demo)
    go = f"Launch {name} ↗" if demo.get("tool") else "Launch ↗"
    heading = f"{demo['title']} ({name})" if demo.get("tool") else "On this Spark"
    with st.container(key=f"d1launch-{key}"):
        st.markdown(f'<style>.st-key-d1launch-{key} {{ border-left-color: {KINDS[kind][0]} !important; }}</style>'
                    f'<div class="d1-launch-k">{escape(heading)}</div>'
                    f'<div class="d1-launch-status"><span class="d1-tag {KINDS[kind][1]}">'
                    f'{escape(text if not mine else "Starting")}</span></div>',
                    unsafe_allow_html=True)
        with st.container(key=f"launch-page-{key}"):
            if kind == "running" and link:
                st.link_button(f"Open {name} ↗", launch_href(demo), type="primary",
                               width="stretch", help=f"Opens {name} in a new tab")
                note = f"Running at <code>{escape(link)}</code>."
            elif not demo.get("container"):
                st.button("Coming soon", key=f"soon-{key}", disabled=True, width="stretch")
                note = "This demo doesn't have a container yet."
            elif checkout_missing(demo):
                # (no Launch: it would run the host checks, maybe stop another demo, then fail)
                st.button("Not on this Spark", key=f"missing-{key}", disabled=True, width="stretch")
                note = (f"{escape(name)}'s code isn't in <code>{escape(demo['checkout'])}</code>. On the "
                        "training's demo Sparks it arrives with the next update; or, in a terminal:"
                        + (f"<br><code>{escape(demo['get'])}</code>" if demo.get("get") else ""))
            elif setup_running() and not mine:
                st.button(go, key=f"busy-{key}", disabled=True, width="stretch",
                          help="Wait for the current setup run to finish")
                note = "Another demo is being set up; this one can start when it finishes."
            elif (others := [] if mine else running_others(demo)) and not st.session_state.get(f"confirm-{key}"):
                # Another demo is running: ask first, since launching this one stops it
                st.button(go, key=f"ask-{key}", type="primary", width="stretch",
                          help=f"{names(others)} is running; you'll be asked before it's stopped",
                          on_click=st.session_state.__setitem__, args=(f"confirm-{key}", True))
                note = f"Only one demo runs at a time on this Spark, and {names(others)} is running now."
            elif others:
                st.warning(f"{names(others)} is running. Only one demo runs at a time, so launching "
                           f"{name} stops {names(others)}. Anyone using it will lose their place.")
                st.link_button(f"Stop {names(others)} and launch ↗",
                               launch_href(demo, stop=",".join(d["container"] for d in others)),
                               type="primary", width="stretch")
                st.button("Cancel", key=f"cancel-{key}", width="stretch",
                          on_click=st.session_state.pop, args=(f"confirm-{key}", None))
                note = "Opens a new tab that stops the other demo, starts this one and then shows it."
            else:
                st.session_state.pop(f"confirm-{key}", None)
                st.link_button("Follow progress ↗" if mine else go, launch_href(demo),
                               type="primary", width="stretch",
                               help=f"Opens a new tab that starts the container and then shows {name}")
                note = ""
            if demo.get("container"):
                st.button(f"Stop {name}", key=f"stop-{key}", width="stretch",
                          disabled=kind != "running" or bool(mine), on_click=stop, args=([demo],),
                          help="Stops the container; its tabs close themselves" if kind == "running"
                          else f"{name} isn't running")
        if note:
            st.markdown(f'<div class="d1-launch-note">{note}</div>', unsafe_allow_html=True)
        if steps and (mine or any(s["status"] == "failed" for s in steps)):
            st.markdown(progress(steps, run["state"]), unsafe_allow_html=True)


# --- The new tab a Launch button opens: start the demo, then go to it ----------------------

def starting_page(demo: dict):
    # Already up (an Open link, or a reload): show it straight away, without the Starting page in between
    kind, _, link = status(demo)
    if kind == "running" and page_up(demo, link):
        st.session_state[f"framed-{uid(demo)}"] = link
    if link := st.session_state.get(f"framed-{uid(demo)}"):
        demo_frame(demo, link)
        return
    st.markdown(f'<div class="d1-page-title">Starting {label(demo)} · {escape(demo["title"])}</div>'
                '<div class="d1-page-sub">This tab switches to the demo as soon as its page answers.</div>',
                unsafe_allow_html=True)
    follow(demo)


def demo_frame(demo: dict, link: str):
    """The demo, filling this tab. The tab stays the launcher's, so it can close itself when the demo stops
    (a tab that had moved on to the demo's own address couldn't)."""
    document_title = json.dumps(f"{label(demo)} · {demo['title']}")
    st.html(f"<script>document.title = {document_title};</script>",
            unsafe_allow_javascript=True)
    with st.container(key="demoframe"):
        st.iframe(link, height=800)
    watch(demo)


@st.fragment(run_every=3)
def watch(demo: dict):
    """Close the tab once the demo's container has stopped (two checks in a row, so one failed
    Docker query doesn't close it). Browsers only let a page close a tab that a link or script opened;
    otherwise the message stays."""
    misses = f"misses-{uid(demo)}"
    kind, _, _ = status(demo)
    st.session_state[misses] = 0 if kind == "running" else st.session_state.get(misses, 0) + 1
    if st.session_state[misses] < 2:
        return
    st.markdown('<style>.st-key-demoframe { display: none !important; }</style>'
                f'<div class="d1-page-title">{label(demo)} was stopped</div>'
                '<div class="d1-page-sub">This tab closes itself. If it stays open, you can close it.</div>',
                unsafe_allow_html=True)
    st.html("<script>window.close();</script>", unsafe_allow_javascript=True)


@st.fragment(run_every=2)
def follow(demo: dict):
    kind, _, link = status(demo)
    if checkout_missing(demo):  # (a Launch link from before it went: nothing to start, nothing to stop)
        st.markdown(f'<div class="d1-launch-note">{escape(label(demo))} isn\'t on this Spark (no '
                    f'<code>{escape(demo["checkout"])}</code>), so it can\'t start. See its page in the dashboard.</div>',
                    unsafe_allow_html=True)
        return
    if kind == "running" and page_up(demo, link):
        st.session_state[f"framed-{uid(demo)}"] = link
        st.rerun()
    # Start it once per tab, as soon as no other setup run holds the lock.
    # Other running demos are stopped first, but only the ones confirmed (in the link, or with the button below).
    key = f"launched-{uid(demo)}"
    if key not in st.session_state and kind != "running" and not setup_running():
        others = running_others(demo)
        confirmed = set(st.query_params.get("stop", "").split(",")) | st.session_state.get("stop-ok", set())
        unconfirmed = [d for d in others if d["container"] not in confirmed]
        if unconfirmed:
            st.warning(f"{names(unconfirmed)} is running. Only one demo runs at a time, so starting "
                       f"{label(demo)} stops {names(unconfirmed)}.")
            st.button(f"Stop {names(unconfirmed)} and launch {label(demo)}", type="primary",
                      on_click=st.session_state.__setitem__, args=("stop-ok", {d["container"] for d in others}))
            return
        if others:
            with st.spinner(f"Stopping {names(others)} …"):
                stop(others)
        st.session_state[key] = time.time()
        launch(demo)
    run = last_run()
    since = st.session_state.get(key)
    # Only a run started from this tab, or one setting up this demo right now; not an old run's leftovers
    if run and not ((since and run["started"] >= since - 5) or (run["state"] == "running" and steps_for(demo, run))):
        run = None
    steps = steps_for(demo, run)
    with st.container(key="d1launch"):
        if steps:
            st.markdown(progress(steps, run["state"]), unsafe_allow_html=True)
        if steps and run["state"] != "running" and any(s["status"] in ("failed", "skipped") for s in steps):
            note = "The demo didn't start. Fix the problem above, then launch it again from its page in the dashboard."
        elif kind == "running":
            note = "The container is running; waiting for its page to answer …"
        elif setup_running() and not steps:
            note = "Another demo is being set up; this one starts when it finishes."
        else:
            note = "Getting ready …"
        st.markdown(f'<div class="d1-launch-note">{note}</div>', unsafe_allow_html=True)
        if steps:
            with st.expander("Command-line output"):
                st.code(output(steps) or "no output yet", language=None, height=320)


# --- Page ------------------------------------------------------------------------------------

st.set_page_config(page_title="AI Weather Lab · Demos", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)

try:
    current = BY_N.get(str(st.query_params.get("demo", 1)), BY_N["1"])
except ValueError:
    current = BY_N["1"]
tool = next((t for t in current.get("tools", []) if t["tool"] == st.query_params.get("tool")), None)
launching = current["n"] != 1 and st.query_params.get("launch") == "1" and (tool or current).get("container")
# The tab a Launch or Open button opens shows only the demo or tool (or its start-up progress), without the bar
if launching and link_works(tool or current):
    starting_page(tool or current)
else:
    if launching:  # An old link: forget it, and show the demo's page
        st.query_params.clear()
        st.query_params["demo"] = str(current["n"])
    navbar(current["n"])
    if launching:
        st.info(f"That link has expired, so {label(tool or current)} wasn't started. Use the button on this page.")
    # (Statements, not `a() if … else b()`: Streamlit's "magic" would write that expression's None on the page.)
    if current["n"] == 1:
        setup_page()
    else:
        demo_page(current)
