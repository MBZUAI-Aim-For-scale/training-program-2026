"""Visual style matching the Demo 5 Onset Forecast Platform."""
from __future__ import annotations

from html import escape

import streamlit as st

NAVY = "#1f2f6b"
NAVY_DARK = "#172457"
GREEN = "#5fb58a"
TEAL = "#3f8f7a"
BLUE = "#4b6fd6"
ORANGE = "#d4913a"
PURPLE = "#7b61c9"
INK = "#111827"
MUTED = "#6b7280"
LINE = "#e5e7eb"
PAGE = "#f6f7f9"

CSS = f"""
<style>
#MainMenu, footer, header[data-testid="stHeader"] {{ visibility: hidden; height: 0; }}
.stApp {{ background: {PAGE}; }}
.block-container {{ padding-top: 0.5rem; max-width: 1200px; }}
h1, h2, h3, h4 {{ color: {INK}; }}

.d3-header {{ background: linear-gradient(90deg, {NAVY_DARK}, {NAVY}); color: #fff;
  border-radius: 0 0 10px 10px; padding: 18px 28px; margin: 0 -1rem 1.4rem -1rem;
  display: flex; align-items: center; justify-content: space-between;
  border-bottom: 3px solid {BLUE}; }}
.d3-header .title {{ font-size: 1.35rem; font-weight: 700; }}
.d3-header .title small {{ font-weight: 400; opacity: .75; font-size: .8rem; margin-left: 8px; }}
.d3-live {{ background: rgba(255,255,255,.12); border: 1px solid rgba(255,255,255,.25);
  border-radius: 999px; padding: 5px 14px; font-family: ui-monospace, monospace; font-size: .8rem; }}
.d3-live .dot {{ display: inline-block; width: 8px; height: 8px; border-radius: 50%;
  margin-right: 8px; background: {GREEN}; }}
.d3-live .dot.off {{ background: {ORANGE}; }}

.d3-page-title {{ font-size: 1.9rem; font-weight: 800; color: {INK}; margin: .2rem 0 .1rem; }}
.d3-page-sub {{ color: {MUTED}; margin-bottom: 1.2rem; }}

.d3-stats {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 1.6rem; }}
@media (max-width: 800px) {{ .d3-stats {{ grid-template-columns: repeat(2, 1fr); }} }}
.d3-stat {{ background: #fff; border: 1px solid {LINE}; border-left: 3px solid var(--c);
  border-radius: 8px; padding: 14px 18px; }}
.d3-stat .k {{ font-size: .72rem; letter-spacing: .08em; color: {MUTED}; text-transform: uppercase; }}
.d3-stat .v {{ font-size: 1.35rem; font-weight: 700; color: {INK}; margin-top: 4px;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}

.d3-section {{ font-size: 1.2rem; font-weight: 700; color: {INK}; border-left: 4px solid {TEAL};
  padding-left: 10px; margin: 1.6rem 0 .8rem; }}
.d3-section .step {{ color: {MUTED}; font-weight: 500; font-size: .9rem; margin-right: 6px; }}
.d3-rule {{ border-bottom: 1px solid {LINE}; margin: -.3rem 0 1rem; }}

.d3-cards {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(230px, 1fr)); gap: 14px; margin-bottom: .6rem; }}
.d3-card {{ background: #fff; border: 1px solid {LINE}; border-top: 3px solid var(--c);
  border-radius: 8px; padding: 14px 18px; }}
.d3-card.sel {{ box-shadow: 0 0 0 2px var(--c); }}
.d3-card.dim {{ opacity: .6; }}
.d3-card .n {{ font-weight: 700; color: {INK}; font-size: 1.02rem; }}
.d3-card .o {{ color: {MUTED}; font-size: .85rem; margin: 2px 0 8px; }}
.d3-card .t {{ color: {MUTED}; font-size: .8rem; margin-top: 8px; }}

.d3-tag {{ display: inline-block; font-size: .68rem; font-weight: 700; letter-spacing: .06em;
  border-radius: 6px; padding: 3px 8px; margin-right: 6px; text-transform: uppercase; }}
.d3-tag.det {{ background: #e7f3ee; color: #2f6f57; }}
.d3-tag.ens {{ background: #efeafb; color: #6a4fb3; }}
.d3-tag.warn {{ background: #fdf1e2; color: #9a5b12; }}
.d3-tag.ok {{ background: #e7f3ee; color: #2f6f57; }}

.d3-pills {{ display: flex; flex-wrap: wrap; gap: 8px; margin: .4rem 0 1rem; }}
.d3-pill {{ background: #fff; border: 1px solid {LINE}; border-radius: 8px; padding: 7px 12px;
  font-size: .85rem; color: {MUTED}; }}
.d3-pill b {{ color: {INK}; }}

div.stButton > button[kind="primary"] {{ background: {NAVY}; border-color: {NAVY};
  border-radius: 8px; height: 2.8rem; }}
div.stButton > button[kind="primary"]:hover {{ background: {NAVY_DARK}; border-color: {NAVY_DARK}; }}
.stTabs [aria-selected="true"] {{ color: #c2410c !important; }}
div[data-testid="stRadio"] label p {{ color: {INK} !important; }}
</style>
"""


def apply() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def header(title: str, subtitle: str, live_text: str, live: bool = True) -> None:
    dot = "dot" if live else "dot off"
    st.markdown(
        f'<div class="d3-header"><div class="title">{escape(title)}<small>| {escape(subtitle)}</small></div>'
        f'<div class="d3-live"><span class="{dot}"></span>{escape(live_text)}</div></div>',
        unsafe_allow_html=True)


def page_title(title: str, sub: str) -> None:
    st.markdown(f'<div class="d3-page-title">{escape(title)}</div><div class="d3-page-sub">{escape(sub)}</div>',
                unsafe_allow_html=True)


def stats(items: list[tuple[str, str, str]]) -> None:
    """items: (label, value, color)"""
    cells = "".join(f'<div class="d3-stat" style="--c:{c}"><div class="k">{escape(k)}</div>'
                    f'<div class="v">{escape(v)}</div></div>' for k, v, c in items)
    st.markdown(f'<div class="d3-stats">{cells}</div>', unsafe_allow_html=True)


def section(title: str, step: str | None = None) -> None:
    s = f'<span class="step">{escape(step)}</span>' if step else ""
    st.markdown(f'<div class="d3-section">{s}{escape(title)}</div><div class="d3-rule"></div>',
                unsafe_allow_html=True)


def tag(text: str, kind: str) -> str:
    return f'<span class="d3-tag {kind}">{escape(text)}</span>'


def cards(items: list[dict]) -> None:
    """items: name, org, tags (html), note, color, selected, dim"""
    html = ""
    for it in items:
        cls = "d3-card" + (" sel" if it.get("selected") else "") + (" dim" if it.get("dim") else "")
        html += (f'<div class="{cls}" style="--c:{it["color"]}"><div class="n">{escape(it["name"])}</div>'
                 f'<div class="o">{escape(it["org"])}</div>{it["tags"]}'
                 f'<div class="t">{escape(it.get("note", ""))}</div></div>')
    st.markdown(f'<div class="d3-cards">{html}</div>', unsafe_allow_html=True)


def pills(items: list[tuple[str, str]]) -> None:
    html = "".join(f'<div class="d3-pill">{escape(k)}: <b>{escape(v)}</b></div>' for k, v in items)
    st.markdown(f'<div class="d3-pills">{html}</div>', unsafe_allow_html=True)
