# ui/components/nrt_cards.py
from __future__ import annotations

import streamlit as st
from datetime import datetime
from typing import Dict
from utils.http import fetch_nrt_image
from nrt.sources import NRT_SOURCES, _TAG_COLORS

def nrt_card(col, prod_name: str, meta: Dict) -> bool:
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    tag_color = _TAG_COLORS.get(meta.get("tag", ""), "#718096")
    with col:
        st.markdown(f"""
        <div class="nrt-card">
          <div class="nrt-hdr">
            <span class="nrt-hdr-title">
              <span style="background:{tag_color}33;color:{tag_color};padding:1px 7px;
                border-radius:10px;font-size:.62rem;margin-right:6px;">{meta.get('tag','')}</span>
              {prod_name}
            </span>
            <span class="nrt-hdr-ts">{ts}</span>
          </div>
          <div class="nrt-body">
        """, unsafe_allow_html=True)
        img, err = fetch_nrt_image(meta["url"], meta.get("referer", ""))
        if img:
            st.image(img,width='stretch', caption=meta["desc"][:150])
            st.markdown(
                f'<div style="font-size:.63rem;color:#718096;">⏱ {meta["update"]} · 📍 {meta["region"]}</div>',
                unsafe_allow_html=True,
            )
            st.markdown("</div></div>", unsafe_allow_html=True)
            return True
        else:
            st.markdown(f"""
            <div class="nrt-unavail">
              <div style="font-size:1.2rem;margin-bottom:6px;">🛰</div>
              <div style="font-weight:600;margin-bottom:4px;">Image unavailable</div>
              <div style="font-size:.72rem;color:#9CA3AF;margin-bottom:8px;">{err}</div>
              <a href="{meta['url']}" target="_blank"
                style="color:#1A6B9A;font-size:.72rem;">Open directly ↗</a>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("</div></div>", unsafe_allow_html=True)
            return False

def render_image_pair(left_name: str, right_name: str, label: str) -> None:
    c1, c2 = st.columns(2)
    nrt_card(c1, left_name, NRT_SOURCES[left_name])
    nrt_card(c2, right_name, NRT_SOURCES[right_name])