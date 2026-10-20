from __future__ import annotations

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from PIL import Image
from io import BytesIO
from typing import Tuple, Optional
import streamlit as st

@st.cache_resource
def _http_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(total=4, backoff_factor=1.2, status_forcelist=[429,500,502,503,504],
                  allowed_methods=["GET"], raise_on_status=False)
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.mount("http://",  HTTPAdapter(max_retries=retry))
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "image/gif,image/webp,image/*,*/*;q=0.8",
    })
    return s

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_nrt_image(url: str, referer: str = "") -> Tuple[Optional[Image.Image], str]:
    session = _http_session()
    headers = {"Referer": referer} if referer else {}
    try:
        resp = session.get(url, headers=headers, timeout=28)
        if resp.status_code == 404:
            return None, "404 Not Found — product URL may have changed."
        resp.raise_for_status()
        ct = resp.headers.get("Content-Type", "")
        if "text/html" in ct or "text/plain" in ct:
            return None, "Server returned HTML — likely a redirect or access restriction."
        if len(resp.content) < 800:
            return None, "Response too small — likely an error page."
        raw = Image.open(BytesIO(resp.content))
        if raw.mode in ("RGBA", "P"):
            bg = Image.new("RGB", raw.size, (255, 255, 255))
            mask = raw.split()[3] if raw.mode == "RGBA" else raw.convert("RGBA").split()[3]
            bg.paste(raw.convert("RGB"), mask=mask)
            return bg, ""
        return raw.convert("RGB"), ""
    except Exception as e:
        return None, f"Error: {str(e)[:120]}"