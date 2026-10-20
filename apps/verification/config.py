# Authors: The africlimate AI team
# SPDX-License-Identifier: MIT
"""Deployment configuration — the single knob that makes this a per-agency platform.

One env var, ``VERIFICATION_COUNTRY`` (e.g. ``"Ethiopia"``, ``"Kenya"``), scopes a local
deployment to one met agency. It drives both the on-disk **domain** path segment (so EMI
writes/reads under ``.../ethiopia/...`` and KMD under ``.../kenya/...``) and the UI's
Region dropdown (which then shows only that country). Unset ⇒ the legacy pan-African
behaviour (domain ``africa``, all regions) so existing data and tests keep working.

Read at call time (not import) so a deployment sets it before launch and tests can vary it.
"""
from __future__ import annotations

import os


def country() -> str | None:
    """The deployment country display name (a REGION_COUNTRIES key), or None if unset."""
    value = os.environ.get("VERIFICATION_COUNTRY", "").strip()
    return value or None


def domain() -> str:
    """The path-domain segment: the country lowercased, or 'africa' when unscoped."""
    c = country()
    return c.lower() if c else "africa"
