"""AIFS ENS v2 (ECMWF) via Earth2Studio, initialized from IFS analyses (ECMWF open data).

Members come from the model's own noise conditioning: each member is a separate
rollout from the same initial state, with no initial-condition perturbation."""
import os

from _e2s import run_e2s

# ECMWF mirrors: "aws" (default) or "azure"; switch when one is throttling.
SOURCE = os.environ.get("DEMO3_IFS_SOURCE", "aws")


def run(init, lead_hours, members, report):
    from earth2studio.data import IFS
    from earth2studio.models.px import AIFS2ENS
    return run_e2s(AIFS2ENS, IFS(source=SOURCE, async_timeout=3600), init, lead_hours, members,
                   report, ensemble=True)
