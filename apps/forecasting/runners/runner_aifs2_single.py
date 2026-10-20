"""AIFS Single v2.0 (ECMWF) via Earth2Studio, initialized from IFS analyses (ECMWF open data on AWS)."""
from _e2s import run_e2s


import os

# ECMWF mirrors: "aws" (default) or "azure"; switch when one is throttling.
SOURCE = os.environ.get("DEMO3_IFS_SOURCE", "aws")


def run(init, lead_hours, members, report):
    from earth2studio.data import IFS
    from earth2studio.models.px import AIFS2
    # ECMWF open-data GRIB downloads can be slow; the default 600 s timeout is too short.
    return run_e2s(AIFS2, IFS(source=SOURCE, async_timeout=3600), init, lead_hours, 1, report)
