"""AIFS Single v2.0 (ECMWF) via Earth2Studio.

Starting conditions: ECMWF open data (IFS analyses) from 13 May 2026, when it has every
input AIFS v2 needs; before that (00Z, 2000-2025), ocean-wave fields from the ECMWF wave
dataset plus everything else from ARCO ERA5 (see wave_source.py)."""
from _e2s import run_e2s


import os

# ECMWF mirrors: "aws" (default) or "azure"; switch when one is throttling.
SOURCE = os.environ.get("DEMO3_IFS_SOURCE", "aws")


def run(init, lead_hours, members, report):
    from earth2studio.models.px import AIFS2
    from wave_source import INIT_SOURCE, available, source_for
    ds = run_e2s(AIFS2, source_for(init, SOURCE), init, lead_hours, 1, report)
    if available(init):  # recorded as what this run started from
        ds.attrs["run_init_source"] = INIT_SOURCE
    return ds
