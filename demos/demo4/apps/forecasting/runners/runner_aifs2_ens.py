"""AIFS ENS v2 (ECMWF) via Earth2Studio, initialized from IFS analyses (ECMWF open data)
from 13 May 2026, or for 00Z starts in 2000-2025 from the ECMWF wave dataset plus ARCO
ERA5 (see wave_source.py).

Members come from the model's own noise conditioning: each member is a separate
rollout from the same initial state, with no initial-condition perturbation."""
import os

from _e2s import run_e2s

# ECMWF mirrors: "aws" (default) or "azure"; switch when one is throttling.
SOURCE = os.environ.get("DEMO3_IFS_SOURCE", "aws")


def run(init, lead_hours, members, report):
    from earth2studio.models.px import AIFS2ENS
    from wave_source import INIT_SOURCE, available, source_for
    ds = run_e2s(AIFS2ENS, source_for(init, SOURCE), init, lead_hours, members, report, ensemble=True)
    if available(init):  # recorded as what this run started from
        ds.attrs["run_init_source"] = INIT_SOURCE
    return ds
