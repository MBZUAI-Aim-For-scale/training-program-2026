"""GraphCast (operational 0.25 deg, 13 levels) via Earth2Studio, initialized from ARCO ERA5."""
from _e2s import arco, run_e2s


def run(init, lead_hours, members, report):
    from earth2studio.models.px import GraphCastOperational
    return run_e2s(GraphCastOperational, arco(), init, lead_hours, 1, report)
