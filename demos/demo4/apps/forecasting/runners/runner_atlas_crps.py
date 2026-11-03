"""Atlas CRPS (NVIDIA) via Earth2Studio, initialized from ARCO ERA5.

Each ensemble member is a separate rollout; the model draws fresh noise on
every forward pass, so no initial-condition perturbation is applied."""
from _e2s import arco, run_e2s


def run(init, lead_hours, members, report):
    from earth2studio.models.px import AtlasCRPS
    return run_e2s(AtlasCRPS, arco(), init, lead_hours, members, report, ensemble=True)
