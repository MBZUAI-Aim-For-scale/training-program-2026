"""Aurora 1.5 (Microsoft) via Earth2Studio, initialized from ARCO ERA5.

Aurora 1.5 was fine-tuned on IFS analyses, but open-data IFS lacks sea-ice
concentration, so ARCO is the recommended alternative. The wrapper yields
hourly steps; the helper keeps 6-hourly fields and sums hourly rain.

Precipitation fix: Earth2Studio 0.18 applies Aurora's log-untransform
eps*(exp(x)-1), eps=1e-3, to tp1h that Aurora has already untransformed, which
shrinks rain ~1000x. Inverting the extra step, x = log1p(raw / eps), restores
metres (checked against ERA5: mean hourly rain ~6e-5 m vs ERA5 ~1e-4 m).
"""
import numpy as np

from _e2s import arco, run_e2s

EPS = 1e-3


def run(init, lead_hours, members, report):
    from earth2studio.models.px import Aurora1p5
    return run_e2s(Aurora1p5, arco(), init, lead_hours, 1, report, step_hours=1,
                   precip_fix=lambda tp: np.log1p(tp / EPS))
