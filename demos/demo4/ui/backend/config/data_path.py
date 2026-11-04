# Authors: The africlimate AI team
# SPDX-License-Identifier: MIT
"""UI data root + the climatology result path.

Model result paths are no longer composed here — discovery + the engine's
``ui.services.engine`` own the result-path scheme. Only the (distinct,
``{metric}.nc``-shaped) climatology path remains, used by the climatology
loader until the climatology toggle is reworked (issue #44).
"""
import os as _os
from pathlib import Path

# Root folder where the analysis repo writes its results.
# Met offices can override with the VERIFICATION_DATA_DIR environment variable.
DATA_DIR = Path(_os.environ.get("VERIFICATION_DATA_DIR", Path.home() / ".africlimateai" / "data"))


def _normalise(**kwargs) -> dict:
    return {k: str(v).lower().strip() for k, v in kwargs.items()}


def get_climatology_data_path(
    forecast_variable: str,
    truth_source: str,
    metric: str,
    year: str,
    *,
    root: Path,
) -> Path:
    """Returns the climatology NetCDF path under ``root`` (the results root).

    ``root`` is the UI's results root (``ui_results_root()``) so the
    ``africa`` segment lives in exactly one place. Layout:
      {root}/climatology/{variable}/{truth}/{metric}/{year}/annual_monthly_averages/{metric}.nc
    """
    n = _normalise(
        forecast_variable=forecast_variable,
        truth_source=truth_source,
        metric=metric,
        year=year,
    )
    return (
        Path(root) / "climatology"
        / n["forecast_variable"] / n["truth_source"]
        / n["metric"] / n["year"] / "annual_monthly_averages"
        / f"{n['metric']}.nc"
    )
