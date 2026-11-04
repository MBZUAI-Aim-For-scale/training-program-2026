"""
ROMP Main Entry Script
apps/benchmarking/ROMP/momp/driver.py
----------------------
Primary execution point for the Rainy season Onset Metrics Package (ROMP).
Orchestrates calculation of binned skill scores and spatial performance maps.
"""

import logging
import sys
import traceback

from momp.lib.loader import get_cfg, get_setting
from momp.app.bin_skill_score import skill_score_in_bins
from momp.app.spatial_far_mr_mae import spatial_far_mr_mae_map
from momp.utils.printing import print_momp_banner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)


def run_momp(cfg=None, setting=None):
    """
    Executes the standard ROMP evaluation workflow.
    Evaluates cfg and setting dynamically if not explicitly provided.
    """
    if cfg is None:
        cfg = get_cfg()
    if setting is None:
        setting = get_setting()

    print_momp_banner(cfg)
    logger.info("Starting ROMP Workflow...")

    try:
        # 1. Calculate and save Skill Scores in defined day bins
        skill_score_in_bins(cfg=cfg, setting=setting)

        # 2. Generate spatial metrics and maps (MAE, FAR, Miss Rate)
        spatial_far_mr_mae_map(cfg=cfg, setting=setting)

        logger.info("ROMP Workflow completed successfully!")

    except Exception as e:
        logger.error(f"ROMP failed during execution: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    run_momp()
