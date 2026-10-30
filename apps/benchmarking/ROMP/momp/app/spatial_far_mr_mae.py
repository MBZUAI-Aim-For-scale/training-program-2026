"""
momp/app/spatial_far_mr_mae.py
Spatial FAR, Miss Rate, and MAE map calculation and export workflow.
Ensures explicit resolution of evaluation years, canonical model key normalization,
and standardized directory routing for deterministic and reference model benchmarks.
"""

from ai_weather.spatial import climatology_boundary_path
import os
import sys
import copy
from pathlib import Path
from dataclasses import asdict
from itertools import product
import xarray as xr

# Safe import for config and params
try:
    import config
except ImportError:
    try:
        from ai_weather import config
    except ImportError:
        _curr = Path(__file__).resolve()
        for _parent in _curr.parents:
            if (_parent / "ai_weather" / "config.py").exists() or (_parent / "config.py").exists():
                sys.path.insert(0, str(_parent / "ai_weather"))
                sys.path.insert(0, str(_parent))
                break
        import config

from momp.metrics.error import create_spatial_far_mr_mae
from momp.stats.benchmark import compute_metrics_multiple_years
from momp.lib.control import iter_list, make_case, ref_model_case
from momp.lib.convention import Case
from momp.lib.loader import get_cfg, get_setting
from momp.graphics.maps import plot_spatial_metrics
from momp.graphics.onset_map import plot_spatial_climatology_onset
from momp.graphics.panel_portrait_error import panel_portrait_mae_far_mr
from momp.io.output import save_metrics_to_netcdf
from momp.io.output import set_nested
from momp.utils.printing import tuple_to_str_range
from momp.stats.parallel import parallel_metrics_multiple_years


cfg, setting = get_cfg(validate=False), get_setting(validate=False)


def get_canonical_model_key(model_name: str) -> str:
    """
    Maps various model naming variations and capitalization styles
    to a standardized, lowercase canonical folder key.
    """
    if not model_name:
        return "model"

    key = str(model_name).strip().lower()

    if key in ["aifs", "aifs_single", "aifs-single"]:
        return "aifs_single"
    elif key in ["graphcast", "graph_cast", "graphcast_single"]:
        return "graphcast"
    elif key in ["fuxi", "fuxi_single"]:
        return "fuxi"
    elif key in ["aifs_ens", "aifs_ensemble", "aifs-ens"]:
        return "aifs_ens"
    elif key in ["climatology", "clim"]:
        return "climatology"

    return key


def spatial_far_mr_mae_map(cfg=cfg, setting=setting):
    """
    Computes spatial metrics (MAE, FAR, Miss Rate) across verification windows
    and exports NetCDF metrics and PNG maps to standardized model output directories.
    """
    if getattr(cfg, "probabilistic", False):
        return {}

    results = {}
    layout_pool = iter_list(vars(cfg))

    for combi in product(*layout_pool):
        case = make_case(Case, combi, vars(cfg))

        canonical_model = get_canonical_model_key(getattr(case, "model", "aifs_single"))

        print(f"{'='*50}")
        print(
            f"processing {case.model} onset evaluation for verification window "
            f"{case.verification_window}, case: {case.case_name}"
        )

        case_cfg = {**asdict(case), **asdict(setting)}
        case_cfg["model"] = canonical_model

        # Derive evaluation years directly from configure dates rather than defaulting to 2003
        if not case_cfg.get("years"):
            start_yr = case_cfg.get("start_date", (2019, 5, 1))[0]
            end_yr = case_cfg.get("end_date", (2024, 10, 31))[0]
            case_cfg["years"] = (
                getattr(setting, "years", None)
                or getattr(cfg, "years", None)
                or tuple(range(start_yr, end_yr + 1))
            )

        # Standardize output directory path routing for forecast model
        base_out_dir = case_cfg.get("dir_out") or f"/app/data/ROMP_OUT/et/output"
        base_fig_dir = case_cfg.get("dir_fig") or f"/app/data/ROMP_OUT/et/figure"

        if not str(base_out_dir).endswith(canonical_model):
            case_cfg["dir_out"] = os.path.join(base_out_dir, canonical_model)
        else:
            case_cfg["dir_out"] = base_out_dir

        if not str(base_fig_dir).endswith(canonical_model):
            case_cfg["dir_fig"] = os.path.join(base_fig_dir, canonical_model)
        else:
            case_cfg["dir_fig"] = base_fig_dir

        os.makedirs(case_cfg["dir_out"], exist_ok=True)
        os.makedirs(case_cfg["dir_fig"], exist_ok=True)

        # Model-observation onset benchmarking
        if setting.parallel:
            metrics_df_dict, onset_da_dict = parallel_metrics_multiple_years(**case_cfg)
        else:
            metrics_df_dict, onset_da_dict = compute_metrics_multiple_years(**case_cfg)

        # Create spatial metrics
        spatial_metrics = create_spatial_far_mr_mae(metrics_df_dict, onset_da_dict)

        # Log case result into combined multi-case results dictionary
        results = set_nested(results, combi, spatial_metrics)

        # Save spatial metrics to NetCDF
        if case_cfg.get("save_nc_spatial_far_mr_mae", True):
            desc_dict = {
                "title": "Monsoon Onset MAE, FAR, MR Analysis",
                "contingency_definition": "timing_matched_v2",
                "description": """Spatial maps of Mean Absolute Error, False Alarm Rate, and Miss Rate 
                for monsoon onset predictions""",
            }
            save_metrics_to_netcdf(spatial_metrics, case_cfg, desc_dict=desc_dict)

        # Render spatial metrics plot
        if case_cfg.get("plot_spatial_far_mr_mae", True):
            plot_spatial_metrics(spatial_metrics, **case_cfg)

    # ------------------------------------------------------------------------
    # Baseline metrics (climatology or user-specified reference model)
    if not getattr(cfg, "ref_model", None):
        return results

    cfg_ref = copy.copy(cfg)
    cfg_ref.model_list = (cfg.ref_model,)
    layout_pool = iter_list(vars(cfg_ref))

    for combi in product(*layout_pool):
        case = make_case(Case, combi, vars(cfg_ref))
        print(f"{'='*50}")
        print(
            f"processing {case.model} onset evaluation for verification window "
            f"{case.verification_window}, case: {case.case_name}"
        )

        case_ref = {
            "model_dir": case_cfg.get("ref_model_dir"),
            "model_var": case_cfg.get("ref_model_var"),
            "file_pattern": case_cfg.get("ref_model_file_pattern"),
            "unit_cvt": case_cfg.get("ref_model_unit_cvt"),
        }

        # Safely update case attributes if present
        for k, v in case_ref.items():
            if hasattr(case, k):
                setattr(case, k, v)

        case_cfg_ref = {**asdict(case), **asdict(setting), **case_ref}
        canonical_ref_model = get_canonical_model_key(getattr(case, "model", "climatology"))
        case_cfg_ref["model"] = canonical_ref_model

        # Route reference model output/figures to its own directory (e.g. .../climatology)
        ref_base_out = getattr(setting, "dir_out", "/app/data/ROMP_OUT/et/output")
        ref_base_fig = getattr(setting, "dir_fig", "/app/data/ROMP_OUT/et/figure")

        if Path(ref_base_out).name in ["fuxi", "aifs_single", "graphcast", "aifs_ens", "gencast", "climatology"]:
            ref_base_out = str(Path(ref_base_out).parent)
        if Path(ref_base_fig).name in ["fuxi", "aifs_single", "graphcast", "aifs_ens", "gencast", "climatology"]:
            ref_base_fig = str(Path(ref_base_fig).parent)

        case_cfg_ref["dir_out"] = os.path.join(ref_base_out, canonical_ref_model)
        case_cfg_ref["dir_fig"] = os.path.join(ref_base_fig, canonical_ref_model)

        os.makedirs(case_cfg_ref["dir_out"], exist_ok=True)
        os.makedirs(case_cfg_ref["dir_fig"], exist_ok=True)

        if not case_cfg_ref.get("years"):
            case_cfg_ref["years"] = (
                getattr(setting, "years_clim", None)
                or getattr(cfg, "years_clim", None)
                or case_cfg.get("years")
                or tuple(range(2003, 2024))
            )

        # Model-observation onset benchmarking for reference model
        if setting.parallel:
            metrics_df_dict, onset_da_dict = parallel_metrics_multiple_years(**case_cfg_ref)
        else:
            metrics_df_dict, onset_da_dict = compute_metrics_multiple_years(**case_cfg_ref)

        # Create spatial metrics
        spatial_metrics = create_spatial_far_mr_mae(metrics_df_dict, onset_da_dict)

        # Mask climatology grids before summaries and NetCDF serialization.
        if canonical_ref_model == "climatology":
            from ai_weather import config as weather_config
            from ai_weather.spatial import clip_to_boundary

            spatial_metrics = {key: clip_to_boundary(value, climatology_boundary_path(weather_config))
                               for key, value in spatial_metrics.items()}
        # Log case result into combined multi-case results dictionary
        results = set_nested(results, combi, spatial_metrics)

        # Save spatial metrics to NetCDF
        if case_cfg_ref.get("save_nc_spatial_far_mr_mae", True):
            desc_dict = {
                "title": "Monsoon Onset MAE, FAR, MR Analysis",
                "contingency_definition": "timing_matched_v2",
                "description": """Spatial maps of Mean Absolute Error, False Alarm Rate, and Miss Rate 
                for monsoon onset predictions""",
            }
            save_metrics_to_netcdf(spatial_metrics, case_cfg_ref, desc_dict=desc_dict)

        # Render spatial metrics plot
        if case_cfg_ref.get("plot_spatial_far_mr_mae", True):
            plot_spatial_metrics(spatial_metrics, **case_cfg_ref)

    # Save climatological onset to NetCDF
    if getattr(case, "model", None) == "climatology" and case_cfg_ref.get("save_nc_climatology", False):
        fout = os.path.join(case_cfg_ref["dir_out"], "climatology_onset_doy_{}.nc")
        fout = fout.format(tuple_to_str_range(case_cfg_ref.get("years_clim", list(range(2003, 2024)))))
        if onset_da_dict:
            climatological_onset_doy = next(iter(onset_da_dict.values()))
            from ai_weather import config as weather_config
            from ai_weather.spatial import clip_to_boundary
            climatological_onset_doy = clip_to_boundary(climatological_onset_doy, climatology_boundary_path(weather_config))
            climatological_onset_doy.attrs["years"] = case_cfg_ref.get("years_clim", list(range(2003, 2024)))
            climatological_onset_doy.to_netcdf(fout)

    # Render spatial map of climatology onset day
    if case_cfg_ref.get("plot_climatology_onset", False):
        plot_spatial_climatology_onset(onset_da_dict, **case_cfg_ref)

    # Render panel portrait plot of MAE, FAR, and Miss Rate
    if case_cfg_ref.get("plot_panel_heatmap_error", False):
        panel_portrait_mae_far_mr(results, **case_cfg_ref)

    return results


if __name__ == "__main__":
    results = spatial_far_mr_mae_map()
