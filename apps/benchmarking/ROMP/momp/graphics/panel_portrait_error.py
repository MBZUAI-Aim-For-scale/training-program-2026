import copy
import os
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from momp.io.output import analyze_nested_dict, nested_dict_to_array, set_nested
from momp.lib.loader import get_cfg, get_setting
from momp.utils.visual import portrait_plot


def panel_portrait_mae_far_mr(results, *, dir_fig, show_panel=True, **kwargs):
    """Generate 3-panel portrait plots for MAE, FAR, and Miss Rate anomalies."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 3))

    # 1. Mean MAE data & anomaly wrt climatology
    arr, row_labels, col_labels = nested_dict_to_array(results, "mean_mae")
    data = arr[:-1] - arr[-1]
    v = np.nanmax(np.abs(data)) if np.any(~np.isnan(data)) else 1.0
    vrange = (-v, v)

    fig, ax1, im = portrait_plot(
        data,
        col_labels,
        row_labels[:-1],
        fig=fig,
        ax=ax1,
        vrange=vrange,
        annotate=True,
        annotate_data=data,
        title=r"$\Delta MAE$ (days)",
        colorbar_off=True,
    )

    # 2. False Alarm Rate anomaly relative to climatology
    arr, row_labels, col_labels = nested_dict_to_array(results, "false_alarm_rate")
    data = arr[:-1] - arr[-1]
    v = np.nanmax(np.abs(data)) if np.any(~np.isnan(data)) else 1.0
    vrange = (-v, v)

    fig, ax2, im = portrait_plot(
        data,
        col_labels,
        row_labels[:-1],
        fig=fig,
        ax=ax2,
        vrange=vrange,
        annotate=True,
        annotate_data=data,
        title=r"$\Delta FAR (\%)$",
        colorbar_off=True,
    )

    ax2.set_xlabel("Forecast window (days)")

    # 3. Miss Rate anomaly relative to climatology
    arr, row_labels, col_labels = nested_dict_to_array(results, "miss_rate")
    data = arr[:-1] - arr[-1]
    v = np.nanmax(np.abs(data)) if np.any(~np.isnan(data)) else 1.0
    vrange = (-v, v)

    fig, ax3, im = portrait_plot(
        data,
        col_labels,
        row_labels[:-1],
        fig=fig,
        ax=ax3,
        vrange=vrange,
        annotate=True,
        annotate_data=data,
        title=r"$\Delta MR (\%)$",
        colorbar_off=True,
    )

    fig.tight_layout()

    # Save figure
    first_model = kwargs.get("model_list", ["model"])[0]
    max_forecast_day = kwargs.get("max_forecast_day", "")
    figure_filename = f"panel_portrait_mae_far_mr_{first_model}_{max_forecast_day}day.png"
    figure_path = os.path.join(dir_fig, figure_filename)

    os.makedirs(dir_fig, exist_ok=True)
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    print(f"Figure saved as '{figure_path}'")

    # Safe interactive check
    backend = plt.get_backend().lower()
    non_interactive_backends = ["agg", "pdf", "ps", "svg", "cairo"]

    if show_panel and (backend not in non_interactive_backends):
        plt.show()
    else:
        plt.close(fig)

    return fig, (ax1, ax2, ax3)


def panel_portrait_probabilistic_skills(results, *, dir_fig, show_panel=True, **kwargs):
    """Generate 3-panel portrait plots for Probabilistic Skill Scores (BSS, RPSS, AUC)."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 3))

    # 1. Brier Skill Score (BSS)
    arr, row_labels, col_labels = nested_dict_to_array(results, "bss")
    fig, ax1, im = portrait_plot(
        arr,
        col_labels,
        row_labels,
        fig=fig,
        ax=ax1,
        vrange=(-0.5, 0.5),
        annotate=True,
        annotate_data=arr,
        title="Brier Skill Score (BSS)",
        colorbar_off=True,
        cmap="RdBu",
    )

    # 2. Ranked Probability Skill Score (RPSS)
    arr, row_labels, col_labels = nested_dict_to_array(results, "rpss")
    fig, ax2, im = portrait_plot(
        arr,
        col_labels,
        row_labels,
        fig=fig,
        ax=ax2,
        vrange=(-0.5, 0.5),
        annotate=True,
        annotate_data=arr,
        title="RPSS",
        colorbar_off=True,
        cmap="RdBu",
    )
    ax2.set_xlabel("Forecast window (days)")

    # 3. Area Under ROC Curve (AUC)
    arr, row_labels, col_labels = nested_dict_to_array(results, "auc")
    fig, ax3, im = portrait_plot(
        arr,
        col_labels,
        row_labels,
        fig=fig,
        ax=ax3,
        vrange=(0.0, 1.0),
        annotate=True,
        annotate_data=arr,
        title="ROC Area (AUC)",
        colorbar_off=True,
        cmap="YlGnBu",
    )

    fig.tight_layout()

    first_model = kwargs.get("model_list", ["model"])[0]
    max_forecast_day = kwargs.get("max_forecast_day", "")
    figure_filename = f"panel_portrait_skill_scores_{first_model}_{max_forecast_day}day.png"
    figure_path = os.path.join(dir_fig, figure_filename)

    os.makedirs(dir_fig, exist_ok=True)
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    print(f"Figure saved as '{figure_path}'")

    backend = plt.get_backend().lower()
    non_interactive_backends = ["agg", "pdf", "ps", "svg", "cairo"]

    if show_panel and (backend not in non_interactive_backends):
        plt.show()
    else:
        plt.close(fig)

    return fig, (ax1, ax2, ax3)


if __name__ == "__main__":
    from momp.lib.control import iter_list, make_case
    from momp.lib.convention import Case
    from momp.utils.printing import tuple_to_str

    cfg, setting = get_cfg(), get_setting()

    result = {}
    is_probabilistic = getattr(cfg, "probabilistic", False)

    if is_probabilistic:
        var_list = ["bss", "rpss", "auc"]
    else:
        var_list = ["mean_mae", "false_alarm_rate", "miss_rate"]

    layout_pool = iter_list(vars(cfg))

    for combi in product(*layout_pool):
        case = make_case(Case, combi, vars(cfg))
        window_bin_str = tuple_to_str(case.verification_window)

        fi = os.path.join(cfg.dir_out, "spatial_metrics_{}_{}.nc")
        fi = fi.format(case.model, window_bin_str)

        if os.path.exists(fi):
            with xr.open_dataset(fi) as ds:
                spatial_metrics_dict = {
                    var: ds[var] for var in var_list if var in ds
                }
                result = set_nested(result, combi, spatial_metrics_dict)

    cfg_ref = copy.copy(cfg)
    cfg_ref.model_list = (cfg.ref_model,)
    layout_pool = iter_list(vars(cfg_ref))

    for combi in product(*layout_pool):
        case = make_case(Case, combi, vars(cfg_ref))
        window_bin_str = tuple_to_str(case.verification_window)

        fi = os.path.join(cfg.dir_out, "spatial_metrics_{}_{}.nc")
        fi = fi.format(case.ref_model, window_bin_str)

        if result.get(case.ref_model, {}).get(window_bin_str) is None and os.path.exists(fi):
            with xr.open_dataset(fi) as ds:
                spatial_metrics_dict = {
                    var: ds[var] for var in var_list if var in ds
                }
                result = set_nested(result, combi, spatial_metrics_dict)

    # Route to appropriate panel function depending on configuration
    if is_probabilistic:
        panel_portrait_probabilistic_skills(result, **vars(cfg))
    else:
        panel_portrait_mae_far_mr(result, **vars(cfg))