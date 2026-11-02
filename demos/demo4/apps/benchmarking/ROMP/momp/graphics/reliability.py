from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from momp.io.dict import extract_pd_bins
from momp.utils.printing import tuple_to_str


def plot_reliability_diagram(combined_forecast_obs, *, model, verification_window, day_bins,
                             save_fig, dir_fig, extract_bins=True, show_plot=True, **kwargs):
    """Plot reliability for one full 1–15 or 16–30 day event window."""

    verification_window = tuple(verification_window)
    if verification_window not in ((1, 15), (16, 30)):
        raise ValueError("Reliability diagrams support only days 1–15 and 16–30.")
    # Always enforce the full event window, including for prefiltered callers.
    combined_forecast_obs = extract_pd_bins(combined_forecast_obs, (verification_window,))
    if combined_forecast_obs.empty:
        raise ValueError(f"No full-window forecast/observation pairs for {verification_window}; regenerate benchmark results.")

    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    reliability_y = np.full(n_bins, np.nan)
    mean_forecast_prob = np.full(n_bins, np.nan)
    frequency = np.zeros(n_bins)
    n_forecasts_array = np.zeros(n_bins)

    print("\nReliability Analysis:")
    print("Bin Range\t\tN_Forecasts\tMean_Forecast_Prob\tReliability\tFrequency\tError_Bar")
    print("-" * 90)

    results_for_csv = []
    total = len(combined_forecast_obs)

    for i in range(n_bins):
        if i == 0:
            in_bin = ((combined_forecast_obs['predicted_prob'] >= bin_edges[i]) &
                      (combined_forecast_obs['predicted_prob'] <= bin_edges[i + 1]))
        else:
            in_bin = ((combined_forecast_obs['predicted_prob'] > bin_edges[i]) &
                      (combined_forecast_obs['predicted_prob'] <= bin_edges[i + 1]))

        n_forecasts = int(in_bin.sum())
        n_forecasts_array[i] = n_forecasts

        if n_forecasts > 0:
            mean_forecast_prob[i] = combined_forecast_obs.loc[in_bin, 'predicted_prob'].mean()
            reliability_y[i] = combined_forecast_obs.loc[in_bin, 'observed_onset'].mean()
            frequency[i] = n_forecasts / total
            error_bar = np.sqrt(reliability_y[i] * (1 - reliability_y[i]) / n_forecasts)
        else:
            frequency[i] = 0
            error_bar = np.nan

        bin_range = f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}"
        print(f"{bin_range}\t\t{n_forecasts}\t\t{mean_forecast_prob[i]:.3f}\t\t\t"
              f"{reliability_y[i]:.3f}\t\t{frequency[i]:.3f}\t\t{error_bar:.3f}")

        results_for_csv.append({
            'Bin_Range': bin_range,
            'N_Forecasts': n_forecasts,
            'Mean_Forecast_Prob': round(mean_forecast_prob[i], 3) if not np.isnan(mean_forecast_prob[i]) else np.nan,
            'Observed_Frequency': round(reliability_y[i], 3) if not np.isnan(reliability_y[i]) else np.nan,
            'Frequency': round(frequency[i], 3),
            'Error_Bar': round(error_bar, 3) if not np.isnan(error_bar) else np.nan,
        })

    results_df = pd.DataFrame(results_for_csv)

    with np.errstate(invalid='ignore', divide='ignore'):
        error_bars = np.sqrt(reliability_y * (1 - reliability_y) / n_forecasts_array)
    error_bars = np.where(n_forecasts_array > 0, error_bars, 0)

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    valid_bins = ~np.isnan(reliability_y) & ~np.isnan(mean_forecast_prob)
    ax.errorbar(mean_forecast_prob[valid_bins], reliability_y[valid_bins],
                yerr=error_bars[valid_bins], fmt='o-',
                color='blue', linewidth=2, markersize=8, capsize=5, capthick=2,
                label='Reliability')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect Reliability')

    ax2 = ax.twinx()
    ax2.set_yscale('log')
    ax2.bar(bin_centers, frequency, width=0.08, alpha=0.3, color='gray', label='Frequency')
    positive = frequency[frequency > 0]
    if len(positive):
        ax2.set_ylim(positive.min() * 0.5, positive.max() * 2)
    ax2.set_ylabel('Forecast frequency', fontsize=12)

    ax.set_xlabel('Forecast Probability', fontsize=12)
    ax.set_ylabel('Observed Frequency', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Title, e.g. "Reliability Diagram - AIFS-ENS - Days 1-15"
    if isinstance(verification_window, (tuple, list)) and len(verification_window) == 2:
        days_str = f"{verification_window[0]}-{verification_window[1]}"
    else:
        days_str = str(verification_window)
    ax.set_title(f"Reliability Diagram - {model} - Days {days_str}", fontsize=14)

    plt.tight_layout()

    if save_fig:
        os.makedirs(dir_fig, exist_ok=True)
        window_str = tuple_to_str(verification_window)
        fig_fn = os.path.join(dir_fig, f'reliability_{model}_{window_str}.png')
        fig.savefig(fig_fn, dpi=600, bbox_inches='tight')
        print(f"Figure saved to: {fig_fn}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax, results_df
