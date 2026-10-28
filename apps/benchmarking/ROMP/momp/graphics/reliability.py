# apps/benchmarking/ROMP/momp/graphics/reliability.py

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from momp.io.dict import extract_pd_bins
from momp.utils.printing import tuple_to_str
from momp.lib.control import filter_bins_in_window

#def plot_reliability_diagram(combined_forecast_obs, years, max_forecast_day, save_fig, dir_fig, **kwargs):
#def plot_reliability_diagram(combined_forecast_obs, *, model, max_forecast_day, day_bins,
def plot_reliability_diagram(combined_forecast_obs, *, model, verification_window, day_bins,
                             save_fig, dir_fig, extract_bins=True, show_plot=True, **kwargs):
    """Plot reliability diagram from forecast-observation pairs."""

#    day_bins_filtered = filter_bins_in_window(day_bins, verification_window)

    if extract_bins:
        combined_forecast_obs = extract_pd_bins(combined_forecast_obs, day_bins)
#        combined_forecast_obs = extract_pd_bins(combined_forecast_obs, day_bins_filtered)

    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    data = combined_forecast_obs.copy()
    data['predicted_prob'] = pd.to_numeric(data['predicted_prob'], errors='coerce')
    data['observed_onset'] = pd.to_numeric(data['observed_onset'], errors='coerce')
    data = data.dropna(subset=['predicted_prob', 'observed_onset'])
    data = data[data['predicted_prob'].between(0, 1)]
    if data.empty:
        raise ValueError("No valid forecast probabilities and observations for reliability diagram")

    event_column = 'bin_label' if 'bin_label' in data else None
    event_groups = (
        [(str(label), group) for label, group in data.groupby(event_column, sort=False, dropna=False)]
        if event_column else [('All events', data)]
    )
    results_for_csv = []

    print("\nReliability Analysis (separate curve for each onset-time event):")
    print("Event bin\tProbability range\tN forecasts\tMean forecast\tObserved frequency")
    print("-" * 100)

    fig, axes = plt.subplots(
        len(event_groups), 1,
        figsize=(10, max(4, 3.8 * len(event_groups))),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    axes = axes[:, 0]
    for panel_index, (event_label, event_data) in enumerate(event_groups):
        ax = axes[panel_index]
        mean_forecast_prob = np.full(n_bins, np.nan)
        observed_frequency = np.full(n_bins, np.nan)
        error_bars = np.full(n_bins, np.nan)
        counts = np.zeros(n_bins, dtype=int)

        for i in range(n_bins):
            lower, upper = bin_edges[i], bin_edges[i + 1]
            if i == 0:
                mask = event_data['predicted_prob'].between(lower, upper, inclusive='both')
            else:
                mask = (event_data['predicted_prob'] > lower) & (event_data['predicted_prob'] <= upper)
            group = event_data.loc[mask]
            counts[i] = len(group)
            if counts[i] == 0:
                continue

            mean_forecast_prob[i] = group['predicted_prob'].mean()
            observed_frequency[i] = group['observed_onset'].mean()
            error_bars[i] = np.sqrt(
                observed_frequency[i] * (1 - observed_frequency[i]) / counts[i]
            )
            print(
                f"{event_label}\t{lower:.1f}-{upper:.1f}\t{counts[i]}\t"
                f"{mean_forecast_prob[i]:.3f}\t{observed_frequency[i]:.3f}"
            )
            results_for_csv.append({
                'Event_Bin': event_label,
                'Bin_Range': f"{lower:.1f}-{upper:.1f}",
                'N_Forecasts': counts[i],
                'Mean_Forecast_Prob': mean_forecast_prob[i],
                'Observed_Frequency': observed_frequency[i],
                'Frequency': counts[i] / len(event_data),
                'Error_Bar': error_bars[i],
            })

        valid = ~np.isnan(observed_frequency)
        if valid.any():
            ax.errorbar(
                mean_forecast_prob[valid], observed_frequency[valid],
                yerr=error_bars[valid], fmt='o-', linewidth=1.8,
                markersize=6, capsize=3, color='tab:blue', label='Observed frequency',
            )
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect reliability')
        ax.set_title(event_label)
        ax.set_ylabel('Observed frequency', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        event_frequency = np.histogram(event_data['predicted_prob'], bins=bin_edges)[0] / len(event_data)
        ax2 = ax.twinx()
        positive_frequency = event_frequency[event_frequency > 0]
        ax2.bar(
            bin_centers,
            event_frequency,
            width=0.08,
            alpha=0.2,
            color='gray',
            label='Forecast frequency',
            zorder=0,
        )
        if len(positive_frequency):
            ax2.set_yscale('log')
            ax2.set_ylim(positive_frequency.min() * 0.5, positive_frequency.max() * 2)
        ax2.set_ylabel('Forecast frequency', fontsize=9)
        handles, labels = ax.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(handles + handles2, labels + labels2, loc='best', fontsize=8)

    results_df = pd.DataFrame(results_for_csv)
    axes[-1].set_xlabel('Forecast probability', fontsize=12)

# this block has not been used, so no need years in the function args
#    if len(years) > 1:
#        year_str = f"{min(years)}-{max(years)}"
#    else:
#        year_str = str(years[0])

    fig.suptitle(f"Reliability by onset-time bin — {model} — {tuple_to_str(verification_window)}")
    fig.tight_layout(rect=(0, 0, 1, 0.98))

    if save_fig:
        os.makedirs(dir_fig, exist_ok=True)
        #model = kwargs.get("model")
        window_str = tuple_to_str(verification_window)
        #fig_fn = os.path.join(dir_fig, f'reliability_{model}_{max_forecast_day}day.png')
        fig_fn = os.path.join(dir_fig, f'reliability_by_onset_bin_{model}_{window_str}.png')
        fig.savefig(fig_fn, dpi=600, bbox_inches='tight')
        print(f"Figure saved to: {fig_fn}")

    if show_plot:
        plt.show()

    return fig, axes[0], results_df

