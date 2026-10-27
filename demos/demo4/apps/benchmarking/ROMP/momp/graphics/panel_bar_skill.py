import os
#import pickle
import numpy as np
import matplotlib.pyplot as plt
from momp.lib.loader import get_cfg, get_setting
#from momp.io.output import nested_dict_to_array, analyze_nested_dict
#from momp.utils.visual import portrait_plot
from momp.io.dict import extract_overall_dict
from momp.utils.printing import tuple_to_str


def panel_bar_bss_rpss_auc(result_overall, verification_window, *, dir_fig, 
                           legend=True, show_panel=True, **kwargs):

    fig, ax = plt.subplots(figsize=(8, 5))

    #window = kwargs.get("verification_window_list")[-1]
    #window = kwargs.get("verification_window")
    window = verification_window
    title = f"{tuple_to_str(window)} day forecast"

#    if verification_window[0] != 1:
#        raise ValueError("verification_window must start with 1 for RPSS")

    
    # load binned BSS, add climatology on top row
    bss, model_list  = extract_overall_dict(result_overall, 'Fair_Brier_Skill_Score')
    rps, _  = extract_overall_dict(result_overall, 'Fair_RPS_Skill_Score')
    auc, _  = extract_overall_dict(result_overall, 'AUC')
    print("/n/n result_overall = ", result_overall)
    auc_ref, _  = extract_overall_dict(result_overall, 'AUC_ref')#[0]
    auc_ref = auc_ref[0]

#    bss, model_list  = extract_overall_dict(result_overall, 'Fair Brier Skill Score')
#    rps, _  = extract_overall_dict(result_overall, 'Fair RPS Skill Score')

    bss *= 100
    rps *= 100

#    print("\n model list = ", model_list)

#    print("auc = ", auc)
#    print("auc_ref = ", auc_ref)
#    print("bss = ", bss)
#    print("rps = ", rps)
    
    # Create second x-axis (top)
    ax2 = ax.twiny()
    
    # Bar height
    height = 0.2

    y_pos = np.arange(len(model_list))

    # Add colors
    auc_col = np.array([217, 95, 14]) / 256
    rpss_col = np.array([33, 102, 172]) / 256
    bss_col = np.array([146, 197, 222]) / 256
    
    # Plot AUC bars on top
    bars3 = ax2.barh(y_pos + height, auc, height,
                    label='AUC', alpha=0.8, color=auc_col)
    
    # Plot skill scores on bottom axis
    bars1 = ax.barh(y_pos, bss, height,
                   label='BSS', alpha=0.8, color=bss_col)
    bars2 = ax.barh(y_pos - height, rps, height,
                   label='RPSS', alpha=0.8, color=rpss_col)

    # Plot Climatology AUC as vertical line
    ax2.axvline(x=auc_ref, color=auc_col, linestyle='-', linewidth=1.25, alpha=0.8)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1.25, alpha=0.8)


    # Customize axes
    ax.set_xlabel(r'BSS/RPSS ($\%$)', fontsize=12)
    ax.set_title(f'{title}', fontsize=15, fontweight='normal', loc='left')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(model_list, rotation=0, ha='right', fontsize=15)

    min_val = np.min(np.concatenate([rps, bss, auc])) *1.05 # * 1.1
    max_val = np.max(np.concatenate([rps, bss, auc])) *1.3 #* 1.2
    ax.set_xlim(min_val, 25) 
    ax.set_ylim(-0.5, 1.5)

    ax2.set_xlabel('AUC', fontsize=12, color=auc_col)
    ax2.tick_params(axis='x', colors=auc_col)
    ax2.spines['top'].set_color(auc_col)
    ax2.set_xlim(0.6, 1.0)
    ax2.set_xticks(np.arange(0.6, 1.02, 0.05))

    # Control tick label visibility
    if not isinstance(ax, np.ndarray): # single panel plot
        ax.tick_params(labelbottom=True, labeltop=False)
        ax2.tick_params(labelbottom=False, labeltop=True)
        ax.tick_params(labelleft=True)
    else:
        ax.tick_params(labelbottom=True, labeltop=False)
        ax2.tick_params(labeltop=True)
        ax.tick_params(labelleft=False)


    clim_line = plt.Line2D([0], [0], color=auc_col, linestyle='-', linewidth=1.25,
                              label='Climatology')

    # Add legends - only for panel 2
    if legend:
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        legend = ax.legend(lines1 + lines2 + [clim_line],
                          labels1 + labels2 + ['climatology'],
                          loc='lower right', frameon=False)
    
    fig.tight_layout()
    if show_panel:
        plt.show()

    # save figure
    window_str = tuple_to_str(verification_window)
    #figure_filename = f"panel_bar_BSS_RPSS_AUC_{tuple_to_str(window)}.png"
    #figure_filename = f"panel_bar_BSS_RPSS_AUC_{kwargs['max_forecast_day']}.png"
    figure_filename = f"panel_bar_BSS_RPSS_AUC_{window_str}.png"
    figure_filename = os.path.join(dir_fig, figure_filename)
    fig.savefig(figure_filename, dpi=300, bbox_inches='tight')
    print(f"Figure saved as '{figure_filename}'")

    return fig, (ax, ax2)


if __name__ == "__main__":

    import pandas as pd
    from collections import defaultdict

    cfg, setting = get_cfg(), get_setting()

    #results = {}
    results = defaultdict(dict)

    #model_list = cfg.get("model_list")
    #max_forecast_day = cfg.get("max_forecast_day")
    model_list = cfg.model_list
    max_forecast_day = cfg.max_forecast_day
    verification_window = cfg.verification_window_list[0]
    window_str = tuple_to_str(verification_window)

    for model in model_list:
        #fout = os.path.join(cfg.dir_out,"overall_skill_scores_{}_{}day.csv")
        fout = os.path.join(cfg.dir_out,"overall_skill_scores_{}_{}.csv")
        #fout = fout.format(model, max_forecast_day)
        fout = fout.format(model, window_str)
        df = pd.read_csv(fout)
        #dic = df.to_dict(orient='list') # this option return key values as list, cause error plotting
        dic_list = df.to_dict(orient='list') # this option return key values as list, cause error plotting
        #df = pd.read_csv(fout, index_col=0)
        #dic = df.to_dict(orient='index')
        dic = {k: v[0] for k, v in dic_list.items()} # convert list[np.float64] into np.float64
        results[model] = dic
    
#    from pprint import pprint
#    pprint(results)

##    fout = os.path.join(cfg.dir_out,f"combi_binned_skill_scores_{max_forecast_day}day.pkl")
#    fout = os.path.join(cfg.dir_out,f"combi_overall_skill_scores_{max_forecast_day}day.pkl")
#    with open(fout, "rb") as f:
#        import pickle
#        results = pickle.load(f)
    
    panel_bar_bss_rpss_auc(results, verification_window, **vars(cfg))


#mae = nested_dict_to_array(results, "mean_mae") # "miss_rate", "false_alarm_rate"
#print(mae)

#model_list = cfg["model_list"]
#window_list = cfg["verification_window_list"]
