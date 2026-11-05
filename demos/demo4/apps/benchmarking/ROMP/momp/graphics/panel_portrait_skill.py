import os
#import pickle
import numpy as np
import matplotlib.pyplot as plt
from momp.lib.loader import get_cfg, get_setting
#from momp.io.output import nested_dict_to_array, analyze_nested_dict
from momp.utils.visual import portrait_plot
from momp.io.dict import extract_binned_dict #, extract_overall_dict
from momp.utils.printing import tuple_to_str


def panel_portrait_bss_auc(result_binned, verification_window, *, dir_fig, show_panel=True, **kwargs):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
    
    # load binned BSS, add climatology on top row
    arr, row_labels, col_labels = extract_binned_dict(result_binned, 'Fair_Brier_Skill_Score')
    data = np.vstack([np.zeros(arr.shape[1]), arr])
    data *= 100
    row_labels = ["Climatology"] + row_labels

    fig, ax1, im = portrait_plot(data, col_labels, row_labels, fig=fig, ax=ax1, 
                                 annotate=True, annotate_data=data, title=r'(a) Brier Skill Score ($\%$)', 
                                 colorbar_off=True)
    ax1.set_xlabel('Forecast window (days)')
    

    # load binned AUC, add climatology on top row
    arr, row_labels, col_labels = extract_binned_dict(result_binned, 'AUC')
    arr_clim, _, _ = extract_binned_dict(result_binned, 'AUC_ref')
    print("arr = ", arr)
    print("arr_clim = ", arr_clim)
    data = np.vstack([arr_clim[0], arr])
    row_labels = ["Climatology"] + row_labels
    
    fig, ax2, im = portrait_plot(data, col_labels, row_labels, fig=fig, ax=ax2, 
                                 annotate=True, annotate_data=data, title='(b) AUC', 
                                 colorbar_off=True)#, cbar_kw={"orientation":"horizontal"})
    ax2.set_xlabel('Forecast window (days)')
    
    fig.tight_layout()
    if show_panel:
        plt.show()

    # save figure
    window_str = tuple_to_str(verification_window)
    #figure_filename = f"panel_portrait_BSS_AUC_{kwargs['max_forecast_day']}day.png"
    figure_filename = f"panel_portrait_BSS_AUC_{window_str}.png"
    figure_filename = os.path.join(dir_fig, figure_filename)
    fig.savefig(figure_filename, dpi=300, bbox_inches='tight')
    print(f"Figure saved as '{figure_filename}'")

    return fig, (ax1, ax2)



if __name__ == "__main__":

    import pandas as pd

    cfg, setting = get_cfg(), get_setting()

    results = {}

    #model_list = cfg.get("model_list")
    #max_forecast_day = cfg.get("max_forecast_day")
    model_list = cfg.model_list
    max_forecast_day = cfg.max_forecast_day
    verification_window = cfg.verification_window_list[0]
    window_str = tuple_to_str(verification_window)

    for model in model_list:
        #fout = os.path.join(cfg['dir_out'],"binned_skill_scores_{}_{}day.csv")
        fout = os.path.join(cfg.dir_out,"binned_skill_scores_{}_{}.csv")
        #fout = fout.format(model, max_forecast_day)
        fout = fout.format(model, window_str)
        df = pd.read_csv(fout)
        dic = df.to_dict(orient='list')
        results[model] = dic
    
#    fout = os.path.join(cfg['dir_out'],"combi_binned_skill_scores_results.pkl")
#    with open(fout, "rb") as f:
#        import pickle
#        results = pickle.load(f)
    
    panel_portrait_bss_auc(results, verification_window, **vars(cfg))


#mae = nested_dict_to_array(results, "mean_mae") # "miss_rate", "false_alarm_rate"
#print(mae)

#model_list = cfg["model_list"]
#window_list = cfg["verification_window_list"]
