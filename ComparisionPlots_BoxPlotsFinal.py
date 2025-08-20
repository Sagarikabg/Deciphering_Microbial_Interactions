import numpy as np
import os
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import seaborn as sns
import sys
from matplotlib.lines import Line2D


plt.rcParams.update({'font.size': 16})

#Parameters
n_errors = 2
num_microbe_list = [4,5,6]
num_exp_list = [2,4,6,8,10,12]
num_exp_list_str = list(map(str, num_exp_list))
num_runs = 5
RESULTSFOLDER = ['inf_results_mu_0_3_NoIsolatedGrowth_n96','inf_results_mu_0_3_SingleIsolatedGrowth_n96','inf_results_mu_0_3_DoubleIsolatedGrowth_n96','inf_results_mu_0_3_NegInteraction_n96']
SIMFOLDER = ['sim_results_mu_0_3_NoIsolatedGrowth_n96','sim_results_mu_0_3_SingleIsolatedGrowth_n96','sim_results_mu_0_3_DoubleIsolatedGrowth_n96','sim_results_mu_0_3_NegInteraction_n96']


# colorblind-friendly colors 
colors = ['r',  # orange-yellow (similar to 'y')
                      'b',  # sky blue (similar to 'b')
                      'y',  # bluish green (alternative to 'g')
                      'g']  # reddish orange (alternative to 'r')
run_markers = ['o', '+', '*', 's', 'D']

#Plot setup
fig, ax = plt.subplots(n_errors, len(num_microbe_list),sharex=True,sharey='row', figsize=(12,8))#, sharex=True
if len(num_microbe_list) == 1:
    ax = ax[:, np.newaxis]
# fig.suptitle(RESULTSFOLDER, fontsize=10)
fig.supxlabel("Number of experiments")

ticks_list =[]

for set_idx, (result_set, sim_set) in enumerate (zip(RESULTSFOLDER , SIMFOLDER)): 

    main_resultsfolder = os.path.join(os.getcwd(), result_set)
    main_simfolder = os.path.join(os.getcwd(), sim_set)

    #NumPy 2D object array with num_microbe_list x num_exp_list dimensions and only the third dimension varies as a 1D array stored at each [i,j] position
    data_inference_error_Cij = np.empty ((len(num_microbe_list), len(num_exp_list),num_runs), dtype=object) 
    data_inference_error_fij = np.empty ((len(num_microbe_list), len(num_exp_list),num_runs), dtype=object)

    for i, n_microb in enumerate(num_microbe_list):
        n_combis = int(n_microb * (n_microb - 1) / 2)
        for j in range (len(num_exp_list)):
            for k in range (num_runs):
                data_inference_error_Cij[i,j,k] = np.zeros (n_combis-1)
                data_inference_error_fij[i,j,k] = np.zeros (n_combis)


    for i, n_microb in enumerate(num_microbe_list):
        
        C_ij_simulation_df = pd.read_excel(os.path.join(main_simfolder, 'C_ij_means_simulation_n%i.xlsx' % n_microb),
                                       header=None)
        C_ij_simulation = C_ij_simulation_df.to_numpy(dtype=float)
        f_ij_simulation_df = pd.read_excel(os.path.join(main_simfolder, 'f_ij_means_simulation_n%i.xlsx' % n_microb),
                                       header=None)
        f_ij_simulation = f_ij_simulation_df.to_numpy(dtype=float)

        results_Microb_folder = os.path.join(main_resultsfolder, 'N_microbe_%i' % n_microb)

        for j, num_exp in enumerate(num_exp_list):        

            results_exp_folder = os.path.join(results_Microb_folder, 'N_exp_%i' % num_exp)

            for run in range(num_runs):
                results_run_folder = os.path.join(results_exp_folder, 'Run_%i' % run)

                C_ij_final = np.genfromtxt(os.path.join(results_run_folder, 'Simulation_Consortium_C_ij_final.csv'))
                f_ij_final = np.genfromtxt(os.path.join(results_run_folder, 'Simulation_Consortium_f_ij_final.csv'))
                diff_Cij = np.zeros(C_ij_final.shape)
                diff_fij = np.zeros(C_ij_final.shape)
                diff_Cij = np.abs(C_ij_final - C_ij_simulation)  
                diff_fij = np.abs(f_ij_final - f_ij_simulation)


                if np.isnan (diff_Cij).any():
                    print (f"Found NaN! Skipping: Case: N microbes : {n_microb}, Experiment: {num_exp}, Run: {run}  Error Cij: \n{diff_Cij}")
                    continue
                if np.isnan (diff_fij).any():
                    print (f"Found NaN! Skipping: Case: N microbes : {n_microb}, Experiment: {num_exp}, Run: {run}  Error fij: \n{diff_fij}")
                    continue

                triu_inds = np.triu_indices_from(diff_Cij, k=1) #excluding the diagonal
                Cij_sim_upper= C_ij_simulation [triu_inds] 
                diff_Cij_upper = diff_Cij[triu_inds]
                diff_fij_upper = diff_fij[triu_inds]

                mask = Cij_sim_upper > 1e-1 # True if Cij >= 0.1
                filtered_diff_fij = diff_fij_upper[mask]  # keep only corresponding diff_fij where Cij >= 0.1
                selected_diff_Cij = diff_Cij_upper [1:] #exclude the first index
                

                data_inference_error_Cij[i,j,run]= selected_diff_Cij
                data_inference_error_fij [i,j,run]= filtered_diff_fij     





    for mic_ind in range(len(num_microbe_list)):
        combined_data_Cij =[]
        width_between_boxes = 0.68# or even smaller
        spacing = 0
        total_sets = len(RESULTSFOLDER)
        

        for exp_ind, num_exp in enumerate(num_exp_list):
            combined_Cij = np.concatenate(data_inference_error_Cij[mic_ind, exp_ind, :])
            combined_fij = np.concatenate(data_inference_error_fij[mic_ind, exp_ind, :]) 
            # current_data_Cij = data_inference_error_Cij[mic_ind, exp_ind, :]# Combine data for each experiment
            # current_data_Cij = np.asarray(current_data_Cij).flatten()
            # combined_data_Cij.extend(current_data_Cij)
            # combined_data_Cij_np = np.array(combined_data_Cij)
            # flat_data_Cij =combined_data_Cij_np.flatten()

            if set_idx == 2 and mic_ind == 0:
                continue



            if set_idx == 3 and mic_ind == 0:
                continue



            offset = (set_idx - (total_sets - 1)/2) * width_between_boxes #offset around the num_exp
            position = num_exp + offset + spacing
            if set_idx==0:
                ticks_list.append(num_exp+spacing)
            spacing +=1
        

            
            # ax[0].plot(num_exp_list, data_inference_error[idx,:], '-o', label='N microbe = %i' % num_microbe_list[idx])
            # ax[0, idx].scatter(num_exp_list, data_inference_error_Cij[idx, :, run], color=colors[set], marker=run_markers[run],alpha=0.75,
            #                 label='run %i' % run, s=15)

            ax[0,mic_ind].boxplot(combined_Cij, widths=0.4, positions=[position],
                                boxprops=dict(color=colors[set_idx], linewidth=1), capprops=dict(color=colors[set_idx],linewidth=1), whiskerprops=dict(color=colors[set_idx], linewidth=1),
                                medianprops=dict(color=colors[set_idx], linewidth=1), showfliers=False
                                ) #flierprops=  dict(marker='o', markerfacecolor=colors[set],markeredgewidth=0)
            
            ax[1,mic_ind].boxplot(combined_fij, widths=0.4, positions=[position],
                                boxprops=dict(color=colors[set_idx], linewidth=1),capprops=dict(color=colors[set_idx],linewidth=1),whiskerprops=dict(color=colors[set_idx], linewidth=1),
                                medianprops=dict(color=colors[set_idx], linewidth=1), showfliers=False
                                ) #flierprops=  dict(marker='o', markerfacecolor=colors[set],markeredgewidth=0)
                
        print(ticks_list)
        if mic_ind == 0:
                ax[0, 0].set_ylabel(r'Inference Error $G_{ij}$')
                ax[1, 0].set_ylabel(r'Inference Error $f_{i|ij}$')
        ax[0, mic_ind].set_xticks(ticks_list[:6])
        ax[0, mic_ind].set_xticklabels(num_exp_list)
        ax[0, mic_ind].set_ylim(-0.5,2.5)
        ax[0, mic_ind].set_yticks(np.arange(0,2.5,0.5))
        ax[1, mic_ind].set_ylim(-0.2,1.2)
        ax[1, mic_ind].set_yticks(np.arange(0,1.2,0.2))
        # ax[0, mic_ind].grid(alpha=0.5, axis='x')
        # ax[1, mic_ind].grid(alpha=0.5, axis='x')
        # ax[0, mic_ind].set_title('Consortium size = %i' % num_microbe_list[mic_ind])
        # ax[0, mic_ind].set_yscale('log')
        # ax[0, mic_ind].set_ylim ( 2.4)
        # ax[1, mic_ind].set_ylim ( 2.4)
        # ax[0, mic_ind].set_yticks(np.arange (0.4,2.4,0.4))
        # ax[1, mic_ind].set_yticks(np.arange (0.4,2.4,0.4))



plt.tight_layout()
plt.savefig(os.path.join(main_resultsfolder, "Overview of fit_test_comparision_n96.png"), dpi=800)
plt.show()
