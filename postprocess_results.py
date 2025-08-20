import numpy as np
import os
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd

# plt.style.use(['science', 'no-latex'])

n_errors = 4
num_microbe_list = [3,4,5,6]
# num_microbe_list = [3, 4]
num_exp_list = [2,4,6,8,10,12]
num_runs = 5

# colorblind-friendly colors
colors = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]
markers = ['o', '+', '*', 's', 'D']

RESULTSFOLDER = 'inf_results_mu_0_3_NoIsolatedGrowth'
main_resultsfolder = os.path.join(os.getcwd(), RESULTSFOLDER)
SIMFOLDER = 'sim_results_mu_0_3_NoIsolatedGrowth'
main_simfolder = os.path.join(os.getcwd(), SIMFOLDER)

data_inference_error_Cij = np.zeros((len(num_microbe_list), len(num_exp_list), num_runs))
data_inference_error_fij = np.zeros((len(num_microbe_list), len(num_exp_list), num_runs))
# data_inference_error_std = np.zeros((len(num_microbe_list), len(num_exp_list)))

data_coverage = np.zeros((len(num_microbe_list), len(num_exp_list), num_runs))
# data_coverage_std = np.zeros((len(num_microbe_list), len(num_exp_list)))

data_confidence_error = np.zeros((len(num_microbe_list), len(num_exp_list), num_runs))
# data_confidence_error_std = np.zeros((len(num_microbe_list), len(num_exp_list)))

for microb_ind, n_microb in enumerate(num_microbe_list):
    results_Microb_folder = os.path.join(main_resultsfolder, 'N_microbe_%i' % n_microb)
    C_ij_simulation_df = pd.read_excel(os.path.join(main_simfolder, 'C_ij_means_simulation_n%i.xlsx' % n_microb),
                                       header=None)
    C_ij_simulation = C_ij_simulation_df.to_numpy(dtype=float)
    f_ij_simulation_df = pd.read_excel(os.path.join(main_simfolder, 'f_ij_means_simulation_n%i.xlsx' % n_microb),
                                       header=None)
    f_ij_simulation = f_ij_simulation_df.to_numpy(dtype=float)
    for exp_ind, num_exp in enumerate(num_exp_list):
        data_n = np.zeros((num_runs, n_errors))
        results_exp_folder = os.path.join(results_Microb_folder, 'N_exp_%i' % num_exp)
        for run in range(num_runs):
            results_run_folder = os.path.join(results_exp_folder, 'Run_%i' % run)
            # error_data = np.genfromtxt(results_run_folder+'/Fit_Error_values.txt', delimiter=',', skip_header=1)
            C_ij_final = np.genfromtxt(os.path.join(results_run_folder, 'Simulation_Consortium_C_ij_final.csv'))
            C_ij_ub_final = np.genfromtxt(os.path.join(results_run_folder, 'Simulation_Consortium_C_ij_ub_final.csv'))
            C_ij_lb_final = np.genfromtxt(os.path.join(results_run_folder, 'Simulation_Consortium_C_ij_lb_final.csv'))
            f_ij_final = np.genfromtxt(os.path.join(results_run_folder, 'Simulation_Consortium_f_ij_final.csv'))
            f_ij_ub_final = np.genfromtxt(os.path.join(results_run_folder, 'Simulation_Consortium_f_ij_ub_final.csv'))
            f_ij_lb_final = np.genfromtxt(os.path.join(results_run_folder, 'Simulation_Consortium_f_ij_lb_final.csv'))


            # inference_error = 0
            # confidence_error = 0
            # coverage = 0
            # inside_bounds = []
            # log_diff_Cij = np.zeros(C_ij_final.shape)
            # for ind1 in range(C_ij_final.shape[0]):
            #     for ind2 in range(ind1 + 1, C_ij_final.shape[0]):
            #         if (ind1 == 0) and (ind2 == 1):
            #             continue
            #         # inference error = abs (inferred-simulated)
            #         if (C_ij_simulation[ind1, ind2] == 0):  # condtion if either are equal to zero, then ln(0) = inf
            #             log_diff_Cij[ind1, ind2] = abs(
            #                 np.log(C_ij_final[ind1, ind2]) - (-10))  # because lower bound is at -10 in log
            #         else:
            #             log_diff_Cij[ind1, ind2] = abs(np.log(C_ij_final[ind1, ind2]) - np.log(
            #                 C_ij_simulation[ind1, ind2]))  # Normal log difference otherwise
            #         diff_c = log_diff_Cij[ind1, ind2]

            inference_error_c = 0
            inference_error_f = 0
            n_strains = C_ij_final.shape[0]
            n_combis = int(n_strains * (n_strains - 1) / 2)
            count_C = n_combis - 1  # We have one parameter less than n_combis, because C[0,1] is normalized
            count_f = 0
            confidence_error = 0
            coverage = 0
            inside_bounds = []
            diff_Cij = np.zeros(C_ij_final.shape)
            diff_fij = np.zeros(C_ij_final.shape)

            for ind1 in range(C_ij_final.shape[0]):
                for ind2 in range(ind1 + 1, C_ij_final.shape[0]):

                    # inference error = abs (inferred-simulated)
                    # if (C_ij_simulation[ind1, ind2] == 0):  # condtion if either are equal to zero, then ln(0) = inf
                    #     log_diff_Cij[ind1, ind2] = abs(
                    #         np.log(C_ij_final[ind1, ind2]) - (-10))  # because lower bound is at -10 in log
                    # else:
                    diff_Cij[ind1, ind2] = abs((C_ij_final[ind1, ind2]) - (C_ij_simulation[ind1, ind2]))  
                    diff_c = diff_Cij[ind1, ind2]

                    if (ind1 == 0) and (ind2 == 1):
                        diff_c = 0
                    inference_error_c += diff_c

                    if C_ij_simulation[ind1, ind2] > 1e-1:
                        diff_fij[ind1, ind2] = abs((f_ij_final[ind1, ind2]) - (f_ij_simulation[ind1, ind2]))  
                        diff_f = diff_fij[ind1, ind2]
                        inference_error_f += diff_f
                        count_f += 1

            for ind1 in range(C_ij_final.shape[0]):
                for ind2 in range(ind1 + 1, C_ij_final.shape[0]):
                    # coverage = % of parameters for which the simulated parameters fall b/w the ub and lb of the inferred parameters
                    if ind1 == 0 and ind2 == 1:
                        pass
                    elif (round(C_ij_lb_final[ind1, ind2], 1) <= round(C_ij_simulation[ind1, ind2], 1) <= round(
                            C_ij_ub_final[ind1, ind2], 1)):  # rounds to 0 decimal points
                        inside_bounds.append(True)  # 1 if true and 0 if false, list
                    else:
                        inside_bounds.append(False)
                    # print (f"Cij_sim={C_ij_simulation[ind1,ind2]}, bounds ={inside_bounds}")
                    if C_ij_simulation[ind1, ind2] < 1e-1:
                        continue
                    elif (round(f_ij_lb_final[ind1, ind2], 2) <= round(f_ij_simulation[ind1, ind2], 2) <= round(
                            f_ij_ub_final[ind1, ind2], 2)):
                        inside_bounds.append(True)
                    else:
                        inside_bounds.append(False)

            # print (f"fij_sim={f_ij_simulation[ind1,ind2]}, bounds ={inside_bounds}")

            for ind1 in range(C_ij_final.shape[0]):
                for ind2 in range(ind1 + 1, C_ij_final.shape[0]):
                    # confidence error
                    # if (C_ij_simulation[ind1, ind2] == 0):  # condition if either are equal to zero, then ln(0) = inf
                    #     C_ij_simulation[ind1, ind2] = np.exp(-10)

                    if ind1 == 0 and ind2 == 1:
                        conf_err = 0
                    elif C_ij_final[ind1, ind2] > C_ij_simulation[ind1, ind2]:
                        conf_err = abs((C_ij_final[ind1, ind2]) - (C_ij_simulation[ind1, ind2])) / (
                                    (C_ij_final[ind1, ind2]) - (C_ij_lb_final[ind1, ind2]))
                    elif C_ij_final[ind1, ind2] < C_ij_simulation[ind1, ind2]:
                        conf_err = abs((C_ij_final[ind1, ind2]) - (C_ij_simulation[ind1, ind2])) / (
                                    (C_ij_ub_final[ind1, ind2]) - (C_ij_final[ind1, ind2]))
                    else:
                        conf_err = 0

                    confidence_error += conf_err
                    # print("C[{},{}]: {}, confidence error: {}".format(ind1, ind2, C_ij_final[ind1, ind2], conf_err))

                    if C_ij_simulation[ind1, ind2] > 1e-1:
                        if f_ij_final[ind1, ind2] > f_ij_simulation[ind1, ind2]:
                            conf_err = diff_fij[ind1, ind2] / (f_ij_final[ind1, ind2] - f_ij_lb_final[ind1, ind2])
                        elif f_ij_final[ind1, ind2] < f_ij_simulation[ind1, ind2]:
                            conf_err = diff_fij[ind1, ind2] / (f_ij_ub_final[ind1, ind2] - f_ij_final[ind1, ind2])
                        else:
                            conf_err = 0

                        confidence_error += conf_err
                        # print("f[{},{}]: {}, confidence error: {}".format(ind1, ind2, f_ij_final[ind1, ind2], conf_err))

            coverage = 100 * np.sum(inside_bounds) / (count_C + count_f)
            normalized_inference_error_c = inference_error_c / count_C
            normalized_inference_error_f = inference_error_f / count_f
            normalized_confidence_error = confidence_error / (count_C + count_f)

            print (f"\nCase N_microbe={n_microb}, N_exp={num_exp}, Run ={run}")
            print(f"Cij Error for simulation and inference model fit: {normalized_inference_error_c}")
            print(f"fij Error for simulation and inference model fit: {normalized_inference_error_f}")
            print(f'Inside_bounds \n', inside_bounds)
            print(f"No. of parameters processed:{len(inside_bounds)}")
            print(f"Percentage of parameters predicted within the certainity region:{coverage}")
            print(f"Error for simulation and inference model fit, relative to ub or lab: {normalized_confidence_error}")

            error_file = os.path.join(results_run_folder, "Fit_Error_values.txt")
            with open(error_file, "w") as f:
                f.write(f"Cij Inference Error,fij Inference Error,Coverage,Confidence error\n")
                f.write(
                    f"{normalized_inference_error_c:.4f},{normalized_inference_error_f:.4f},{coverage:.4f},{normalized_confidence_error:.4f}\n")

            data_n[run, 0] = normalized_inference_error_c
            data_n[run, 1] = normalized_inference_error_f
            data_n[run, 2] = coverage
            data_n[run, 3] = normalized_confidence_error

        data_inference_error_Cij[microb_ind, exp_ind, :] = data_n[:, 0]
        data_inference_error_fij[microb_ind, exp_ind, :] = data_n[:, 1]
        # data_inference_error_std[microb_ind, exp_ind] = np.std(data_n[:,0])
        data_coverage[microb_ind, exp_ind, :] = data_n[:, 2]
        # data_coverage_std[microb_ind, exp_ind] = np.std(data_n[:,1])
        data_confidence_error[microb_ind, exp_ind, :] = data_n[:, 3]
        # data_confidence_error_std[microb_ind, exp_ind] = np.std(data_n[:,2])

fig, ax = plt.subplots(n_errors, len(num_microbe_list), sharex=True,sharey='row', figsize=(10, 9))
fig.suptitle(RESULTSFOLDER, fontsize=16)
if len(num_microbe_list) == 1:
    ax = ax[:, np.newaxis]

for idx in range(len(num_microbe_list)):
    for run in range(num_runs):
        # ax[0].plot(num_exp_list, data_inference_error[idx,:], '-o', label='N microbe = %i' % num_microbe_list[idx])
        ax[0, idx].scatter(num_exp_list, data_inference_error_Cij[idx, :, run], color=colors[idx], alpha=0.75,
                           marker=markers[run], label='run %i' % run, s=15)
        ax[0, idx].set_xticks(num_exp_list)
        ax[0, idx].grid(alpha=0.25)
        # ax[0, idx].set_ylim(-0.1,42)
        if idx == 0:
            ax[0, 0].set_ylabel('Average Inference Error Cij')
            ax[0, 0].legend(loc='lower left')  # , bbox_to_anchor=(0, 1.5)
        # ax[0, idx].set_yticks([0,0.5,1,1.5,2,2.5,3,30])
    ax[0, idx].set_title('Consortium size = %i' % num_microbe_list[idx])
    ax[0, idx].set_yscale('log')
    ax[0, idx].set_ylim (10**-5, 10**1)
    ax[0, idx].set_yticks([10**-4, 10**-3,10**-2,10**-1,10**0, 10**1])


for idx in range(len(num_microbe_list)):
    for run in range(num_runs):
        # ax[0].plot(num_exp_list, data_inference_error[idx,:], '-o', label='N microbe = %i' % num_microbe_list[idx])
        ax[1, idx].scatter(num_exp_list, data_inference_error_fij[idx, :, run], color=colors[idx], alpha=0.75,
                           marker=markers[run], label='run %i' % run, s=15)
        if idx == 0:
            ax[1, 0].set_ylabel('Average Inference Error fij')
    ax[1, idx].set_xticks(num_exp_list)
    ax[1, idx].set_xlabel('Number of Experiments')
    ax[1, idx].grid(alpha=0.25)
    ax[1, idx].set_yscale('log')
    ax[1, idx].set_ylim (10**-3, 10**0)
    ax[1, idx].set_yticks([10**-3,10**-2,10**-1])


for idx in range(len(num_microbe_list)):
    for run in range(num_runs):
        # ax[1].plot(num_exp_list, data_coverage[idx,:], '-o', label='N microbe = %i' % num_microbe_list[idx])
        ax[2, idx].scatter(num_exp_list, data_coverage[idx, :, run], color=colors[idx], alpha=0.75, marker=markers[run],
                           s=15)
        if idx == 0:
            ax[2, 0].set_ylabel('Coverage %')
    ax[2, idx].set_xticks(num_exp_list)
    ax[2, idx].grid(alpha=0.25)
    ax[2, idx].set_ylim(0, 120)


for idx in range(len(num_microbe_list)):
    for run in range(num_runs):
        # ax[2].plot(num_exp_list, data_confidence_error[idx,:], '-o', label='N microbe = %i' % num_microbe_list[idx])
        ax[3, idx].scatter(num_exp_list, data_confidence_error[idx, :, run], color=colors[idx], alpha=0.75,
                           marker=markers[run], s=15)
        if idx == 0:
            ax[3, 0].set_ylabel('Confidence Error')
        ax[3, idx].set_xlabel('Number of Experiments')
        ax[3, idx].set_xticks(num_exp_list)
        ax[3, idx].grid(alpha=0.25)
        ax[3, idx].set_yscale('log')

plt.tight_layout()
plt.savefig(os.path.join(main_resultsfolder, "Overview of fit_test.png"), dpi=300)
plt.show()
