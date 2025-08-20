import numpy as np
import os
import pandas as pd
from scipy.stats import poisson

seed = 1231
np.random.seed(seed)

# Input parameters
poisson_lambda = 0.3
num_microbes_list = [3,4,5,6]
num_experiments_list = [2,4,6,8,10,12]
n_random_runs = 5
sample_size = 100


# Define folder and file paths
DATAFOLDER = os.path.join(os.getcwd(),
                          'sim_results_mu_0_3_NegInteraction_n96')  
files_name = [
    'C_ij_means_simulation',
    'f_ij_means_simulation'
]


target_perc_runs = {}
for N_mic in num_microbes_list:
    target_perc_runs[N_mic] = []
    for run_idx in range(n_random_runs):
        max_num_exps = np.max(num_experiments_list)
        target_perc = np.zeros((N_mic, max_num_exps), dtype=np.float64)
        for i in range(max_num_exps):
            fractions = np.random.dirichlet(np.ones(N_mic))
            target_perc[:, i] = np.round(fractions * 100, decimals=6)
            
        target_perc_runs[N_mic].append(target_perc)

print(f'target_perc_runs: \n {target_perc_runs}')


# Calculation Function

def get_final_fracs(target_fracs_ic, C_ij, f_ij, poisson_lambda):

    # Loop over different experiments
    n_exps = target_fracs_ic.shape[1]  # retrives number of columns , in this case experiments
    final_fracs = np.zeros_like(target_fracs_ic)
    total_cells_i = np.zeros(n_exps)

    for exp_ind in range(n_exps):
        # First get all pairwise products of the initial fracs:
        target_fracs = target_fracs_ic[:, exp_ind]
        p_times_p_ij = np.matmul(target_fracs[:, None], target_fracs[None, :])
        p_times_p_ij *= 2 * poisson.pmf(2, poisson_lambda)  # DHdG: times probability of 2-cell droplet

        # Add term for cell being on its own
        # DHdG: p_i = initial_fracs * probability of 1-cell droplet
        p_i = target_fracs * poisson.pmf(1, poisson_lambda)
        # SbG: but there will also be droplets with 2 cells of only one cell-type
        p_2i = poisson.pmf(2, poisson_lambda) * target_fracs ** 2

        # In the denominator of the fractions, we will sum for all the combinations i,j the product of  C * p * p:
        # First we only select the upper triangle of the matrix, to not count (i, j) and (j, i)
        upper_tri_indices = np.triu_indices_from(p_times_p_ij, k=1)
        total_cells_i[exp_ind] = np.sum(C_ij[upper_tri_indices] * p_times_p_ij[upper_tri_indices])

        # DHdG: Add cells created by isolated growth total_cells_i[exp_ind] += np.sum(C_i * p_i)
        total_cells_i[exp_ind] += np.sum(np.diag(C_ij) * p_i)
        total_cells_i[exp_ind] += np.sum(np.diag(C_ij) * p_2i)

        # The numerator of the fraction of strain i will take only the i-th row of the multiplication f_ij, C_ij, p*p_ij
        # Create a mask to ignore diagonal elements
        diagonal_mask = np.eye(f_ij.shape[0], f_ij.shape[1])  # Identity matrix of the same shape
        non_diagonal_mask = 1 - diagonal_mask  # Set diagonal to 0 and off-diagonal to 1
        # Apply the mask to the computation
        cells_i = np.sum(f_ij * C_ij * p_times_p_ij * non_diagonal_mask, axis=1)

        # DHdG: Add cells created by isolated growth: cells_i += C_i * p_i
        cells_i += (np.diag(C_ij) * p_i)
        cells_i += (np.diag(C_ij) * p_2i)

        # Finally, return the fractions
        final_fracs[:, exp_ind] = cells_i / total_cells_i[exp_ind]

    return final_fracs


# Iter over given parameters

for idx, N_mic in enumerate(num_microbes_list):

    Microb_folder = os.path.join(DATAFOLDER, 'N_microbe_%i' % N_mic)
    if not os.path.exists(Microb_folder):
        os.makedirs(Microb_folder)

    # Load data into numpy arrays
    files = [f + '_n%i' % (N_mic) + '.xlsx' for f in files_name]
    C_ij, f_ij = [
        pd.read_excel(os.path.join(DATAFOLDER, file), header=None).to_numpy()
        for file in files
    ]
    # generating random target fraction sets
    num_cell_types = C_ij.shape[1]

    for run_idx in range(n_random_runs):
        # Simulate data for the maximal number of experiments, then take subsequent subsets
        max_num_exps = np.max(num_experiments_list)

        # First simulate target fractions
        target_perc = np.zeros((num_cell_types, max_num_exps), dtype=np.float64)
        existing_files = 0

        target_perc = target_perc_runs[N_mic][run_idx]
        target_fracs_ic = np.round(target_perc / 100, decimals=6)

        # Use these target fractions to get the final fractions
        final_fracs = get_final_fracs(target_fracs_ic, C_ij, f_ij, poisson_lambda)
        # final_counts = final_fracs*sample_size

        # Use final fractions to sample final counts
        # final_counts = np.array(
        #     [np.random.multinomial(sample_size, final_fracs[:, i]) for i in range(final_fracs.shape[1])]).T
        final_counts = np.zeros_like(target_perc)
        for f_idx in range(final_fracs.shape[1]):
            final_counts[:,f_idx] = np.random.multinomial(sample_size, final_fracs[:, f_idx])
        # final_counts = final_counts.T
        # check_final_fracs =np.sum(final_fracs,axis=0)
        # check_target_fracs = np.sum(target_fracs_ic,axis=0)
        # preventing zeros in initial counts
        # initial_counts_ic = target_fracs_ic*sample_size

        # Use target fractions to sample initial counts
        initial_counts_ic = np.zeros_like(target_perc)
        # initial_counts_ic = np.array(
        #     [np.random.multinomial(sample_size, target_fracs_ic[:, i]) for i in range(target_fracs_ic.shape[1])]).T

        for i_idx in range(target_fracs_ic.shape[1]):
            initial_counts_ic[:,i_idx] = np.random.multinomial(sample_size, target_fracs_ic[:, i_idx])
        # initial_counts_ic = initial_counts_ic.T
        print(f'initial counts: \n {initial_counts_ic}')

        # Now for each "num_experiments = number of experiments" take only the first "num_experiments" columns
        for num_exp_idx in range(len(num_experiments_list)):
            num_experiments = num_experiments_list[num_exp_idx]

            # Create folders first:
            experim_folder = os.path.join(Microb_folder, 'N_exp_%i' % num_experiments)
            if not os.path.exists(experim_folder):
                os.makedirs(experim_folder)
            run_folder = os.path.join(experim_folder, 'Run_%i' % run_idx)
            if not os.path.exists(run_folder):
                os.makedirs(run_folder)

            # Loop through each experiment (columns in target_fracs, initial_fracs_ic, final_fracs)
            for exp_idx in range(num_experiments):
                # Combine the target, initial, and final fractions for the current experiment
                export_data = pd.DataFrame(
                    np.vstack([
                        target_perc[:, exp_idx],  # Row for target fractions
                        initial_counts_ic[:, exp_idx],  # Row for initial fractions
                        final_counts[:, exp_idx]  # Row for final fractions
                    ]),
                    index=["Target", "Initial", "Final"],  # Row labels
                    columns=[f"Cell_type_{i + 1}" for i in range(target_perc.shape[0])]  # Column labels
                )            
                export_file = os.path.join(run_folder, f"Simulation_Consortium_Set_{exp_idx + 1}.csv")

                export_data.to_csv(export_file)
                # print(target_perc[:,exp_idx])
                # print(f"Exported: {export_file}")
