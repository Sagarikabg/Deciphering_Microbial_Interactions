import os, re, json
import pandas as pd
import numpy as np
from droplet_interactions_helpers import get_dirichlet_precisions, extract_params_from_matrices, calc_loglikelihood, \
    loglik_wrapper, \
    compile_parameter_mats_from_vect, get_final_fracs, get_uncertainty_region
from scipy import optimize
import time
import seaborn as sns
from matplotlib import pyplot as plt
from pathlib import Path
import sys

COMPARE_TO_TRUTH = False
VERY_VERBOSE = False

t1 = time.time()

# Set the precision parameter (nu) of the experimental preparation. This values should be larger than one, and can be
# interpreted as "pseudocounts", so a larger nu-value will put more weight on the prior. This nu is optimized in the
# "nu optimization_DHdG.py" script at the moment.
exp_prep_precision = 91.66
poisson_lambda = 0.3

num_microbe_list =[5,6]
num_exp_list = [2,4,6,8,10,12]
num_runs = 5

DATAFOLDER = 'sim_results_mu_0_3_NegInteraction_n96'  # data_simulation_N_exp --> consortia of 3 and 4 microbes, but experiments varied
RESULTSFOLDER = 'inf_results_mu_0_3_NegInteraction_n96'  # sim_results_N_exp --> consortia of 3 and 4 microbes, but experiments varied
PARSFOLDER = 'prior_hyperparameters_sim'
main_datafolder = os.path.join(os.getcwd(), DATAFOLDER)
parsfolder = os.path.join(os.getcwd(), PARSFOLDER)
main_resultsfolder = os.path.join(os.getcwd(), RESULTSFOLDER)
Path(main_resultsfolder).mkdir(parents=True, exist_ok=True)

sim_datafiles = []

for microb_ind, n_microb in enumerate(num_microbe_list):
    C_ij_means_consortium_sim_file = 'C_ij_means_consortium_sim_n%i.xlsx' % n_microb
    C_ij_stds_consortium_sim_file = 'C_ij_stds_consortium_sim_n%i.xlsx' % n_microb
    f_ij_means_consortium_sim_file = 'f_ij_means_consortium_sim_n%i.xlsx' % n_microb
    f_ij_stds_consortium_sim_file = 'f_ij_stds_consortium_sim_n%i.xlsx' % n_microb

    results_Microb_folder = os.path.join(main_resultsfolder, 'N_microbe_%i' % n_microb)
    if not os.path.exists(results_Microb_folder):
        os.makedirs(results_Microb_folder)

    for num_exp in num_exp_list:
        results_exp_folder = os.path.join(results_Microb_folder, 'N_exp_%i' % num_exp)
        if not os.path.exists(results_exp_folder):
            os.makedirs(results_exp_folder)
        for run in range (num_runs):
            print(f"Running case N_microbe={n_microb}, N_exp={num_exp}, Run ={run}")
            sys.stdout.flush()

            sim_datafiles = []
            results_run_folder = os.path.join(results_exp_folder, 'Run_%i' % run)
            if not os.path.exists(results_run_folder):
                os.makedirs(results_run_folder)
            working_folder_name = 'N_microbe_%i/N_exp_%i/Run_%i' % (n_microb, num_exp, run)
            current_datafolder = os.path.join(main_datafolder, working_folder_name)
            for file in os.listdir(current_datafolder):
                if re.search(".*" + "Simulation_Consortium" + ".*.csv$", file) is not None:
                    sim_datafiles.append(os.path.join(current_datafolder, file))
            sim_datafiles.sort()

            datafiles = sim_datafiles
            true_initial_est_ic = []
            final_counts_ic = []
            initial_counts_ic = []
            for ind_file, file in enumerate(datafiles):
                # Read in data
                data = pd.read_csv(file, header=0, index_col=0)
                strains = list(data.columns)

                target_fracs = data.loc['Target'].values.astype(dtype=float)
                target_fracs /= np.sum(target_fracs)
                initial_counts = data.loc['Initial'].values
                initial_counts_ic.append(data.loc['Initial'].values)
                final_counts_ic.append(data.loc['Final'].values)

                # Get mean of posterior distribution for the initial fractions
                # As described in the notes_rinke_droplet_interactions.tex-file this is equal to
                # p_i(0) = (nu * t_i + d_i(0)) / (nu + \sum_i d_i(0))
                true_initial_est_ic.append((exp_prep_precision * target_fracs + initial_counts) / (
                        exp_prep_precision + np.sum(initial_counts)))

            true_initial_est_ic = np.vstack(true_initial_est_ic).T  # fractions
            final_counts_ic = np.vstack(final_counts_ic).T
            initial_counts_ic = np.vstack(initial_counts_ic).T  # counts

            """Here one can set per consortium, the prior distribution for the parameters. """
            """The carrying capacities C_{ij}"""
            # Since we only have relative measurements of the growth, we can only measure the C_{ij} up to a common
            # multiplicative factor. Therefore, I will assume that the first of these parameters is always 1, i.e. C_{01}=1.
            # The other parameters thus indicate the growth of those combinations *relative to* the growth of C_{01}.
            # Therefore, the prior information should also be given in those units.

            # I assume the prior is a lognormal-distribution, which is thus determined by its mode and standard deviation.
            # One can set this by filling up the (upper-diagonal part of) the matrix of C_{ij} for the means, and a matrix
            # for the standard-deviations. If one sets the standard-deviation to 0, the prior will be a delta-peak, if the
            # std is set to -1, the prior will be uniform in log-scale: P(C_{ij}) = 1/C_{ij}
            # Important note: this std will thus be in units of "e-fold changes".
            C_ij_means_df = pd.read_excel(os.path.join(parsfolder, C_ij_means_consortium_sim_file), header=None)
            C_ij_means = C_ij_means_df.to_numpy()

            # The following values are completely made-up by me. These have to be decided on either by doing experiments
            # that provide means and stds for the pars, or we have to marginalize over these parameters.
            # Reminder: when there is no information on a parameter at all, you should set the std to -1.
            C_ij_stds_df = pd.read_excel(os.path.join(parsfolder, C_ij_stds_consortium_sim_file), header=None)
            C_ij_stds = C_ij_stds_df.to_numpy()

            # With this std, the 95% confidence interval is thus approximately:
            # np.exp(np.log(1) - 2*1), np.exp(np.log(1) + 2*1) = (0.1353352832366127, 7.38905609893065)

            # Finally, we make these matrices symmetric, as C_ij = C_ji
            # C_ij_means = C_ij_means + C_ij_means.T  # Make symmetric
            # C_ij_stds = C_ij_stds + C_ij_stds.T  # Make symmetric

            """The parameters f^i_{ij}"""
            # I assume that the prior distribution for the fraction parameters are Dirichlet distributions. Again, we can
            # just specify the mean and the std. However, note that not all variances/stds are possible given a certain
            # mean and given that we want the distribution to be peaked around the mean (that is concentrated instead of
            # sparse). Therefore, we will use the following formula to determine the variance, but will set it to the
            # uniform distribution if the inferred nu is negative.
            # Var = p(1-p)/(nu + 1), --> nu = p(1-p)/std**2 - 1

            f_ij_means_df = pd.read_excel(os.path.join(parsfolder, f_ij_means_consortium_sim_file), header=None)
            f_ij_means = f_ij_means_df.to_numpy()

            # The following values are completely made-up by me. These have to be decided on either by doing experiments
            # that provide means and stds for the pars, or we have to marginalize over these parameters.
            # Reminder: when there is no information on a parameter at all, you should set the std to -1.
            f_ij_stds_df = pd.read_excel(os.path.join(parsfolder, f_ij_stds_consortium_sim_file), header=None)
            f_ij_stds = f_ij_stds_df.to_numpy()

            # I have a small function that calculates the precisions of the Dirichlet priors based on the given stds
            f_ij_precs = get_dirichlet_precisions(f_ij_means, f_ij_stds)

            # C_ij, f_ij = compile_parameter_mats(C_ij, f_ij, unknown_pars, init_unknown_pars)
            C_ij_0 = C_ij_means.copy()
            # C_ij_0 = np.ones_like(C_ij_0) - np.eye(C_ij_0.shape[0])
            C_ij_0[0, 1] = 1.0
            C_ij_0[1, 0] = 1.0
            f_ij_0 = f_ij_means.copy()

            # Then there's a function that gets the final fractions, given some parameter values
            # true_final_est_i = get_final_fracs(true_initial_est_i, C_ij_0, f_ij_0)

            log_C_ij_means = C_ij_means.copy()
            log_C_ij_means[log_C_ij_means > 0] = np.log(log_C_ij_means[log_C_ij_means > 0])

            if COMPARE_TO_TRUTH:
                C_ij_simulation_df = pd.read_excel(
                    os.path.join(main_datafolder, 'C_ij_means_simulation_n%i.xlsx' % n_microb), header=None)
                C_ij_simulation = C_ij_simulation_df.to_numpy(dtype=float)
                C_ij_simulation[C_ij_simulation == 0] = 1e-3

                f_ij_simulation_df = pd.read_excel(
                    os.path.join(main_datafolder, 'f_ij_means_simulation_n%i.xlsx' % n_microb), header=None)
                f_ij_simulation = f_ij_simulation_df.to_numpy(dtype=float)

            # Finally, we can calculate the likelihood using the multinomial distribution and the expressions for the prior
            # loglikelihood = calc_loglikelihood(true_fracs=true_final_est_i, final_counts=final_counts, C_ijs=C_ij_0,
            #                                    f_ijs=f_ij_0, log_C_ij_means=log_C_ij_means, C_ij_stds=C_ij_stds,
            #                                    f_ij_means=f_ij_means, f_ij_precs=f_ij_precs)

            # DHdG: The lambda-parameter of your experiment determines the chance of getting 0, 1, 2 cells in a droplet. It's
            # most efficient to calculate those probabilities already here and give them as a parameter to the function below.
            # In that way, you only have to calculate these probabilites once, instead of recalculating it every time you run
            # the loglik_wrappper function. You can use "numpy.random.poisson" for this.

            # We can now use this function to optimize the parameters using a wrapper function
            # DHdG: This setting of the bounds and the params0 needs to be updated to include bounds and initial guesses for
            # the self-growth parameters
            params0 = extract_params_from_matrices(C_ij_0, f_ij_0)
            n_strains = C_ij_0.shape[0]
            n_combis = int((n_strains * (n_strains - 1) / 2) + n_strains)
            bounds = [(-10, 10)] * (n_combis - 1) + [(1e-6, 1 - 1e-6)] * (n_combis - n_strains)

            if COMPARE_TO_TRUTH:
                log_C_ij_means_simulation = C_ij_simulation.copy()
                log_C_ij_means_simulation[log_C_ij_means_simulation > 0] = np.log(
                    log_C_ij_means_simulation[log_C_ij_means_simulation > 0])

                C_ij_stds_sim = C_ij_stds.copy()
                np.fill_diagonal(C_ij_stds_sim, -1)
                f_ij_stds_sim = f_ij_stds.copy()
                np.fill_diagonal(f_ij_stds_sim, -1)

                f_ij_precs_sim = get_dirichlet_precisions(f_ij_simulation, f_ij_stds_sim)

                true_fracs_simulation, total_cells_simulation = get_final_fracs(true_initial_est_ic, C_ij_simulation,
                                                                                f_ij_simulation, poisson_lambda)
                # optres_initial= calc_loglikelihood(true_fracs_simulation, final_counts_ic, C_ij_simulation, f_ij_simulation, log_C_ij_means_simulation,
                #                                C_ij_stds_sim, f_ij_simulation, f_ij_precs_sim,)
                # diff=true_fracs_simulation-(final_counts_ic)/1e7
                params_sim = extract_params_from_matrices(C_ij_simulation, f_ij_simulation)

                optres_initial_sim_wrapper = - loglik_wrapper(params_sim, true_initial_est_ic, final_counts_ic,
                                                              log_C_ij_means_simulation, C_ij_stds_sim, f_ij_simulation,
                                                              f_ij_precs_sim, poisson_lambda, False)
                # print("Loglikelihood with simulation parameters:", optres_initial)
                print("Loglikelihood with simulation parameters:", optres_initial_sim_wrapper)
                C_ij_simulation[C_ij_simulation == 1e-3] = 0

            observed_final_fracs = final_counts_ic / np.sum(final_counts_ic, axis=0)
            if VERY_VERBOSE:
                true_fracs_initial_guess, total_cells_initial_guess = get_final_fracs(true_initial_est_ic, C_ij_0,
                                                                                      f_ij_0,
                                                                                      poisson_lambda)
                optres_initial_wrapper = -loglik_wrapper(params0, true_initial_est_ic, final_counts_ic, log_C_ij_means,
                                                         C_ij_stds, f_ij_means, f_ij_precs, poisson_lambda, False)
                print('Loglikelihood with initial guess', optres_initial_wrapper)

            t2 = time.time()
            if VERY_VERBOSE:
                print(f'Time for PreProcess = {t2 - t1:.4f} s\n')

            success = False
            attempts = 0
            MAX_ATTEMPTS = 3
            while (not success) and (attempts < MAX_ATTEMPTS):
                optres = optimize.minimize(loglik_wrapper, params0,
                                           args=(true_initial_est_ic, final_counts_ic, log_C_ij_means,
                                                 C_ij_stds, f_ij_means, f_ij_precs, poisson_lambda, False),
                                           bounds=bounds, options={'maxiter': 50000,
                                                                   'maxfun': 1e10})  # method='L-BFGS-B',jac='2-point',options={'maxfun': 1e10,'maxcor':4}) #options={'maxiter': 50000, 'maxfun': 1e10}
                attempts += 1
                t3 = time.time()
                if VERY_VERBOSE:
                    print('Optimized loglikelihood', -optres.fun)

                C_ij_final, f_ij_final = compile_parameter_mats_from_vect(pars_vect=optres.x, n_strains=C_ij_0.shape[0])
                # optres = optimize.basinhopping(loglik_wrapper, params0, minimizer_kwargs={'args':(true_initial_est_ic, final_counts_ic, log_C_ij_means,
                #                                                           C_ij_stds, f_ij_means, f_ij_precs, False), 'bounds':bounds})

                # DHdG: In the following function I treat the C_ij different than the f_ij at some points, and I distinguish them by
                # using that the n_combis - 1 first parameters are C_ij. This is no longer true when you include C_i
                lbs_uncertainty, ubs_uncertainty, report = get_uncertainty_region(opt_pars=optres.x,
                                                                          true_initial_est_ic=true_initial_est_ic,
                                                                          final_counts_ic=final_counts_ic,
                                                                          log_C_ij_means=log_C_ij_means,
                                                                          C_ij_stds=C_ij_stds,
                                                                          f_ij_means=f_ij_means, f_ij_precs=f_ij_precs,
                                                                          poisson_lambda=poisson_lambda,
                                                                          bounds=bounds, make_plots=False,
                                                                          savePath=results_run_folder,
                                                                          return_if_error=attempts < MAX_ATTEMPTS)
                success = report['success']
                if not success:
                    print("Finding optimum was unsuccessful. "
                          "Setting initial parameter {} to {}, "
                          "and trying again.".format(report['new_param'][0], report['new_param'][1]))
                    params0[report['new_param'][0]] = report['new_param'][1]
                    sys.stdout.flush()
                    sys.stderr.flush()
                t4 = time.time()

            print(f'Time for Uncertainty Region Calc = {t4 - t3:.4f} s\n')
            sys.stdout.flush()
            sys.stderr.flush()

            # DHdG: This compile_parameter_mats_from_vect needs to be updated to account for the new parameters.

            C_ij_lb_final, f_ij_lb_final = compile_parameter_mats_from_vect(pars_vect=lbs_uncertainty,
                                                                            n_strains=C_ij_0.shape[0])
            C_ij_ub_final, f_ij_ub_final = compile_parameter_mats_from_vect(pars_vect=ubs_uncertainty,
                                                                            n_strains=C_ij_0.shape[0])

            np.set_printoptions(precision=3)
            np.set_printoptions(suppress=True)

            # print("loglikelihood:\n{}".format(loglikelihood))
            observed_initial_fracs = initial_counts_ic / np.sum(initial_counts_ic, axis=0)
            # print (f"observed initial fracs:\n {observed_initial_fracs}")
            # print ("corrected initial fracs :\n", true_initial_est_ic)
            observed_final_fracs = final_counts_ic / np.sum(final_counts_ic, axis=0)
            # print("observed final fracs:\n{}".format(observed_final_fracs))
            true_final_est_ic, total_cells = get_final_fracs(true_initial_est_ic, C_ij_final, f_ij_final,
                                                             poisson_lambda)
            # print("estimated final fracs:\n{}".format(true_final_est_ic))
            # print("")
            if VERY_VERBOSE:
                print(f"Running case N_microbe={n_microb}, N_exp={num_exp}, Run ={run}")
                print("Final carrying capacities (C_ij):\n{}".format(C_ij_final))
                print("Upper Bounds: Final carrying capacities (C_ij):\n{}".format(C_ij_ub_final))
                print("Lower Bounds: Final carrying capacities (C_ij):\n{}\n".format(C_ij_lb_final))
                print("")
                print("Final cross-feeding proportions (f_ij):\n{}".format(f_ij_final))
                print("Upper Bounds: Final cross-feeding proportions (f_ij):\n{}".format(f_ij_ub_final))
                print("Lower Bounds: Final cross-feeding proportions (f_ij):\n{}\n".format(f_ij_lb_final))
                print("")

                # optimization messages
                print("Optimization Sucess:", optres.success)
                print("Message:", optres.message)
                print("Optimized loglikelihood value:", optres.fun)
                print("Number of iterations", optres.nit)
                print("Number of function evaluations:", optres.nfev)
                print("Number of Gradient evaluations", optres.njev)
                print(f'Time for Optimization = {t3 - t2:.4f} s\n')
                print("")

            # export data to plot

            np.savetxt(os.path.join(results_run_folder, 'Simulation_Consortium_Obsereved_initial_fracs.csv'),
                       observed_initial_fracs, fmt='%f')
            np.savetxt(os.path.join(results_run_folder, 'Simulation_Consortium_Corrected_initial_fracs.csv'),
                       true_initial_est_ic)
            np.savetxt(os.path.join(results_run_folder, 'Simulation_Consortium_Obsereved_final_fracs.csv'),
                       observed_final_fracs)
            np.savetxt(os.path.join(results_run_folder, 'Simulation_Consortium_Estimated_final_fracs.csv'),
                       true_final_est_ic)
            np.savetxt(os.path.join(results_run_folder, 'Simulation_Consortium_C_ij_final.csv'), C_ij_final)
            np.savetxt(os.path.join(results_run_folder, 'Simulation_Consortium_C_ij_ub_final.csv'), C_ij_ub_final)
            np.savetxt(os.path.join(results_run_folder, 'Simulation_Consortium_C_ij_lb_final.csv'), C_ij_lb_final)
            np.savetxt(os.path.join(results_run_folder, 'Simulation_Consortium_f_ij_final.csv'), f_ij_final)
            np.savetxt(os.path.join(results_run_folder, 'Simulation_Consortium_f_ij_ub_final.csv'), f_ij_ub_final)
            np.savetxt(os.path.join(results_run_folder, 'Simulation_Consortium_f_ij_lb_final.csv'), f_ij_lb_final)

            file_path = os.path.join(results_run_folder, "optres.json")
            x = optres.x
            fun = optres.fun
            success = optres.success
            message = optres.message
            nfun = optres.nfev
            # Write to text file
            with open(file_path, "w") as f:
                f.write("### Optimization Results ###\n")
                f.write(f"Optimal Parameters (x): {x}\n")
                f.write(f"Objective Function Value (fun): {fun}\n")
                f.write(f"Number of function evaluations: {nfun}\n")
                f.write(f"Success: {success}\n")
                f.write(f"Message: {message}\n")

            res_infoo_file = os.path.join(results_run_folder, "Run_Summary.txt")
            with open(res_infoo_file, "w") as f:
                f.write(f"Running case N_microbe={n_microb}, N_exp={num_exp}, Run ={run}\n")
                f.write(f'Poission Lambda = {poisson_lambda}\n')
                f.write(f'nu = {exp_prep_precision}\n')
                if COMPARE_TO_TRUTH:
                    f.write(f'Simulation_param_loglikelihood = {optres_initial_sim_wrapper}\n')
                    f.write(f'Initial_guess_loglikelihood = {optres_initial_wrapper}\n')

            if COMPARE_TO_TRUTH:
                plt.figure()
                diff_final_fracs = abs(observed_final_fracs - true_final_est_ic)

                sns.heatmap(diff_final_fracs, annot=True, fmt=".2f", cmap='coolwarm',
                            cbar=True)  # annotate with diff, precision
                plt.title("Final fracs")
                plt.savefig(os.path.join(results_run_folder, "final_fracs.png"), dpi=300)

                diff_Cij = abs(C_ij_final - C_ij_simulation)

                diff_fij = abs(f_ij_final - f_ij_simulation)

                plt.figure()
                sns.heatmap(diff_Cij, annot=True, fmt=".2f", cmap='coolwarm',
                            cbar=True)  # annotate with diff, precision
                plt.title("Cij")
                plt.savefig(os.path.join(results_run_folder, "diff_Cij.png"), dpi=300)
                # plt.show()
                plt.figure()
                d_fij = np.full(C_ij_final.shape, np.nan)
                for ind1 in range(C_ij_final.shape[0]):
                    for ind2 in range(C_ij_final.shape[0]):

                        if C_ij_final[ind1, ind2] > 1e-1:
                            d_fij[ind1, ind2] = abs(f_ij_final[ind1, ind2] - f_ij_simulation[ind1, ind2])

                sns.heatmap(d_fij, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)  # annotate with diff, precision
                plt.title("fij")
                plt.savefig(os.path.join(results_run_folder, "diff_fij.png"), dpi=300)
                # plt.show()

t5 = time.time()
print(f'Time to Complete Simulation = {t5 - t1:.4f} s\n')
