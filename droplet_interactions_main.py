import os, re
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




VERY_VERBOSE=False


t1 = time.time()
# Set the precision parameter (nu) of the experimental preparation. This values should be larger than one, and can be
# interpreted as "pseudocounts", so a larger nu-value will put more weight on the prior. This nu is optimized in the
# "nu optimization_DHdG.py" script at the moment.
exp_prep_precision = 91.66
poisson_lambda = 0.3

C_ij_means_consortium1_file = 'C_ij_means_consortium1_uniform_noisolatedgrowth.xlsx'
C_ij_means_consortium2_file = 'C_ij_means_consortium2_uniform_noisolatedgrowth.xlsx'
C_ij_stds_consortium1_file = 'C_ij_stds_consortium1_uniform_noisolatedgrowth.xlsx'
C_ij_stds_consortium2_file = 'C_ij_stds_consortium2_uniform_noisolatedgrowth.xlsx'
f_ij_means_consortium1_file = 'f_ij_means_consortium1_uniform_noisolatedgrowth.xlsx'
f_ij_means_consortium2_file = 'f_ij_means_consortium2_uniform_noisolatedgrowth.xlsx'
f_ij_stds_consortium1_file = 'f_ij_stds_consortium1_uniform_noisolatedgrowth.xlsx'
f_ij_stds_consortium2_file = 'f_ij_stds_consortium2_uniform_noisolatedgrowth.xlsx'

DATAFOLDER = 'data'  # Relative to project folder or absolute path
RESULTSFOLDER = 'results'
PARSFOLDER = 'prior_hyperparameters'
datafolder = os.path.join(os.getcwd(), DATAFOLDER)
parsfolder = os.path.join(os.getcwd(), PARSFOLDER)
resultsfolder = os.path.join(os.getcwd(), RESULTSFOLDER)
Path(resultsfolder).mkdir(parents=True, exist_ok=True)

consortia = ["Consortium1", "Consortium2"]
experiment_datafiles = [[] for _ in consortia]

for cons_ind, consortium in enumerate(consortia):
    for file in os.listdir(datafolder):
        if re.search(".*" + consortium + ".*.csv$", file) is not None:
            experiment_datafiles[cons_ind].append(os.path.join(datafolder, file))
    experiment_datafiles[cons_ind].sort()

    datafiles = experiment_datafiles[cons_ind]
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

    # The following print message can be used to get a quick view of what the effect of the prior distribution
    # is on the inferred initial counts
    # print("Target: {}".format(target_fracs))
    # print("Initial: {}".format(initial_counts/np.sum(initial_counts)))
    # print("True: {}".format(true_initial_est_i))
    # print('')

    # Then we should calculate the final fractions given these initial fractions, but this depends on the growth
    # parameters C_{ij} and f^i_{ij}. Therefore, we use a function that takes these as arguments, and we first set the
    # known parameters in these matrices. Note that f[i,j] = f^i_{ij} which indicates the eventual fraction of i in a
    # droplet with ij, and f[j,i] is thus 1 - f[i,j].

    # DHdG: When including isolated growth, you need to define a prior for the C_i (isolated growth of each strain)
    # parameters as well.
    if consortium == 'Consortium1':
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
        C_ij_means_df = pd.read_excel(os.path.join(parsfolder, C_ij_means_consortium1_file), header=None)
        C_ij_means = C_ij_means_df.to_numpy()

        C_ij_stds_df = pd.read_excel(os.path.join(parsfolder, C_ij_stds_consortium1_file), header=None)
        C_ij_stds = C_ij_stds_df.to_numpy()

        """The parameters f^i_{ij}"""
        # I assume that the prior distribution for the fraction parameters are Dirichlet distributions. Again, we can
        # just specify the mean and the std. However, note that not all variances/stds are possible given a certain
        # mean and given that we want the distribution to be peaked around the mean (that is concentrated instead of
        # sparse). Therefore, we will use the following formula to determine the variance, but will set it to the
        # uniform distribution if the inferred nu is negative.
        # Var = p(1-p)/(nu + 1), --> nu = p(1-p)/std**2 - 1
        f_ij_means_df = pd.read_excel(os.path.join(parsfolder, f_ij_means_consortium1_file), header=None)
        f_ij_means = f_ij_means_df.to_numpy()

        # The following values are completely made-up by me. These have to be decided on either by doing experiments
        # that provide means and stds for the pars, or we have to marginalize over these parameters.
        # Reminder: when there is no information on a parameter at all, you should set the std to -1.
        f_ij_stds_df = pd.read_excel(os.path.join(parsfolder, f_ij_stds_consortium1_file), header=None)
        f_ij_stds = f_ij_stds_df.to_numpy()

    elif consortium == 'Consortium2':
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
        C_ij_means_df = pd.read_excel(os.path.join(parsfolder, C_ij_means_consortium2_file), header=None)
        C_ij_means = C_ij_means_df.to_numpy()

        # The following values are completely made-up by me. These have to be decided on either by doing experiments
        # that provide means and stds for the pars, or we have to marginalize over these parameters.
        # Reminder: when there is no information on a parameter at all, you should set the std to -1.
        C_ij_stds_df = pd.read_excel(os.path.join(parsfolder, C_ij_stds_consortium2_file), header=None)
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

        f_ij_means_df = pd.read_excel(os.path.join(parsfolder, f_ij_means_consortium2_file), header=None)
        f_ij_means = f_ij_means_df.to_numpy()

        # The following values are completely made-up by me. These have to be decided on either by doing experiments
        # that provide means and stds for the pars, or we have to marginalize over these parameters.
        # Reminder: when there is no information on a parameter at all, you should set the std to -1.
        f_ij_stds_df = pd.read_excel(os.path.join(parsfolder, f_ij_stds_consortium2_file), header=None)
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

    print(consortium)

    observed_final_fracs = final_counts_ic / np.sum(final_counts_ic, axis=0)
    true_fracs_initial_guess, total_cells_initial_guess = get_final_fracs(true_initial_est_ic, C_ij_0, f_ij_0,
                                                                          poisson_lambda)
    optres_initial_wrapper = loglik_wrapper(params0, true_initial_est_ic, final_counts_ic, log_C_ij_means,
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


    optres = optimize.minimize(loglik_wrapper, params0, args=(true_initial_est_ic, final_counts_ic, log_C_ij_means,
                                                              C_ij_stds, f_ij_means, f_ij_precs, poisson_lambda, False),
                               bounds=bounds, options={'maxiter': 50000,
                                                       'maxfun': 1e10})  # method='L-BFGS-B',jac='2-point',options={'maxfun': 1e10,'maxcor':4}) #options={'maxiter': 50000, 'maxfun': 1e10}

    t3 = time.time()

    C_ij_final, f_ij_final = compile_parameter_mats_from_vect(pars_vect=optres.x, n_strains=C_ij_0.shape[0])
    # optres = optimize.basinhopping(loglik_wrapper, params0, minimizer_kwargs={'args':(true_initial_est_ic, final_counts_ic, log_C_ij_means,
    #                                                           C_ij_stds, f_ij_means, f_ij_precs, False), 'bounds':bounds})

    # DHdG: In the following function I treat the C_ij different than the f_ij at some points, and I distinguish them by
    # using that the n_combis - 1 first parameters are C_ij. This is no longer true when you include C_i
    lbs_uncertainty, ubs_uncertainty, report = get_uncertainty_region(opt_pars=optres.x,
                                                              true_initial_est_ic=true_initial_est_ic,
                                                              final_counts_ic=final_counts_ic,
                                                              log_C_ij_means=log_C_ij_means, C_ij_stds=C_ij_stds,
                                                              f_ij_means=f_ij_means, f_ij_precs=f_ij_precs,
                                                              poisson_lambda=poisson_lambda,
                                                              bounds=bounds, make_plots=True, savePath=RESULTSFOLDER,
                                                            return_if_error=attempts < MAX_ATTEMPTS)

    t4 = time.time()

    print(f'Time for Uncertainity Region Calc = {t4 - t3:.4f} s\n')

    # DHdG: This compile_parameter_mats_from_vect needs to be updated to account for the new parameters.

    C_ij_lb_final, f_ij_lb_final = compile_parameter_mats_from_vect(pars_vect=lbs_uncertainty,
                                                                    n_strains=C_ij_0.shape[0])
    C_ij_ub_final, f_ij_ub_final = compile_parameter_mats_from_vect(pars_vect=ubs_uncertainty,
                                                                    n_strains=C_ij_0.shape[0])

    np.set_printoptions(precision=2)
    np.set_printoptions(suppress=True)

    # print("loglikelihood:\n{}".format(loglikelihood))
    observed_initial_fracs = initial_counts_ic / np.sum(initial_counts_ic, axis=0)
    # print (f"observed initial fracs:\n {observed_initial_fracs}")
    # print ("corrected initial fracs :\n", true_initial_est_ic)
    observed_final_fracs = final_counts_ic / np.sum(final_counts_ic, axis=0)
    # print("observed final fracs:\n{}".format(observed_final_fracs))
    true_final_est_ic, total_cells = get_final_fracs(true_initial_est_ic, C_ij_final, f_ij_final, poisson_lambda)
    # print("estimated final fracs:\n{}".format(true_final_est_ic))
    # print("")

    print(consortium)
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
    if consortium == 'Consortium1':
        np.savetxt(os.path.join(resultsfolder, 'Consortium1_Observed_initial_fracs.csv'), observed_initial_fracs,
                   fmt='%f')
        np.savetxt(os.path.join(resultsfolder, 'Consortium1_Corrected_initial_fracs.csv'), true_initial_est_ic)
        np.savetxt(os.path.join(resultsfolder, 'Consortium1_Observed_final_fracs.csv'), observed_final_fracs)
        np.savetxt(os.path.join(resultsfolder, 'Consortium1_Estimated_final_fracs.csv'), true_final_est_ic)
        np.savetxt(os.path.join(resultsfolder, 'Consortium1_C_ij_final.csv'), C_ij_final)
        np.savetxt(os.path.join(resultsfolder, 'Consortium1_C_ij_ub_final.csv'), C_ij_ub_final)
        np.savetxt(os.path.join(resultsfolder, 'Consortium1_C_ij_lb_final.csv'), C_ij_lb_final)
        np.savetxt(os.path.join(resultsfolder, 'Consortium1_fij_final.csv'), f_ij_final)
        np.savetxt(os.path.join(resultsfolder, 'Consortium1_f_ij_ub_final.csv'), f_ij_ub_final)
        np.savetxt(os.path.join(resultsfolder, 'Consortium1_f_ij_lb_final.csv'), f_ij_lb_final)

    if consortium == 'Consortium2':
        np.savetxt(os.path.join(resultsfolder, 'Consortium2_Obsereved_initial_fracs.csv'), observed_initial_fracs, fmt='%f')
        np.savetxt(os.path.join(resultsfolder, 'Consortium2_Corrected_initial_fracs.csv'), true_initial_est_ic)
        np.savetxt(os.path.join(resultsfolder, 'Consortium2_Obsereved_final_fracs.csv'), observed_final_fracs)
        np.savetxt(os.path.join(resultsfolder, 'Consortium2_Estimated_final_fracs.csv'), true_final_est_ic)
        np.savetxt(os.path.join(resultsfolder, 'Consortium2_C_ij_final.csv'), C_ij_final)
        np.savetxt(os.path.join(resultsfolder, 'Consortium2_C_ij_ub_final.csv'), C_ij_ub_final)
        np.savetxt(os.path.join(resultsfolder, 'Consortium2_C_ij_lb_final.csv'), C_ij_lb_final)
        np.savetxt(os.path.join(resultsfolder, 'Consortium2_f_ij_final.csv'), f_ij_final)
        np.savetxt(os.path.join(resultsfolder, 'Consortium2_f_ij_ub_final.csv'), f_ij_ub_final)
        np.savetxt(os.path.join(resultsfolder, 'Consortium2_f_ij_lb_final.csv'), f_ij_lb_final)



   
