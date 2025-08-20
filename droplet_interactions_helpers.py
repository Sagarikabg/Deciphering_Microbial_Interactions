import numpy as np
from scipy.special import gammaln
from scipy import optimize
from scipy.stats import multinomial, poisson
import matplotlib.pyplot as plt
import os


def get_dirichlet_precisions(means, stds):
    precs = (means * (1 - means)) / (stds ** 2) - 1
    precs[precs < 0] = -1
    precs[np.isnan(precs)] = 0.0
    return precs


def extract_params_from_matrices(C_ij, f_ij):
    upper_tri_indices = np.triu_indices_from(C_ij, k=0)  # includes the diagonal
    C_ij_vals_all = np.log(C_ij[upper_tri_indices])
    C_ij_vals = np.concatenate((C_ij_vals_all[:1], C_ij_vals_all[2:]))  # C_ij_vals[:1]: This selects the first element of the array, as :1 means from the start (index 0) up to, but not including, index 1.
    #C_ij_vals[2:]: This selects all elements starting from index 2 to the end, skipping the element at index 1.
    upper_tri_indices = np.triu_indices_from(C_ij, k=1)  # does not include the diagonal
    f_ij_vals = f_ij[upper_tri_indices]
    return np.concatenate((C_ij_vals, f_ij_vals))


def compile_parameter_mats_from_vect(pars_vect, n_strains):
    C_ij = np.zeros((n_strains, n_strains))
    f_ij = np.eye(n_strains)
    n_combis = int((n_strains * (n_strains - 1) / 2) + n_strains)
    C_pars = np.exp(pars_vect[: n_combis-1])
    f_pars = pars_vect[n_combis - 1:]
    triu_inds = np.triu_indices_from(C_ij, k=0)
    C_ij[0, 1] = 1
    all_indices = np.arange(0, n_combis) # returns [0,1,2,3,4,5...55]
    selected_indices = all_indices[all_indices != 1] # exlcude 1 [0,2,3,4,5...55]
    C_ij[triu_inds[0][[selected_indices]], triu_inds[1][[selected_indices]]] = C_pars # exlcudes the second element which is C_ij[0,1]
    #[0,2,3,4,5....55] [0,2,3,4,5....55]
    f_triu_inds = np.triu_indices_from(C_ij, k=1)
    f_ij[f_triu_inds] = f_pars
    C_ij_exclude_diag = C_ij.copy()
    np.fill_diagonal(C_ij_exclude_diag,0)
    f_ij_exclude_diag = f_ij.copy()
    np.fill_diagonal(f_ij_exclude_diag,0)
    C_ij = C_ij + C_ij_exclude_diag.T
    f_ij = f_ij + f_ij_exclude_diag.T
    tril_inds = np.tril_indices_from(f_ij, k=-1)
    f_ij[tril_inds] = 1 - f_ij[tril_inds]
    return C_ij, f_ij


def get_final_fracs(initial_fracs_ic, C_ij, f_ij, poisson_lambda):
    # poisson_lambda=0.3

    # Loop over different experiments
    n_exps = initial_fracs_ic.shape[1] #retrives number of columns , in this case experiments
    final_fracs_ic = np.zeros_like(initial_fracs_ic)
    total_cells_i = np.zeros(n_exps)

    for exp_ind in range(n_exps):

        # First get all pairwise products of the initial fracs:
        initial_fracs = initial_fracs_ic[:, exp_ind]
        p_times_p_ij = np.matmul(initial_fracs[:, None], initial_fracs[None, :])
        p_times_p_ij *= 2 * poisson.pmf(2, poisson_lambda) # DHdG: times probability of 2-cell droplet

        # Add term for cell being on its own
        # DHdG: p_i = initial_fracs * probability of 1-cell droplet
        p_i = initial_fracs* poisson.pmf(1,poisson_lambda)
        #SbG: but there will also be droplets with 2 cells of only one cell-type
        p_2i = poisson.pmf(2, poisson_lambda)*initial_fracs**2


        # In the denominator of the fractions, we will sum for all the combinations i,j the product of  C * p * p:
        # First we only select the upper triangle of the matrix, to not count (i, j) and (j, i)
        upper_tri_indices = np.triu_indices_from(p_times_p_ij, k=1)
        total_cells_i[exp_ind] = np.sum(C_ij[upper_tri_indices] * p_times_p_ij[upper_tri_indices])

        # DHdG: Add cells created by isolated growth total_cells_i[exp_ind] += np.sum(C_i * p_i)
        total_cells_i[exp_ind] += np.sum(np.diag(C_ij)*p_i)
        total_cells_i[exp_ind] += np.sum(np.diag(C_ij)*p_2i)

        # The numerator of the fraction of strain i will take only the i-th row of the multiplication f_ij, C_ij, p*p_ij
        # Create a mask to ignore diagonal elements
        diagonal_mask = np.eye(f_ij.shape[0], f_ij.shape[1])  # Identity matrix of the same shape
        non_diagonal_mask = 1 - diagonal_mask  # Set diagonal to 0 and off-diagonal to 1
        # Apply the mask to the computation
        cells_i = np.sum(f_ij * C_ij * p_times_p_ij * non_diagonal_mask, axis=1)


        # DHdG: Add cells created by isolated growth: cells_i += C_i * p_i
        cells_i += (np.diag(C_ij)*p_i)
        cells_i += (np.diag(C_ij)*p_2i)

        # Finally, return the fractions
        final_fracs_ic[:, exp_ind] = cells_i / total_cells_i[exp_ind]

    return final_fracs_ic, total_cells_i


def calc_loglikelihood(true_fracs_ic, final_counts_ic, C_ijs, f_ijs, log_C_ij_means, C_ij_stds, f_ij_means, f_ij_precs):
    # First get the likelihood part of the loglikelihood (later get the contribution of the prior)
    summed_counts = np.sum(final_counts_ic, axis=0)
    # See the description of the loglikelihood in the .tex-file (it's just the log of the multinomial). Taking
    # the log of the factorials is way more efficient using the gammaln function, which is why I use it
    loglik = np.sum(
        gammaln(summed_counts + 1) + np.sum(- gammaln(final_counts_ic + 1) + final_counts_ic * np.log(true_fracs_ic),
                                            axis=0))
    # loglik = 0
    # loglik = np.sum(np.log(true_fracs_ic[0, :]))

    # The following should be the same:
    # from scipy import stats
    # rv=stats.multinomial(n=summed_counts, p=true_fracs)
    # print(np.log(rv.pmf(final_counts)), loglik)

    # Loop over all parameters and calculate the prior contribution
    for ind1 in range(C_ijs.shape[0]):
        for ind2 in range(ind1,C_ijs.shape[0]):
            # First for C_ij where the prior is a lognormal
            C_ij = C_ijs[ind1, ind2]
            log_C_ij_mean = log_C_ij_means[ind1, ind2]
            log_C_ij = np.log(C_ij)
            C_ij_std = C_ij_stds[ind1, ind2]
            if C_ij_stds[ind1, ind2] < 0:
                #  In this case, the prior is just uniform in logscale
                loglik -= log_C_ij
            else:
                loglik += - log_C_ij - ((log_C_ij - log_C_ij_mean) ** 2) / (2 * C_ij_std ** 2)

            # TODO: Remove this later
            # import matplotlib.pyplot as plt
            # xrange = np.linspace(-2, 4, 100)
            # liks = np.zeros(len(xrange))
            # for ind in range(len(xrange)):
            #     log_C_ij = xrange[ind]
            #     # log_C_ij = np.log(x)
            #     liks[ind] = np.exp(-log_C_ij - ((log_C_ij - log_C_ij_mean) ** 2) / (2 * C_ij_std ** 2))
            #
            # fig, ax = plt.subplots()
            # ax.plot(np.exp(xrange), liks)
            # ax.plot(np.array([np.exp(log_C_ij_mean - C_ij_std ** 2), np.exp(log_C_ij_mean - C_ij_std ** 2)]), np.array([-100, 100]))
            # ax.set_xscale('log')
            # ax.set_yscale('log')
            # plt.show()

            # Then for f_ij where the prior is a Dirichlet
            f_ij_mean = f_ij_means[ind1, ind2]
            f_ij_prec = f_ij_precs[ind1, ind2]
            f_ij = f_ijs[ind1, ind2]
            if f_ij_prec < 0:
                # In this case, the prior is just uniform over the unit interval
                continue
            else:
                loglik += (f_ij_prec * f_ij_mean - 1) * np.log(f_ij) + (f_ij_prec * (1 - f_ij_mean) - 1) * np.log(
                    1 - f_ij)

                # TODO: Remove this later
                # from scipy import stats
                # rv = stats.dirichlet(alpha=np.array([f_ij_prec * f_ij_mean, f_ij_prec*(1- f_ij_mean)]))
                # print(rv.pdf(np.array([f_ij, 1- f_ij])))
                #
                # import matplotlib.pyplot as plt
                # print("mean {}, prec {}".format(f_ij_mean, f_ij_prec))
                # f_ij_prec = 1
                # xrange = np.linspace(0, 1, 100)
                # liks = np.zeros(len(xrange))
                # for ind in range(len(xrange)):
                #     f_ij = xrange[ind]
                #     # log_C_ij = np.log(x)
                #     liks[ind] = (f_ij_prec * f_ij_mean - 1) * np.log(f_ij) + (f_ij_prec * (1 - f_ij_mean) - 1) * np.log(1 - f_ij)
                #
                # fig, ax = plt.subplots()
                # ax.plot(xrange, liks)
                # # ax.plot(np.array([np.exp(log_C_ij_mean - C_ij_std ** 2), np.exp(log_C_ij_mean - C_ij_std ** 2)]), np.array([-100, 100]))
                # plt.show()
    return loglik


def calc_dloglik_dCij(true_fracs_ic, true_initial_est_ic, total_cells_i, final_counts_ic, C_ijs, f_ijs, log_C_ij_means,
                      C_ij_stds):
    # First get the likelihood part of the loglikelihood (later get the contribution of the prior)
    # See the derivation of the derivative in the .tex-file.
    n_strains = C_ijs.shape[0]
    dloglik_dCij_all = []
    for ind1 in range(n_strains):
        for ind2 in range(ind1, n_strains):
            dloglik_dCij = np.sum((true_initial_est_ic[ind1, :] * true_initial_est_ic[ind2, :] / total_cells_i) * (
                    final_counts_ic[ind1, :] * f_ijs[ind1, ind2] / true_fracs_ic[ind1, :] + final_counts_ic[ind2, :] *
                    f_ijs[ind2, ind1] /
                    true_fracs_ic[ind2, :] - np.sum(final_counts_ic, axis=0)))

            # Contribution of the prior
            # Loop over all parameters and calculate the prior contribution
            C_ij = C_ijs[ind1, ind2]
            log_C_ij_mean = log_C_ij_means[ind1, ind2]
            log_C_ij = np.log(C_ij)
            C_ij_std = C_ij_stds[ind1, ind2]
            if C_ij_stds[0, 1] < 0:
                #  In this case, the prior is just uniform in logscale
                dloglik_dCij -= 1 / C_ij
            else:
                dloglik_dCij -= (1 / C_ij) * (1 + (log_C_ij - log_C_ij_mean) / (C_ij_std ** 2))

            dloglik_dCij_all.append(dloglik_dCij)
    return np.array(dloglik_dCij_all)


def calc_dloglik_dfij(true_fracs_ic, true_initial_est_ic, total_cells_i, final_counts_ic, C_ijs, f_ijs, log_C_ij_means,
                      C_ij_stds, f_ij_means, f_ij_precs):
    # First get the likelihood part of the loglikelihood (later get the contribution of the prior)
    # See the derivation of the derivative in the .tex-file.
    n_strains = C_ijs.shape[0]
    dloglik_dfij_all = []
    for ind1 in range(n_strains):
        for ind2 in range(ind1, n_strains):
            C_ij = C_ijs[ind1, ind2]
            dloglik_dfij = np.sum(
                (C_ij * true_initial_est_ic[ind1, :] * true_initial_est_ic[ind2, :] / total_cells_i) * (
                        final_counts_ic[ind1, :] / true_fracs_ic[ind1, :] - final_counts_ic[ind2,
                                                                            :] / true_fracs_ic[ind2, :]))

            # Contribution of the prior
            # Loop over all parameters and calculate the prior contribution
            # Then for f_ij where the prior is a Dirichlet
            f_ij_mean = f_ij_means[ind1, ind2]
            f_ij_prec = f_ij_precs[ind1, ind2]
            f_ij = f_ijs[ind1, ind2]
            if f_ij_prec >= 0:
                # Otherwise, the prior is just uniform over the unit interval
                dloglik_dfij += (f_ij_prec * f_ij_mean - 1) * (1 / f_ij) - (f_ij_prec * (1 - f_ij_mean) - 1) * (1 / (
                        1 - f_ij))

            dloglik_dfij_all.append(dloglik_dfij)
    return np.array(dloglik_dfij_all)


def loglik_wrapper(pars, true_initial_est_ic, final_counts_ic, log_C_ij_means, C_ij_stds, f_ij_means, f_ij_precs, poisson_lambda,
                   verbose):
    n_strains = true_initial_est_ic.shape[0]
    C_ij, f_ij = compile_parameter_mats_from_vect(pars_vect=pars, n_strains=n_strains)
    true_final_est_ic, total_cells_i = get_final_fracs(true_initial_est_ic, C_ij, f_ij, poisson_lambda)
    loglikelihood = calc_loglikelihood(true_fracs_ic=true_final_est_ic, final_counts_ic=final_counts_ic, C_ijs=C_ij,
                                       f_ijs=f_ij, log_C_ij_means=log_C_ij_means, C_ij_stds=C_ij_stds,
                                       f_ij_means=f_ij_means, f_ij_precs=f_ij_precs)
    # delta_C_ij = C_ij.copy()
    # delta_f_ij = f_ij.copy()
    # # delta_C_ij[1, 2] += 1e-6
    # # delta_C_ij[2, 1] += 1e-6
    # delta_f_ij[1, 2] += 1e-6
    # delta_f_ij[2, 1] -= 1e-6
    # true_final_est_ic_delta, total_cells_i_delta = get_final_fracs(true_initial_est_ic, delta_C_ij, delta_f_ij)
    #
    # # approx_d_pij_dC12 = (np.log(true_final_est_ic_delta) - np.log(true_final_est_ic)) / 1e-6
    #
    # delta_loglikelihood = calc_loglikelihood(true_fracs_ic=true_final_est_ic_delta, final_counts_ic=final_counts_ic,
    #                                          C_ijs=delta_C_ij,
    #                                          f_ijs=delta_f_ij, log_C_ij_means=log_C_ij_means, C_ij_stds=C_ij_stds,
    #                                          f_ij_means=f_ij_means, f_ij_precs=f_ij_precs)
    # approx_dloglik = (delta_loglikelihood - loglikelihood) / 1e-6
    # dloglik_dCij = calc_dloglik_dCij(true_fracs_ic=true_final_est_ic, true_initial_est_ic=true_initial_est_ic,
    #                                  total_cells_i=total_cells_i,
    #                                  final_counts_ic=final_counts_ic, C_ijs=C_ij,
    #                                  f_ijs=f_ij, log_C_ij_means=log_C_ij_means, C_ij_stds=C_ij_stds)
    # dloglik_dfij = calc_dloglik_dfij(true_fracs_ic=true_final_est_ic, true_initial_est_ic=true_initial_est_ic,
    #                                  total_cells_i=total_cells_i,
    #                                  final_counts_ic=final_counts_ic, C_ijs=C_ij,
    #                                  f_ijs=f_ij, log_C_ij_means=log_C_ij_means, C_ij_stds=C_ij_stds,
    #                                  f_ij_means=f_ij_means, f_ij_precs=f_ij_precs)
    # print(np.vstack((dloglik_dCij, dloglik_dfij)))
    if verbose:
        print("loglikelihood:\n{}".format(loglikelihood))
        print("observed fracs:\n{}".format(final_counts_ic / np.sum(final_counts_ic, axis=0)))
        print("estimated fracs:\n{}".format(true_final_est_ic))
        print("")
    return -loglikelihood


def loglik_wrapper_onefixed(other_pars, fixed_par_val, fixed_par_ind, true_initial_est_ic, final_counts_ic,
                            log_C_ij_means, C_ij_stds, f_ij_means, f_ij_precs, poisson_lambda,verbose):
    all_pars = np.insert(other_pars, fixed_par_ind, fixed_par_val)
    return loglik_wrapper(all_pars, true_initial_est_ic, final_counts_ic, log_C_ij_means, C_ij_stds, f_ij_means,
                          f_ij_precs, poisson_lambda, verbose)


def get_d_optim_loglik_onefixed(fixed_par_val, fixed_par_ind, orig_loglik, pars_0, true_initial_est_ic, final_counts_ic,
                                log_C_ij_means, C_ij_stds, f_ij_means, f_ij_precs, poisson_lambda, verbose, bounds, return_pars=False):
    other_pars_0 = np.delete(pars_0, fixed_par_ind)
    other_bounds = [bound_set for ind_bound, bound_set in enumerate(bounds) if ind_bound != fixed_par_ind]
    optres = optimize.minimize(loglik_wrapper_onefixed, other_pars_0, args=(
        fixed_par_val, fixed_par_ind, true_initial_est_ic, final_counts_ic, log_C_ij_means,
        C_ij_stds, f_ij_means, f_ij_precs, poisson_lambda, verbose), bounds=other_bounds, options={'maxiter': 50000,
                                                      'maxfun': 1e10})
    if not optres.success:
        print("Can't find optimum when fixing one parameter, and searching others from optimum. Trying others.")
        init_pars = extract_params_from_matrices(np.exp(log_C_ij_means), f_ij_means)
        other_pars_0 = np.delete(init_pars, fixed_par_ind)
        optres = optimize.minimize(loglik_wrapper_onefixed, other_pars_0, args=(
            fixed_par_val, fixed_par_ind, true_initial_est_ic, final_counts_ic, log_C_ij_means,
            C_ij_stds, f_ij_means, f_ij_precs, poisson_lambda, verbose), bounds=other_bounds, options={'maxiter': 50000,
                                                                                                       'maxfun': 1e10})
    new_loglik = -optres.fun  # returns optimized value
    if not return_pars:
        if optres.success:
            return new_loglik - orig_loglik
        else:
            return np.nan
    else:
        if optres.success:
            return new_loglik - orig_loglik, np.insert(optres.x, fixed_par_ind, fixed_par_val)  # optres.x returns the parameters for the optimized value
        else:
            return np.nan, np.nan


def get_uncertainty_region(opt_pars=None, true_initial_est_ic=None, final_counts_ic=None, log_C_ij_means=None,
                           C_ij_stds=None, f_ij_means=None, f_ij_precs=None, poisson_lambda=None, bounds=None,
                           make_plots=False, savePath=None, return_if_error=False):
    lb_uncertainty = []
    ub_uncertainty = []
    opt_loglik = -loglik_wrapper(opt_pars, true_initial_est_ic, final_counts_ic, log_C_ij_means, C_ij_stds, f_ij_means,
                                 f_ij_precs, poisson_lambda, verbose=False)
    report = {"success": True, 'new_param': (None, None)}
    if make_plots:
        n_pars = len(opt_pars)
        if n_pars == 5:
            labels = [r'$C_{02}$', r'$C_{12}$', r'$f_{01}$', r'$f_{02}$', r'$f_{12}$']
        elif n_pars == 8:
            labels = [r'$C_{00}$', r'$C_{02}$', r'$C_{11}$', r'$C_{12}$', r'$C_{22}$', r'$f_{01}$', r'$f_{02}$', r'$f_{12}$']
        else:
            labels = ['Par {}'.format(ind) for ind in range(n_pars)]
        n_cols = int(np.ceil(np.sqrt(n_pars)))
        n_rows = int(np.ceil(n_pars / n_cols))
        fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(20,8))
        axs = axs.flatten()
        fig2, axs2 = plt.subplots(ncols=n_pars, nrows=2, figsize=(20,8))
    loglik_bound = opt_loglik - 0.5
    for ind_par, opt_par in enumerate(opt_pars): #looping through all the parameters (Cij, fij)
        pars = opt_pars.copy()
        par_bounds = bounds[ind_par]

        # We first do the lower bound of the uncertainty region
        # First check whether loglik is lower than bound at bounds for the parameter
        lb_par = par_bounds[0] #the parameter is set at its lower bound
        pars[ind_par] = lb_par
        diff_lb_loglik = get_d_optim_loglik_onefixed(lb_par, ind_par, opt_loglik, pars, true_initial_est_ic,
                                                     final_counts_ic, log_C_ij_means, C_ij_stds, f_ij_means, f_ij_precs, poisson_lambda,
                                                     False, bounds) #returns the difference of the loglikelihood calculated by fixing the parameter at its lower bound value and the optimal loglikelihood value
        if np.isnan(diff_lb_loglik):
            diff_lb_loglik = -1.0
        if return_if_error and (diff_lb_loglik > 0.1):
            # Checks whether lower bound actually gives a larger loglikelihood, in which case, we do the optimization
            # again, with that lower bound as initial value
            report["success"] = False
            report['new_param'] = (ind_par, lb_par)
            return None, None, report

        if diff_lb_loglik > -0.5: #checks if the difference between the loglikelihood for the lower bound value and the optimal loglikehood is low, if the difference is low, then the lower bound is equal to the lower bound
            # In this case, even going to the lower bound for the parameter does not decrease the loglikelihood by .5.
            # Therefore, we set the lower bound for the parameter as the lower bound of the uncertainty region
            lb_uncertainty.append(lb_par)
        else:
            # In this case, the loglikelihood must be lower by .5 somewhere between lower bound and optimum. We will use
            # bisection to get it.
            root_res = optimize.root_scalar(get_d_optim_loglik_onefixed,
                                            args=(ind_par, loglik_bound, pars, true_initial_est_ic, final_counts_ic,
                                                  log_C_ij_means, C_ij_stds, f_ij_means, f_ij_precs, poisson_lambda, False, bounds),
                                            bracket=[lb_par, opt_par], xtol=1e-2, rtol=1e-2) #loglik_bound = opt_loglik- 0.5, calculates the roots (i.e parameters) for which the diff_lb_loglikelihood is 0.5.
            #bracket --> sets the limit for the rootscalar optimizer. loglik_bound sets the value for which we need to find the roots.
            if root_res.converged:
                lb_uncertainty.append(root_res.root)  # sets the lower bound to the corresponding root for which the diff_lb_loglikelihood is 0.5.
            else:
                print("Something went wrong with finding lower bound of uncertainty region for parameter {}.\n"
                      "Setting lower bound to lower bound of the parameter.")
                lb_uncertainty.append(lb_par)

        # We then do the upper bound of the uncertainty region
        # First check whether loglik is lower than bound at bounds for the parameter
        ub_par = par_bounds[1]
        pars[ind_par] = ub_par
        diff_ub_loglik = get_d_optim_loglik_onefixed(ub_par, ind_par, opt_loglik, pars, true_initial_est_ic,
                                                     final_counts_ic, log_C_ij_means, C_ij_stds, f_ij_means, f_ij_precs,poisson_lambda,
                                                     False, bounds)
        if np.isnan(diff_ub_loglik):
            diff_ub_loglik = -1.0

        if return_if_error and (diff_ub_loglik > 0.1):
            # Checks whether lower bound actually gives a larger loglikelihood, in which case, we do the optimization
            # again, with that lower bound as initial value
            report["success"] = False
            report['new_params'] = (ind_par, ub_par)
            return None, None, report

        if diff_ub_loglik > -0.5:
            # In this case, even going to the lower bound for the parameter does not decrease the loglikelihood by .5.
            # Therefore, we set the lower bound for the parameter as the lower bound of the uncertainty region
            ub_uncertainty.append(ub_par)
        else:
            # In this case, the loglikelihood must be lower by .5 somewhere between lower bound and optimum. We will use
            # bisection to get it.
            root_res = optimize.root_scalar(get_d_optim_loglik_onefixed,
                                            args=(ind_par, loglik_bound, pars, true_initial_est_ic, final_counts_ic,
                                                  log_C_ij_means, C_ij_stds, f_ij_means, f_ij_precs, poisson_lambda,False, bounds),
                                            bracket=[opt_par, ub_par], xtol=1e-2, rtol=1e-2)
            if root_res.converged:
                ub_uncertainty.append(root_res.root)
            else:
                print("Something went wrong with finding lower bound of uncertainty region for parameter {}.\n"
                      "Setting lower bound to lower bound of the parameter.".format(ind_par))
                ub_uncertainty.append(lb_par)

        print("Parameter {}: optimal val = {}, uncertainty region = [{}, {}]".format(ind_par, opt_par,
                                                                                     lb_uncertainty[-1],
                                                                                     ub_uncertainty[-1]))

        if make_plots:
            n_points_per_side = 10
            lb_plot = max(opt_par - 3 * (opt_par - lb_uncertainty[-1]), lb_par)
            ub_plot = min(opt_par + 3 * (ub_uncertainty[-1] - opt_par), ub_par)
            if lb_plot == ub_plot:
                par_vals_plot = np.array([lb_plot, ub_plot])
            elif (lb_plot == opt_par) or (ub_plot == opt_par):
                par_vals_plot = np.linspace(lb_plot, ub_plot, 2 * n_points_per_side)
            else:
                par_vals_plot = np.concatenate(
                    (np.linspace(lb_plot, opt_par, n_points_per_side), np.linspace(opt_par, ub_plot, n_points_per_side+1)[1:]))
            n_points = len(par_vals_plot)
            lik_vals_plot = np.zeros(n_points)
            opt_par_vals_plot = np.zeros((n_pars, n_points))
            for ind_plot in range(n_points):
                lik_vals, opt_par_vals = get_d_optim_loglik_onefixed(
                    par_vals_plot[ind_plot], ind_par, 0, opt_pars,
                    true_initial_est_ic, final_counts_ic,
                    log_C_ij_means, C_ij_stds, f_ij_means, f_ij_precs, poisson_lambda,
                    False, bounds, return_pars=True)
                if np.isnan(lik_vals):
                    lik_vals_plot[ind_plot] = np.nan
                    opt_par_vals_plot[:, ind_plot] = np.nan
                else:
                    lik_vals_plot[ind_plot] = lik_vals
                    opt_par_vals_plot[:, ind_plot] = opt_par_vals

            n_combis = (f_ij_means.shape[0] * (f_ij_means.shape[0] - 1) / 2)+ f_ij_means.shape[0]
            if ind_par < (n_combis - 1):
                x_vals = np.exp(par_vals_plot)
            else:
                x_vals = par_vals_plot
            for ax in [axs[ind_par], axs2[0, ind_par]]:
                ax.plot(x_vals, np.exp(lik_vals_plot), '-*', linewidth=2)
                if ind_par == 0:
                    ax.set_xlabel("Parameter value")
                    ax.set_ylabel("Likelihood")
                ax.set_title(labels[ind_par])
                ax.set_yscale('log')
                ymin, ymax = ax.get_ylim()
                ymin = np.minimum(np.exp(opt_loglik - 0.5), ymin)
                ax.set_ylim(ymin, ymax)
                if ind_par < (n_combis - 1):
                    ax.axvline(x=np.exp(lb_uncertainty[-1]), linestyle='--', linewidth=1, label='LB uncertainty')
                    ax.axvline(x=np.exp(ub_uncertainty[-1]), linestyle='--', linewidth=1, label='UB uncertainty')
                    ax.axvline(x=np.exp(opt_par), linestyle='--', c='red', linewidth=1.5, label='optimal value')
                else:
                    ax.axvline(x=lb_uncertainty[-1], linestyle='--', linewidth=1, label='LB uncertainty')
                    ax.axvline(x=ub_uncertainty[-1], linestyle='--', linewidth=1, label='UB uncertainty')
                    ax.axvline(x=opt_par, linestyle='--', c='red', linewidth=1.5, label='optimal value')
                if ind_par == 0:
                    ax.legend()
            ax = axs2[1, ind_par]
            for ind_curr in range(n_pars):
                opt_pars_curr = opt_par_vals_plot[ind_curr, :]
                # Exponentiate C_ij-values
                if ind_curr < (n_combis - 1):
                    opt_pars_curr = np.exp(opt_pars_curr)
                ax.plot(x_vals, opt_pars_curr, label=labels[ind_curr])
            if ind_par == 0:
                ax.legend()
            ax.set_ylabel('Parameter values')

    if make_plots and (savePath is not None):
        fig.savefig(os.path.join(savePath, 'Uncetainity_region_1.png'), dpi=300)
        fig2.savefig(os.path.join(savePath, 'Uncetainity_region_2.png'), dpi=300)
        plt.show()
        plt.close(fig)
        plt.close(fig2)
    elif make_plots:
        plt.show()
    return np.array(lb_uncertainty), np.array(ub_uncertainty), report
