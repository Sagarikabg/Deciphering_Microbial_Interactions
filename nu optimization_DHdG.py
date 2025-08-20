import os, re
import pandas as pd
import numpy as np
from scipy.special import gammaln
from scipy import optimize
import matplotlib.pyplot as plt

# nu is the precision parameter. It's a weight given to the prior information (target_fracs_ic). 
# The larger the nu is, the more influence the prior (target fractions) has on the final estimate.

# os.chdir(
#     '/Users/sbangalorego1/surfdrive - Sagarika Bangalore Govindaraju@surfdrive.surf.nl/PhD/Git_folder/microbial-interaction-in-droplets')

DATAFOLDER = 'data_files_nu'
# DATAFOLDER = 'data'
datafolder = os.path.join(os.getcwd(), DATAFOLDER)

consortia = ["Consortium1", "Consortium2"]
experiment_datafiles = []

for consortium in consortia:
    for file in os.listdir(datafolder):
        if re.search(".*" + consortium + ".*.csv$", file) is not None:
            experiment_datafiles.append(os.path.join(datafolder, file))

# Sort the collected files
experiment_datafiles.sort()

combined_initial_counts_ic = []
combined_target_fracs_ic = []
combined_initial_fracs_ic = []
# Process each file in the combined list
for file in experiment_datafiles:
    data = pd.read_csv(file, header=0, index_col=0)

    # Extract strains, target fractions, and initial counts
    target_fracs = data.loc['Target'].values.astype(dtype=float)
    target_fracs /= np.sum(target_fracs)
    initial_counts = data.loc['Initial'].values
    initial_fracs = data.loc['Initial'].values / np.sum(initial_counts)

    combined_initial_counts_ic.append(initial_counts)
    combined_target_fracs_ic.append(target_fracs)
    combined_initial_fracs_ic.append(initial_fracs)

combined_initial_counts_ic = np.vstack(combined_initial_counts_ic).T
combined_initial_fracs_ic = np.vstack(combined_initial_fracs_ic).T
combined_target_fracs_ic = np.vstack(combined_target_fracs_ic).T

# np.set_printoptions(precision=2) #precision of numbers
# np.set_printoptions(suppress=True)
print("Target initial fractions:\n {}".format(combined_target_fracs_ic))
print("Observed initial counts:\n {}".format(combined_initial_counts_ic))
print("Observed initial fractions: \n {}".format(combined_initial_fracs_ic))

N_EXP = combined_initial_counts_ic.shape[1]


def logposterior_prec(nu, combined_initial_counts_ic, combined_target_fracs_ic):
    loglik = -np.log(nu)
    # loglik = 0
    loglik += N_EXP * gammaln(nu)
    loglik -= np.sum(gammaln(nu + np.sum(combined_initial_counts_ic, axis=0)))
    loglik += np.sum(
        gammaln(nu * combined_target_fracs_ic + combined_initial_counts_ic) - gammaln(nu * combined_target_fracs_ic))
    return loglik


def neg_logposterior_wrapper(lognu, combined_initial_counts_ic, combined_target_fracs_ic):
    nu = np.exp(lognu)
    loglik = logposterior_prec(nu, combined_initial_counts_ic, combined_target_fracs_ic)
    return -loglik


# Optimize nu using the calculated likelihood function
initial_guess = 3
optimum = optimize.minimize(neg_logposterior_wrapper, initial_guess,
                            args=(combined_initial_counts_ic, combined_target_fracs_ic))

optimal_nu = np.exp(optimum.x[0])

# Output the result
print("Optimization message:", optimum.message)
print("Optimal nu: {:.2f}".format(optimal_nu))


nu_test = np.exp(np.linspace(np.log(1e-3), np.log(10000), 100, endpoint=True))
likel_test = np.zeros(100)
for ind_nu, nu in enumerate(nu_test):
    likel_test[ind_nu] = logposterior_prec(nu, combined_initial_counts_ic, combined_target_fracs_ic)
# plot to see the trend
fig, ax = plt.subplots()
ax.plot(nu_test, likel_test, color='black')
ax.axvline(optimal_nu, linestyle='--', color='black', linewidth=2, label=r'Optimum at $\nu={:.2f}$'.format(optimal_nu))
ax.set_ylabel(r'Posterior for precision: $\operatorname{L}(\nu~|~ d_{i}^{e}(0), t_{i}^{e})$')
ax.set_xlabel(r'Experimental preparation precision: $\nu$')
ax.set_xscale('log')
ax.legend()
plt.savefig('nu_optimization.png', dpi=600, bbox_inches='tight')
plt.show()
