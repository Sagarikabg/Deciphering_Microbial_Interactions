from matplotlib import pyplot as plt
import os
import numpy as np


plt.rcParams.update({'font.size': 16})

# Directory where the files are stored
home_directory = os.getcwd()  # Replace with your directory path if not the current directory
results_folder = os.path.join(home_directory, 'results')

# Get all Consortium1 and Consortium2 files
files_cons1 = [f for f in os.listdir(results_folder) if f.startswith('Consortium1_') and f.endswith('.csv')]
files_cons2 = [f for f in os.listdir(results_folder) if f.startswith('Consortium2_') and f.endswith('.csv')]

# Load Consortium1 files into a dictionary
data_cons1 = {}
for file in files_cons1:
    # Generate a variable name by removing 'Consortium1_' and '.csv'
    var_name = file.replace('Consortium1_', '').replace('.csv', '').replace(',', '_')

    # Load the file as a numpy array
    data_cons1[var_name] = np.loadtxt(os.path.join(results_folder, file))

# Load Consortium2 files into a dictionary
data_cons2 = {}
for file in files_cons2:
    # Generate a variable name by removing 'Consortium2_' and '.csv'
    var_name = file.replace('Consortium2_', '').replace('.csv', '').replace(',', '_')

    # Load the file as a numpy array
    data_cons2[var_name] = np.loadtxt(os.path.join(results_folder, file))

# Access the arrays for Consortium1
observed_initial_fracs_cons1 = data_cons1['Observed_initial_fracs']
corrected_initial_fracs_cons1 = data_cons1['Corrected_initial_fracs']
observed_final_fracs_cons1 = data_cons1['Observed_final_fracs']
estimated_final_fracs_cons1 = data_cons1['Estimated_final_fracs']
C_ij_final_cons1 = data_cons1['C_ij_final']
f_ij_final_cons1 = data_cons1['f_ij_final']

# Access the arrays for Consortium2
observed_initial_fracs_cons2 = data_cons2['Observed_initial_fracs']
corrected_initial_fracs_cons2 = data_cons2['Corrected_initial_fracs']
observed_final_fracs_cons2 = data_cons2['Observed_final_fracs']
estimated_final_fracs_cons2 = data_cons2['Estimated_final_fracs']
C_ij_final_cons2 = data_cons2['C_ij_final']
f_ij_final_cons2 = data_cons2['f_ij_final']

# Print to confirm
print("Consortium1 Data:")
for name, array in data_cons1.items():
    print(f"{name}:")
    print(array)

print("\nConsortium2 Data:")
for name, array in data_cons2.items():
    print(f"{name}:")
    print(array)

conditions = ['Initial', 'Final']
strains = ['Strain A', 'Strain B', 'Strain C']
strains_1 = ['Strain A', 'Strain B', 'Strain C*']
colors = ['r', 'b', 'y']
colors_1 = ['r', 'b', 'g']

fig, axes = plt.subplots(3, 3, figsize=(10, 10))

for i in range(3):
    axes[0, i].plot(conditions, [observed_initial_fracs_cons2[i, 0:4], observed_final_fracs_cons2[i, 0:4]], marker='o',markersize=8,
                    color=colors[i], alpha=0.5)
    axes[1, i].plot(conditions, [observed_initial_fracs_cons2[i, 4:7], observed_final_fracs_cons2[i, 4:7]], marker='o', markersize=8,
                    color=colors[i], alpha=0.5)
    axes[2, i].plot(conditions, [observed_initial_fracs_cons2[i, 7:11], observed_final_fracs_cons2[i, 7:11]],
                    marker='o', markersize=8,color=colors[i], alpha=0.5)

for i in range(3):
    for j in range(3):
        axes[i, j].set_ylim(0, 1)
        axes[0, j].set_title(strains[j], fontsize=24)
        axes[i, 0].set_ylabel("Strain fraction", fontsize=24)
        axes[i, j].set_xticklabels(conditions, fontsize=24)
        axes[i, j].grid(True, alpha=0.6)
plt.tight_layout()
plt.savefig("NZ5500.png", dpi=300)
plt.show()

fig2, axes2 = plt.subplots(2, 3, figsize=(5, 5))

for i in range(3):
    axes2[0, i].plot(conditions, [observed_initial_fracs_cons1[i, 0:3], observed_final_fracs_cons1[i, 0:3]], marker='o',
                     color=colors_1[i], alpha=0.5)
    axes2[1, i].plot(conditions, [observed_initial_fracs_cons2[i, 4:7], observed_final_fracs_cons2[i, 4:7]], marker='o',
                     color=colors[i], alpha=0.5)

for i in range(2):
    for j in range(3):
        axes2[i, j].set_ylim(0, 1)
        axes2[0, j].set_title(strains_1[j], fontsize=12)
        axes2[1, j].set_title(strains[j], fontsize=12)
        axes2[i, 0].set_ylabel("Strain fraction", fontsize=12)
        axes2[i, j].set_xticklabels(conditions, fontsize=12)
        axes2[i, j].grid(True, alpha=0.6)
plt.tight_layout()
plt.savefig("MG1363.png", dpi=300)
plt.show()
