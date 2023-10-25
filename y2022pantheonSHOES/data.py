# Source: https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/Pantheon%2BSH0ES.dat
import os
import pandas as pd
import numpy as np

def get_data():
    path_to_data = os.path.dirname(os.path.abspath(__file__)) + '/raw-data/'
    data_frame = pd.read_csv(path_to_data + 'distances.txt', sep = ' ')
    convariances_file = pd.read_csv(path_to_data + 'covariance_stat_sys.txt', sep = ' ')
    selected_columns = data_frame[['zCMB', 'MU_SH0ES']].sort_values(by = 'zCMB')
    n = int(np.sqrt(convariances_file.size))
    variances = np.array(convariances_file['cov_mu_shoes'].values, dtype=np.float64, copy=False).reshape((n, n)).diagonal()
    std_values = np.sqrt(variances)

    legend = 'Pantheon+SHOES'
    z_values = np.array(selected_columns['zCMB'], dtype = np.float64, copy = False)
    distance_moduli = np.array(selected_columns['MU_SH0ES'], dtype = np.float64, copy = False)
    distance_moduli_std = np.array(std_values, dtype = np.float64, copy = False)

    return (legend, z_values, distance_moduli, distance_moduli_std)
