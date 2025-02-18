# Source: https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/Pantheon%2BSH0ES.dat
import os
import pandas as pd
import numpy as np

path_to_data = os.path.dirname(os.path.abspath(__file__)) + '/raw-data/'
data_frame = pd.read_csv(path_to_data + 'distances.txt', sep = ' ')
convariances_file = pd.read_csv(path_to_data + 'covariance_stat_sys.txt', sep = ' ')
selected_columns = data_frame[['zHD', 'MU_SH0ES', 'IS_CALIBRATOR']]
n = int(np.sqrt(convariances_file.size))
variances = np.array(convariances_file['cov_mu_shoes'].values, dtype=np.float64, copy=False).reshape((n, n))

legend = 'Pantheon+SHOES'
z_values = np.array(selected_columns['zHD'], dtype = np.float64, copy = False)
distance_moduli = np.array(selected_columns['MU_SH0ES'], dtype = np.float64, copy = False)
covariance_matrix = np.array(variances, dtype = np.float64, copy = False)

range_0 = [0, 0.1]
range_1 = [0.01, 0.1]
range_2 = [0.1, 0.25]
range_3 = [0.25, 0.42]
range_4 = [0.42, 2.3]
range_pantheon_full = [0.01, 2.3]
range_shoes_full = [0, 2.3]
bin_range = range_shoes_full
bin = np.where((z_values >= bin_range[0]) & (z_values < bin_range[1]))[0]

def get_data():
    return (
        legend,
        z_values[bin],
        distance_moduli[bin],
        covariance_matrix[bin, :][:, bin],
    )
