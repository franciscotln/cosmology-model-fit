# Source: https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/Pantheon%2BSH0ES.dat
# https://arxiv.org/pdf/2202.04077
import os
import pandas as pd
import numpy as np

path_to_data = os.path.dirname(os.path.abspath(__file__)) + '/raw-data/'
data_frame = pd.read_csv(path_to_data + 'distances.txt', sep = ' ')
convariances_file = pd.read_csv(path_to_data + 'covariance_stat_sys.txt', sep = ' ')
selected_columns = data_frame[['zHD', 'zHEL', 'm_b_corr']]

legend = 'Pantheon+ (2022)'
z_values = selected_columns['zHD'].to_numpy(dtype=np.float64)
z_hel_values = selected_columns['zHEL'].to_numpy(dtype=np.float64)
apparent_magnitudes = selected_columns['m_b_corr'].to_numpy(dtype=np.float64)

n = z_values.size
covariance_matrix = convariances_file['cov_mu_shoes'].to_numpy(dtype=np.float64).reshape((n, n))

pantheon_plus_range = np.where(z_values > 0.01)[0]

def get_data():
    return (
        legend,
        z_values[pantheon_plus_range],
        z_hel_values[pantheon_plus_range],
        apparent_magnitudes[pantheon_plus_range],
        covariance_matrix[np.ix_(pantheon_plus_range, pantheon_plus_range)],
    )
