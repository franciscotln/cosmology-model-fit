# Source: https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/Pantheon%2BSH0ES.dat
import os
import pandas as pd
import numpy as np

path_to_data = os.path.dirname(os.path.abspath(__file__)) + '/raw-data/'
data_frame = pd.read_csv(path_to_data + 'distances.txt', sep = ' ')
convariances_file = pd.read_csv(path_to_data + 'covariance_stat_sys.txt', sep = ' ')
selected_columns = data_frame[['zHD', 'IS_CALIBRATOR', 'm_b_corr']]

legend = 'Pantheon+'
z_values = selected_columns['zHD'].to_numpy()
apparent_magnitudes = selected_columns['m_b_corr'].to_numpy()

n = z_values.size
covariance_matrix = convariances_file['cov_mu_shoes'].to_numpy().reshape((n, n))

pantheon_range = np.where(z_values > 0.01)[0]
sort_indices = np.argsort(z_values[pantheon_range])

def get_data():
    return (
        legend,
        z_values[sort_indices],
        apparent_magnitudes[sort_indices],
        covariance_matrix[np.ix_(sort_indices, sort_indices)],
    )
