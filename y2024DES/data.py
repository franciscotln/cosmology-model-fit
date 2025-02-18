# Source: https://github.com/des-science/DES-SN5YR/blob/main/4_DISTANCES_COVMAT/DES-SN5YR_HD.csv
import os
import pandas as pd
import numpy as np

path_to_data = os.path.dirname(os.path.abspath(__file__)) + '/raw-data/'
distances_file = pd.read_csv(path_to_data + 'distances.txt')
covariance_file = pd.read_csv(path_to_data + 'covariance_stat_sys.txt')
selected_columns = distances_file[['zHD', 'MU', 'MUERR_FINAL']]

n = int(np.sqrt(covariance_file.size))
variances = np.array(covariance_file['cov_mu'].values, dtype=np.float64, copy=False).reshape((n, n))

# Get the indices of supernovae that were kept
kept_indices = selected_columns.index.to_numpy()

# Filter covariance matrix to only include kept indices
filtered_cov_matrix = variances[np.ix_(kept_indices, kept_indices)]

# Add statistical uncertainties (only for selected data)
covariance_matrix = filtered_cov_matrix + np.diag(selected_columns['MUERR_FINAL'].values ** 2)

# Sort by redshift
z_values = selected_columns['zHD'].to_numpy()
sort_indices = np.argsort(z_values)

def get_data():
    return (
        'DES-SN5YR',
        z_values[sort_indices],
        selected_columns['MU'].to_numpy()[sort_indices],
        selected_columns['MUERR_FINAL'].to_numpy()[sort_indices],
        covariance_matrix[sort_indices, :][:, sort_indices]
    )
