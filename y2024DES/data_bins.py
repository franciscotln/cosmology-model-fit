import os
import pandas as pd
import numpy as np

path_to_data = os.path.dirname(os.path.abspath(__file__)) + '/raw-data/'
data_frame = pd.read_csv(path_to_data + 'distances.txt')
covariance_file = pd.read_csv(path_to_data + 'covariance_stat_sys.txt')

selected_columns = data_frame[['zHD', 'MU', 'MUERR_FINAL']]
n = selected_columns['zHD'].size
covariances_stat = covariance_file['cov_mu'].to_numpy().reshape((n, n))

# Add statistical uncertainties
covariance_matrix = covariances_stat + np.diag(selected_columns['MUERR_FINAL'].values ** 2)

# Sort by redshift
z_values = selected_columns['zHD'].to_numpy()
sort_indices = np.argsort(z_values)

z_sorted = z_values[sort_indices]
mu_sorted = selected_columns['MU'].to_numpy()[sort_indices]
mu_err_sorted = selected_columns['MUERR_FINAL'].to_numpy()[sort_indices]
full_covariance_sorted = covariance_matrix[sort_indices, :][:, sort_indices]

# Number of elements per bin
bin_size = 50
num_bins = int(np.ceil(n / bin_size))

# Lists to store binned results
z_bins = []
mu_bins = []
full_covariance_bins = np.zeros((num_bins, num_bins))  

# Bin the data
bin_indices = np.array_split(np.arange(n), num_bins)

weights_list = []
for i, indices in enumerate(bin_indices):
    z_bin = z_sorted[indices]
    mu_bin = mu_sorted[indices]
    cov_bin = full_covariance_sorted[np.ix_(indices, indices)]

    # Compute inverse variance weights
    inv_cov_bin = np.linalg.inv(cov_bin)  # Inverse of the full covariance matrix
    weights = np.sum(inv_cov_bin, axis=1)
    weights /= np.sum(weights)  # Normalize weights

    # Store results
    z_bins.append(np.mean(z_bin))
    mu_bins.append(np.sum(weights * mu_bin))
    weights_list.append(weights)

# Compute full binned covariance matrix
for i in range(num_bins):
    for j in range(num_bins):
        indices_i = bin_indices[i]
        indices_j = bin_indices[j]
        cov_ij = full_covariance_sorted[np.ix_(indices_i, indices_j)]
        full_covariance_bins[i, j] = weights_list[i].T @ cov_ij @ weights_list[j]


def get_data():
    return (
        f"DES-SN5YR - {bin_size} bins",
        np.array(z_bins),
        np.array(mu_bins),
        np.array(full_covariance_bins),
    )
