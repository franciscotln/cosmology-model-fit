# Source: https://github.com/dscolnic/Pantheon/blob/master/lcparam_full_long.txt
import os
import pandas as pd
import numpy as np

path_to_data = os.path.dirname(os.path.abspath(__file__)) + "/raw-data/"
data_frame = pd.read_csv(path_to_data + "mb.txt", sep=" ")
convariances_file = pd.read_csv(path_to_data + "mb_covariance_sys.txt")

n = data_frame["zcmb"].values.size
variances = convariances_file["cov_mu"].to_numpy(dtype=np.float64).reshape((n, n))

z_values = data_frame["zcmb"].to_numpy(dtype=np.float64)
z_hel_values = data_frame["zhel"].to_numpy(dtype=np.float64)
apparent_magnitude_vals = data_frame["mb"].to_numpy(dtype=np.float64)
sigma_magnitudes = data_frame["dmb"].to_numpy(dtype=np.float64)
full_covariance_matrix = variances + np.diag(sigma_magnitudes**2)

sort_indices = np.argsort(z_values)


def get_data():
    return (
        "Pantheon2018",
        z_values[sort_indices],
        z_hel_values[sort_indices],
        apparent_magnitude_vals[sort_indices],
        full_covariance_matrix[sort_indices, :][:, sort_indices],
    )
