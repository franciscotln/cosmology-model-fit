# Source: https://github.com/des-science/DES-SN5YR/blob/main/4_DISTANCES_COVMAT/DES-SN5YR_HD.csv
import os
import pandas as pd
import numpy as np

path_to_data = os.path.dirname(os.path.abspath(__file__)) + "/raw-data/"
data_frame = pd.read_csv(path_to_data + "distances.txt")
covariance_file = pd.read_csv(path_to_data + "covariance_stat_sys.txt")
selected_columns = data_frame[["zHD", "zHEL", "MU", "MUERR_FINAL"]]

n = selected_columns["zHD"].size
covariance_stat = covariance_file["cov_mu"].to_numpy().reshape((n, n))

# Add statistical uncertainties
covariance_matrix = covariance_stat + np.diag(
    selected_columns["MUERR_FINAL"].values ** 2
)

# Sort by redshift
z_values = selected_columns["zHD"].to_numpy()
z_hel_values = selected_columns["zHEL"].to_numpy()
sort_indices = np.argsort(z_values)

"""
effective_sample_size:
np.where(selected_columns['PROB_SNNV19'] < 0)[0].size + np.where(selected_columns['PROB_SNNV19'] > 0.1)[0].size
=> 1735
"""


def get_data():
    return (
        "DES-SN5YR",
        z_values[sort_indices],
        z_hel_values[sort_indices],
        selected_columns["MU"].to_numpy()[sort_indices],
        covariance_matrix[sort_indices, :][:, sort_indices],
    )
