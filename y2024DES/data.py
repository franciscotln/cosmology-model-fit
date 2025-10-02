# Source: https://github.com/des-science/DES-SN5YR/blob/main/4_DISTANCES_COVMAT/DES-SN5YR_HD.csv
# https://arxiv.org/pdf/2401.02929
import os
import pandas as pd
import numpy as np

path_to_data = os.path.dirname(os.path.abspath(__file__)) + "/raw-data/"
data_frame = pd.read_csv(path_to_data + "distances.txt")
covariance_file = pd.read_csv(path_to_data + "covariance_stat_sys.txt.zip")
selected_columns = data_frame[
    ["zHD", "zHEL", "MU", "MUERR_FINAL", "PROB_SNNV19", "PROBCC_BEAMS", "IDSURVEY"]
]

n = selected_columns["zHD"].size
covariance_stat = covariance_file["cov_mu"].to_numpy(dtype=np.float64).reshape((n, n))

# Add statistical uncertainties
covariance_matrix = covariance_stat + np.diag(
    selected_columns["MUERR_FINAL"].values ** 2
)

# Sort by redshift
z_values = selected_columns["zHD"].to_numpy(dtype=np.float64)
z_hel_values = selected_columns["zHEL"].to_numpy(dtype=np.float64)
sort_indices = np.argsort(z_values)

"""
effective_sample_size:
sum (1 - PROBCC_BEAMS) + low_z size => 1735
"""

effective_sample_size = np.round((1 - selected_columns["PROBCC_BEAMS"]).sum()).astype(
    int
)


def get_data():
    return (
        f"DES-SN5YR - effective: {effective_sample_size} SNe",
        z_values[sort_indices],
        z_hel_values[sort_indices],
        selected_columns["MU"].to_numpy(dtype=np.float64)[sort_indices],
        covariance_matrix[sort_indices, :][:, sort_indices],
    )
