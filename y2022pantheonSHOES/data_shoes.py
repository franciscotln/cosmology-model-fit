# Source: https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/Pantheon%2BSH0ES.dat
import os
import pandas as pd
import numpy as np

path_to_data = os.path.dirname(os.path.abspath(__file__)) + "/raw-data/"
data_frame = pd.read_csv(path_to_data + "distances.txt", sep=" ")
convariances_file = pd.read_csv(path_to_data + "covariance_stat_sys.txt", sep=" ")
selected_columns = data_frame[["zHD", "zHEL", "m_b_corr", "CEPH_DIST", "IS_CALIBRATOR"]]

legend = "Pantheon+ and SH0ES"
z_values = selected_columns["zHD"].to_numpy(dtype=np.float64)
z_hel_values = selected_columns["zHEL"].to_numpy(dtype=np.float64)
apparent_mag = selected_columns["m_b_corr"].to_numpy(dtype=np.float64)
cepheid_distances = selected_columns["CEPH_DIST"].to_numpy(dtype=np.float64)

n = z_values.size
covariance_matrix = convariances_file["cov_mu_shoes"].to_numpy().reshape((n, n))

pantheon_SH0ES_range = np.where(
    (selected_columns["IS_CALIBRATOR"] == 1) | (z_values > 0.01)
)[0]


def get_data():
    return (
        legend,
        z_values[pantheon_SH0ES_range],
        z_hel_values[pantheon_SH0ES_range],
        apparent_mag[pantheon_SH0ES_range],
        cepheid_distances[pantheon_SH0ES_range],
        covariance_matrix[np.ix_(pantheon_SH0ES_range, pantheon_SH0ES_range)],
    )
