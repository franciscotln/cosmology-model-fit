# Source: https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/Pantheon%2BSH0ES.dat
import os
import pandas as pd
import numpy as np

path_to_data = os.path.dirname(os.path.abspath(__file__)) + "/raw-data/"
data_frame = pd.read_csv(path_to_data + "distances.txt", sep=" ")
cov_file = pd.read_csv(path_to_data + "covariance_stat_sys.txt", sep=" ")
selected_columns = data_frame[["zHD", "zHEL", "m_b_corr", "CEPH_DIST", "IS_CALIBRATOR"]]

legend = "Pantheon+ and SH0ES"
z_values = selected_columns["zHD"].to_numpy(dtype=np.float64)
z_hel_values = selected_columns["zHEL"].to_numpy(dtype=np.float64)
apparent_mag = selected_columns["m_b_corr"].to_numpy(dtype=np.float64)
cepheid_distances = selected_columns["CEPH_DIST"].to_numpy(dtype=np.float64)

n = z_values.size
covariance_matrix = cov_file["cov_mu_shoes"].to_numpy(dtype=np.float64).reshape((n, n))


def get_data(z_cut_ceph=0.0):
    """
    :param float z_cut_ceph: Minimum redshift for including Cepheid-calibrated SNe
    """
    ceph_mask = (z_values >= z_cut_ceph) & (selected_columns["IS_CALIBRATOR"] == 1)
    pantheon_SH0ES_range = np.where((ceph_mask) | (z_values > 0.01))[0]

    return (
        legend,
        z_values[pantheon_SH0ES_range],
        z_hel_values[pantheon_SH0ES_range],
        apparent_mag[pantheon_SH0ES_range],
        cepheid_distances[pantheon_SH0ES_range],
        covariance_matrix[np.ix_(pantheon_SH0ES_range, pantheon_SH0ES_range)],
    )


"""
Ceph only < 0.0055 (constant true magnitude M0):
Fitted M0: -19.285 +- 0.042 mag

Ceph only >= 0.0055 (outflow model):
M_inf: -19.403 +0.148 -0.147 mag
v_flow: 183 +147 -148 km/s

DESI + CMB + Pantheon+ + SH0ES (Ceph_z >= 0.0055)
M_inf: -19.427 +0.009 -0.009 mag (perfect agreement with Ceph only >= 0.0055)
v_flow: 160.4 +29.6 -29.5 km/s (perfect agreement with Ceph only >= 0.0055)
M(z=0.0055) = -19.427 + 160.4 * (5/ln(10)) / (c * 0.0055) = -19.216 +- 0.048 mag
(1.08 sigma agreement with Ceph only M0 for z < 0.0055)
"""
