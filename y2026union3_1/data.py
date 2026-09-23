# arXiv:2601.19854v1 UNITY1.8
# https://github.com/rubind/union3/blob/master/data_release/union31_unity18_binned_mu/mu_binned.ecsv

import pandas as pd
import numpy as np

data_frame = pd.read_csv("y2026union3_1/raw-data/bins_union3_1_unity1_8.csv")
covariance_matrix = np.genfromtxt("y2026union3_1/raw-data/covariance_unity1_8.txt")

z_cmb = data_frame["z"].to_numpy()
z_hel = z_cmb.copy()
mu_residual = data_frame["mu_residual"].to_numpy()
mu = data_frame["mu"].to_numpy()


def get_data():
    return ("Union3.1 - 22 Bins", z_cmb, z_hel, mu, covariance_matrix)
