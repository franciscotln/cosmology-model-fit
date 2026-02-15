# arXiv:2311.12098
import pandas as pd
import numpy as np

data_frame = pd.read_csv("y2026union3_1/raw-data/bins_union_3_1.csv")
cov_data = np.genfromtxt("y2026union3_1/raw-data/covariance.txt", dtype=np.float64)

n = data_frame["zcmb"].size
covariance_matrix = cov_data.reshape((n, n))
z_cmb = data_frame["zcmb"].to_numpy()
z_hel = data_frame["zhel"].to_numpy()
mu_values = data_frame["mb"].to_numpy()
sort_indices = np.argsort(z_cmb)


def get_data():
    return (
        "Union3.1 - 22 Bins",
        z_cmb[sort_indices],
        z_hel[sort_indices],
        mu_values[sort_indices],
        covariance_matrix[sort_indices, :][:, sort_indices],
    )
