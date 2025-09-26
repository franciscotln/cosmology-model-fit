# arXiv:2311.12098
import pandas as pd
import numpy as np

data_frame = pd.read_csv("y2023union3/raw-data/bins_union_3.csv")
cov_data = np.genfromtxt("y2023union3/raw-data/covariance.txt")

n = data_frame["zcmb"].size
covariance_matrix = cov_data.reshape((n, n))
z_values = data_frame["zcmb"].to_numpy(dtype=np.float64)
mu_values = data_frame["mu"].to_numpy(dtype=np.float64)
sort_indices = np.argsort(z_values)


def get_data():
    return (
        "Union3 - 22 Bins",
        z_values[sort_indices],
        mu_values[sort_indices],
        covariance_matrix[sort_indices, :][:, sort_indices],
    )
