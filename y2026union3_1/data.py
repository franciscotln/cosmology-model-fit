# arXiv:2601.19854v1 UNITY1.8
import pandas as pd
import numpy as np

data_frame = pd.read_csv("y2026union3_1/raw-data/bins_union_3_1.csv")
cov_data = np.genfromtxt("y2026union3_1/raw-data/covariance.txt", dtype=np.float64)

n = data_frame["zcmb"].size
covariance_matrix = cov_data.reshape((n, n))
z_cmb = data_frame["zcmb"].to_numpy()
z_hel = data_frame["zhel"].to_numpy()
mu_values = data_frame["mb"].to_numpy()


def get_data():
    return (
        "Union3.1 - 22 Bins",
        z_cmb,
        z_hel,
        mu_values,
        covariance_matrix,
    )
