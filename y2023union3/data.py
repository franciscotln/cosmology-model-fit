# arXiv:2311.12098
import pandas as pd
import numpy as np

df = pd.read_csv("y2023union3/raw-data/bins_union_3.csv")
cov_data = np.genfromtxt("y2023union3/raw-data/covariance.txt")

n = df["zcmb"].size
covariance_matrix = cov_data.reshape((n, n))
zcmb = df["zcmb"].to_numpy()
zhel = df["zhel"].to_numpy()
mu_vals = df["mu"].to_numpy()


def get_data():
    return (
        "Union3.0 - 22 Bins",
        zcmb,
        zhel,
        mu_vals,
        covariance_matrix,
    )
