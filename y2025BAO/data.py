import numpy as np

# Source: https://arxiv.org/pdf/2503.14738
# https://github.com/CobayaSampler/bao_data/tree/master/desi_bao_dr2

file_path = "y2025BAO/raw-data/"

data = np.genfromtxt(
    file_path + "data.csv",
    dtype=[("z", np.float64), ("value", np.float64), ("quantity", "U10")],
    delimiter=",",
    names=True,
)

cov_matrix = np.loadtxt(file_path + "covariance.txt", delimiter=" ", dtype=np.float64)


def get_data():
    return ("DESI BAO DR2", data, cov_matrix)
