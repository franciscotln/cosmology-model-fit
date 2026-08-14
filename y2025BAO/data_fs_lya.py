import numpy as np

# Source: https://arxiv.org/pdf/2607.27410
# https://github.com/CobayaSampler/bao_data/pull/6/changes

file_path = "y2025BAO/raw-data/"

data = np.genfromtxt(
    file_path + "data_fs_lya.csv",
    dtype=[("z", np.float64), ("value", np.float64), ("quantity", "U10")],
    delimiter=",",
    names=True,
)

cov_matrix = np.loadtxt(file_path + "covariance_fs_lya.txt", delimiter=" ", dtype=np.float64)


def get_data():
    return ("DESI BAO DR2 + FS Lyα", data, cov_matrix)
