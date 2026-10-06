import numpy as np

# DESY6 BAO (https://arxiv.org/abs/2601.14559 independent of DESI DR2)
data = np.array(
    [(0.85, 19.74, "DM_over_rs")],
    dtype=[("z", np.float64), ("value", np.float64), ("quantity", "U20")],
)
covariance = np.array([[0.60**2]])


def get_data():
    return ("DES Y6 BAO", data, covariance)
