import numpy as np

# https://arxiv.org/pdf/1106.3366
data = np.array(
    [(0.106, 2.9761904762, "DV_over_rs")],
    dtype=[("z", np.float64), ("value", np.float64), ("quantity", "U20")],
)
covariance = np.array([[0.0176517796]])


def get_data():
    return ("6dF BAO", data, covariance)
