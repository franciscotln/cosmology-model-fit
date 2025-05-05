import numpy as np

# Source: https://arxiv.org/pdf/2503.14738
# https://github.com/CobayaSampler/bao_data/tree/master/desi_bao_dr2
data = np.genfromtxt(
    "y2025BAO/raw-data/data.txt",
    dtype=[("z", float), ("value", float), ("quantity", "U10")],
    delimiter=" ",
    names=True,
)

cov_matrix = np.loadtxt("y2025BAO/raw-data/covariance.txt", delimiter=" ", dtype=float)

legend = "DESI BAO DR2"

def get_data():
    return (
        legend,
        data,
        cov_matrix,
    )
