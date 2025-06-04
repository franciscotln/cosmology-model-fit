import numpy as np

# Source: https://www.sdss4.org/science/final-bao-and-rsd-measurements-table/
data = np.genfromtxt(
    "y2020SDSSBAO/raw-data/data.txt",
    dtype=[("z", float), ("value", float), ("quantity", "U10")],
    delimiter=" ",
    names=True,
)

cov_matrix = np.loadtxt(
    "y2020SDSSBAO/raw-data/covariance.txt", delimiter=" ", dtype=float
)

legend = "SDSS BAO DR17"


def get_data():
    return (
        legend,
        data,
        cov_matrix,
    )
