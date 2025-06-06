import numpy as np

"""
Sources:
https://www.sdss4.org/science/final-bao-and-rsd-measurements-table/
https://svn.sdss.org/public/data/eboss/DR16cosmo/tags/v1_0_1/likelihoods/BAO-only/README.txt
https://arxiv.org/pdf/2007.08995 (z_eff = 2.334)

For the compilation data, the following references were added:
DV/rd z=0.106: https://arxiv.org/pdf/1106.3366
DV/rd z=0.32 and z=0.57: https://arxiv.org/pdf/1312.4877
"""

data = np.genfromtxt(
    "y2020SDSSBAO/raw-data/data.csv",
    dtype=[("z", float), ("value", float), ("quantity", "U10")],
    delimiter=",",
    names=True,
)

cov_matrix = np.loadtxt(
    "y2020SDSSBAO/raw-data/covariance.txt", delimiter=" ", dtype=float
)

legend = "SDSS BAO DR16"


def get_data():
    return (
        legend,
        data,
        cov_matrix,
    )
