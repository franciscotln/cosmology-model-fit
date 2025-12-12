import numpy as np

pathname = "y2018fs8/raw/"
data = np.loadtxt(
    pathname + "fs8.csv",
    delimiter=",",
    skiprows=1,
    dtype=[
        ("z", np.float64),
        ("fs8", np.float64),
        ("fs8_err", np.float64),
        ("omega_fid", np.float64),
        ("year", np.int16),
    ],
)
cov_mat = np.loadtxt(pathname + "fs8_cov.dat", dtype=np.float64)

# covariances from arXiv:1806.10822v2
# data
# arXiv:1803.01337v4
# arXiv:2110.08498v2
# arXiv:2007.08999v2
