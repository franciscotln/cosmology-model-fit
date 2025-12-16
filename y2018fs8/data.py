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

# Covariance for the data points z = 0.3, 0.4, 0.5, 0.6 respectively
# taken from arXiv:1203.6565v2, section 5.1
# estimated_corr = [
#     [1.00, 0.84, 0.50, 0.15],
#     [0.84, 1.00, 0.88, 0.65],
#     [0.50, 0.88, 1.00, 0.92],
#     [0.15, 0.65, 0.92, 1.00],
# ]
