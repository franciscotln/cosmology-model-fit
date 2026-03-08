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
        ("s8_fid", np.float64),
        ("year", np.int16),
        ("cov_id", np.int16),
    ],
)
cov_mat = np.loadtxt(pathname + "fs8_cov.dat", dtype=np.float64)

mask = (
    ((data["omega_fid"] >= 0.28) & (data["s8_fid"] >= 0.8))
    | (data["cov_id"] == 1)
    | (data["cov_id"] == 2)
    | (data["cov_id"] == 3)
)

data = data[mask]
cov_mat = cov_mat[mask, :][:, mask]

# covariances from arXiv:1806.10822v2

# data
# arXiv:1803.01337v4
# arXiv:2110.08498v2
# arXiv:2007.08999v2

# Covariance for the data points z = 0.3, 0.4, 0.5, 0.6 respectively (cov_id = 2)
# taken from arXiv:1203.6565v2, section 5.1
# estimated_corr = [
#     [1.00, 0.84, 0.50, 0.15],
#     [0.84, 1.00, 0.88, 0.65],
#     [0.50, 0.88, 1.00, 0.92],
#     [0.15, 0.65, 0.92, 1.00],
# ]

cov1 = 1e-3 * np.array(
    [
        [6.400, 2.570, 0.000],
        [2.570, 3.969, 2.540],
        [0.000, 2.540, 5.184],
    ]
)

cov2 = 1e-3 * np.array(
    [
        [3.02500, 1.89420, 1.18250, 0.55275],
        [1.89420, 1.68100, 1.55144, 1.78555],
        [1.18250, 1.55144, 1.84900, 2.65052],
        [0.55275, 1.78555, 2.65052, 4.48900],
    ]
)

cov3 = 1e-2 * np.array(
    [
        [3.0976, 0.8920, 0.3290, -0.021],
        [0.8920, 0.9801, 0.4360, 0.0760],
        [0.3290, 0.4360, 0.4900, 0.3500],
        [-0.021, 0.0760, 0.3500, 1.1236],
    ]
)

inv_cov1 = np.linalg.inv(cov1)
inv_cov2 = np.linalg.inv(cov2)
inv_cov3 = np.linalg.inv(cov3)
