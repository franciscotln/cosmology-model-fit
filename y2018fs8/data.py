import numpy as np

"""
Data:
arXiv:1803.01337v4
arXiv:2110.08498v2 (ALFALFA z=0.013)

Covariances WiggleZ and SDSS-IV (cov_id = 1 and cov_id = 4 respectively):
arXiv:1806.10822v2

Covariance for SDSS-III (cov_id = 3):
arXiv:1607.03155v1

Covariance for the data points z = 0.3, 0.4, 0.5, 0.6 respectively (cov_id = 2)
arXiv:1203.6565v2, section 5.1
Estimated correlation:
[[1.         0.82185005 0.49472079 0.1540818 ]
 [0.82185005 1.         0.86260762 0.63950074]
 [0.49472079 0.86260762 1.         0.90248676]
 [0.1540818  0.63950074 0.90248676 1.        ]]
"""

data = np.loadtxt(
    "y2018fs8/raw/fs8.csv",
    delimiter=",",
    skiprows=1,
    dtype=[
        ("z", np.float64),
        ("fs8", np.float64),
        ("fs8_err", np.float64),
        ("omega_fid", np.float64),
        ("s8_fid", np.float64),
        ("H0_fid", np.float64),
        ("year", np.int16),
        ("cov_id", np.int16),
    ],
)

outliers_idx = [3, 5, 33, 36, 54]
data = np.delete(data, outliers_idx)

missing_H0_fid = data["H0_fid"] == 0
data["H0_fid"][missing_H0_fid] = 71.0  # WMAP-based data

cov1 = 1e-3 * np.array(
    [
        [6.400, 2.570, 0.000],
        [2.570, 3.969, 2.540],
        [0.000, 2.540, 5.184],
    ]
)

cov2 = 1e-3 * np.array(
    [
        [3.025, 1.85327186, 1.17001468, 0.56779143],
        [1.85327186, 1.681, 1.52077723, 1.75670853],
        [1.17001468, 1.52077723, 1.849, 2.60006435],
        [0.56779143, 1.75670853, 2.60006435, 4.489],
    ]
)

cov3 = 1e-3 * np.array(
    [
        [2.025, 0.816183, 0.260712],
        [0.816183, 1.444, 0.6593076],
        [0.260712, 0.6593076, 1.156],
    ]
)

cov4 = 1e-2 * np.array(
    [
        [3.0976, 0.8920, 0.3290, -0.021],
        [0.8920, 0.9801, 0.4360, 0.0760],
        [0.3290, 0.4360, 0.4900, 0.3500],
        [-0.021, 0.0760, 0.3500, 1.1236],
    ]
)

cov_mat = np.diag(data["fs8_err"] ** 2)

cov_blocks = {1: cov1, 2: cov2, 3: cov3, 4: cov4}
for cov_id, block in cov_blocks.items():
    idx = np.where(data["cov_id"] == cov_id)[0]
    cov_mat[np.ix_(idx, idx)] = block
