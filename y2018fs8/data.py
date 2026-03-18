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
[[1.    0.88  0.448 0.147]
 [0.88  1.    0.886 0.455]
 [0.448 0.886 1.    0.895]
 [0.147 0.455 0.895 1.   ]]
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

outliers_idx = [3, 5, 33, 35, 36, 54]
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
        [3.025, 1.9844, 1.05952, 0.541695],
        [1.9844, 1.681, 1.562018, 1.249885],
        [1.05952, 1.562018, 1.849, 2.578495],
        [0.541695, 1.249885, 2.578495, 4.489],
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
