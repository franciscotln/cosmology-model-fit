# covariances from arXiv:1806.10822v2
# data
# arXiv:1803.01337v4
# arXiv:2110.08498v2
import numpy as np

pathname = "y2018fs8/raw/"
data = np.genfromtxt(pathname + "fs8.csv", delimiter=",", names=True, dtype=np.float64)
cov_mat = np.genfromtxt(pathname + "fs8_cov.dat", dtype=np.float64)

z_values = data["z"]
fs8_values = data["fs8"]
fs8_err = data["fs8_err"]
om_fid = data["omega_fid"]
