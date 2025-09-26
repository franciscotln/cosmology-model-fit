import pandas as pd
import numpy as np

data = pd.read_csv("y2005cc/raw-data/data.csv")
cov_components = pd.read_csv("y2005cc/raw-data/cov_components.csv")

z = data["z"].to_numpy(dtype=np.float64)
Hz = data["H"].to_numpy(dtype=np.float64)
sigma_H = data["sigma_H"].to_numpy(dtype=np.float64)

zmod = cov_components["z"].to_numpy(dtype=np.float64)
imf_intp = np.interp(z, zmod, cov_components["imf"]) / 100
slib_intp = np.interp(z, zmod, cov_components["stlib"]) / 100
sps_intp = np.interp(z, zmod, cov_components["sps"]) / 100
spsooo_intp = np.interp(z, zmod, cov_components["spsooo"]) / 100

N = len(z)
cov_mat_diag = np.zeros((N, N), dtype="float64")
cov_mat_imf = np.zeros((N, N), dtype="float64")
cov_mat_slib = np.zeros((N, N), dtype="float64")
cov_mat_sps = np.zeros((N, N), dtype="float64")
cov_mat_spsooo = np.zeros((N, N), dtype="float64")

for i in range(N):
    cov_mat_diag[i, i] = sigma_H[i] ** 2

for i in range(N):
    for j in range(N):
        cov_mat_imf[i, j] = Hz[i] * imf_intp[i] * Hz[j] * imf_intp[j]
        cov_mat_slib[i, j] = Hz[i] * slib_intp[i] * Hz[j] * slib_intp[j]
        cov_mat_sps[i, j] = Hz[i] * sps_intp[i] * Hz[j] * sps_intp[j]
        cov_mat_spsooo[i, j] = Hz[i] * spsooo_intp[i] * Hz[j] * spsooo_intp[j]

# suggested covariance matrix
cov_matrix = cov_mat_spsooo + cov_mat_imf + cov_mat_diag


def get_data():
    return (
        f"Cosmic Chronometers ({N} data points)",
        z,
        Hz,
        cov_matrix,
    )


"""
Covariance components: https://arxiv.org/pdf/2003.07362
Covariance matrix construction: https://gitlab.com/mmoresco/CCcovariance/-/blob/master/examples/CC_covariance.ipynb

Current data from https://arxiv.org/pdf/2307.09501
and https://arxiv.org/pdf/2506.03836
z,H,sigma_H,method,ref
0.09,69,12,F,72
0.27,77,14,F,72
0.4,95,17,F,72
0.9,117,23,F,72

Read off of plot in publication [72] https://arxiv.org/pdf/astro-ph/0412269
[
    (z, H, sigma_H)
    (0.09, 70.7, 12),
    (0.17, 83, 8),
    (0.27, 70, 14),
    (0.40, 87, 17),
    (0.88, 117, 23),
    (1.30, 168, 17),
    (1.43, 177, 18),
    (1.53, 140, 14),
    (1.75, 202, 40),
]
"""
