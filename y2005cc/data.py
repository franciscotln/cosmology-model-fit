# source: https://arxiv.org/pdf/2307.09501
# Covariance components: https://arxiv.org/pdf/2003.07362
# Covariance matrix construction: https://gitlab.com/mmoresco/CCcovariance/-/blob/master/examples/CC_covariance.ipynb
import pandas as pd
import numpy as np

data = pd.read_csv("y2005cc/raw-data/data.csv")
cov_components = pd.read_csv("y2005cc/raw-data/cov_components.csv")

z = data["z"].to_numpy()
Hz = data["H"].to_numpy()
sigma_H = data["sigma_H"].to_numpy()

zmod = cov_components["z"].to_numpy()
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
