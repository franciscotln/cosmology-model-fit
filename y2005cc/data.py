import pandas as pd
import numpy as np
from interpolator import interp_pchip

data = pd.read_csv("y2005cc/raw-data/data.csv")
cov_components = pd.read_csv("y2005cc/raw-data/cov_components.csv")

z = data["z"].to_numpy()
Hz = data["H"].to_numpy()
sigma_H = data["sigma_H"].to_numpy()

zmod = cov_components["z"].to_numpy(dtype=np.float64)
imf_intp = interp_pchip(z, zmod, cov_components["imf"].to_numpy()) / 100
spsooo_intp = interp_pchip(z, zmod, cov_components["spsooo"].to_numpy()) / 100
# slib_intp = interp_pchip(z, zmod, cov_components["stlib"].to_numpy()) / 100
# sps_intp = interp_pchip(z, zmod, cov_components["sps"].to_numpy()) / 100

cov_mat_diag = np.diag(sigma_H**2)
cov_mat_imf = np.outer(Hz * imf_intp, Hz * imf_intp)
cov_mat_spsooo = np.outer(Hz * spsooo_intp, Hz * spsooo_intp)
# cov_mat_slib = np.outer(Hz * slib_intp, Hz * slib_intp)
# cov_mat_sps = np.outer(Hz * sps_intp, Hz * sps_intp)

# suggested covariance matrix
cov_matrix_sys = cov_mat_imf + cov_mat_spsooo
cov_matrix = cov_matrix_sys + cov_mat_diag


def get_data(split_sys=False):
    legend = f"Cosmic Chronometers ({len(z)} data points)"
    if split_sys:
        return (legend, z, Hz, sigma_H, cov_matrix_sys)
    return (legend, z, Hz, cov_matrix)


# *********************************
# Current data compilation
# arXiv:2412.01994v1: 32 data points
#
# Covariance components:
# https://arxiv.org/pdf/2003.07362
# 
# Covariance matrix construction:
# https://gitlab.com/mmoresco/CCcovariance/-/blob/master/examples/CC_covariance.ipynb
# *********************************


# ------ Latest Measurements ------
# arXiv:2512.02109v1
# H(z=0.542) = 66 +82 -32 (stat) ± 13 (sys)  km/s/Mpc

# arXiv:2506.03836v1
# H(z=0.5) = 72.1 ± 34.7

# arXiv:2511.02730v1
# H(z=0.46) = 88.48 ± 12.33
# H(z=0.67) = 119.45 ± 17.82
# H(z=0.83) = 108.28 ± 18.13

# arXiv:2606.07298v1
# H(0.65) = 93.68 ± 30.22

# arXiv:2608.13178v1
# H(z=0.61) = 88.5 ± 8.1 (sys) +6.7 -12.6 (stat)
# ---------------------------------


# ---- Functional Forms for Covariance Components (curve_fit) ----
# def spsoo(z):
#   a, b, c, A, mu, sig = 18.764, 13.814,  2.297,  3.924,  0.532,  0.385
#   return a * np.exp(-b * z) + c + A * np.exp(-(((z - mu) / sig) ** 2))


# def imf(z):
#   top, dip_amp, z_dip, w_dip, bottom, z_step, k_step = 0.468, 0.350, 0.547, 0.0355, 0.193, 1.001, 39.128
#   dip = dip_amp * np.exp(-(((z - z_dip) / w_dip) ** 2))
#   step = (top - bottom) / (1.0 + np.exp(k_step * (z - z_step)))
#   return bottom + step - dip
