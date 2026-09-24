import pandas as pd
import numpy as np
from interpolator import interp_pchip

data = pd.read_csv("y2005cc/raw-data/data.csv")
cov_components = pd.read_csv("y2005cc/raw-data/cov_components.csv")

z = data["z"].to_numpy()
Hz = data["H"].to_numpy()
sigma_H = data["sigma_H"].to_numpy()
method= data["M"].to_numpy()

_stat_H = data["stat"]
_syst_H = data["sys"]
_ref = data["reference"]


# ------ Handling missing systematic uncertainties ------
# Where systematic uncertainties were not published
# separately (basically all full spectral fitting data)
# There are only two D4000A data points without published
# systematic uncertainties separately.

_syst_mask = data["sys"] == 0
_method_D_mask = method == "D"
_method_F_mask = method == "F"

# Method D4000 (D)
_mask_D4000 = _syst_mask & _method_D_mask
data.loc[_mask_D4000, "sys"] = 0.64 * sigma_H[_mask_D4000]
data.loc[_mask_D4000, "stat"] = np.sqrt(1.0 - 0.64**2) * sigma_H[_mask_D4000]

# Method FSF (F)
_mask_FSF  = _syst_mask & _method_F_mask
data.loc[_mask_FSF, "sys"]  = np.minimum(0.42 * sigma_H[_mask_FSF], 0.07 * Hz[_mask_FSF])
data.loc[_mask_FSF, "stat"] = np.sqrt(sigma_H[_mask_FSF]**2 - data.loc[_mask_FSF, "sys"]**2)

diag_syst = _syst_H.to_numpy()
diag_stat = _stat_H.to_numpy()
# -------------------------------------------------------


# ---- Constructing covariance matrix as per Moresco ----
zmod = cov_components["z"].to_numpy(dtype=np.float64)
imf_intp = interp_pchip(z, zmod, cov_components["imf"].to_numpy()) / 100
spsooo_intp = interp_pchip(z, zmod, cov_components["spsooo"].to_numpy()) / 100

imf_component = np.zeros_like(Hz, dtype=np.float64)
spsooo_component = np.zeros_like(Hz, dtype=np.float64)
imf_component[_method_D_mask] = Hz[_method_D_mask] * imf_intp[_method_D_mask]
spsooo_component[_method_D_mask] = Hz[_method_D_mask] * spsooo_intp[_method_D_mask]

cov_mat_imf = np.outer(imf_component, imf_component)
cov_mat_spsooo = np.outer(spsooo_component, spsooo_component)

# suggested systematic covariance matrix D4000A
cov_matrix_sys = cov_mat_imf + cov_mat_spsooo + np.diag(diag_syst**2)
# -------------------------------------------------------


# ----- Handle Loubser et al. systematic correlation ----
# Reference arXiv:2511.02730v1 (2025)
loubser_mask = _ref == "arXiv:2511.02730v1 (2025)"
loubser_corr = np.array([
    [1.0, 0.932, 0.830],
    [0.932, 1.0, 0.776],
    [0.830, 0.776, 1.0],
])
loubser_syst = diag_syst[loubser_mask]
loubser_cov_sys = loubser_corr * np.outer(loubser_syst, loubser_syst)
cov_matrix_sys[np.ix_(loubser_mask, loubser_mask)] = loubser_cov_sys
# -------------------------------------------------------


def get_data(split_sys=False):
    legend = f"Cosmic Chronometers ({len(z)} data points)"
    if split_sys:
        return (legend, z, Hz, diag_stat, cov_matrix_sys)
    return (legend, z, Hz, cov_matrix_sys + np.diag(diag_stat**2))


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
# H(z=0.542) = 66 ± 13 syst +82 -32 stat   km/s/Mpc

# arXiv:2506.03836v1
# H(z=0.5) = 72.1 ± 7.3 syst ± 33.9 stat

# arXiv:2511.02730v1
# H(z=0.46) = 88.48 ± 12.32 syst ± 0.57 stat
# H(z=0.67) = 119.45 ± 16.64 syst ± 6.39 stat
# H(z=0.83) = 108.28 ± 15.08 syst ± 10.07 stat

# arXiv:2606.07298v1
# H(0.65) = 93.68 ± 10.67 syst ± 28.27 stat

# arXiv:2608.13178v1
# H(z=0.61) = 88.5 ± 8.1 syst +6.7 -12.6 stat
# ---------------------------------
