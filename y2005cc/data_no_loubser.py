import pandas as pd
import numpy as np
from interpolator import interp_pchip

_data = pd.read_csv("y2005cc/raw-data/data.csv")
_cov_components = pd.read_csv("y2005cc/raw-data/cov_components.csv")

_loubser_reference = "arXiv:2511.02730v1 (2025)"
_data = _data.loc[_data["reference"] != _loubser_reference].reset_index(drop=True)

z = _data["z"].to_numpy()
Hz = _data["H"].to_numpy()
sigma_H = _data["sigma_H"].to_numpy()
method= _data["M"].to_numpy()

# ------ Handling missing systematic uncertainties ------
# Where systematic uncertainties were not published
# separately (all full spectral fitting data)
# we set them to 18% floor of the total error budget for F
# and 65% for D4000A (mean value based on other data points)
# There are only two D4000A data points without published systematic uncertainties

_syst_mask = _data["sys"] == 0
_syst_fraction = np.where(method[_syst_mask] == "F", 0.18, 0.65)
_data.loc[_syst_mask, "sys"] = _syst_fraction * sigma_H[_syst_mask]
_data.loc[_syst_mask, "stat"] = sigma_H[_syst_mask] * np.sqrt(1.0 - _syst_fraction**2)

diag_syst = _data["sys"].to_numpy()
diag_stat = _data["stat"].to_numpy()
# -------------------------------------------------------


# ---- Constructing covariance matrix as per Moresco ----
zmod, imf, spsooo = _cov_components[["z", "imf", "spsooo"]].to_numpy(dtype=np.float64).T
imf_intp = interp_pchip(z, zmod, imf) / 100
spsooo_intp = interp_pchip(z, zmod, spsooo) / 100
cov_mat_imf = np.outer(Hz * imf_intp, Hz * imf_intp)
cov_mat_spsooo = np.outer(Hz * spsooo_intp, Hz * spsooo_intp)

# suggested systematic covariance matrix
cov_matrix_sys = cov_mat_imf + cov_mat_spsooo + np.diag(diag_syst**2)
# -------------------------------------------------------


# --- Total covariance matrix as suggested by Moresco ---
cov_matrix_tot = cov_matrix_sys + np.diag(diag_stat**2)
# -------------------------------------------------------


def get_data(split_sys=False):
    legend = f"Cosmic Chronometers ({len(z)} data points)"
    if split_sys:
        return (legend, z, Hz, diag_stat, cov_matrix_sys)
    return (legend, z, Hz, cov_matrix_tot)


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

# arXiv:2606.07298v1
# H(0.65) = 93.68 ± 10.67 syst ± 28.27 stat

# arXiv:2608.13178v1
# H(z=0.61) = 88.5 ± 8.1 syst +6.7 -12.6 stat

# THESE ARE EXCLUDED FROM THIS DATASET
# arXiv:2511.02730v1
# H(z=0.46) = 88.48 ± 12.32 syst ± 0.57 stat
# H(z=0.67) = 119.45 ± 16.64 syst ± 6.39 stat
# H(z=0.83) = 108.28 ± 15.08 syst ± 10.07 stat
# ---------------------------------