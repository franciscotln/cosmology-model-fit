# https://arxiv.org/pdf/1811.02590
# Quasar data set provided by Risaliti & Lusso, Nature Astronomy, 2018
import pandas as pd
import numpy as np
from scipy.stats import linregress

df = pd.read_csv('y2018quasars/raw-data/data.txt', sep='\s+').sort_values(by='xz')

z = df['xz'].to_numpy()
log_LUV = df['xlo'].to_numpy()
log_LX = df['xlx'].to_numpy()
log_FUV = df['xfo'].to_numpy()
log_FX = df['xfx'].to_numpy()
err_log_FX = df['xefx'].to_numpy()


def bin_and_fit_gamma(z, log_LUV, log_FX, err_log_FX):
    log_z = np.log10(z)
    bin_width = 0.1
    bins = np.arange(min(log_z), max(log_z) + bin_width, bin_width)
    gamma_values = []

    for i in range(len(bins) - 1):
        mask = (log_z >= bins[i]) & (log_z < bins[i + 1])
        if np.sum(mask) > 1:
            weights = 1 / err_log_FX[mask]**2
            A = np.vstack([log_LUV[mask], np.ones_like(log_LUV[mask])]).T 
            W = np.diag(weights)
            b = log_FX[mask]
            ATA = A.T @ W @ A
            ATb = A.T @ W @ b
            gamma_z = np.linalg.solve(ATA, ATb)[0]
            gamma_values.append(gamma_z)

    return np.mean(gamma_values)

mask = z >= 0.5

gamma = bin_and_fit_gamma(z[mask], log_LUV[mask], log_FX[mask], err_log_FX[mask])

sigma_mu = (5 / (2 - 2 * gamma)) * err_log_FX  # Error propagation (without s)


def mu_quasar(beta_prime):
    return 5 / (2 - 2 * gamma) * (beta_prime + gamma * log_FUV - log_FX)


def get_data():
    return (
        f'Quasars (γ={gamma:.3f})',
        z,
        mu_quasar(0),
        sigma_mu,
    )
