# https://arxiv.org/pdf/1811.02590
# Quasar data set provided by Risaliti & Lusso, Nature Astronomy, 2018
import pandas as pd
import numpy as np

df = pd.read_csv('y2018quasars/raw-data/data.txt', sep='\s+').sort_values(by='xz')

z = df['xz'].to_numpy()
log_LUV = df['xlo'].to_numpy()
log_LX = df['xlx'].to_numpy()
log_FUV = df['xfo'].to_numpy()
log_FX = df['xfx'].to_numpy()
err_log_FX = df['xefx'].to_numpy()


def bin_and_fit_gamma(z, log_FUV, log_FX, err_log_FX):
    log_z = np.log10(z)
    bin_width = 0.1
    bins = np.arange(min(log_z), max(log_z) + bin_width, bin_width)
    gamma_values = []

    for i in range(len(bins) - 1):
        mask = (log_z >= bins[i]) & (log_z < bins[i + 1])
        if np.sum(mask) > 1:
            weights = 1 / err_log_FX[mask]**2
            A = np.vstack([log_FUV[mask], np.ones_like(log_FUV[mask])]).T 
            W = np.diag(weights)
            b = log_FX[mask]
            ATA = A.T @ W @ A
            ATb = A.T @ W @ b
            gamma_z = np.linalg.solve(ATA, ATb)[0]
            gamma_values.append(gamma_z)

    return np.mean(gamma_values)

# https://arxiv.org/abs/1505.07118
mask = z >= 0.5

gamma = bin_and_fit_gamma(z[mask], log_FUV[mask], log_FX[mask], err_log_FX[mask])
factor = 5 / (2 - 2 * gamma)
sigma_mu = factor * err_log_FX  # Error propagation (without s - intrinsic dispersion)


def mu_quasar(beta_prime):
    return factor * (beta_prime + gamma * log_FUV - log_FX)


def get_data():
    return (
        f'Quasars (γ={gamma:.3f})',
        z,
        mu_quasar(0),
        sigma_mu,
    )


def bin_quasar_data_fixed_size(z, mu_qso, sigma_mu, bin_size):
    sorted_indices = np.argsort(z)
    z_sorted = z[sorted_indices]
    mu_sorted = mu_qso[sorted_indices]
    sigma_sorted = sigma_mu[sorted_indices]

    N = len(z)
    z_binned = []
    mu_binned = []
    sigma_binned = []

    for i in range(0, N, bin_size):
        z_bin = z_sorted[i:i + bin_size]
        mu_bin = mu_sorted[i:i + bin_size]
        sigma_bin = sigma_sorted[i:i + bin_size]

        if len(z_bin) == 0:
            continue

        weights = 1 / sigma_bin**2
        mu_avg = np.sum(mu_bin * weights) / np.sum(weights)
        sigma_avg = np.sqrt(1 / np.sum(weights))
        z_avg = np.mean(z_bin)

        z_binned.append(z_avg)
        mu_binned.append(mu_avg)
        sigma_binned.append(sigma_avg)

    return np.array(z_binned), np.array(mu_binned), np.array(sigma_binned)


def get_binned_data(bin_size=30):
    mu_qso = mu_quasar(0)
    z_binned, mu_binned, sigma_binned = bin_quasar_data_fixed_size(z, mu_qso, sigma_mu, bin_size)
    return (
        f'Quasars (γ={gamma:.3f})',
        z_binned,
        mu_binned,
        sigma_binned,
    )
