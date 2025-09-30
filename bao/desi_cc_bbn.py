from numba import njit
import numpy as np
import emcee
import corner
from scipy.linalg import cho_factor, cho_solve
import matplotlib.pyplot as plt
from multiprocessing import Pool
from cmb.data_desi_compression import r_drag, c
import y2024BBN.prior_lcdm as bbn
from y2005cc.data import get_data as get_cc_data
from y2025BAO.data import get_data as get_bao_data
from .plot_predictions import plot_bao_predictions
from cosmic_chronometers.plot_predictions import plot_cc_predictions

cc_legend, z_cc_vals, H_cc_vals, cc_cov_matrix = get_cc_data()
bao_legend, data, bao_cov_matrix = get_bao_data()
cho_bao = cho_factor(bao_cov_matrix)
cho_cc = cho_factor(cc_cov_matrix)
logdet_cc = np.linalg.slogdet(cc_cov_matrix)[1]
N_cc = len(z_cc_vals)


@njit
def rd(H0, Om, Obh2):  # Mpc
    Omh2 = Om * (H0 / 100) ** 2
    return r_drag(wb=Obh2, wm=Omh2, n_eff=bbn.N_eff)


@njit
def Ez(z, params):
    Om, w0 = params[1], params[3]
    cubic = (1 + z) ** 3
    rho_DE = (2 * cubic / (1 + cubic)) ** (2 * (1 + w0))
    return np.sqrt(Om * cubic + (1 - Om) * rho_DE)


@njit
def H_z(z, params):
    return params[0] * Ez(z, params)


@njit
def DH_z(z, params):
    return c / H_z(z, params)


@njit
def DM_z(z, params):
    x = np.linspace(0, z, num=max(250, int(250 * z)))
    y = DH_z(x, params)
    return np.trapz(y=y, x=x)


@njit
def DV_z(z, params):
    DH = DH_z(z, params)
    DM = DM_z(z, params)
    return (z * DH * DM**2) ** (1 / 3)


qty_map = {
    "DV_over_rs": 0,
    "DM_over_rs": 1,
    "DH_over_rs": 2,
}

quantities = np.array([qty_map[q] for q in data["quantity"]], dtype=np.int32)


@njit
def bao_theory(z, qty, params):
    H0, Om, Obh2 = params[0], params[1], params[2]
    results = np.empty(z.size, dtype=np.float64)
    for i in range(z.size):
        q = qty[i]
        if q == 0:
            results[i] = DV_z(z[i], params) / rd(H0, Om, Obh2)
        elif q == 1:
            results[i] = DM_z(z[i], params) / rd(H0, Om, Obh2)
        elif q == 2:
            results[i] = DH_z(z[i], params) / rd(H0, Om, Obh2)
    return results


bounds = np.array(
    [
        (50, 80),  # H0
        (0.15, 0.45),  # Ωm
        (0.019, 0.025),  # Ωb h^2
        (-2, 0),  # w0
        (0.5, 2.5),  # f_cc
    ],
    dtype=np.float64,
)


def chi_squared(params):
    f_cc = params[-1]
    delta_cc = H_cc_vals - H_z(z_cc_vals, params)
    chi_cc = np.dot(delta_cc, cho_solve(cho_cc, delta_cc, check_finite=False)) * f_cc**2

    delta_bao = data["value"] - bao_theory(data["z"], quantities, params)
    chi_bao = np.dot(delta_bao, cho_solve(cho_bao, delta_bao, check_finite=False))

    bbn_delta = bbn.Obh2 - params[2]
    chi_bbn = (bbn_delta / bbn.Obh2_sigma) ** 2
    return chi_cc + chi_bao + chi_bbn


@njit
def log_prior(params):
    if np.all((bounds[:, 0] < params) & (params < bounds[:, 1])):
        return 0.0
    return -np.inf


def log_likelihood(params):
    f_cc = params[-1]
    normalization_cc = N_cc * np.log(2 * np.pi) + logdet_cc - 2 * N_cc * np.log(f_cc)
    return -0.5 * chi_squared(params) - 0.5 * normalization_cc


def log_probability(params):
    lp = log_prior(params)
    if np.isinf(lp):
        return -np.inf
    return lp + log_likelihood(params)


def main():
    ndim = len(bounds)
    nwalkers = 100 * ndim
    burn_in = 100
    nsteps = 1400 + burn_in
    initial_pos = np.zeros((nwalkers, ndim))

    for dim, (lower, upper) in enumerate(bounds):
        initial_pos[:, dim] = np.random.uniform(lower, upper, nwalkers)

    with Pool(10) as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers,
            ndim,
            log_probability,
            pool=pool,
            moves=[
                (emcee.moves.KDEMove(), 0.5),
                (emcee.moves.DEMove(), 0.4),
                (emcee.moves.DESnookerMove(), 0.1),
            ],
        )
        sampler.run_mcmc(initial_pos, nsteps, progress=True)

    try:
        tau = sampler.get_autocorr_time()
        print("auto-correlation time", tau)
    except emcee.autocorr.AutocorrError as e:
        print("Autocorrelation time could not be computed", e)

    samples = sampler.get_chain(discard=burn_in, flat=True)

    [
        [H0_16, H0_50, H0_84],
        [Om_16, Om_50, Om_84],
        [Obh2_16, Obh2_50, Obh2_84],
        [w0_16, w0_50, w0_84],
        [f_cc_16, f_cc_50, f_cc_84],
    ] = np.percentile(samples, [15.9, 50, 84.1], axis=0).T

    best_fit = np.array([H0_50, Om_50, Obh2_50, w0_50, f_cc_50])
    rd_samples = rd(samples[:, 0], samples[:, 1], samples[:, 2])
    rd_16, rd_50, rd_84 = np.percentile(rd_samples, [15.9, 50, 84.1], axis=0)

    print(f"H0: {H0_50:.2f} +{(H0_84 - H0_50):.2f} -{(H0_50 - H0_16):.2f} km/s/Mpc")
    print(
        f"Ωb h^2: {Obh2_50:.5f} +{(Obh2_84 - Obh2_50):.5f} -{(Obh2_50 - Obh2_16):.5f}"
    )
    print(f"Ωm: {Om_50:.3f} +{(Om_84 - Om_50):.3f} -{(Om_50 - Om_16):.3f}")
    print(f"w0: {w0_50:.3f} +{(w0_84 - w0_50):.3f} -{(w0_50 - w0_16):.3f}")
    print(f"f_cc: {f_cc_50:.2f} +{(f_cc_84 - f_cc_50):.2f} -{(f_cc_50 - f_cc_16):.2f}")
    print(f"rd: {rd_50:.2f} +{(rd_84 - rd_50):.2f} -{(rd_50 - rd_16):.2f} Mpc")
    print(f"Chi squared: {chi_squared(best_fit):.2f}")
    print(f"log likelihood: {log_likelihood(best_fit):.2f}")
    print(f"Degs of freedom: {1 + data['value'].size + z_cc_vals.size - len(best_fit)}")

    plot_bao_predictions(
        theory_predictions=lambda z, qty: bao_theory(z, qty, best_fit),
        data=data,
        errors=np.sqrt(np.diag(bao_cov_matrix)),
        title=f"{bao_legend}: $H_0$={H0_50:.2f} km/s/Mpc",
    )
    plot_cc_predictions(
        H_z=lambda z: H_z(z, best_fit),
        z=z_cc_vals,
        H=H_cc_vals,
        H_err=np.sqrt(np.diag(cc_cov_matrix)) / f_cc_50,
        label=f"{cc_legend}: $H_0$={H0_50:.1f} km/s/Mpc",
    )

    labels = ["$H_0$", "$Ω_m$", "$Ω_b h^2$", "$w_0$", "$f_{CCH}$"]
    corner.corner(
        samples,
        labels=labels,
        quantiles=[0.159, 0.5, 0.841],
        show_titles=True,
        title_fmt=".3f",
        smooth=1.5,
        smooth1d=1.5,
        bins=100,
        levels=(0.393, 0.864),  # 1 and 2 sigmas in 2D
        fill_contours=False,
        plot_datapoints=False,
    )
    plt.show()


if __name__ == "__main__":
    main()

"""
*******************************
Dataset: DESI DR2 2025
*******************************

Flat ΛCDM
H0: 68.66 +0.52 -0.52 km/s/Mpc
Ωb h^2: 0.02220 +0.00054 -0.00054
Ωm: 0.299 +0.008 -0.008
w0: -1
f_cc: 1.49 +0.19 -0.18
rd: 147.69 +1.30 -1.25 Mpc
Chi squared: 43.50
log likelihood: -135.83
Degs of freedom: 43

===============================

Flat wCDM
H0: 67.14 +1.85 -1.83 km/s/Mpc
Ωb h^2: 0.02222 +0.00054 -0.00054
Ωm: 0.299 +0.009 -0.008
w0: -0.937 +0.070 -0.074
f_cc: 1.48 +0.19 -0.18
rd: 149.12 +2.28 -2.07 Mpc
Chi squared: 42.28
log likelihood: -135.41
Degs of freedom: 42

===============================

Flat -1 + 2 * (1 + w0) / (1 + (1 + z)**3)
H0: 66.57 +2.02 -1.92 km/s/Mpc
Ωb h^2: 0.02222 +0.00054 -0.00054
Ωm: 0.307 +0.011 -0.011
w0: -0.868 +0.117 -0.124
f_cc: 1.48 +0.19 -0.18
rd: 148.84 +1.72 -1.67 Mpc
Chi squared: 41.63
log likelihood: -135.19
Degs of freedom: 42
"""
