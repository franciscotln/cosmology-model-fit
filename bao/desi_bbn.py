from numba import njit
import numpy as np
import emcee
import corner
from scipy.linalg import cho_factor, cho_solve
import matplotlib.pyplot as plt
from multiprocessing import Pool
from y2025BAO.data import get_data
import y2024BBN.prior_lcdm as bbn
from cmb.data_desi_compression import r_drag, c
from .plot_predictions import plot_bao_predictions

legend, data, cov_matrix = get_data()
cho = cho_factor(cov_matrix)


@njit
def rd(H0, Om, Obh2):
    h = H0 / 100
    Omh2 = Om * h**2
    return r_drag(wb=Obh2, wm=Omh2, n_eff=bbn.N_eff)


@njit
def Ez(z, params):
    Om, w0 = params[1], params[3]
    OL = 1 - Om
    one_plus_z = 1 + z
    cubic = one_plus_z**3
    rho_de = (2 * cubic / (1 + cubic)) ** (2 * (1 + w0))
    return np.sqrt(Om * one_plus_z**3 + OL * rho_de)


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
def theory_predictions(z, qty, params):
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
        (0.15, 0.55),  # Ωm
        (0.015, 0.030),  # Ωb h^2
        (-1.5, 0),  # w0
    ],
    dtype=np.float64,
)


def chi_squared(params):
    bbn_delta = (bbn.Obh2 - params[2]) / bbn.Obh2_sigma
    delta = data["value"] - theory_predictions(data["z"], quantities, params)
    return np.dot(delta, cho_solve(cho, delta, check_finite=False)) + bbn_delta**2


@njit
def log_prior(params):
    if not np.all((bounds[:, 0] < params) & (params < bounds[:, 1])):
        return -np.inf
    return 0.0


def log_likelihood(params):
    return -0.5 * chi_squared(params)


def log_probability(params):
    lp = log_prior(params)
    if np.isinf(lp):
        return -np.inf
    return lp + log_likelihood(params)


def main():
    ndim = len(bounds)
    nwalkers = 125 * ndim
    burn_in = 100
    nsteps = 1400 + burn_in
    initial_pos = np.zeros((nwalkers, ndim))

    for dim, (lower, upper) in enumerate(bounds):
        initial_pos[:, dim] = np.random.uniform(lower, upper, nwalkers)

    with Pool(5) as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers,
            ndim,
            log_probability,
            pool=pool,
            moves=[(emcee.moves.KDEMove(), 0.5), (emcee.moves.StretchMove(), 0.5)],
        )
        sampler.run_mcmc(initial_pos, nsteps, progress=True)

    try:
        tau = sampler.get_autocorr_time()
        print("auto-correlation time", tau)
    except emcee.autocorr.AutocorrError as e:
        print("Autocorrelation time could not be computed", e)

    chains_samples = sampler.get_chain(discard=0, flat=False)
    samples = sampler.get_chain(discard=burn_in, flat=True)

    [
        [H0_16, H0_50, H0_84],
        [Om_16, Om_50, Om_84],
        [Obh2_16, Obh2_50, Obh2_84],
        [w0_16, w0_50, w0_84],
    ] = np.percentile(samples, [15.9, 50, 84.1], axis=0).T

    best_fit = np.array([H0_50, Om_50, Obh2_50, w0_50], dtype=np.float64)

    h_samples = samples[:, 0] / 100
    Omh2_samples = samples[:, 1] * h_samples**2
    rd_samples = rd(samples[:, 0], samples[:, 1], samples[:, 2])
    Omh2_16, Omh2_50, Omh2_84 = np.percentile(Omh2_samples, [15.9, 50, 84.1])
    rd_16, rd_50, rd_84 = np.percentile(rd_samples, [15.9, 50, 84.1])

    residuals = data["value"] - theory_predictions(data["z"], quantities, best_fit)
    SS_res = np.sum(residuals**2)
    SS_tot = np.sum((data["value"] - np.mean(data["value"])) ** 2)
    r2 = 1 - SS_res / SS_tot

    print(f"H0: {H0_50:.2f} +{(H0_84 - H0_50):.2f} -{(H0_50 - H0_16):.2f} km/s/Mpc")
    print(
        f"Ωb h^2: {Obh2_50:.5f} +{(Obh2_84 - Obh2_50):.5f} -{(Obh2_50 - Obh2_16):.5f}"
    )
    print(
        f"Ωm h^2: {Omh2_50:.5f} +{(Omh2_84 - Omh2_50):.5f} -{(Omh2_50 - Omh2_16):.5f}"
    )
    print(f"Ωm: {Om_50:.4f} +{Om_84-Om_50:.4f} -{Om_50-Om_16:.4f}")
    print(f"w0: {w0_50:.3f} +{(w0_84 - w0_50):.3f} -{(w0_50 - w0_16):.3f}")
    print(f"r_d: {rd_50:.2f} +{(rd_84 - rd_50):.2f} -{(rd_50 - rd_16):.2f} Mpc")
    print(f"Chi squared: {chi_squared(best_fit):.2f}")
    print(f"Degs of freedom: {1 + data['value'].size  - len(best_fit)}")
    print(f"R^2: {r2:.4f}")
    print(f"RMSD: {np.sqrt(np.mean(residuals**2)):.3f}")

    plot_bao_predictions(
        theory_predictions=lambda z, qty: theory_predictions(z, qty, best_fit),
        data=data,
        errors=np.sqrt(np.diag(cov_matrix)),
        title=f"{legend}: $H_0$={H0_50:.2f} km/s/Mpc, $\\Omega_m$={Om_50:.4f}",
    )
    labels = ["$H_0$", "$Ω_m$", "$Ω_b x h^2$", "$w_0$"]
    corner.corner(
        samples,
        labels=labels,
        quantiles=[0.159, 0.5, 0.841],
        show_titles=True,
        title_fmt=".3f",
        bins=100,
        fill_contours=False,
        plot_datapoints=False,
        smooth=1.5,
        smooth1d=1.5,
        levels=(0.393, 0.864),  # 1 and 2 sigmas in 2D
    )
    plt.show()

    _, axes = plt.subplots(ndim, figsize=(10, 7))
    for i in range(ndim):
        axes[i].plot(chains_samples[:, :, i], color="black", alpha=0.3, lw=0.4)
        axes[i].set_ylabel(labels[i])
        axes[i].axvline(x=burn_in, color="red", linestyle="--", alpha=0.5)
        axes[i].axhline(y=best_fit[i], color="white", linestyle="--", alpha=0.5)
    axes[ndim - 1].set_xlabel("chain step")
    plt.show()


if __name__ == "__main__":
    main()


"""
*******************************
Dataset: DESI DR2 2025
*******************************

Flat ΛCDM:
H0: 68.64 +0.53 -0.54 km/s/Mpc
Ωb h^2: 0.02218 +0.00055 -0.00054
Ωm h^2: 0.14029 +0.00461 -0.00449
Ωm: 0.2978 +0.0087 -0.0084
w0: -0.753 +0.509 -0.506
r_d: 147.90 +1.36 -1.34 Mpc
Chi squared: 10.27
Degs of freedom: 11
R^2: 0.9987
RMSD: 0.305

===============================

Flat wCDM:
H0: 66.55 +2.05 -2.08 km/s/Mpc
Ωb h^2: 0.02218 +0.00055 -0.00054
Ωm h^2: 0.13199 +0.00914 -0.00962
Ωm: 0.2971 +0.0090 -0.0087
w0: -0.916 +0.076 -0.079
r_d: 150.00 +2.75 -2.42 Mpc
Chi squared: 9.05
Degs of freedom: 10
R^2: 0.9989
RMSD: 0.280

===============================

Flat alternative: w(z) = -1 + 2 * (1 + w0) / (1 + (1 + z)**3)
H0: 65.95 +2.13 -2.00 km/s/Mpc
Ωb h^2: 0.02218 +0.00054 -0.00055
Ωm h^2: 0.13404 +0.00671 -0.00646
Ωm: 0.3079 +0.0117 -0.0116
w0: -0.833 +0.121 -0.129
r_d: 149.46 +1.89 -1.83 Mpc
Chi squared: 8.43
Degs of freedom: 10
R^2: 0.9990
RMSD: 0.266
"""
