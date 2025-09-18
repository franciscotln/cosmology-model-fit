import numpy as np
import emcee
import corner
from scipy.integrate import quad
from scipy.linalg import cho_factor, cho_solve
import matplotlib.pyplot as plt
from multiprocessing import Pool
from y2025BAO.data import get_data as get_bao_data
from .plot_predictions import plot_bao_predictions


c = 299792.458  # km/s

# --- PLANCK DISTANCE PRIORS (Chen+2018 arXiv:1808.05724v1) ---
PLANCK_R_mean = 1.750235
PLANCK_lA_mean = 301.4707
PLANCK_Ob_h2_mean = 0.02235976
planck_priors = np.array([PLANCK_R_mean, PLANCK_lA_mean, PLANCK_Ob_h2_mean])
inv_cov_mat = np.array(
    [
        [94392.3971, -1360.4913, 1664517.2916],
        [-1360.4913, 161.4349, 3671.618],
        [1664517.2916, 3671.618, 79719182.5162],
    ]
)
TCMB = 2.7255  # K
O_GAMMA_H2 = 2.38095e-5 * (TCMB / 2.7) ** 4.0


bao_legend, bao_data, bao_cov_matrix = get_bao_data()
cho_bao = cho_factor(bao_cov_matrix)


def Ez(z, params):
    H0, Om, w0 = params[0], params[1], params[3]
    h = H0 / 100
    z_eq = 2.5 * 10**4 * Om * h**2 * (2.7 / TCMB) ** 4
    Or = Om / (1 + z_eq)
    Ode = 1 - Om - Or

    one_plus_z = 1 + z
    cubed = one_plus_z**3
    rho_de = (2 * cubed / (1 + cubed)) ** (2 * (1 + w0))

    return np.sqrt(Or * one_plus_z**4 + Om * cubed + Ode * rho_de)


def z_star(Ob_h2, Om_h2):
    g1 = 0.0783 * Ob_h2**-0.238 / (1 + 39.5 * Ob_h2**0.763)
    g2 = 0.560 / (1 + 21.1 * Ob_h2**1.81)
    return 1048 * (1 + 0.00124 * Ob_h2**-0.738) * (1 + g1 * Om_h2**g2)


def z_drag(Ob_h2, Om_h2):
    b1 = 0.313 * Om_h2**-0.419 * (1 + 0.607 * Om_h2**0.674)
    b2 = 0.238 * Om_h2**0.223
    # Calibrated 1340 to reproduce Planck r_drag (see discussion: EH98 vs CAMB/Planck)
    # see arXiv:astro-ph/9510117v2, equation E-2
    return (1340 * Om_h2**0.251 / (1 + 0.659 * Om_h2**0.828)) * (1 + b1 * Ob_h2**b2)


def rs_z(z, params):
    H0, Ob_h2 = params[0], params[2]
    Rb = 3 * Ob_h2 / (4 * O_GAMMA_H2)

    def integrand(a):
        zp = -1 + 1 / a
        denom = a**2 * Ez(zp, params) * np.sqrt(3 * (1 + Rb * a))
        return 1 / denom

    a_lower = 1e-8
    a_upper = 1 / (1 + z)
    I = quad(integrand, a_lower, a_upper)[0]
    return (c / H0) * I


def DA_z(z, params):
    integral = quad(lambda zp: 1 / Ez(zp, params), 0, z)[0]
    return (c / params[0]) * integral / (1 + z)


def cmb_distances(params):
    H0, Om, Ob_h2 = params[0], params[1], params[2]
    Om_h2 = Om * (H0 / 100) ** 2
    zstar = z_star(Ob_h2, Om_h2)
    rs_star = rs_z(zstar, params)
    DA_star = DA_z(zstar, params)
    lA = (1 + zstar) * np.pi * DA_star / rs_star
    R = np.sqrt(Om) * H0 * (1 + zstar) * DA_star / c
    return np.array([R, lA, Ob_h2])


def H_z(z, params):
    return params[0] * Ez(z, params)


def DH_z(z, params):
    return c / H_z(z, params)


def DM_z(z, params):
    return quad(lambda zp: DH_z(zp, params), 0, z)[0]


def DV_z(z, params):
    DH = c / H_z(z, params)
    DM = DM_z(z, params)
    return (z * DH * DM**2) ** (1 / 3)


bao_funcs = {
    "DV_over_rs": lambda z, p: DV_z(z, p),
    "DM_over_rs": lambda z, p: DM_z(z, p),
    "DH_over_rs": lambda z, p: DH_z(z, p),
}


def bao_predictions(z, qty, params):
    H0_50, Om_50, Obh2_50 = params[0], params[1], params[2]
    Omh2_50 = Om_50 * (H0_50 / 100) ** 2
    rd = rs_z(z_drag(Obh2_50, Omh2_50), params)

    return np.array([bao_funcs[q](zi, params) / rd for zi, q in zip(z, qty)])


def chi_squared(params):
    delta = planck_priors - cmb_distances(params)
    chi2_cmb = delta @ inv_cov_mat @ delta

    delta_bao = bao_data["value"] - bao_predictions(
        bao_data["z"], bao_data["quantity"], params
    )
    chi_bao = np.dot(delta_bao, cho_solve(cho_bao, delta_bao))

    return chi2_cmb + chi_bao


bounds = np.array(
    [
        (60, 75),  # H0
        (0.1, 0.6),  # Ωm
        (0.019, 0.025),  # Ωb * h^2
        (-2, 0),  # w0
    ]
)


def log_prior(params):
    if np.all((bounds[:, 0] < params) & (params < bounds[:, 1])):
        return 0
    return -np.inf


def log_likelihood(params):
    return -0.5 * chi_squared(params)


def log_probability(params):
    lp = log_prior(params)
    if np.isinf(lp):
        return -np.inf
    return lp + log_likelihood(params)


def main():
    ndim = len(bounds)
    nwalkers = 16 * ndim
    burn_in = 500
    nsteps = 12000 + burn_in
    initial_pos = np.zeros((nwalkers, ndim))

    for dim, (lower, upper) in enumerate(bounds):
        initial_pos[:, dim] = np.random.uniform(lower, upper, nwalkers)

    with Pool(10) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool=pool)
        sampler.run_mcmc(initial_pos, nsteps, progress=True)

    try:
        tau = sampler.get_autocorr_time()
        print("auto-correlation time", tau)
    except emcee.autocorr.AutocorrError as e:
        print("Autocorrelation time could not be computed", e)

    chains_samples = sampler.get_chain(discard=0, flat=False)
    samples = sampler.get_chain(discard=burn_in, flat=True)

    pct = np.percentile(samples, [15.9, 50, 84.1], axis=0).T
    (H0_16, H0_50, H0_84) = pct[0]
    (Om_16, Om_50, Om_84) = pct[1]
    (Obh2_16, Obh2_50, Obh2_84) = pct[2]
    (w0_16, w0_50, w0_84) = pct[3]

    best_fit = [H0_50, Om_50, Obh2_50, w0_50]

    Omh2_50 = Om_50 * (H0_50 / 100) ** 2
    z_st = z_star(Obh2_50, Omh2_50)
    z_dr = z_drag(Obh2_50, Omh2_50)

    print(f"H0: {H0_50:.2f} +{(H0_84 - H0_50):.2f} -{(H0_50 - H0_16):.2f} km/s/Mpc")
    print(f"Ωm: {Om_50:.3f} +{(Om_84 - Om_50):.3f} -{(Om_50 - Om_16):.3f}")
    print(
        f"Ωb h^2: {Obh2_50:.5f} +{(Obh2_84 - Obh2_50):.5f} -{(Obh2_50 - Obh2_16):.5f}"
    )
    print(f"w0: {w0_50:.3f} +{(w0_84 - w0_50):.3f} -{(w0_50 - w0_16):.3f}")
    print(f"z*: {z_st:.2f}")
    print(f"z_drag: {z_dr:.2f}")
    print(f"r_s(z*) = {rs_z(z_st, best_fit):.4f} Mpc")
    print(f"r_s(z_drag) = {rs_z(z_dr, best_fit):.4f} Mpc")
    print(f"Chi squared: {chi_squared(best_fit):.4f}")

    plot_bao_predictions(
        theory_predictions=lambda z, qty: bao_predictions(z, qty, best_fit),
        data=bao_data,
        errors=np.sqrt(np.diag(bao_cov_matrix)),
        title=bao_legend,
    )
    labels = ["$H_0$", "$Ω_m$", "$Ω_b h^2$", "$w_0$"]
    corner.corner(
        samples,
        labels=labels,
        quantiles=[0.159, 0.5, 0.841],
        show_titles=True,
        title_fmt=".4f",
        bins=100,
        fill_contours=False,
        plot_datapoints=False,
        smooth=1.5,
        smooth1d=1.5,
        levels=(0.393, 0.864),
    )
    plt.show()

    _, axes = plt.subplots(ndim, figsize=(10, 7))
    for i in range(ndim):
        axes[i].plot(chains_samples[:, :, i], color="black", alpha=0.3, lw=0.4)
        axes[i].set_ylabel(labels[i])
        axes[i].set_xlabel("chain step")
        axes[i].axvline(x=burn_in, color="red", linestyle="--", alpha=0.5)
        axes[i].axhline(y=best_fit[i], color="white", linestyle="--", alpha=0.5)
    plt.show()


if __name__ == "__main__":
    main()

"""
Flat ΛCDM w(z) = -1
H0: 68.45 +0.30 -0.30 km/s/Mpc
Ωm: 0.303 +0.004 -0.004
Ωb h^2: 0.02252 +0.00012 -0.00012
w0: -1
z*: 1091.59
z_drag: 1059.68
r_s(z*) = 144.62 Mpc
r_s(z_drag) = 147.46 Mpc
Chi squared: 13.87
Degs of freedom: 11

===============================

Flat wCDM w(z) = w0
H0: 69.15 +0.99 -0.96 km/s/Mpc
Ωm: 0.298 +0.008 -0.008
Ωb h^2: 0.02248 +0.00013 -0.00013
w0: -1.030 +0.040 -0.041
z*: 1091.69
z_drag: 1059.64
r_s(z*) = 144.52 Mpc
r_s(z_drag) = 147.36 Mpc
Chi squared: 13.36
Degs of freedom: 10

===============================

Flat w(z) = -1 + 2 * (1 + w0) / (1 + (1 + z)**3)
H0: 68.55 +1.45 -1.38 km/s/Mpc
Ωm: 0.302 +0.012 -0.012
Ωb h^2: 0.02251 +0.00013 -0.00013
w0: -1.007 +0.089 -0.092
z*: 1091.60
z_drag: 1059.67
r_s(z*) = 144.62 Mpc
r_s(z_drag) = 147.45 Mpc
Chi squared: 13.87
Degs of freedom: 10

===============================

Flat w(z) = w0 + wa * z / (1 + z)
H0: 63.81 +2.11 -1.86 km/s/Mpc
Ωm: 0.354 +0.023 -0.023
Ωb h^2: 0.02235 +0.00014 -0.00014
w0: -0.435 +0.231 -0.234
wa: -1.699 +0.675 -0.699
z*: 1092.02
z_drag: 1059.50
r_s(z*) = 144.1382 Mpc
r_s(z_drag) = 147.0117 Mpc
Chi squared: 6.60
Degs of freedom: 9
"""
