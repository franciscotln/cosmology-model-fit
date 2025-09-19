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
N_EFF = 3.046
TCMB = 2.7255  # K
O_GAMMA_H2 = 2.4728e-5 * (TCMB / 2.7255) ** 4


bao_legend, bao_data, bao_cov_matrix = get_bao_data()
cho_bao = cho_factor(bao_cov_matrix)


def Omega_r_h2(Neff=N_EFF):
    return O_GAMMA_H2 * (1 + 0.2271 * Neff)


def Ez(z, params):
    H0, Om, w0 = params[0], params[1], params[3]
    h = H0 / 100
    Or = Omega_r_h2() / h**2
    Ode = 1 - Om - Or

    one_plus_z = 1 + z
    cubed = one_plus_z**3
    rho_de = (2 * cubed / (1 + cubed)) ** (2 * (1 + w0))

    return np.sqrt(Or * one_plus_z**4 + Om * cubed + Ode * rho_de)


def z_star(wb, wm):
    # arXiv:2106.00428v2
    return wm**-0.731631 + (
        (391.672 * wm**-0.372296 + 937.422 * wb**-0.97966) * wm**0.0192951 * wb**0.93681
    )


def z_drag(wb, wm):
    # arXiv:2106.00428v2
    return (
        1 + 428.169 * wb**0.256459 * wm**0.616388 + 925.56 * wm**0.751615
    ) * wm**-0.714129


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
    "DV_over_rs": DV_z,
    "DM_over_rs": DM_z,
    "DH_over_rs": DH_z,
}


def bao_predictions(z, qty, params):
    H0, Om, Obh2 = params[0], params[1], params[2]
    Omh2 = Om * (H0 / 100) ** 2
    rd = rs_z(z_drag(Obh2, Omh2), params)

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
    print(f"r_s(z*) = {rs_z(z_st, best_fit):.2f} Mpc")
    print(f"r_s(z_drag) = {rs_z(z_dr, best_fit):.2f} Mpc")
    print(f"Chi squared: {chi_squared(best_fit):.2f}")

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
H0: 68.62 ± 0.30 km/s/Mpc
Ωm: 0.300 ± 0.004
Ωb h^2: 0.02254 ± 0.00012
w0: -1
z*: 1088.55
z_drag: 1060.17
r_s(z*) = 144.68 Mpc
r_s(z_drag) = 147.21 Mpc
Chi squared: 15.26
Degs of freedom: 14

===============================

Flat wCDM w(z) = w0
H0: 69.27 +0.99 -0.95 km/s/Mpc
Ωm: 0.296 ± 0.008
Ωb h^2: 0.02251 ± 0.00013
w0: -1.028 +0.039 -0.041
z*: 1088.62
z_drag: 1060.12
r_s(z*) = 144.59 Mpc
r_s(z_drag) = 147.12 Mpc
Chi squared: 14.81 (Δ chi2 0.45)
Degs of freedom: 13

===============================

Flat w(z) = -1 + 2 * (1 + w0) / (1 + (1 + z)**3)
H0: 68.63 +1.45 -1.37 km/s/Mpc
Ωm: 0.300 ± 0.012
Ωb h^2: 0.02254 ± 0.00013
w0: -1.000 +0.088 -0.091
z*: 1088.55
z_drag: 1060.17
r_s(z*) = 144.68 Mpc
r_s(z_drag) = 147.21 Mpc
Chi squared: 15.26 (Δ chi2 0.00)
Degs of freedom: 13

===============================

Flat w(z) = w0 + wa * z / (1 + z)
H0: 63.81 +2.01 -1.78 km/s/Mpc
Ωm: 0.353 +0.021 -0.022
Ωb h^2: 0.02238 +0.00014 -0.00014
w0: -0.423 +0.217 -0.223
wa: -1.723 +0.644 -0.654
z*: 1088.87
z_drag: 1059.96
r_s(z*) = 144.22 Mpc
r_s(z_drag) = 146.78 Mpc
Chi squared: 7.24 (Δ chi2 8.02)
Degs of freedom: 12
"""
