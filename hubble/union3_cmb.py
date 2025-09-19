import numpy as np
import emcee
import corner
from scipy.integrate import cumulative_trapezoid, quad
from scipy.linalg import cho_factor, cho_solve
import matplotlib.pyplot as plt
from multiprocessing import Pool
from y2023union3.data import get_data
from .plotting import plot_predictions


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

sn_legend, z_sn_vals, mu_vals, cov_matrix_sn = get_data()
cho_sn = cho_factor(cov_matrix_sn)

sn_grid = np.linspace(0, np.max(z_sn_vals), num=3000)


def Ez(z, params):
    H0, Om, w0 = params[0], params[1], params[3]
    h = H0 / 100
    z_eq = 2.5 * 10**4 * Om * h**2 * (2.7 / TCMB) ** 4
    Or = Om / (1 + z_eq)
    Ode = 1 - Om - Or

    one_plus_z = 1 + z
    rho_de = (2 * one_plus_z**3 / (1 + one_plus_z**3)) ** (2 * (1 + w0))

    return np.sqrt(Or * one_plus_z**4 + Om * one_plus_z**3 + Ode * rho_de)


def integral_Ez(params):
    integral_values = cumulative_trapezoid(1 / Ez(sn_grid, params), sn_grid, initial=0)
    return np.interp(z_sn_vals, sn_grid, integral_values)


def distance_modulus(params):
    H0, offset_mag = params[0], params[-1]
    dL = (1 + z_sn_vals) * integral_Ez(params) * c / H0
    return offset_mag + 25 + 5 * np.log10(dL)


def z_star(Ob_h2, Om_h2):
    # Wayne Hu, Naoshi Sugiyama (arXiv:astro-ph/9510117v2 equation E-1)
    g1 = 0.0783 * Ob_h2**-0.238 / (1 + 39.5 * Ob_h2**0.763)
    g2 = 0.560 / (1 + 21.1 * Ob_h2**1.81)
    return 1048 * (1 + 0.00124 * Ob_h2**-0.738) * (1 + g1 * Om_h2**g2)


def z_drag(Ob_h2, Om_h2):
    # Calibrated 1340 to reproduce Planck 2018 r_drag
    # Wayne Hu, Naoshi Sugiyama use 1345 (arXiv:astro-ph/9510117v2 equation E-2)
    # Daniel J. Eisenstein, Wayne Hu use 1291 (arXiv:astro-ph/9709112v1 equation 4)
    b1 = 0.313 * Om_h2**-0.419 * (1 + 0.607 * Om_h2**0.674)
    b2 = 0.238 * Om_h2**0.223
    return (1340 * Om_h2**0.251 / (1 + 0.659 * Om_h2**0.828)) * (1 + b1 * Ob_h2**b2)


def rs_z(z, params):
    H0, Ob_h2 = params[0], params[2]
    Rb = 3 * Ob_h2 / (4 * O_GAMMA_H2)

    def integrand(a):
        zp = -1 + 1 / a
        denominator = a**2 * Ez(zp, params) * np.sqrt(3 * (1 + Rb * a))
        return 1 / denominator

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

    lA = np.pi * (1 + zstar) * DA_star / rs_star
    R = np.sqrt(Om) * H0 * (1 + zstar) * DA_star / c
    return np.array([R, lA, Ob_h2])


def chi_squared(params):
    delta = planck_priors - cmb_distances(params)
    chi2_cmb = delta @ inv_cov_mat @ delta

    delta_sn = mu_vals - distance_modulus(params)
    chi_sn = np.dot(delta_sn, cho_solve(cho_sn, delta_sn))

    return chi2_cmb + chi_sn


bounds = np.array(
    [
        (60, 75),  # H0
        (0.1, 0.45),  # Ωm
        (0.019, 0.025),  # Ωb * h^2
        (-1.5, 0),  # w0
        (-0.7, 0.7),  # ΔM
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
    nwalkers = 20 * ndim
    burn_in = 500
    nsteps = 10000 + burn_in
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
    H0_16, H0_50, H0_84 = pct[0]
    Om_16, Om_50, Om_84 = pct[1]
    Obh2_16, Obh2_50, Obh2_84 = pct[2]
    w0_16, w0_50, w0_84 = pct[3]
    dM_16, dM_50, dM_84 = pct[4]

    best_fit = [H0_50, Om_50, Obh2_50, w0_50, dM_50]

    Omh2_50 = Om_50 * (H0_50 / 100) ** 2
    z_st = z_star(Obh2_50, Omh2_50)
    z_dr = z_drag(Obh2_50, Omh2_50)

    print(f"H0: {H0_50:.2f} +{(H0_84 - H0_50):.2f} -{(H0_50 - H0_16):.2f} km/s/Mpc")
    print(f"Ωm: {Om_50:.3f} +{(Om_84 - Om_50):.3f} -{(Om_50 - Om_16):.3f}")
    print(
        f"Ωb h^2: {Obh2_50:.5f} +{(Obh2_84 - Obh2_50):.5f} -{(Obh2_50 - Obh2_16):.5f}"
    )
    print(f"w0: {w0_50:.3f} +{(w0_84 - w0_50):.3f} -{(w0_50 - w0_16):.3f}")
    print(f"ΔM: {dM_50:.3f} +{(dM_84 - dM_50):.3f} -{(dM_50 - dM_16):.3f}")
    print(f"z*: {z_st:.2f}")
    print(f"z_drag: {z_dr:.2f}")
    print(f"r_s(z*) = {rs_z(z_st, best_fit):.2f} Mpc")
    print(f"r_s(z_drag) = {rs_z(z_dr, best_fit):.2f} Mpc")
    print(f"Chi squared: {chi_squared(best_fit):.2f}")

    plot_predictions(
        legend=sn_legend,
        x=z_sn_vals,
        y=mu_vals,
        y_err=np.sqrt(np.diag(cov_matrix_sn)),
        y_model=distance_modulus(best_fit),
        label=f"Best fit: $Ω_m$={Om_50:.3f}",
        x_scale="log",
    )
    labels = ["$H_0$", "$Ω_m$", "$Ω_b h^2$", "$w_0$", "$Δ_M$"]
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
*******************************
Dataset: Union 3 Bins
z range: 0.050 - 2.262
Sample size: 22
*******************************

Flat ΛCDM w(z) = -1
H0: 67.15 ± 0.58 km/s/Mpc
Ωm: 0.320 ± 0.008
Ωb h^2: 0.02231 ± 0.00015
w0: -1
ΔM: -0.165 ± 0.090
z*: 1092.12
z_drag: 1059.44
r_s(z*) = 144.04 Mpc
r_s(z_drag) = 146.92 Mpc
Chi squared: 26.01
Degrees of freedom: 21

===============================

Flat wCDM w(z) = w0
H0: 65.31 ± 1.22 km/s/Mpc
Ωm: 0.337 +0.014 -0.013
Ωb h^2: 0.02236 ± 0.00015
w0: -0.927 ± 0.043
ΔM: -0.217 ± 0.095
z*: 1091.99
z_drag: 1059.51
r_s(z*) = 144.16 Mpc
r_s(z_drag) = 147.03 Mpc
Chi squared: 23.18
Degrees of freedom: 20

===============================

Flat w(z) = -1 + 2 * (1 + w0) / (1 + (1 + z)^3)
H0: 65.41 +1.10 -1.08 km/s/Mpc
Ωm: 0.336 ± 0.012
Ωb h^2: 0.02237 ± 0.00015
w0: -0.877 +0.068 -0.067
ΔM: -0.209 +0.092 -0.093
z*: 1091.98
z_drag: 1059.52
r_s(z*) = 144.17 Mpc
r_s(z_drag) = 147.04 Mpc
Chi squared: 22.57
Degrees of freedom: 20

===============================

Flat w0waCDM w(z) = w0 + wa * z / (1 + z)
H0: 66.64 +1.32 -1.39 km/s/Mpc
Ωm: 0.324 +0.015 -0.013
Ωb h^2: 0.02235 ±0.00015
w0: -0.687 ± 0.160
wa: -1.131 +0.740 -0.770
ΔM: -0.154 +0.100 -0.102
z*: 1092.02
z_drag: 1059.50
r_s(z*) = 144.13 Mpc
r_s(z_drag) = 147.01 Mpc
Chi squared: 21.53
Degrees of freedom: 20
"""
