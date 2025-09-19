import numpy as np
import emcee
import corner
from scipy.integrate import quad
import matplotlib.pyplot as plt
from multiprocessing import Pool


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


def Ez(z, params):
    H0, Om = params[0], params[1]
    h = H0 / 100
    z_eq = 2.5 * 10**4 * Om * h**2 * (2.7 / TCMB) ** 4
    Or = Om / (1 + z_eq)
    Ode = 1 - Om - Or
    one_plus_z = 1 + z

    return np.sqrt(Or * one_plus_z**4 + Om * one_plus_z**3 + Ode)


def z_star(Ob_h2, Om_h2):
    # Wayne Hu, Naoshi Sugiyama use 1345 (arXiv:astro-ph/9510117v2 equation E-1)
    g1 = 0.0783 * Ob_h2**-0.238 / (1 + 39.5 * Ob_h2**0.763)
    g2 = 0.560 / (1 + 21.1 * Ob_h2**1.81)
    return 1048 * (1 + 0.00124 * Ob_h2**-0.738) * (1 + g1 * Om_h2**g2)


def z_drag(Ob_h2, Om_h2):
    b1 = 0.313 * Om_h2**-0.419 * (1 + 0.607 * Om_h2**0.674)
    b2 = 0.238 * Om_h2**0.223
    # Calibrated 1340 to reproduce Planck 2018 r_drag
    # Wayne Hu, Naoshi Sugiyama use 1345 (arXiv:astro-ph/9510117v2 equation E-2)
    # Daniel J. Eisenstein, Wayne Hu use 1291 (arXiv:astro-ph/9709112v1 equation 4)
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


def chi_squared(params):
    delta = planck_priors - cmb_distances(params)
    return delta @ inv_cov_mat @ delta


bounds = np.array(
    [
        (60, 75),  # H0
        (0.1, 0.6),  # Ωm
        (0.019, 0.025),  # Ωb * h^2
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
    (H0_16, H0_50, H0_84) = pct[0]
    (Om_16, Om_50, Om_84) = pct[1]
    (Obh2_16, Obh2_50, Obh2_84) = pct[2]

    best_fit = [H0_50, Om_50, Obh2_50]

    Omh2_50 = Om_50 * (H0_50 / 100) ** 2
    z_st = z_star(Obh2_50, Omh2_50)
    z_dr = z_drag(Obh2_50, Omh2_50)

    print(f"H0: {H0_50:.2f} +{(H0_84 - H0_50):.2f} -{(H0_50 - H0_16):.2f} km/s/Mpc")
    print(f"Ωm: {Om_50:.3f} +{(Om_84 - Om_50):.3f} -{(Om_50 - Om_16):.3f}")
    print(
        f"Ωb h^2: {Obh2_50:.5f} +{(Obh2_84 - Obh2_50):.5f} -{(Obh2_50 - Obh2_16):.5f}"
    )
    print(f"z*: {z_st:.2f}")
    print(f"z_drag: {z_dr:.2f}")
    print(f"r_s(z*) = {rs_z(z_st, best_fit):.4f} Mpc")
    print(f"r_s(z_drag) = {rs_z(z_dr, best_fit):.4f} Mpc")
    print(f"Chi squared: {chi_squared(best_fit):.4f}")

    labels = ["$H_0$", "$Ω_m$", "$Ω_b h^2$"]
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
H0: 67.44 +0.61 -0.61 km/s/Mpc
Ωm: 0.316 +0.009 -0.008
Ωb h^2: 0.02236 +0.00015 -0.00015
z*: 1092.00
r_s(*) = 144.16 Mpc
z_drag: 1059.50
r_s(drag) = 147.03 Mpc
Chi squared: 0.0014
"""
