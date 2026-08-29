from numba import njit
import numpy as np
from scipy.linalg import cho_factor
import cmb.data_planck_act_compression as cmb
from solve_triangular import solve_triangular
from y2005cc.data import get_data

c = cmb.c  # Speed of light in km/s
Orh2 = cmb.Or_h2
Onuh2 = cmb.Omnu_h2

legend, z_values, H_values, cov_matrix_cc = get_data()
L_cc = cho_factor(cov_matrix_cc, lower=True)[0]
logdet_base = 2.0 * np.log(np.diag(L_cc)).sum()
N = len(z_values)


@njit
def H_z(z, params):
    H0, Obh2, Och2 = params[0], params[1], params[2]
    h = H0 / 100
    Obc = (Obh2 + Och2) / h**2
    Onu = Onuh2 / h**2
    Or = Orh2 / h**2
    Ode = 1.0 - Obc - Or - Onu

    zp1 = 1.0 + z

    radiation_term = Or * zp1**4
    matter_term = Obc * zp1**3
    neutrino_term = Onu * cmb.Omnu_z(z)
    dark_energy_term = Ode

    return H0 * np.sqrt(radiation_term + matter_term + dark_energy_term + neutrino_term)


cmb.set_HZ(H_z)


bounds = np.array(
    [
        (63.0, 73.0),  # H0
        (0.0210, 0.0235),  # Ωb * h^2
        (0.05, 0.30),  # Ωc * h^2
        (0.5, 4.0),  # f0
        (-2.0, 2.0),  # fa
    ]
)


@njit
def chi_squared(params, f_array):
    delta_cc = H_values - H_z(z_values, params)
    chi2_cc = solve_triangular(L_cc, f_array * delta_cc)

    delta_cm = cmb.DISTANCE_PRIORS - cmb.cmb_distances(params[1], params[2], params)
    chi2_cmb = delta_cm @ cmb.inv_cov_mat @ delta_cm
    return chi2_cc + chi2_cmb


@njit
def log_likelihood(params):
    f_array = params[3] + params[4] * z_values
    if np.any(f_array <= 1e-4):
        return -np.inf

    logdet = logdet_base - 2.0 * np.log(f_array).sum()
    normalization = N * np.log(2 * np.pi) + logdet
    return -0.5 * (chi_squared(params, f_array) + normalization)


@njit
def log_prior(params):
    if np.all((bounds[:, 0] < params) & (params < bounds[:, 1])):
        return 0.0
    return -np.inf


def log_probability(params):
    lp = log_prior(params)
    if np.isinf(lp):
        return -np.inf
    return lp + log_likelihood(params)


def main():
    import emcee
    from multiprocessing import Pool
    from log_evidence import log_evidence
    from corner_plot import plot_corner_and_chains
    from ohd.plot_predictions import plot_cc_predictions

    ndim = len(bounds)
    nwalkers = 150
    burn_in = 500
    nsteps = 5000 + burn_in
    initial_pos = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(nwalkers, ndim))
    moves = [
        (emcee.moves.KDEMove(bw_method="silverman"), 0.20),
        (emcee.moves.DEMove(), 0.80),
    ]

    with Pool(6) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool, moves)
        sampler.run_mcmc(
            initial_pos, nsteps, progress=True, progress_kwargs={"colour": "#ff5a00"}
        )

    try:
        tau = sampler.get_autocorr_time()
        print("auto-correlation time", tau)
        print("acceptance fraction", np.mean(sampler.acceptance_fraction))
        print("effective samples", nwalkers * ndim * (nsteps - burn_in) / np.max(tau))
    except emcee.autocorr.AutocorrError as e:
        print("Autocorrelation time could not be computed", e)

    samples = sampler.get_chain(discard=burn_in, flat=True)
    chains_samples = sampler.get_chain(discard=burn_in, flat=False)
    log_probs = sampler.get_log_prob(discard=burn_in, flat=True)
    log_evd = log_evidence(samples, log_probs, log_probability, bounds)

    [
        (H0_16, H0_50, H0_84),
        (Obh2_16, Obh2_50, Obh2_84),
        (Och2_16, Och2_50, Och2_84),
        (f0_16, f0_50, f0_84),
        (fa_16, fa_50, fa_84),
    ] = np.percentile(samples, [15.9, 50, 84.1], axis=0).T

    best_fit = samples[np.argmax(log_probs)]
    DOF = len(cmb.DISTANCE_PRIORS) + N - len(best_fit)

    Om_samples = (samples[:, 1] + samples[:, 2] + Onuh2) / (samples[:, 0] / 100) ** 2
    Om_16, Om_50, Om_84 = np.percentile(Om_samples, [15.9, 50, 84.1], axis=0)
    f_array = best_fit[3] + best_fit[4] * z_values

    print(f"H0: {H0_50:.2f} +{(H0_84 - H0_50):.2f} -{(H0_50 - H0_16):.2f} km/s/Mpc")
    print(f"Ωm: {Om_50:.4f} +{(Om_84 - Om_50):.4f} -{(Om_50 - Om_16):.4f}")
    print(f"ωb: {Obh2_50:.5f} +{(Obh2_84 - Obh2_50):.5f} -{(Obh2_50 - Obh2_16):.5f}")
    print(f"ωc: {Och2_50:.5f} +{(Och2_84 - Och2_50):.5f} -{(Och2_50 - Och2_16):.5f}")
    print(f"f0: {f0_50:.2f} +{(f0_84 - f0_50):.2f} -{(f0_50 - f0_16):.2f}")
    print(f"fa: {fa_50:.2f} +{(fa_84 - fa_50):.2f} -{(fa_50 - fa_16):.2f}")
    print(f"Chi squared: {chi_squared(best_fit, f_array):.2f}")
    print(f"Log likelihood: {log_likelihood(best_fit):.2f}")
    print(f"Log evidence: {log_evd:.2f}")
    print(f"Degs of freedom: {DOF}")

    labels = ["$H_0$", "$ω_b$", "$ω_c$", "$f_0$", "$f_a$"]
    plot_corner_and_chains(labels=labels, flat_samples=samples, samples=chains_samples)
    plot_cc_predictions(
        H_z=lambda z: H_z(z, best_fit),
        z=z_values,
        H=H_values,
        H_err=np.sqrt(np.diag(cov_matrix_cc)) / f_array,
        label=f"{legend} $H_0$: {H0_50:.2f} ± {(H0_84 - H0_50):.2f} km/s/Mpc",
    )


if __name__ == "__main__":
    main()


# *******************************************
# CMB (ACT+Planck) + Cosmic Chronometers (CC)
# Model: Flat ΛCDM
# *******************************************


# ------ Overestimation factor f(z) = f0 + fa * z -----
# H0: 67.62 +0.49 -0.49 km/s/Mpc
# Ωm: 0.3116 +0.0070 -0.0069
# ωb: 0.02250 +0.00011 -0.00011
# ωc: 0.11931 +0.00120 -0.00119
# f0: 2.30 +0.36 -0.35
# fa: -0.84 +0.31 -0.29
# Chi squared: 38.51
# Log likelihood: -145.73 (3.74 sigma significance)
# Log evidence: -162.47 (Δ logZ = 7.53 compared to no scaling)
# Degs of freedom: 36


# ------------ Fixed factor f0 = 1, fa = 0 ------------
# f0: 1, fa: 0 (assuming no overestimaded errors in CCH sample)
# H0: 67.60 +0.50 -0.50 km/s/Mpc
# Ωm: 0.3118 +0.0072 -0.0070
# ωb: 0.02249 +0.00011 -0.00011
# ωc: 0.11935 +0.00121 -0.00121
# Chi squared: 16.45
# Log likelihood: -154.34
# Log evidence: -170.00
# Degs of freedom: 38
# """
