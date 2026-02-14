from numba import njit
import numpy as np
import cmb.data_planck_act_compression as cmb
from y2005cc.data import get_data

c = cmb.c  # Speed of light in km/s
Orh2 = cmb.Or_h2
Onuh2 = cmb.Omnu_h2

legend, z_values, H_values, cov_matrix_cc = get_data()
inv_cov_cc = np.linalg.inv(cov_matrix_cc)
logdet = np.linalg.slogdet(cov_matrix_cc)[1]


@njit
def Ez(z, H0, Obh2, Och2):
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

    return np.sqrt(radiation_term + matter_term + dark_energy_term + neutrino_term)


@njit
def H_z(z, params):
    H0, Obh2, Och2 = params[0:3]
    return H0 * Ez(z, H0, Obh2, Och2)


cmb.set_HZ(H_z)


bounds = np.array(
    [
        (50.0, 85.0),  # H0
        (0.0210, 0.0235),  # Ωb * h^2
        (0.05, 0.30),  # Ωc * h^2
        (0.30, 2.75),  # f_cc
    ]
)


def chi_squared(params):
    delta_cc = H_values - H_z(z_values, params)
    chi2_cc = params[3] ** 2 * delta_cc @ inv_cov_cc @ delta_cc

    delta_cm = cmb.DISTANCE_PRIORS - cmb.cmb_distances(params[1], params[2], params)
    chi2_cmb = delta_cm @ cmb.inv_cov_mat @ delta_cm
    return chi2_cc + chi2_cmb


def log_likelihood(params):
    N = len(z_values)
    normalization = N * np.log(2 * np.pi) + logdet - 2 * N * np.log(params[3])
    return -0.5 * (chi_squared(params) + normalization)


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
    from .plot_predictions import plot_cc_predictions

    ndim = len(bounds)
    nwalkers = 150
    burn_in = 500
    nsteps = 3500 + burn_in
    initial_pos = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(nwalkers, ndim))
    moves = [
        (emcee.moves.KDEMove(bw_method="silverman"), 0.20),
        (emcee.moves.DEMove(), 0.80),
    ]

    with Pool(5) as pool:
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
        [H0_16, H0_50, H0_84],
        [Obh2_16, Obh2_50, Obh2_84],
        [Och2_16, Och2_50, Och2_84],
        [f_16, f_50, f_84],
    ] = np.percentile(samples, [15.9, 50, 84.1], axis=0).T

    best_fit = np.percentile(samples, 50, axis=0)
    degs_of_freedom = len(cmb.DISTANCE_PRIORS) + len(z_values) - len(best_fit)

    Om_samples = (samples[:, 1] + samples[:, 2] + Onuh2) / (samples[:, 0] / 100) ** 2
    Om_16, Om_50, Om_84 = np.percentile(Om_samples, [15.9, 50, 84.1], axis=0)

    print(f"H0: {H0_50:.2f} +{(H0_84 - H0_50):.2f} -{(H0_50 - H0_16):.2f} km/s/Mpc")
    print(f"Ωm: {Om_50:.4f} +{(Om_84 - Om_50):.4f} -{(Om_50 - Om_16):.4f}")
    print(f"ωb: {Obh2_50:.5f} +{(Obh2_84 - Obh2_50):.5f} -{(Obh2_50 - Obh2_16):.5f}")
    print(f"ωc: {Och2_50:.5f} +{(Och2_84 - Och2_50):.5f} -{(Och2_50 - Och2_16):.5f}")
    print(f"fcc: {f_50:.2f} +{(f_84 - f_50):.2f} -{(f_50 - f_16):.2f}")
    print(f"Chi squared: {chi_squared(best_fit):.2f}")
    print(f"Log likelihood: {log_likelihood(best_fit):.2f}")
    print(f"Log evidence: {log_evd:.2f}")
    print(f"Degs of freedom: {degs_of_freedom}")

    labels = ["$H_0$", "$ω_b$", "$ω_c$", "$f_{CC}$"]
    plot_corner_and_chains(labels=labels, flat_samples=samples, samples=chains_samples)
    plot_cc_predictions(
        H_z=lambda z: H_z(z, best_fit),
        z=z_values,
        H=H_values,
        H_err=np.sqrt(np.diag(cov_matrix_cc)) / f_50,
        label=f"{legend} $H_0$: {H0_50:.2f} ± {(H0_84 - H0_50):.2f} km/s/Mpc",
    )


if __name__ == "__main__":
    main()

"""
*******************************
Results for data from
https://arxiv.org/pdf/2307.09501
and 4 data points from
https://arxiv.org/pdf/2506.03836
https://arxiv.org/pdf/2511.02730v1
*******************************
"""

"""
Flat ΛCDM

-------------------------------

f: 1.50 +0.19 -0.18
H0: 67.59 +0.49 -0.49
Ωm: 0.3120 +0.0071 -0.0069
ωb: 0.02249 +0.00011 -0.00011
ωc: 0.11939 +0.00121 -0.00120
f: 1.51 +0.18 -0.17
Chi squared: 36.36
Log likelihood: -141.72
Log evidence: -158.19 (Δ logZ = 3.79 compared to fixed f)
Degs of freedom: 35

-------------------------------

f: 1.0 (fixed - assuming no overestimaded errors in CCH sample)
H0: 67.61 +0.50 -0.50
Ωm: 0.3117 +0.0072 -0.0070
ωb: 0.02250 +0.00011 -0.00011
ωc: 0.11934 +0.00122 -0.00120
Chi squared: 16.00
Log likelihood: -146.32
Log evidence: -161.98
Degs of freedom: 36
"""
