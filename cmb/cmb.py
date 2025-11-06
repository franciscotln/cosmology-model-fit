from numba import njit
import numpy as np
import cmb.data_desi_compression as cmb

Or_h2 = cmb.Omega_r_h2()


@njit
def Ez(z, params):
    h, Om = params[0] / 100, params[1]
    Or = Or_h2 / h**2
    Ode = 1 - Om - Or
    one_plus_z = 1 + z

    return np.sqrt(Or * one_plus_z**4 + Om * one_plus_z**3 + Ode)


def chi_squared(params):
    H0, Om, Ob_h2 = params
    delta = cmb.DISTANCE_PRIORS - cmb.cmb_distances(Ez, params, H0, Om, Ob_h2)
    return np.dot(delta, np.dot(cmb.inv_cov_mat, delta))


bounds = np.array(
    [
        (60, 70),  # H0
        (0.15, 0.45),  # Ωm
        (0.020, 0.024),  # Ωb * h^2
    ],
    dtype=np.float64,
)

normalization = -np.sum(np.log(bounds[:, 1] - bounds[:, 0]))


@njit
def log_prior(params):
    if np.all((bounds[:, 0] < params) & (params < bounds[:, 1])):
        return normalization
    return -np.inf


def log_likelihood(params):
    return -0.5 * chi_squared(params)


def log_probability(params):
    lp = log_prior(params)
    if np.isinf(lp):
        return -np.inf
    return lp + log_likelihood(params)


def main():
    import emcee
    from multiprocessing import Pool
    from corner_plot import plot_corner_and_chains

    ndim = len(bounds)
    nwalkers = 200
    burn_in = 200
    nsteps = 2000 + burn_in
    np.random.seed(42)
    initial_pos = np.random.uniform(bounds[:, 0], bounds[:, 1], (nwalkers, ndim))
    moves = [
        (emcee.moves.KDEMove(), 0.30),
        (emcee.moves.DEMove(), 0.56),
        (emcee.moves.DESnookerMove(), 0.14),
    ]

    with Pool(5) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool, moves)
        sampler.run_mcmc(initial_pos, nsteps, progress=True)

    try:
        tau = sampler.get_autocorr_time()
        print("auto-correlation time", tau)
        print("acceptance fraction", np.mean(sampler.acceptance_fraction))
        print("effective samples", ndim * nwalkers * (nsteps - burn_in) / np.max(tau))
    except emcee.autocorr.AutocorrError as e:
        print("Autocorrelation time could not be computed", e)

    samples = sampler.get_chain(discard=burn_in, flat=True)
    chains_samples = sampler.get_chain(discard=burn_in, flat=False)

    one_sigma_percentiles = [15.9, 50, 84.1]
    pct = np.percentile(samples, one_sigma_percentiles, axis=0).T
    [
        (H0_16, H0_50, H0_84),
        (Om_16, Om_50, Om_84),
        (Obh2_16, Obh2_50, Obh2_84),
    ] = pct

    best_fit = np.array([H0_50, Om_50, Obh2_50], dtype=np.float64)

    h_samples = samples[:, 0] / 100
    Omh2_samples = samples[:, 1] * h_samples**2
    z_eq_samples = -1 + Omh2_samples / Or_h2
    z_st_samples = cmb.z_star(samples[:, 2], Omh2_samples)
    z_dr_samples = cmb.z_drag(samples[:, 2], Omh2_samples)
    Omh2_16, Omh2_50, Omh2_84 = np.percentile(Omh2_samples, one_sigma_percentiles)
    z_eq_16, z_eq_50, z_eq_84 = np.percentile(z_eq_samples, one_sigma_percentiles)
    z_st_16, z_st_50, z_st_84 = np.percentile(z_st_samples, one_sigma_percentiles)
    z_d_16, z_d_50, z_d_84 = np.percentile(z_dr_samples, one_sigma_percentiles)
    rd_samples = cmb.r_drag(wb=samples[:, 2], wm=Omh2_samples)
    rd_16, rd_50, rd_84 = np.percentile(rd_samples, one_sigma_percentiles)

    print(f"H0: {H0_50:.2f} +{(H0_84 - H0_50):.2f} -{(H0_50 - H0_16):.2f} km/s/Mpc")
    print(f"Ωm: {Om_50:.4f} +{(Om_84 - Om_50):.4f} -{(Om_50 - Om_16):.4f}")
    print(f"ωm: {Omh2_50:.4f} +{(Omh2_84 - Omh2_50):.4f} -{(Omh2_50 - Omh2_16):.4f}")
    print(f"ωb: {Obh2_50:.5f} +{(Obh2_84 - Obh2_50):.5f} -{(Obh2_50 - Obh2_16):.5f}")
    print(f"z_eq: {z_eq_50:.1f} +{(z_eq_84 - z_eq_50):.1f} -{(z_eq_50 - z_eq_16):.1f}")
    print(f"z*: {z_st_50:.2f} +{(z_st_84 - z_st_50):.2f} -{(z_st_50 - z_st_16):.2f}")
    print(f"z_drag: {z_d_50:.2f} +{(z_d_84 - z_d_50):.2f} -{(z_d_50 - z_d_16):.2f}")
    print(f"r*: {cmb.rs_z(Ez, z_st_50, best_fit, H0_50, Obh2_50):.2f} Mpc")
    print(f"r_d: {rd_50:.2f} +{(rd_84 - rd_50):.2f} -{(rd_50 - rd_16):.2f} Mpc")
    print(f"Chi squared: {chi_squared(best_fit):.4f}")

    plot_corner_and_chains(
        labels=["$H_0$", "$Ω_m$", "$ω_b$"],
        flat_samples=samples,
        samples=chains_samples,
    )


if __name__ == "__main__":
    main()

"""
Flat ΛCDM w(z) = -1 

===============================

Chen+2018 compression
H0: 67.41 +0.61 -0.60 km/s/Mpc
Ωm: 0.3167 +0.0085 -0.0082
ωm: 0.1439 +0.0013 -0.0013
ωb: 0.02236 +0.00015 -0.00015
z_eq: 3438.3 +30.2 -30.1
z*: 1088.92 +0.22 -0.22
z_drag: 1059.93 +0.29 -0.29
r*: 144.16 Mpc
r_d: 146.72 +0.29 -0.29 Mpc
Chi squared: 0.0004

===============================

Prakhar Bansal+ (Planck + ACT) compression
H0: 67.25 +0.50 -0.50 km/s/Mpc
Ωm: 0.3170 +0.0072 -0.0070
ωm: 0.1433 +0.0011 -0.0011
ωb: 0.02237 +0.00014 -0.00014
z_eq: 3426.1 +27.3 -26.7
z*: 1088.87 +0.21 -0.20
z_drag: 1059.92 +0.28 -0.29
r*: 144.30 Mpc
r_d: 146.86 Mpc
Chi squared: 0.0002

===============================

Early ΛCDM (arXiv:2302.12911v2)
H0: 67.48 +0.59 -0.58 km/s/Mpc
Ωm: 0.3121 +0.0080 -0.0079
ωm: 0.1421 +0.0012 -0.0012
ωb: 0.02223 +0.00014 -0.00014
z_eq: 3396.8 +28.5 -28.8
z*: 1088.86 +0.22 -0.21
z_drag: 1057.92 +0.28 -0.29
r*: 144.70 Mpc
r_d: 147.46 +0.25 -0.24 Mpc
Chi squared: 0.0005

===============================

Rubin+ Union3 compression
H0: 67.39 +0.60 -0.60 km/s/Mpc
Ωm: 0.3150 +0.0084 -0.0081
ωm: 0.1430 +0.0013 -0.0013
ωb: 0.02239 +0.00014 -0.00015
z_eq: 3420.2 +30.3 -30.2
z*: 1091.88 +0.28 -0.28
z_drag: 1059.94 +0.29 -0.29
r*: 144.13 Mpc
r_d: 146.95 Mpc
Chi squared: 0.0017
"""
