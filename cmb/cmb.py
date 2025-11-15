from numba import njit
import numpy as np
import cmb.data_desi_compression as cmb

Or_h2 = cmb.Omega_r_h2()


@njit
def Ez(z, Obc, Or, w0=-1, wa=0):
    Ode = 1 - Obc - Or
    one_plus_z = 1 + z
    return np.sqrt(Or * one_plus_z**4 + Obc * one_plus_z**3 + Ode)


def chi_squared(params):
    delta = cmb.DISTANCE_PRIORS - cmb.cmb_distances(Ez, *params)
    return np.dot(delta, np.dot(cmb.inv_cov_mat, delta))


bounds = np.array(
    [
        (55, 75),  # H0
        (0.018, 0.025),  # Ωb * h^2
        (0.05, 0.25),  # Ωc * h^2
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
        (emcee.moves.DEMove(), 0.70),
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
        (Obh2_16, Obh2_50, Obh2_84),
        (Och2_16, Och2_50, Och2_84),
    ] = pct

    best_fit = np.percentile(samples, 50, axis=0)

    h_samples = samples[:, 0] / 100
    Omh2_samples = samples[:, 1] + samples[:, 2] + cmb.Omnu_h2
    Om_samples = Omh2_samples / h_samples**2
    z_eq_samples = -1 + Omh2_samples / Or_h2
    z_st_samples = cmb.z_star(samples[:, 1], Omh2_samples)
    z_dr_samples = cmb.z_drag(samples[:, 1], Omh2_samples)
    Om_16, Om_50, Om_84 = np.percentile(Om_samples, one_sigma_percentiles)
    Omh2_16, Omh2_50, Omh2_84 = np.percentile(Omh2_samples, one_sigma_percentiles)
    z_eq_16, z_eq_50, z_eq_84 = np.percentile(z_eq_samples, one_sigma_percentiles)
    z_st_16, z_st_50, z_st_84 = np.percentile(z_st_samples, one_sigma_percentiles)
    z_d_16, z_d_50, z_d_84 = np.percentile(z_dr_samples, one_sigma_percentiles)
    rd_samples = cmb.r_drag(wb=samples[:, 1], wm=Omh2_samples)
    rd_16, rd_50, rd_84 = np.percentile(rd_samples, one_sigma_percentiles)

    print(f"H0: {H0_50:.2f} +{(H0_84 - H0_50):.2f} -{(H0_50 - H0_16):.2f} km/s/Mpc")
    print(f"ωc: {Och2_50:.4f} +{(Och2_84 - Och2_50):.4f} -{(Och2_50 - Och2_16):.4f}")
    print(f"ωb: {Obh2_50:.5f} +{(Obh2_84 - Obh2_50):.5f} -{(Obh2_50 - Obh2_16):.5f}")
    print(f"ωm: {Omh2_50:.5f} +{(Omh2_84 - Omh2_50):.5f} -{(Omh2_50 - Omh2_16):.5f}")
    print(f"Ωm: {Om_50:.4f} +{(Om_84 - Om_50):.4f} -{(Om_50 - Om_16):.4f}")
    print(f"z_eq: {z_eq_50:.1f} +{(z_eq_84 - z_eq_50):.1f} -{(z_eq_50 - z_eq_16):.1f}")
    print(f"z*: {z_st_50:.2f} +{(z_st_84 - z_st_50):.2f} -{(z_st_50 - z_st_16):.2f}")
    print(f"z_drag: {z_d_50:.2f} +{(z_d_84 - z_d_50):.2f} -{(z_d_50 - z_d_16):.2f}")
    print(f"r*: {cmb.rs_z(Ez, z_st_50, H0_50, Obh2_50, Och2_50):.2f} Mpc")
    print(f"r_d: {rd_50:.2f} +{(rd_84 - rd_50):.2f} -{(rd_50 - rd_16):.2f} Mpc")
    print(f"Chi squared: {chi_squared(best_fit):.4f}")

    plot_corner_and_chains(
        labels=["$H_0$", "$ω_b$", "$ω_c$"],
        flat_samples=samples,
        samples=chains_samples,
    )


if __name__ == "__main__":
    main()

"""
Flat ΛCDM w(z) = -1 

*******************************

Chen+2018 compression (Planck 2018)
H0: 67.39 +0.61 -0.61 km/s/Mpc
ωc: 0.1203 +0.0014 -0.0014
ωb: 0.02236 +0.00015 -0.00015
ωm: 0.14327 +0.00130 -0.00129
Ωm: 0.3155 +0.0086 -0.0084
z_eq: 3423.5 +31.1 -30.9
z*: 1089.11 +0.29 -0.29
z_drag: 1059.89 +0.30 -0.30
r*: 144.46 Mpc
r_d: 146.89 +0.30 -0.30 Mpc
Chi squared: 0.0001

===============================

Early ΛCDM (arXiv:2302.12911v2)
H0: 67.48 +0.59 -0.57 km/s/Mpc
ωc: 0.1192 +0.0013 -0.0013
ωb: 0.02223 +0.00015 -0.00015
ωm: 0.14208 +0.00122 -0.00122
Ωm: 0.3120 +0.0080 -0.0079
z_eq: 3396.0 +29.1 -29.1
z*: 1090.05 +0.28 -0.28
z_drag: 1059.72 +0.29 -0.29
r*: 144.76 Mpc
r_d: 147.46 +0.28 -0.28 Mpc
Chi squared: 0.0000

===============================

ACT DR6 compression
H0: 66.12 +0.79 -0.78 km/s/Mpc
ωc: 0.1238 +0.0021 -0.0021
ωb: 0.02259 +0.00017 -0.00017
ωm: 0.14701 +0.00213 -0.00211
Ωm: 0.3363 +0.0129 -0.0126
z_eq: 3514.0 +50.8 -50.5
z*: 1089.96 +0.30 -0.29
z_drag: 1060.72 +0.40 -0.40
r*: 143.32 Mpc
r_d: 145.46 +0.56 -0.56 Mpc
Chi squared: 0.0001

===============================

Planck + ACT DR6 compression
H0: 67.63 +0.50 -0.50 km/s/Mpc
ωc: 0.1193 +0.0012 -0.0012
ωb: 0.02250 +0.00011 -0.00011
ωm: 0.14242 +0.00118 -0.00116
Ωm: 0.3113 +0.0072 -0.0070
z_eq: 3404.1 +28.2 -27.8
z*: 1089.62 +0.21 -0.21
z_drag: 1060.17 +0.23 -0.23
r*: 144.55 Mpc
r_d: 147.15 +0.29 -0.29 Mpc
Chi squared: 0.0002
"""
