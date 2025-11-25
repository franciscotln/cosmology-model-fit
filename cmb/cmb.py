from numba import njit
import numpy as np
import cmb.data_planck_compression as cmb

z_nr = cmb.z_nr
Or_h2 = cmb.Omega_r_h2(2.044)
Omnu_h2 = cmb.Omnu_h2


@njit
def Omnu_z(z):
    """
    Computes the appox. evolution of massive neutrino
    energy density with redshift
    """
    return (
        (1 + z) ** 4
        * (1 + ((1 + z_nr) / (1 + z)) ** 2) ** 0.5
        * (1 + (1 + z_nr) ** 2) ** -0.5
    )


@njit
def Ez(z, H0, Obh2, Och2, w0=-1, wa=0):
    h = H0 / 100
    Onu = Omnu_h2 / h**2
    Or = Or_h2 / h**2
    Obc = (Obh2 + Och2) / h**2
    Ode = 1.0 - Obc - Or - Onu

    radiation_term = Or * (1 + z) ** 4
    matter_term = Obc * (1 + z) ** 3
    dark_energy_term = Ode
    neutrino_term = Onu * Omnu_z(z)

    return np.sqrt(radiation_term + matter_term + dark_energy_term + neutrino_term)


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
    z_eq_samples = -1 + (samples[:, 1] + samples[:, 2]) / cmb.Omega_r_h2()
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
    print(f"r*: {cmb.rs_z(Ez, z_st_50, H0_50, Obh2_50, Och2_50):.2f} Mpc")
    print(f"100 θ*: {100 * np.pi / cmb.cmb_distances(Ez, *best_fit)[1]:.7f} radians")
    print(f"z_drag: {z_d_50:.2f} +{(z_d_84 - z_d_50):.2f} -{(z_d_50 - z_d_16):.2f}")
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

Planck compression (2019 - PR3)
H0: 67.26 +0.60 -0.60 km/s/Mpc
ωc: 0.1202 +0.0014 -0.0014
ωb: 0.02236 +0.00015 -0.00015
ωm: 0.14317 +0.00129 -0.00128
Ωm: 0.3164 +0.0085 -0.0083
z_eq: 3406 +31 -31
z*: 1089.95 +0.29 -0.28
r*: 144.41 Mpc
100 θ*: 1.0410885
z_drag: 1059.92 +0.29 -0.30
r_d: 147.06 +0.30 -0.29 Mpc
Chi squared: 0.0001

===============================

Early ΛCDM (arXiv:2302.12911v2)
H0: 67.47 +0.59 -0.58 km/s/Mpc
ωc: 0.1192 +0.0013 -0.0013
ωb: 0.02223 +0.00015 -0.00015
ωm: 0.14208 +0.00122 -0.00123
Ωm: 0.3121 +0.0081 -0.0080
z_eq: 3381 +29 -29
z*: 1090.05 +0.28 -0.28
r*: 144.75 Mpc
100 θ*: 1.041029 radians
z_drag: 1059.72 +0.29 -0.29
r_d: 147.46 +0.28 -0.28 Mpc
Chi squared: 0.0001

===============================

ACT DR6 compression
H0: 66.11 +0.79 -0.79 km/s/Mpc
ωc: 0.1238 +0.0022 -0.0021
ωb: 0.02259 +0.00017 -0.00017
ωm: 0.14703 +0.00215 -0.00210
Ωm: 0.3364 +0.0131 -0.0124
z_eq: 3499 +51 -50
z*: 1089.96 +0.30 -0.29
r*: 143.31 Mpc
100 θ*: 1.0407571 radians
z_drag: 1060.72 +0.39 -0.40
r_d: 145.87 +0.56 -0.56 Mpc
Chi squared: 0.0009

===============================

Planck + ACT DR6 compression
H0: 67.62 +0.50 -0.50 km/s/Mpc
ωc: 0.1193 +0.0012 -0.0012
ωb: 0.02250 +0.00011 -0.00011
ωm: 0.14246 +0.00118 -0.00118
Ωm: 0.3115 +0.0072 -0.0070
z_eq: 3390 +28 -28
z*: 1089.68 +0.21 -0.21
r*: 144.52 Mpc
100 θ*: 1.0409411 radians
z_drag: 1060.17 +0.24 -0.24
r_d: 147.14 +0.29 -0.29 Mpc
Chi squared: 0.0001

===============================

Union3 compression
H0: 67.18 +0.61 -0.60 km/s/Mpc
ωc: 0.1187 +0.0014 -0.0013
ωb: 0.02239 +0.00015 -0.00015
ωm: 0.14171 +0.00128 -0.00126
Ωm: 0.3140 +0.0085 -0.0082
z_eq: 3374 +31 -30
z*: 1089.69 +0.29 -0.28
r*: 144.77 Mpc
100 θ*: 302.3015147 radians
z_drag: 1059.85 +0.29 -0.30
r_d: 147.26 +0.29 -0.29 Mpc
Chi squared: 0.0010
"""
