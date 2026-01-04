from numba import njit
import numpy as np
import cmb.data_planck_compression as cmb

Or_h2 = cmb.Or_h2
Omnu_h2 = cmb.Omnu_h2


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
    neutrino_term = Onu * cmb.Omnu_z(z)

    return np.sqrt(radiation_term + matter_term + neutrino_term + dark_energy_term)


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
    if not np.all((bounds[:, 0] < params) & (params < bounds[:, 1])):
        return -np.inf
    return normalization


def log_likelihood(params):
    return -0.5 * chi_squared(params)


def log_probability(params):
    lp = log_prior(params)
    if np.isinf(lp):
        return -np.inf
    return lp + log_likelihood(params)


def main():
    import emcee, corner
    import matplotlib.pyplot as plt
    from multiprocessing import Pool

    ndim = len(bounds)
    nwalkers = 200
    burn_in = 400
    nsteps = 2000 + burn_in
    np.random.seed(42)
    initial_pos = np.random.uniform(bounds[:, 0], bounds[:, 1], (nwalkers, ndim))
    moves = [
        (emcee.moves.KDEMove(), 0.20),
        (emcee.moves.DEMove(), 0.80),
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

    one_sigma_percentiles = np.array([15.9, 50, 84.1])
    pct = np.percentile(samples, one_sigma_percentiles, axis=0).T
    [
        (H0_16, H0_50, H0_84),
        (Obh2_16, Obh2_50, Obh2_84),
        (Och2_16, Och2_50, Och2_84),
    ] = pct

    best_fit = np.percentile(samples, 50, axis=0)

    h_samples = samples[:, 0] / 100
    Omh2_samples = samples[:, 1] + samples[:, 2] + Omnu_h2
    Om_samples = Omh2_samples / h_samples**2
    z_eq_samples = -1 + (samples[:, 1] + samples[:, 2]) / cmb.Omega_r_h2()
    zst_samples = cmb.z_star(samples[:, 1], Omh2_samples)
    zd_samples = cmb.z_drag(samples[:, 1], Omh2_samples)
    rd_samples = cmb.r_drag(samples[:, 1], Omh2_samples)

    n = len(h_samples)
    DMstar_samples = np.zeros(n, dtype=np.float64)
    rstar_samples = np.zeros(n, dtype=np.float64)
    thetastar_samples = np.zeros(n, dtype=np.float64)
    for i in range(n):
        zst_i = zst_samples[i]
        H0_i = samples[i, 0]
        Obh2_i = samples[i, 1]
        Och2_i = samples[i, 2]
        DMstar_samples[i] = cmb.DM_z(Ez, zst_i, H0_i, Obh2_i, Och2_i)
        rstar_samples[i] = cmb.rs_z(Ez, zst_i, H0_i, Obh2_i, Och2_i)
        thetastar_samples[i] = 100 * rstar_samples[i] / DMstar_samples[i]
        if i % 5000 == 0:
            print(f"Computed {i} of {n}  θ* samples")

    theta_16, theta_50, theta_84 = np.percentile(
        thetastar_samples, one_sigma_percentiles
    )
    rst_16, rs_50, rst_84 = np.percentile(rstar_samples, one_sigma_percentiles)
    Om_16, Om_50, Om_84 = np.percentile(Om_samples, one_sigma_percentiles)
    Omh2_16, Omh2_50, Omh2_84 = np.percentile(Omh2_samples, one_sigma_percentiles)
    z_eq_16, z_eq_50, z_eq_84 = np.percentile(z_eq_samples, one_sigma_percentiles)
    z_st_16, z_st_50, z_st_84 = np.percentile(zst_samples, one_sigma_percentiles)
    z_d_16, z_d_50, z_d_84 = np.percentile(zd_samples, one_sigma_percentiles)
    rd_16, rd_50, rd_84 = np.percentile(rd_samples, one_sigma_percentiles)

    print(f"H0: {H0_50:.2f} +{(H0_84 - H0_50):.2f} -{(H0_50 - H0_16):.2f} km/s/Mpc")
    print(f"ωc: {Och2_50:.4f} +{(Och2_84 - Och2_50):.4f} -{(Och2_50 - Och2_16):.4f}")
    print(f"ωb: {Obh2_50:.5f} +{(Obh2_84 - Obh2_50):.5f} -{(Obh2_50 - Obh2_16):.5f}")
    print(f"ωm: {Omh2_50:.4f} +{(Omh2_84 - Omh2_50):.4f} -{(Omh2_50 - Omh2_16):.4f}")
    print(f"Ωm: {Om_50:.4f} +{(Om_84 - Om_50):.4f} -{(Om_50 - Om_16):.4f}")
    print(f"z_eq: {z_eq_50:.0f} +{(z_eq_84 - z_eq_50):.0f} -{(z_eq_50 - z_eq_16):.0f}")
    print(f"z*: {z_st_50:.2f} +{(z_st_84 - z_st_50):.2f} -{(z_st_50 - z_st_16):.2f}")
    print(f"r*: {rs_50:.2f} +{(rst_84 - rs_50):.2f} -{(rs_50 - rst_16):.2f} Mpc")
    print(
        f"100 θ*: {theta_50:.5f} +{(theta_84 - theta_50):.5f} -{(theta_50 - theta_16):.5f}"
    )
    print(f"z_drag: {z_d_50:.2f} +{(z_d_84 - z_d_50):.2f} -{(z_d_50 - z_d_16):.2f}")
    print(f"r_d: {rd_50:.2f} +{(rd_84 - rd_50):.2f} -{(rd_50 - rd_16):.2f} Mpc")
    print(f"Chi squared: {chi_squared(best_fit):.4f}")

    samples = np.column_stack(
        [samples, thetastar_samples, Om_samples, rd_samples, rstar_samples]
    )
    labels = ["$H_0$", "$ω_b$", "$ω_c$", "$100 θ_*$", "$Ω_m$", "$r_{drag}$", "$r*$"]

    corner.corner(
        samples,
        labels=labels,
        quantiles=one_sigma_percentiles / 100,
        show_titles=True,
        title_fmt=".4f",
        bins=100,
        fill_contours=False,
        plot_datapoints=False,
        smooth=2.0,
        smooth1d=2.0,
        levels=(0.393, 0.864),
        range=np.repeat(0.9999, len(labels)),
    )
    plt.show()


if __name__ == "__main__":
    main()

"""
Flat ΛCDM w(z) = -1 

*******************************

plikHM TT, TE, EE + lowl + lowE compression (Planck 2019 - PR3)
H0: 67.27 +0.61 -0.60 km/s/Mpc
ωc: 0.1202 +0.0014 -0.0014
ωb: 0.02236 +0.00015 -0.00015
ωm: 0.1432 +0.0013 -0.0013
Ωm: 0.3165 +0.0085 -0.0083
z_eq: 3407 +31 -30
z*: 1089.95 +0.28 -0.27
r*: 144.39 +0.30 -0.30 Mpc
100 θ*: 1.04109 +0.00030 -0.00031
z_drag: 1059.93 +0.30 -0.30
r_d: 147.05 +0.29 -0.30 Mpc
Chi squared: 0.0001

===============================

plikHM TT, TE, EE + lowl + lowE + Lensing compression (Planck 2019 - PR3)
H0: 67.36 +0.54 -0.53 km/s/Mpc
ωc: 0.1200 +0.0012 -0.0012
ωb: 0.02237 +0.00015 -0.00015
ωm: 0.1430 +0.0011 -0.0011
Ωm: 0.3153 +0.0074 -0.0072
z_eq: 3402 +27 -26
z*: 1089.92 +0.25 -0.25
r*: 144.43 +0.26 -0.26 Mpc
100 θ*: 1.04110 +0.00031 -0.00031
z_drag: 1059.94 +0.30 -0.30
r_d: 147.09 +0.26 -0.26 Mpc
Chi squared: 0.0003

===============================

Early ΛCDM (arXiv:2302.12911v2)
H0: 67.49 +0.59 -0.58 km/s/Mpc
ωc: 0.1192 +0.0013 -0.0013
ωb: 0.02223 +0.00015 -0.00015
ωm: 0.1421 +0.0012 -0.0012
Ωm: 0.3120 +0.0080 -0.0079
z_eq: 3381 +29 -29
z*: 1090.12 +0.27 -0.27
r*: 144.74 +0.29 -0.28 Mpc
100 θ*: 1.04103 +0.00026 -0.00026
z_drag: 1059.65 +0.29 -0.30
r_d: 147.46 +0.28 -0.28 Mpc
Chi squared: 0.0000

===============================

ACT DR6 compression
H0: 66.11 +0.79 -0.78 km/s/Mpc
ωc: 0.1238 +0.0022 -0.0021
ωb: 0.02259 +0.00017 -0.00017
ωm: 0.1470 +0.0022 -0.0021
Ωm: 0.3364 +0.0130 -0.0125
z_eq: 3499 +52 -50
z*: 1089.96 +0.30 -0.30
r*: 143.31 +0.54 -0.54 Mpc
100 θ*: 1.04075 +0.00031 -0.00031
z_drag: 1060.72 +0.39 -0.40
r_d: 145.88 +0.56 -0.56 Mpc
Chi squared: 0.0015

===============================

ACT DR6 + Planck compression
H0: 67.62 +0.50 -0.50 km/s/Mpc
ωc: 0.1193 +0.0012 -0.0012
ωb: 0.02250 +0.00011 -0.00011
ωm: 0.1425 +0.0012 -0.0012
Ωm: 0.3115 +0.0072 -0.0070
z_eq: 3390 +28 -28
z*: 1089.68 +0.22 -0.21
r*: 144.52 +0.29 -0.29 Mpc
100 θ*: 1.04094 +0.00025 -0.00025
z_drag: 1060.17 +0.23 -0.23
r_d: 147.14 +0.29 -0.29 Mpc
Chi squared: 0.0002
"""
