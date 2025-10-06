from numba import njit
import numpy as np
import emcee
import corner
import matplotlib.pyplot as plt
from multiprocessing import Pool
import cmb.data_chen_compression as cmb


@njit
def Ez(z, params):
    h, Om = params[0] / 100, params[1]
    Or = cmb.Or(h, Om)
    Ode = 1 - Om - Or
    one_plus_z = 1 + z

    return np.sqrt(Or * one_plus_z**4 + Om * one_plus_z**3 + Ode)


def chi_squared(params):
    H0, Om, Ob_h2 = params
    delta = cmb.DISTANCE_PRIORS - cmb.cmb_distances(Ez, params, H0, Om, Ob_h2)
    return np.dot(delta, np.dot(cmb.inv_cov_mat, delta))


bounds = np.array(
    [
        (60, 75),  # H0
        (0.15, 0.45),  # Ωm
        (0.020, 0.024),  # Ωb * h^2
    ],
    dtype=np.float64,
)


@njit
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
    nwalkers = 200
    burn_in = 200
    nsteps = 2000 + burn_in
    initial_pos = np.zeros((nwalkers, ndim))

    for dim, (lower, upper) in enumerate(bounds):
        initial_pos[:, dim] = np.random.uniform(lower, upper, nwalkers)

    with Pool(5) as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers,
            ndim,
            log_probability,
            pool=pool,
            moves=[
                (emcee.moves.KDEMove(), 0.30),
                (emcee.moves.DEMove(), 0.56),
                (emcee.moves.DESnookerMove(), 0.14),
            ],
        )
        sampler.run_mcmc(initial_pos, nsteps, progress=True)

    try:
        tau = sampler.get_autocorr_time()
        print("auto-correlation time", tau)
        print("acceptance fraction", np.mean(sampler.acceptance_fraction))
        print("effective samples", ndim * nwalkers * (nsteps - burn_in) / np.max(tau))
    except emcee.autocorr.AutocorrError as e:
        print("Autocorrelation time could not be computed", e)

    samples = sampler.get_chain(discard=burn_in, flat=True)

    one_sigma_percentiles = [15.9, 50, 84.1]
    pct = np.percentile(samples, one_sigma_percentiles, axis=0).T
    H0_16, H0_50, H0_84 = pct[0]
    Om_16, Om_50, Om_84 = pct[1]
    Obh2_16, Obh2_50, Obh2_84 = pct[2]

    best_fit = np.array([H0_50, Om_50, Obh2_50], dtype=np.float64)

    h_samples = samples[:, 0] / 100
    Orh2_samples = 1e5 * cmb.Or(h=h_samples, Om=samples[:, 1]) * h_samples**2
    Omh2_samples = samples[:, 1] * h_samples**2
    z_eq_samples = 24077.440586 * Omh2_samples
    z_st_samples = cmb.z_star(samples[:, 2], Omh2_samples)
    z_dr_samples = cmb.z_drag(samples[:, 2], Omh2_samples)
    Omh2_16, Omh2_50, Omh2_84 = np.percentile(Omh2_samples, one_sigma_percentiles)
    Or_16, Or_50, Or_84 = np.percentile(Orh2_samples, one_sigma_percentiles)
    z_eq_16, z_eq_50, z_eq_84 = np.percentile(z_eq_samples, one_sigma_percentiles)
    z_st_16, z_st_50, z_st_84 = np.percentile(z_st_samples, one_sigma_percentiles)
    z_d_16, z_d_50, z_d_84 = np.percentile(z_dr_samples, one_sigma_percentiles)

    print(f"H0: {H0_50:.2f} +{(H0_84 - H0_50):.2f} -{(H0_50 - H0_16):.2f} km/s/Mpc")
    print(f"Ωm: {Om_50:.4f} +{(Om_84 - Om_50):.4f} -{(Om_50 - Om_16):.4f}")
    print(f"ωm: {Omh2_50:.4f} +{(Omh2_84 - Omh2_50):.4f} -{(Omh2_50 - Omh2_16):.4f}")
    print(f"ωb: {Obh2_50:.5f} +{(Obh2_84 - Obh2_50):.5f} -{(Obh2_50 - Obh2_16):.5f}")
    print(f"ωr: {Or_50:.3f} +{(Or_84 - Or_50):.3f} -{(Or_50 - Or_16):.3f} x 10^-5")
    print(f"z_eq: {z_eq_50:.1f} +{(z_eq_84 - z_eq_50):.1f} -{(z_eq_50 - z_eq_16):.1f}")
    print(f"z*: {z_st_50:.2f} +{(z_st_84 - z_st_50):.2f} -{(z_st_50 - z_st_16):.2f}")
    print(f"z_drag: {z_d_50:.2f} +{(z_d_84 - z_d_50):.2f} -{(z_d_50 - z_d_16):.2f}")
    print(f"r*: {cmb.rs_z(Ez, z_st_50, best_fit, H0_50, Obh2_50):.2f} Mpc")
    print(f"r_d: {cmb.rs_z(Ez, z_d_50, best_fit, H0_50, Obh2_50):.2f} Mpc")
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
        smooth=2.0,
        smooth1d=2.0,
        levels=(0.393, 0.864),
    )
    plt.show()

    chains_samples = sampler.get_chain(discard=burn_in, flat=False)
    plt.figure(figsize=(16, 1.5 * ndim))
    for n in range(ndim):
        plt.subplot2grid((ndim, 1), (n, 0))
        plt.plot(chains_samples[:, :, n], alpha=0.3)
        plt.ylabel(labels[n])
        plt.xlim(0, None)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

"""
Flat ΛCDM w(z) = -1 

===============================

Chen+2018 compression
H0: 67.43 +0.60 -0.60 km/s/Mpc
Ωm: 0.3165 +0.0085 -0.0082
ωm: 0.1439 +0.0013 -0.0013
ωb: 0.02236 +0.00015 -0.00015
ωr: 4.152 +0.000 -0.000 x 10^-5
z_eq: 3464.5 +31.0 -30.9
z*: 1092.00 +0.29 -0.28
z_drag: 1059.93 +0.29 -0.29
r*: 144.16 Mpc
r_d: 147.00 Mpc
Chi squared: 0.0005

===============================

Prakhar Bansal+ (Planck + ACT) compression
H0: 67.34 +0.52 -0.52 km/s/Mpc
Ωm: 0.3169 +0.0073 -0.0071
ωm: 0.1437 +0.0011 -0.0011
ωb: 0.02237 +0.00014 -0.00014
ωr: 4.181 +0.000 -0.000 x 10^-5
z_eq: 3460.1 +26.1 -26.2
z*: 1090.00 (fixed)
z_drag: 1059.95 +0.29 -0.29
r*: 144.12 Mpc
r_d: 146.77 Mpc
Chi squared: 0.0003

===============================

Karim+ DESI DR2 compression
H0: 67.53 +0.57 -0.57 km/s/Mpc
Ωm: 0.3116 +0.0079 -0.0077
ωm: 0.1421 +0.0012 -0.0012
ωb: 0.02223 +0.00014 -0.00014
ωr: 4.152 +0.000 -0.000 x 10^-5
z_eq: 3421.4 +28.7 -29.0
z*: 1092.03 +0.28 -0.28
z_drag: 1063.00 +0.29 -0.29
r*: 144.68 Mpc
r_d: 147.26 Mpc
Chi squared: 0.0008

===============================

Rubin+ Union3 compression
H0: 67.39 +0.60 -0.59 km/s/Mpc
Ωm: 0.3149 +0.0083 -0.0082
ωm: 0.1430 +0.0013 -0.0013
ωb: 0.02239 +0.00015 -0.00015
ωr: 4.180 +0.000 -0.000 x 10^-5
z_eq: 3443.1 +30.2 -30.5
z*: 1091.88 +0.28 -0.28
z_drag: 1063.46 +0.30 -0.30
r*: 144.13 Mpc
r_d: 146.64 Mpc
Chi squared: 0.0006
"""
