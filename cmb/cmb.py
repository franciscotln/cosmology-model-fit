import numpy as np
import emcee
import corner
import matplotlib.pyplot as plt
from multiprocessing import Pool
import cmb.data_union3_compression as cmb


def Ez(z, params):
    h, Om = params[0] / 100, params[1]
    Or = cmb.Omega_r_h2() / h**2
    Ode = 1 - Om - Or
    one_plus_z = 1 + z

    return np.sqrt(Or * one_plus_z**4 + Om * one_plus_z**3 + Ode)


def chi_squared(params):
    delta = cmb.DISTANCE_PRIORS - cmb.cmb_distances(lambda z: Ez(z, params), *params)
    return np.dot(delta, np.dot(cmb.inv_cov_mat, delta))


bounds = np.array(
    [
        (60, 75),  # H0
        (0.15, 0.45),  # Ωm
        (0.020, 0.024),  # Ωb * h^2
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
    nsteps = 16000 + burn_in
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

    Omh2_samples = samples[:, 1] * (samples[:, 0] / 100) ** 2
    r_drag_samples = cmb.r_drag(samples[:, 2], Omh2_samples)
    z_st_samples = cmb.z_star(samples[:, 2], Omh2_samples)
    z_dr_samples = cmb.z_drag(samples[:, 2], Omh2_samples)
    Omh2_16, Omh2_50, Omh2_84 = np.percentile(Omh2_samples, [15.9, 50, 84.1])
    r_d_16, r_d_50, r_d_84 = np.percentile(r_drag_samples, [15.9, 50, 84.1])
    z_st_16, z_st_50, z_st_84 = np.percentile(z_st_samples, [15.9, 50, 84.1])
    z_d_16, z_d_50, z_d_84 = np.percentile(z_dr_samples, [15.9, 50, 84.1])

    print(f"H0: {H0_50:.2f} +{(H0_84 - H0_50):.2f} -{(H0_50 - H0_16):.2f} km/s/Mpc")
    print(f"Ωm: {Om_50:.4f} +{(Om_84 - Om_50):.4f} -{(Om_50 - Om_16):.4f}")
    print(
        f"Ωm h^2: {Omh2_50:.5f} +{(Omh2_84 - Omh2_50):.5f} -{(Omh2_50 - Omh2_16):.5f}"
    )
    print(
        f"Ωb h^2: {Obh2_50:.5f} +{(Obh2_84 - Obh2_50):.5f} -{(Obh2_50 - Obh2_16):.5f}"
    )
    print(f"z*: {z_st_50:.2f} +{(z_st_84 - z_st_50):.2f} -{(z_st_50 - z_st_16):.2f}")
    print(f"z_drag: {z_d_50:.2f} +{(z_d_84 - z_d_50):.2f} -{(z_d_50 - z_d_16):.2f}")
    print(
        f"r_s(z*) = {cmb.rs_z(lambda z: Ez(z, best_fit), z_st_50, H0_50, Obh2_50):.2f} Mpc"
    )
    print(
        f"r_s(z_drag) = {r_d_50:.2f} +{(r_d_84 - r_d_50):.2f} -{(r_d_50 - r_d_16):.2f} Mpc"
    )
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

===============================

Chen+2018 compression
H0: 67.41 +0.62 -0.61 km/s/Mpc
Ωm: 0.3166 +0.0086 -0.0084
Ωm h^2: 0.14388 +0.00129 -0.00128
Ωb h^2: 0.02236 ± 0.00015
z*: 1088.92 ± 0.22
z_drag: 1059.93 ± 0.30
r_s(z*) = 144.16 Mpc
r_s(z_drag) = 146.72 ± 0.29 Mpc
Chi squared: 0.0003

===============================

Rubin+ Union3 compression
H0: 67.11 ± 0.61 km/s/Mpc
Ωm: 0.3151 +0.0085 -0.0083
Ωm h^2: 0.14194 +0.00126 -0.00125
Ωb h^2: 0.02239 ± 0.00015
z*: 1088.75 +0.22 -0.21
z_drag: 1059.86 ± 0.30
r_s(z*) = 144.67 Mpc
r_s(z_drag) = 147.20 ± 0.29 Mpc
Chi squared: 0.0015

===============================

Karim+ DESI DR2 compression
H0: 67.51 +0.60 -0.59 km/s/Mpc
Ωm: 0.3118 +0.0082 -0.0080
Ωm h^2: 0.14210 +0.00123 -0.00122
Ωb h^2: 0.02223 ± 0.00015
z*: 1088.94 ± 0.22
z_drag: 1059.51 ± 0.29
r_s(z*) = 144.68 Mpc
r_s(z_drag) = 147.30 ± 0.28 Mpc
Chi squared: 0.0004

===============================

Prakhar Bansal+ (Planck + ACT) compression
H0: 67.25 ± 0.51 km/s/Mpc
Ωm: 0.3170 +0.0073 -0.0072
Ωm h^2: 0.14338 ± 0.00115
Ωb h^2: 0.02237 ± 0.00014
z*: 1088.87 ± 0.21
z_drag: 1059.92 ± 0.29
r_s(z*) = 144.28 Mpc
r_s(z_drag) = 146.84 ± 0.26 Mpc
Chi squared: 0.0007
"""
