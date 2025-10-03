from numba import njit
import numpy as np
import emcee
import corner
from scipy.linalg import cho_factor, cho_solve
import matplotlib.pyplot as plt
from multiprocessing import Pool
from y2025BAO.data import get_data as get_bao_data
import cmb.data_desi_compression as cmb
from .plot_predictions import plot_bao_predictions

c = cmb.c  # speed of light in km/s
Or_h2 = cmb.Omega_r_h2()

bao_legend, bao_data, bao_cov_matrix = get_bao_data()
cho_bao = cho_factor(bao_cov_matrix)


@njit
def Ez(z, params):
    H0, Om, w0 = params[0], params[1], params[3]
    h = H0 / 100
    Or = Or_h2 / h**2
    Ode = 1 - Om - Or

    one_plus_z = 1 + z
    cubed = one_plus_z**3
    rho_de = (2 * cubed / (1 + cubed)) ** (2 * (1 + w0))

    return np.sqrt(Or * one_plus_z**4 + Om * cubed + Ode * rho_de)


@njit
def H_z(z, params):
    return params[0] * Ez(z, params)


@njit
def DH_z(z, params):
    return c / H_z(z, params)


@njit
def DM_z(z, params):
    result = np.empty(z.size, dtype=np.float64)
    for i in range(z.size):
        zp = z[i]
        x = np.linspace(0, zp, num=max(250, int(250 * zp)))
        y = DH_z(x, params)
        result[i] = np.trapz(y=y, x=x)
    return result


@njit
def DV_z(z, params):
    DH = DH_z(z, params)
    DM = DM_z(z, params)
    return (z * DH * DM**2) ** (1 / 3)


qty_map = {
    "DV_over_rs": 0,
    "DM_over_rs": 1,
    "DH_over_rs": 2,
}

quantities = np.array([qty_map[q] for q in bao_data["quantity"]], dtype=np.int32)


@njit
def bao_predictions(z, qty, params):
    H0, Om, Obh2 = params[0], params[1], params[2]
    Omh2 = Om * (H0 / 100) ** 2
    rd = cmb.r_drag(wb=Obh2, wm=Omh2)

    results = np.empty(z.size, dtype=np.float64)
    DV_mask = qty == 0
    DM_mask = qty == 1
    DH_mask = qty == 2
    results[DH_mask] = DH_z(z[DH_mask], params)
    results[DM_mask] = DM_z(z[DM_mask], params)
    results[DV_mask] = DV_z(z[DV_mask], params)
    return results / rd


def chi_squared(params):
    H0, Om, Ob_h2 = params[0], params[1], params[2]

    delta_cmb = cmb.DISTANCE_PRIORS - cmb.cmb_distances(Ez, params, H0, Om, Ob_h2)
    chi2_cmb = np.dot(delta_cmb, np.dot(cmb.inv_cov_mat, delta_cmb))

    delta_bao = bao_data["value"] - bao_predictions(bao_data["z"], quantities, params)
    chi_bao = np.dot(delta_bao, cho_solve(cho_bao, delta_bao, check_finite=False))

    return chi2_cmb + chi_bao


bounds = np.array(
    [
        (55, 75),  # H0
        (0.25, 0.45),  # Ωm
        (0.021, 0.023),  # Ωb * h^2
        (-1.5, 0),  # w0
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
    nwalkers = 150
    burn_in = 200
    nsteps = 2200 + burn_in
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
        print("effective samples", ndim * nwalkers * nsteps / np.max(tau))
    except emcee.autocorr.AutocorrError as e:
        print("Autocorrelation time could not be computed", e)

    chains_samples = sampler.get_chain(discard=burn_in, flat=False)
    samples = sampler.get_chain(discard=burn_in, flat=True)

    pct = np.percentile(samples, [15.9, 50, 84.1], axis=0).T
    H0_16, H0_50, H0_84 = pct[0]
    Om_16, Om_50, Om_84 = pct[1]
    Obh2_16, Obh2_50, Obh2_84 = pct[2]
    w0_16, w0_50, w0_84 = pct[3]

    best_fit = np.array([H0_50, Om_50, Obh2_50, w0_50], dtype=np.float64)

    Om_h2_samples = samples[:, 1] * (samples[:, 0] / 100) ** 2
    z_st_samples = cmb.z_star(samples[:, 2], Om_h2_samples)
    z_dr_samples = cmb.z_drag(samples[:, 2], Om_h2_samples)
    r_drag_samples = cmb.r_drag(samples[:, 2], Om_h2_samples)

    Omh2_16, Omh2_50, Omh2_84 = np.percentile(Om_h2_samples, [15.9, 50, 84.1])
    r_d_16, r_d_50, r_d_84 = np.percentile(r_drag_samples, [15.9, 50, 84.1])
    z_st_16, z_st_50, z_st_84 = np.percentile(z_st_samples, [15.9, 50, 84.1])
    z_d_16, z_d_50, z_d_84 = np.percentile(z_dr_samples, [15.9, 50, 84.1])

    print(f"H0: {H0_50:.2f} +{(H0_84 - H0_50):.2f} -{(H0_50 - H0_16):.2f} km/s/Mpc")
    print(f"Ωm: {Om_50:.3f} +{(Om_84 - Om_50):.3f} -{(Om_50 - Om_16):.3f}")
    print(
        f"Ωm h^2: {Omh2_50:.5f} +{(Omh2_84 - Omh2_50):.5f} -{(Omh2_50 - Omh2_16):.5f}"
    )
    print(
        f"Ωb h^2: {Obh2_50:.5f} +{(Obh2_84 - Obh2_50):.5f} -{(Obh2_50 - Obh2_16):.5f}"
    )
    print(f"w0: {w0_50:.3f} +{(w0_84 - w0_50):.3f} -{(w0_50 - w0_16):.3f}")
    print(f"z*: {z_st_50:.2f} +{(z_st_84 - z_st_50):.2f} -{(z_st_50 - z_st_16):.2f}")
    print(f"z_drag: {z_d_50:.2f} +{(z_d_84 - z_d_50):.2f} -{(z_d_50 - z_d_16):.2f}")
    print(f"r_s(z*) = {cmb.rs_z(Ez, z_st_50, best_fit, H0_50, Obh2_50):.2f} Mpc")
    print(
        f"r_s(z_drag) = {r_d_50:.2f} +{(r_d_84 - r_d_50):.2f} -{(r_d_50 - r_d_16):.2f} Mpc"
    )
    print(f"Chi squared: {chi_squared(best_fit):.2f}")

    plot_bao_predictions(
        theory_predictions=lambda z, qty: bao_predictions(z, qty, best_fit),
        data=bao_data,
        errors=np.sqrt(np.diag(bao_cov_matrix)),
        title=bao_legend,
    )
    labels = ["$H_0$", "$Ω_m$", "$Ω_b h^2$", "$w_0$"]
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
*******************************
Dataset: DESI DR2 2024 + (θ∗,ωb,ωbc)CMB
*******************************

Flat ΛCDM w(z) = -1
H0: 68.45 ± 0.30 km/s/Mpc
Ωm: 0.299 ± 0.004
Ωm h^2: 0.14024 +0.00064 -0.00063
Ωb h^2: 0.02238 ± 0.00012
w0: -1
z*: 1088.65 ± 0.14
z_drag: 1059.71 +0.26 -0.27
r_s(z*) = 145.08 Mpc
r_s(z_drag) = 147.74 +0.17 -0.18 Mpc
Chi squared: 13.50
Degs of freedom: 14

===============================

Flat wCDM w(z) = w0
H0: 68.88 +0.96 -0.92 km/s/Mpc
Ωm: 0.296 +0.007 -0.007
Ωm h^2: 0.14056 +0.00089 -0.00090
Ωb h^2: 0.02235 +0.00013 -0.00013
w0: -1.019 +0.038 -0.040
z*: 1088.70 +0.18 -0.17
z_drag: 1059.67 +0.27 -0.28
r_s(z*) = 145.02 Mpc
r_s(z_drag) = 147.69 +0.21 -0.20 Mpc
Chi squared: 13.31 (Δ chi2 0.19)
Degs of freedom: 13

===============================

Flat w(z) = -1 + 2 * (1 + w0) / (1 + (1 + z)**3)
H0: 68.20 +1.41 -1.35 km/s/Mpc
Ωm: 0.301 +0.012 -0.011
Ωm h^2: 0.14016 +0.00080 -0.00080
Ωb h^2: 0.02238 +0.00013 -0.00013
w0: -0.984 +0.088 -0.089
z*: 1088.64 +0.16 -0.16
z_drag: 1059.72 +0.27 -0.28
r_s(z*) = 145.11 Mpc
r_s(z_drag) = 147.76 +0.20 -0.19 Mpc
Chi squared: 13.46 (Δ chi2 0.04)
Degs of freedom: 13

===============================

Flat w(z) = w0 + wa * z / (1 + z)
H0: 63.69 +2.07 -2.09 km/s/Mpc
Ωm: 0.351 +0.025 -0.023
Ωm h^2: 0.14222 +0.00103 -0.00106
Ωb h^2: 0.02222 +0.00014 -0.00014
w0: -0.445 +0.253 -0.230
wa: -1.628 +0.665 -0.760
z*: 1088.96 +0.20 -0.20
z_drag: 1059.49 +0.29 -0.28
r_s(z*) = 144.66 Mpc
r_s(z_drag) = 147.41 +0.22 -0.22 Mpc
Chi squared: 7.07 (Δ chi2 6.50)
Degs of freedom: 12

*******************************
Dataset: SDSS DR16 2020 + Chen+2018 CMB compression
*******************************

Flat ΛCDM w(z) = -1
H0: 67.76 ± 0.44 km/s/Mpc
Ωm: 0.312 ± 0.006
Ωb h^2: 0.02241 ± 0.00013
w0: -1
z*: 1088.81
z_drag: 1060.00
r_s(z*) = 144.30 Mpc
r_s(z_drag) = 146.86 Mpc
Chi squared: 11.35

===============================

Flat wCDM w(z) = w0
H0: 68.16 +1.51 -1.42 km/s/Mpc
Ωm: 0.309 ± 0.013
Ωb h^2: 0.02240 ± 0.00014
w0: -1.017 +0.056 -0.061
z*: 1088.84
z_drag: 1059.98
r_s(z*) = 144.26 Mpc
r_s(z_drag) = 146.82 Mpc
Chi squared: 11.32 (Δ chi2 0.03)

===============================

Flat w(z) = -1 + 2 * (1 + w0) / (1 + (1 + z)**3)
H0: 67.35 +2.14 -1.96 km/s/Mpc
Ωm: 0.315 ± 0.018
Ωb h^2: 0.02242 ± 0.00014
w0: -0.973 +0.130 -0.136
z*: 1088.79
z_drag: 1060.01
r_s(z*) = 144.33 Mpc
r_s(z_drag) = 146.88 Mpc
Chi squared: 11.29 (Δ chi2 0.06)

===============================

Flat w(z) = w0 + wa * z / (1 + z)
H0: 63.67 +2.83 -2.59 km/s/Mpc
Ωm: 0.355 +0.031 -0.030
Ωm h^2: 0.14409 +0.00123 -0.00125
Ωb h^2: 0.02234 +0.00015 -0.00014
w0: -0.495 +0.313 -0.300
wa: -1.436 +0.816 -0.916
z*: 1088.95 ± 0.21
z_drag: 1059.92 ± 0.29
r_s(z*) = 144.13 Mpc
r_s(z_drag) = 146.68 ± 0.28 Mpc
Chi squared: 8.64
"""
