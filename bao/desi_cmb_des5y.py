from numba import njit
import numpy as np
import emcee
import corner
from scipy.integrate import cumulative_trapezoid, quad
from scipy.linalg import cho_factor, cho_solve
import matplotlib.pyplot as plt
from multiprocessing import Pool
from y2024DES.data import get_data
from y2025BAO.data import get_data as get_bao_data
import cmb.data_desi_compression as cmb
from hubble.plotting import plot_predictions as plot_sn_predictions
from .plot_predictions import plot_bao_predictions


c = cmb.c  # km/s
Or_h2 = cmb.Omega_r_h2()

sn_legend, z_cmb, z_hel, mu_values, cov_matrix_sn = get_data()
bao_legend, bao_data, bao_cov_matrix = get_bao_data()
cho_sn = cho_factor(cov_matrix_sn)
cho_bao = cho_factor(bao_cov_matrix)

sn_grid = np.linspace(0, np.max(z_cmb), num=1000)
one_plus_z_hel = 1 + z_hel


@njit
def Ez(z, params):
    H0, Om, w0 = params[0], params[1], params[3]
    h = H0 / 100
    Or = Or_h2 / h**2
    Ode = 1 - Om - Or
    one_plus_z = 1 + z
    rho_de = (2 * one_plus_z**3 / (1 + one_plus_z**3)) ** (2 * (1 + w0))

    return np.sqrt(Or * one_plus_z**4 + Om * one_plus_z**3 + Ode * rho_de)


def theory_mu(params):
    H0, offset_mag = params[0], params[-1]
    integral_vals = cumulative_trapezoid(1 / Ez(sn_grid, params), sn_grid, initial=0)
    I = np.interp(z_cmb, sn_grid, integral_vals)
    dL = one_plus_z_hel * I * c / H0
    return offset_mag + 25 + 5 * np.log10(dL)


@njit
def H_z(z, params):
    return params[0] * Ez(z, params)


@njit
def DH_z(z, params):
    return c / H_z(z, params)


@njit
def DM_z(z, params):
    x = np.linspace(0, z, num=250)
    y = DH_z(x, params)
    return np.trapz(y=y, x=x)


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

quantities = np.array([qty_map[q] for q in bao_data["quantity"]], dtype=np.int64)


@njit
def bao_theory(z, qty, params):
    H0, Om, Obh2 = params[0], params[1], params[2]
    Omh2 = Om * (H0 / 100) ** 2
    rd = cmb.r_drag(wb=Obh2, wm=Omh2)

    results = np.empty(z.size, dtype=np.float64)
    for i in range(z.size):
        q = qty[i]
        if q == 0:
            results[i] = DV_z(z[i], params) / rd
        elif q == 1:
            results[i] = DM_z(z[i], params) / rd
        elif q == 2:
            results[i] = DH_z(z[i], params) / rd
    return results


def chi_squared(params):
    H0, Om, Obh2 = params[0], params[1], params[2]

    delta = cmb.DISTANCE_PRIORS - cmb.cmb_distances(Ez, params, H0, Om, Obh2)
    chi2_cmb = np.dot(delta, np.dot(cmb.inv_cov_mat, delta))

    delta_bao = bao_data["value"] - bao_theory(bao_data["z"], quantities, params)
    chi_bao = np.dot(delta_bao, cho_solve(cho_bao, delta_bao))

    delta_sn = mu_values - theory_mu(params)
    chi_sn = np.dot(delta_sn, cho_solve(cho_sn, delta_sn))

    return chi2_cmb + chi_bao + chi_sn


bounds = np.array(
    [
        (60, 75),  # H0
        (0.1, 0.6),  # Ωm
        (0.019, 0.025),  # Ωb * h^2
        (-2, 0),  # w0
        (-0.7, 0.7),  # ΔM
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
    nwalkers = 8 * ndim
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

    samples = sampler.get_chain(discard=burn_in, flat=True)

    pct = np.percentile(samples, [15.9, 50, 84.1], axis=0).T
    H0_16, H0_50, H0_84 = pct[0]
    Om_16, Om_50, Om_84 = pct[1]
    Obh2_16, Obh2_50, Obh2_84 = pct[2]
    w0_16, w0_50, w0_84 = pct[3]
    dM_16, dM_50, dM_84 = pct[4]

    best_fit = np.array([H0_50, Om_50, Obh2_50, w0_50, dM_50], dtype=np.float64)

    Omh2_samples = samples[:, 1] * (samples[:, 0] / 100) ** 2
    z_star_samples = cmb.z_star(samples[:, 2], Omh2_samples)
    z_drag_samples = cmb.z_drag(samples[:, 2], Omh2_samples)
    r_drag_samples = cmb.r_drag(samples[:, 2], Omh2_samples)

    one_sigma_contours = [15.9, 50, 84.1]

    Omh2_16, Omh2_50, Omh2_84 = np.percentile(Omh2_samples, one_sigma_contours)
    z_star_16, z_star_50, z_star_84 = np.percentile(z_star_samples, one_sigma_contours)
    z_drag_16, z_drag_50, z_drag_84 = np.percentile(z_drag_samples, one_sigma_contours)
    r_drag_16, r_drag_50, r_drag_84 = np.percentile(r_drag_samples, one_sigma_contours)

    print(f"H0: {H0_50:.2f} +{(H0_84 - H0_50):.2f} -{(H0_50 - H0_16):.2f} km/s/Mpc")
    print(f"Ωm: {Om_50:.4f} +{(Om_84 - Om_50):.4f} -{(Om_50 - Om_16):.4f}")
    print(
        f"Ωb h^2: {Obh2_50:.5f} +{(Obh2_84 - Obh2_50):.5f} -{(Obh2_50 - Obh2_16):.5f}"
    )
    print(
        f"Ωm h^2: {Omh2_50:.5f} +{(Omh2_84 - Omh2_50):.5f} -{(Omh2_50 - Omh2_16):.5f}"
    )
    print(f"w0: {w0_50:.3f} +{(w0_84 - w0_50):.3f} -{(w0_50 - w0_16):.3f}")
    print(f"ΔM: {dM_50:.3f} +{(dM_84 - dM_50):.3f} -{(dM_50 - dM_16):.3f}")
    print(
        f"z*: {z_star_50:.2f} +{(z_star_84 - z_star_50):.2f} -{(z_star_50 - z_star_16):.2f}"
    )
    print(
        f"z_drag: {z_drag_50:.2f} +{(z_drag_84 - z_drag_50):.2f} -{(z_drag_50 - z_drag_16):.2f}"
    )
    print(f"r_s(z*) = {cmb.rs_z(Ez, z_star_50, best_fit, H0_50, Obh2_50):.2f} Mpc")
    print(
        f"r_s(z_drag) = {r_drag_50:.2f} +{(r_drag_84 - r_drag_50):.2f} -{(r_drag_50 - r_drag_16):.2f} Mpc"
    )
    print(f"Chi squared: {chi_squared(best_fit):.2f}")

    plot_bao_predictions(
        theory_predictions=lambda z, qty: bao_theory(z, qty, best_fit),
        data=bao_data,
        errors=np.sqrt(np.diag(bao_cov_matrix)),
        title=bao_legend,
    )
    plot_sn_predictions(
        legend=sn_legend,
        x=z_cmb,
        y=mu_values,
        y_err=np.sqrt(np.diag(cov_matrix_sn)),
        y_model=theory_mu(best_fit),
        label=f"Model: $Ω_m$={Om_50:.3f}",
        x_scale="log",
    )
    labels = ["$H_0$", "$Ω_m$", "$Ω_b h^2$", "$w_0$", "$Δ_M$"]
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


if __name__ == "__main__":
    main()

"""
Flat ΛCDM w(z) = -1
H0: 68.22 +0.30 -0.29 km/s/Mpc
Ωm: 0.302 ± 0.004
Ωb h^2: 0.02234 ± 0.00012
w0: -1
wa: 0
ΔM: -0.065 ± 0.008
z*: 1088.72 ± 0.14
z_drag: 1059.65 ± 0.27
r_s(z*) = 144.99 Mpc
r_s(z_drag) = 147.68 ± 0.18 Mpc
Chi squared: 1663.53
Degrees of freedom: 1747

===============================

Flat wCDM w(z) = w0
H0: 67.15 ± 0.54 km/s/Mpc
Ωm: 0.309 ± 0.005
Ωb h^2: 0.02243 ± 0.00013
Ωm h^2: 0.13952 ± 0.00083
w0: -0.948 ± 0.023
wa: 0
ΔM: -0.080 +0.010 -0.011
z*: 1088.54 +0.17 -0.16
z_drag: 1059.79 +0.27 -0.28
r_s(z*) = 145.24 Mpc
r_s(z_drag) = 147.87 ± 0.20 Mpc
Chi squared: 1658.27 (Δ chi2 5.26)
Degrees of freedom: 1746

===============================

Flat w(z) = -1 + 2 * (1 + w0) / (1 + (1 + z)**3)
H0: 66.66 +0.58 -0.55 km/s/Mpc
Ωm: 0.3143 +0.0054 -0.0056
Ωb h^2: 0.02243 +0.00012 -0.00013
Ωm h^2: 0.13963 +0.00071 -0.00070
w0: -0.883 ± 0.037
ΔM: -0.082 ± 0.010
z*: 1088.55 ± 0.15
z_drag: 1059.78 +0.27 -0.28
r_s(z*) = 145.21 Mpc
r_s(z_drag) = 147.85 +0.18 -0.19 Mpc
Chi squared: 1653.31 (Δ chi2 10.22)
Degrees of freedom: 1746

===============================

Flat w(z) = w0 + wa * z / (1 + z)
H0: 66.73 ± 0.56 km/s/Mpc
Ωm: 0.3180 +0.0058 -0.0057
Ωb h^2: 0.02226 ± 0.00013
Ωm h^2: 0.14163 +0.00095 -0.00097
w0: -0.770 +0.058 -0.057
wa: -0.744 +0.231 -0.240
ΔM: -0.064 ± 0.012
z*: 1088.87 ± 0.18
z_drag: 1059.56 ± 0.28
r_s(z*) = 144.79 Mpc
r_s(z_drag) = 147.51 +0.22 -0.21 Mpc
Chi squared: 1646.19 (Δ chi2 17.34)
Degrees of freedom: 1745
"""
