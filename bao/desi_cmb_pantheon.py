from numba import njit
import numpy as np
import emcee
import corner
from scipy.integrate import cumulative_trapezoid, quad
from scipy.linalg import cho_factor, cho_solve
import matplotlib.pyplot as plt
from multiprocessing import Pool
import cmb.data_chen_compression as cmb
from y2022pantheonSHOES.data import get_data
from y2025BAO.data import get_data as get_bao_data
from hubble.plotting import plot_predictions as plot_sn_predictions
from .plot_predictions import plot_bao_predictions


c = cmb.c  # km/s
Or_h2 = cmb.Omega_r_h2()

sn_legend, z_cmb, z_hel, mb_values, cov_matrix_sn = get_data()
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


def integral_Ez(params):
    integral_values = cumulative_trapezoid(1 / Ez(sn_grid, params), sn_grid, initial=0)
    return np.interp(z_cmb, sn_grid, integral_values)


def apparent_mag(params):
    H0, M = params[0], params[-1]
    dL = one_plus_z_hel * (c / H0) * integral_Ez(params)
    return M + 25 + 5 * np.log10(dL)


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
    H0_50, Om_50, Obh2_50 = params[0], params[1], params[2]
    Omh2_50 = Om_50 * (H0_50 / 100) ** 2
    rd = cmb.r_drag(wb=Obh2_50, wm=Omh2_50)

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

    delta_sn = mb_values - apparent_mag(params)
    chi_sn = np.dot(delta_sn, cho_solve(cho_sn, delta_sn))

    return chi2_cmb + chi_bao + chi_sn


bounds = np.array(
    [
        (60, 75),  # H0
        (0.1, 0.6),  # Ωm
        (0.019, 0.025),  # Ωb * h^2
        (-2, 0),  # w0
        (-20, -19),  # M
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
    nwalkers = 10 * ndim
    burn_in = 500
    nsteps = 8000 + burn_in
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
    (H0_16, H0_50, H0_84) = pct[0]
    (Om_16, Om_50, Om_84) = pct[1]
    (Obh2_16, Obh2_50, Obh2_84) = pct[2]
    (w0_16, w0_50, w0_84) = pct[3]
    (M_16, M_50, M_84) = pct[4]

    best_fit = np.array([H0_50, Om_50, Obh2_50, w0_50, M_50], dtype=np.float64)

    Omh2_50 = Om_50 * (H0_50 / 100) ** 2
    z_st = cmb.z_star(Obh2_50, Omh2_50)
    z_dr = cmb.z_drag(Obh2_50, Omh2_50)

    print(f"H0: {H0_50:.2f} +{(H0_84 - H0_50):.2f} -{(H0_50 - H0_16):.2f} km/s/Mpc")
    print(f"Ωm: {Om_50:.3f} +{(Om_84 - Om_50):.3f} -{(Om_50 - Om_16):.3f}")
    print(
        f"Ωb h^2: {Obh2_50:.5f} +{(Obh2_84 - Obh2_50):.5f} -{(Obh2_50 - Obh2_16):.5f}"
    )
    print(f"w0: {w0_50:.3f} +{(w0_84 - w0_50):.3f} -{(w0_50 - w0_16):.3f}")
    print(f"M: {M_50:.3f} +{(M_84 - M_50):.3f} -{(M_50 - M_16):.3f}")
    print(f"z*: {z_st:.2f}")
    print(f"z_drag: {z_dr:.2f}")
    print(f"r_s(z*) = {cmb.rs_z(Ez, z_st, best_fit, H0_50, Obh2_50):.2f} Mpc")
    print(f"r_s(z_drag) = {cmb.rs_z(Ez, z_dr, best_fit, H0_50, Obh2_50):.2f} Mpc")
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
        y=mb_values,
        y_err=np.sqrt(np.diag(cov_matrix_sn)),
        y_model=apparent_mag(best_fit),
        label=f"Model: $Ω_m$={Om_50:.3f}",
        x_scale="log",
    )
    labels = ["$H_0$", "$Ω_m$", "$Ω_b h^2$", "$w_0$", "M"]
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
H0: 68.51 ± 0.30 km/s/Mpc
Ωm: 0.302 ± 0.004
Ωb h^2: 0.02252 ± 0.00012
w0: -1
M: -19.408 ± 0.009
z*: 1088.58
z_drag: 1060.14
r_s(z*) = 144.64 Mpc
r_s(z_drag) = 147.16 Mpc
Chi squared: 1421.09
Degrees of freedom: 1603

===============================

Flat wCDM w(z) = w0
H0: 68.04 +0.59 -0.58 km/s/Mpc
Ωm: 0.305 ± 0.005
Ωb h^2: 0.02256 ± 0.00013
w0: -0.978 ± 0.024
M: -19.418 ± 0.014
z*: 1088.51
z_drag: 1060.19
r_s(z*) = 144.74 Mpc
r_s(z_drag) = 147.26 Mpc
Chi squared: 1420.21
Degrees of freedom: 1602

===============================

Flat w(z) = -1 + 2 * (1 + w0) / (1 + (1 + z)**3)
H0: 67.66 ± 0.59 km/s/Mpc
Ωm: 0.308 +0.006 -0.005
Ωb h^2: 0.02257 ± 0.00013
w0: -0.938 +0.038 -0.037
M: -19.424 ± 0.013
z*: 1088.50
z_drag: 1060.20
r_s(z*) = 144.74 Mpc
r_s(z_drag) = 147.26 Mpc
Chi squared: 1418.36
Degrees of freedom: 1602

===============================

Flat w(z) = w0 + wa * z / (1 + z)
H0: 67.75 +0.61 -0.60 km/s/Mpc
Ωm: 0.311 +0.006 -0.006
Ωb h^2: 0.02244 +0.00014 -0.00014
w0: -0.848 +0.057 -0.055
wa: -0.545 +0.212 -0.231
M: -19.412 +0.015 -0.015
z*: 1088.74
z_drag: 1060.04
r_s(z*) = 144.41 Mpc
r_s(z_drag) = 146.95 Mpc
Chi squared: 1413.56
Degrees of freedom: 1601
"""
