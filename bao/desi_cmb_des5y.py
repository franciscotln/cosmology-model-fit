from numba import njit
import numpy as np
import emcee
import corner
from scipy.integrate import cumulative_trapezoid
from scipy.linalg import cho_factor, cho_solve
import matplotlib.pyplot as plt
from multiprocessing import Pool
from y2024DES.data import get_data
from y2025BAO.data import get_data as get_bao_data
import cmb.data_desi_compression as cmb
from sn.plotting import plot_predictions as plot_sn_predictions
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
    H0, Om, w0 = params[1], params[2], params[4]
    h = H0 / 100
    Or = Or_h2 / h**2
    Ode = 1 - Om - Or
    one_plus_z = 1 + z
    rho_de = (2 * one_plus_z**3 / (1 + one_plus_z**3)) ** (2 * (1 + w0))

    return np.sqrt(Or * one_plus_z**4 + Om * one_plus_z**3 + Ode * rho_de)


def theory_mu(params):
    H0, offset_mag = params[1], params[-1]
    integral_vals = cumulative_trapezoid(1 / Ez(sn_grid, params), sn_grid, initial=0)
    I = np.interp(z_cmb, sn_grid, integral_vals)
    dL = one_plus_z_hel * I * c / H0
    return offset_mag + 25 + 5 * np.log10(dL)


@njit
def H_z(z, params):
    return params[1] * Ez(z, params)


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

quantities = np.array([qty_map[q] for q in bao_data["quantity"]], dtype=np.int64)


@njit
def bao_theory(z, qty, params):
    results = np.empty(z.size, dtype=np.float64)
    DV_mask = qty == 0
    DM_mask = qty == 1
    DH_mask = qty == 2
    results[DH_mask] = DH_z(z[DH_mask], params)
    results[DM_mask] = DM_z(z[DM_mask], params)
    results[DV_mask] = DV_z(z[DV_mask], params)
    return results / params[0]


def chi_squared(params):
    H0, Om, Obh2 = params[1], params[2], params[3]

    delta = cmb.DISTANCE_PRIORS - cmb.cmb_distances(Ez, params, H0, Om, Obh2)
    chi2_cmb = np.dot(delta, np.dot(cmb.inv_cov_mat, delta))

    delta_bao = bao_data["value"] - bao_theory(bao_data["z"], quantities, params)
    chi_bao = np.dot(delta_bao, cho_solve(cho_bao, delta_bao, check_finite=False))

    delta_sn = mu_values - theory_mu(params)
    chi_sn = np.dot(delta_sn, cho_solve(cho_sn, delta_sn, check_finite=False))

    return chi2_cmb + chi_bao + chi_sn


bounds = np.array(
    [
        (120, 160),  # r_d
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
        return 0.0
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
    nsteps = 2000 + burn_in
    initial_pos = np.zeros((nwalkers, ndim))

    for dim, (lower, upper) in enumerate(bounds):
        initial_pos[:, dim] = np.random.uniform(lower, upper, nwalkers)

    with Pool(6) as pool:
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
        print("Auto-correlation time", tau)
        print("Mean acceptance fraction:", np.mean(sampler.acceptance_fraction))
        print("Effective samples:", nwalkers * ndim * (nsteps - burn_in) / np.max(tau))
    except emcee.autocorr.AutocorrError as e:
        print("Autocorrelation time could not be computed", e)

    samples = sampler.get_chain(discard=burn_in, flat=True)

    pct = np.percentile(samples, [15.9, 50, 84.1], axis=0).T
    rd_16, rd_50, rd_84 = pct[0]
    H0_16, H0_50, H0_84 = pct[1]
    Om_16, Om_50, Om_84 = pct[2]
    Obh2_16, Obh2_50, Obh2_84 = pct[3]
    w0_16, w0_50, w0_84 = pct[4]
    dM_16, dM_50, dM_84 = pct[5]

    best_fit = np.array([rd_50, H0_50, Om_50, Obh2_50, w0_50, dM_50], dtype=np.float64)

    one_sigma_contours = [15.9, 50, 84.1]

    Omh2_samples = samples[:, 2] * (samples[:, 1] / 100) ** 2
    z_star_samples = cmb.z_star(samples[:, 3], Omh2_samples)
    Omh2_16, Omh2_50, Omh2_84 = np.percentile(Omh2_samples, one_sigma_contours)
    z_star_16, z_star_50, z_star_84 = np.percentile(z_star_samples, one_sigma_contours)

    print(f"r_d: {rd_50:.2f} +{(rd_84 - rd_50):.2f} -{(rd_50 - rd_16):.2f} Mpc")
    print(f"H0: {H0_50:.2f} +{(H0_84 - H0_50):.2f} -{(H0_50 - H0_16):.2f} km/s/Mpc")
    print(f"Ωm: {Om_50:.4f} +{(Om_84 - Om_50):.4f} -{(Om_50 - Om_16):.4f}")
    print(
        f"Ωb h^2: {Obh2_50:.5f} +{(Obh2_84 - Obh2_50):.5f} -{(Obh2_50 - Obh2_16):.5f}"
    )
    print(
        f"Ωm h^2: {Omh2_50:.5f} +{(Omh2_84 - Omh2_50):.5f} -{(Omh2_50 - Omh2_16):.5f}"
    )
    print(f"w0: {w0_50:.3f} +{(w0_84 - w0_50):.3f} -{(w0_50 - w0_16):.3f}")
    print(f"r*: {cmb.rs_z(Ez, z_star_50, best_fit, H0_50, Obh2_50):.2f} Mpc")
    print(
        f"z*: {z_star_50:.2f} +{(z_star_84 - z_star_50):.2f} -{(z_star_50 - z_star_16):.2f}"
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
    labels = ["$r_d$", "$H_0$", "$Ω_m$", "$Ω_b h^2$", "$w_0$", "$Δ_M$"]
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
r_d: 148.08 +0.50 -0.49 Mpc
H0: 68.01 +0.40 -0.40 km/s/Mpc
Ωm: 0.3079 +0.0054 -0.0053
Ωb h^2: 0.02220 +0.00012 -0.00012
Ωm h^2: 0.14243 +0.00086 -0.00086
w0: -1
r*: 144.34 Mpc
z*: 1092.10 +0.22 -0.21
Chi squared: 1659.19
Degrees of freedom: 1746

===============================

Flat wCDM w(z) = w0
r_d: 148.08 +0.50 -0.50 Mpc
H0: 67.05 +0.57 -0.56 km/s/Mpc
Ωm: 0.3135 +0.0060 -0.0060
Ωb h^2: 0.02232 +0.00014 -0.00014
Ωm h^2: 0.14092 +0.00108 -0.00107
w0: -0.946 +0.023 -0.022
r*: 144.67 Mpc
z*: 1091.79 +0.26 -0.25
Chi squared: 1653.89 (Δ chi2 5.30)
Degrees of freedom: 1745

===============================

Flat w(z) = -1 + 2 * (1 + w0) / (1 + (1 + z)**3)
r_d: 147.85 +0.50 -0.50 Mpc
H0: 66.73 +0.57 -0.56 km/s/Mpc
Ωm: 0.3163 +0.0063 -0.0061
Ωb h^2: 0.02233 +0.00013 -0.00013
Ωm h^2: 0.14086 +0.00100 -0.00100
w0: -0.885 +0.037 -0.037
r*: 144.69 Mpc
z*: 1091.78 +0.24 -0.24
Chi squared: 1650.00 (Δ chi2 9.19)
Degrees of freedom: 1745

===============================

Flat w(z) = w0 + wa * z / (1 + z)
r_d: 147.38 +0.53 -0.53 Mpc
H0: 66.87 +0.57 -0.56 km/s/Mpc
Ωm: 0.3182 +0.0063 -0.0062
Ωb h^2: 0.02221 +0.00014 -0.00014
Ωm h^2: 0.14226 +0.00113 -0.00115
w0: -0.789 +0.057 -0.057
wa: -0.640 +0.220 -0.231
r*: 144.38 Mpc
z*: 1092.07 +0.27 -0.27
Chi squared: 1645.53 (Δ chi2 13.66)
Degrees of freedom: 1744
"""
