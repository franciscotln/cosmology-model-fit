from numba import njit
import numpy as np
import emcee
import corner
from scipy.constants import c as c0
from scipy.integrate import cumulative_trapezoid, quad
from scipy.linalg import cho_factor, cho_solve
import matplotlib.pyplot as plt
from multiprocessing import Pool
from y2024DES.data import effective_sample_size as sn_sample, get_data as get_sn_data
from y2005cc.data import get_data as get_cc_data
from y2025BAO.data import get_data as get_bao_data
from sn.plotting import plot_predictions as plot_sn_predictions
from cosmic_chronometers.plot_predictions import plot_cc_predictions
from .plot_predictions import plot_bao_predictions

cc_legend, z_cc_vals, H_cc_vals, cov_matrix_cc = get_cc_data()
sn_legend, z_sn_vals, z_sn_hel_vals, mu_values, cov_matrix_sn = get_sn_data()
bao_legend, bao_data, cov_matrix_bao = get_bao_data()

cho_sn = cho_factor(cov_matrix_sn)
cho_bao = cho_factor(cov_matrix_bao)
cho_cc = cho_factor(cov_matrix_cc)
logdet_cc = np.linalg.slogdet(cov_matrix_cc)[1]
N_cc = len(z_cc_vals)

c = c0 / 1000  # km/s

z_grid_sn = np.linspace(0, np.max(z_sn_vals), num=1000)
one_plus_z_hel = 1 + z_sn_hel_vals


@njit
def Ez(z, p):
    Om, exp_w0 = p[4], p[5]
    one_plus_z = 1 + z
    cubed = one_plus_z**3
    rho_de = (2 * cubed / (1 + cubed)) ** (2 * (1 + np.log(exp_w0)))
    return np.sqrt(Om * cubed + (1 - Om) * rho_de)


def integral_Ez(params):
    y = 1 / Ez(z_grid_sn, params)
    integral_values = cumulative_trapezoid(y=y, x=z_grid_sn, initial=0)
    return np.interp(z_sn_vals, z_grid_sn, integral_values)


def mu_theory(params):
    mag_offset, H0 = params[1], params[2]
    dL = one_plus_z_hel * (c / H0) * integral_Ez(params)
    return mag_offset + 25 + 5 * np.log10(dL)


@njit
def H_z(z, p):
    return p[2] * Ez(z, p)


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
def bao_theory(z, qty, params):
    DV_mask = qty == 0
    DM_mask = qty == 1
    DH_mask = qty == 2
    results = np.empty(z.size, dtype=np.float64)
    results[DH_mask] = DH_z(z[DH_mask], params)
    results[DM_mask] = DM_z(z[DM_mask], params)
    results[DV_mask] = DV_z(z[DV_mask], params)
    return results / params[3]


bounds = np.array(
    [
        (0.4, 2.5),  # f_cc
        (-0.55, 0.55),  # ΔM
        (50, 80),  # H0
        (110, 175),  # r_d
        (0.2, 0.7),  # Ωm
        (0.15, 0.75),  # e^w0
    ],
    dtype=np.float64,
)


def chi_squared(params):
    delta_sn = mu_values - mu_theory(params)
    chi_sn = delta_sn.dot(cho_solve(cho_sn, delta_sn, check_finite=False))

    delta_bao = bao_data["value"] - bao_theory(bao_data["z"], quantities, params)
    chi_bao = delta_bao.dot(cho_solve(cho_bao, delta_bao, check_finite=False))

    delta_cc = H_cc_vals - H_z(z_cc_vals, params)
    chi_cc = params[0] ** 2 * delta_cc.dot(
        cho_solve(cho_cc, delta_cc, check_finite=False)
    )
    return chi_sn + chi_bao + chi_cc


@njit
def log_prior(params):
    if np.all((bounds[:, 0] < params) & (params < bounds[:, 1])):
        return -np.log(params[5])  # flat prior in w0
    return -np.inf


def log_likelihood(params):
    f_cc = params[0]
    normalization_cc = N_cc * np.log(2 * np.pi) + logdet_cc - 2 * N_cc * np.log(f_cc)
    return -0.5 * chi_squared(params) - 0.5 * normalization_cc


def log_probability(params):
    lp = log_prior(params)
    if not np.isfinite(lp):
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
        print("auto-correlation time", tau)
        print("acceptance fraction", np.mean(sampler.acceptance_fraction))
        print("effective samples", ndim * nwalkers * (nsteps - burn_in) / np.max(tau))
    except emcee.autocorr.AutocorrError as e:
        print("Autocorrelation time could not be computed", e)

    samples = sampler.get_chain(discard=burn_in, flat=True)
    print("correlation matrix:")
    print(np.array2string(np.corrcoef(samples, rowvar=False), precision=5))

    [
        [f_cc_16, f_cc_50, f_cc_84],
        [dM_16, dM_50, dM_84],
        [h0_16, h0_50, h0_84],
        [rd_16, rd_50, rd_84],
        [Om_16, Om_50, Om_84],
        [exp_w0_16, exp_w0_50, exp_w0_84],
    ] = np.percentile(samples, [15.9, 50, 84.1], axis=0).T

    best_fit = np.array(
        [f_cc_50, dM_50, h0_50, rd_50, Om_50, exp_w0_50], dtype=np.float64
    )

    w0_samples = np.log(samples[:, 5])
    w0_16, w0_50, w0_84 = np.percentile(w0_samples, [15.9, 50, 84.1])

    deg_of_freedom = sn_sample + bao_data["value"].size + z_cc_vals.size - ndim

    print(f"f_cc: {f_cc_50:.2f} +{(f_cc_84 - f_cc_50):.2f} -{(f_cc_50 - f_cc_16):.2f}")
    print(f"ΔM: {dM_50:.3f} +{(dM_84 - dM_50):.3f} -{(dM_50 - dM_16):.3f}")
    print(f"H0: {h0_50:.1f} +{(h0_84 - h0_50):.1f} -{(h0_50 - h0_16):.1f}")
    print(f"r_d: {rd_50:.1f} +{(rd_84 - rd_50):.1f} -{(rd_50 - rd_16):.1f}")
    print(f"Ωm: {Om_50:.3f} +{(Om_84 - Om_50):.3f} -{(Om_50 - Om_16):.3f}")
    print(f"w0: {w0_50:.3f} +{(w0_84 - w0_50):.3f} -{(w0_50 - w0_16):.3f}")
    print(f"Chi squared: {chi_squared(best_fit):.2f}")
    print(f"Degrees of freedom: {deg_of_freedom}")

    plot_bao_predictions(
        theory_predictions=lambda z, qty: bao_theory(z, qty, best_fit),
        data=bao_data,
        errors=np.sqrt(np.diag(cov_matrix_bao)),
        title=f"{bao_legend}: $r_d$={rd_50:.1f} Mpc",
    )
    plot_cc_predictions(
        H_z=lambda z: H_z(z, best_fit),
        z=z_cc_vals,
        H=H_cc_vals,
        H_err=np.sqrt(np.diag(cov_matrix_cc)) / f_cc_50,
        label=f"{cc_legend} $H_0$: {h0_50:.1f} km/s/Mpc",
    )
    plot_sn_predictions(
        legend=sn_legend,
        x=z_sn_vals,
        y=mu_values,
        y_err=np.sqrt(np.diag(cov_matrix_sn)),
        y_model=mu_theory(best_fit),
        label=rf"Best fit: $H_0$={h0_50:.1f} km/s/Mpc, $Ω_m$={Om_50:.3f}",
        x_scale="log",
    )

    labels = ["$f_{CCH}$", "ΔM", "$H_0$", "$r_d$", "Ωm", "$e^{w_0}$"]
    corner.corner(
        samples,
        labels=labels,
        quantiles=[0.159, 0.5, 0.841],
        show_titles=True,
        title_fmt=".3f",
        smooth=2.0,
        smooth1d=2.0,
        bins=100,
        levels=(0.393, 0.864),  # 1 and 2 sigmas in 2D
        fill_contours=False,
        plot_datapoints=False,
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
Flat ΛCDM: w(z) = -1
f_cc: 1.48 +0.19 -0.18
ΔM: -0.059 +0.069 -0.072 mag
H0: 68.3 +2.2 -2.3 km/s/Mpc
r_d: 147.3 +4.9 -4.6 Mpc
Ωm: 0.311 +0.008 -0.008
w0: -1
wa: 0
Chi squared: 1691.26
Degrees of freedom: 1776

===============================

Flat wCDM: w(z) = w0
f_cc: 1.46 +0.19 -0.18
ΔM: -0.065 +0.070 -0.072 mag
H0: 67.2 +2.2 -2.3 km/s/Mpc
r_d: 147.2 +4.9 -4.6 Mpc
Ωm: 0.299 +0.009 -0.009
w0: -0.875 +0.038 -0.039
wa: 0
Chi squared: 1680.36 (Δ chi2 10.90)
Degrees of freedom: 1775

===============================

Flat w(z) = -1 + 2 * (1 + w0) / (1 + (1 + z)**3)
f_cc: 1.46 +0.19 -0.18
ΔM: -0.063 +0.070 -0.072 mag
H0: 67.0 +2.3 -2.2 km/s/Mpc
r_d: 147.2 +4.9 -4.6 Mpc
Ωm: 0.308 ± 0.008
w0: -0.839 +0.044 -0.046
wa: -(1 + w0) = -0.161 +0.046 -0.044
Chi squared: 1678.82 (Δ chi2 12.44)
Degrees of freedom: 1775

===============================

Flat w0waCDM: w(z) = w0 + wa * z / (1 + z)
f_cc: 1.458 +0.182 -0.176
ΔM: -0.059 +0.071 -0.073 mag
H0: 67.0 ± 2.3 km/s/Mpc
r_d: 147.1 +5.0 -4.7 Mpc
Ωm: 0.320 +0.013 -0.016
w0: -0.796 +0.071 -0.067
wa: -0.650 ± 0.459
Chi squared: 1677.85 (Δ chi2 13.37)
Degrees of freedom: 1774
"""
