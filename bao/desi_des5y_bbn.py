from numba import njit
import numpy as np
import emcee, corner
from scipy.constants import c as c0
from scipy.integrate import cumulative_trapezoid
from scipy.linalg import cho_factor, cho_solve
import matplotlib.pyplot as plt
from multiprocessing import Pool
import y2024BBN.prior_lcdm_shonberg as bbn
from y2024DES.data import get_data, effective_sample_size as sn_size
from y2025BAO.data import get_data as get_bao_data
from sn.plotting import plot_predictions as plot_sn_predictions
from .plot_predictions import plot_bao_predictions


c = c0 / 1000  # km/s

sn_legend, z_cmb, z_hel, mu_values, cov_matrix_sn = get_data()
bao_legend, bao_data, bao_cov_matrix = get_bao_data()
cho_sn = cho_factor(cov_matrix_sn)
cho_bao = cho_factor(bao_cov_matrix)

sn_grid = np.linspace(0, np.max(z_cmb), num=1000)
one_plus_z_hel = 1 + z_hel


@njit
def r_drag(wb, wm, n_eff=3.04):  # arXiv:2503.14738v2 (eq 2)
    return (
        147.05 * (0.02236 / wb) ** 0.13 * (0.1432 / wm) ** 0.23 * (3.04 / n_eff) ** 0.1
    )


@njit
def Ez(z, params):
    Om, w0 = params[1], params[3]
    Ode = 1 - Om
    one_plus_z = 1 + z
    cubed = one_plus_z**3
    rho_de = (2 * cubed / (1 + cubed)) ** (2 * (1 + w0))

    return np.sqrt(Om * cubed + Ode * rho_de)


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
    H0, Om, Obh2 = params[0], params[1], params[2]
    Omh2 = Om * (H0 / 100) ** 2
    rd = r_drag(wb=Obh2, wm=Omh2)
    results = np.empty(z.size, dtype=np.float64)
    DV_mask = qty == 0
    DM_mask = qty == 1
    DH_mask = qty == 2
    results[DH_mask] = DH_z(z[DH_mask], params)
    results[DM_mask] = DM_z(z[DM_mask], params)
    results[DV_mask] = DV_z(z[DV_mask], params)
    return results / rd


def chi_squared(params):
    delta_bbn = bbn.Obh2 - params[2]
    chi2_bbn = (delta_bbn / bbn.Obh2_sigma) ** 2

    delta_bao = bao_data["value"] - bao_theory(bao_data["z"], quantities, params)
    chi_bao = np.dot(delta_bao, cho_solve(cho_bao, delta_bao, check_finite=False))

    delta_sn = mu_values - theory_mu(params)
    chi_sn = np.dot(delta_sn, cho_solve(cho_sn, delta_sn, check_finite=False))

    return chi2_bbn + chi_bao + chi_sn


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
        print("auto-correlation time", tau)
        print("mean acceptance fraction", np.mean(sampler.acceptance_fraction))
        print("effective samples", ndim * nwalkers * (nsteps - burn_in) / np.max(tau))
    except emcee.autocorr.AutocorrError as e:
        print("Autocorrelation time could not be computed", e)

    samples = sampler.get_chain(discard=burn_in, flat=True)
    one_sigma_contours = [15.9, 50, 84.1]

    pct = np.percentile(samples, one_sigma_contours, axis=0).T
    H0_16, H0_50, H0_84 = pct[0]
    Om_16, Om_50, Om_84 = pct[1]
    Obh2_16, Obh2_50, Obh2_84 = pct[2]
    w0_16, w0_50, w0_84 = pct[3]
    dM_16, dM_50, dM_84 = pct[4]

    best_fit = np.array([H0_50, Om_50, Obh2_50, w0_50, dM_50], dtype=np.float64)

    Omh2_samples = samples[:, 1] * (samples[:, 0] / 100) ** 2
    r_drag_samples = r_drag(samples[:, 2], Omh2_samples)

    Omh2_16, Omh2_50, Omh2_84 = np.percentile(Omh2_samples, one_sigma_contours)
    r_d_16, r_d_50, r_d_84 = np.percentile(r_drag_samples, one_sigma_contours)

    print(f"H0: {H0_50:.2f} +{(H0_84 - H0_50):.2f} -{(H0_50 - H0_16):.2f} km/s/Mpc")
    print(f"Ωm: {Om_50:.4f} +{(Om_84 - Om_50):.4f} -{(Om_50 - Om_16):.4f}")
    print(f"ωb: {Obh2_50:.5f} +{(Obh2_84 - Obh2_50):.5f} -{(Obh2_50 - Obh2_16):.5f}")
    print(f"ωm: {Omh2_50:.5f} +{(Omh2_84 - Omh2_50):.5f} -{(Omh2_50 - Omh2_16):.5f}")
    print(f"w0: {w0_50:.3f} +{(w0_84 - w0_50):.3f} -{(w0_50 - w0_16):.3f}")
    print(f"ΔM: {dM_50:.3f} +{(dM_84 - dM_50):.3f} -{(dM_50 - dM_16):.3f}")
    print(f"r_d: {r_d_50:.2f} +{(r_d_84 - r_d_50):.2f} -{(r_d_50 - r_d_16):.2f} Mpc")
    print(f"Chi squared: {chi_squared(best_fit):.2f}")
    print(f"Degrees of freedom: {1 + bao_data['z'].size + sn_size - len(best_fit)}")

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
    labels = ["$H_0$", "$Ω_m$", "$ω_b$", "$w_0$", "$Δ_M$"]
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
H0: 68.63 +0.53 -0.53 km/s/Mpc
Ωm: 0.3105 +0.0079 -0.0077
ωb: 0.02219 +0.00054 -0.00054
ωm: 0.14622 +0.00435 -0.00424
w0: -1
ΔM: -0.047 +0.018 -0.018
r_d: 146.49 +1.27 -1.23 Mpc
Chi squared: 1658.97
Degrees of freedom: 1745

===============================

Flat wCDM w(z) = w0
H0: 65.37 +1.16 -1.18 km/s/Mpc
Ωm: 0.2979 +0.0090 -0.0089
ωb: 0.02218 +0.00055 -0.00055
ωm: 0.12739 +0.00723 -0.00719
w0: -0.871 +0.038 -0.038
ΔM: -0.123 +0.031 -0.033
r_d: 151.22 +2.18 -2.07 Mpc
Chi squared: 1648.09 (delta chi2 10.88)
Degrees of freedom: 1744

===============================

Flat w(z) = -1 + 2 * (1 + w0) / (1 + (1 + z)**3)
H0: 65.97 +0.91 -0.90 km/s/Mpc
Ωm: 0.3075 +0.0079 -0.0077
ωb: 0.02218 +0.00055 -0.00054
ωm: 0.13386 +0.00533 -0.00519
w0: -0.835 +0.045 -0.045
ΔM: -0.096 +0.023 -0.023
r_d: 149.51 +1.58 -1.55 Mpc
Chi squared: 1646.49 (delta chi2 12.48)
Degrees of freedom: 1744

===============================

Flat w(z) = w0 + wa * z / (1 + z)
H0: 66.88 +1.17 -1.34 km/s/Mpc
Ωm: 0.3215 +0.0127 -0.0150
ωb: 0.02196 +0.00063 -0.00063
ωm: 0.14403 +0.00927 -0.01115
w0: -0.783 +0.072 -0.068
wa: -0.728 +0.452 -0.453
ΔM: -0.059 +0.038 -0.046
r_d: 147.25 +2.90 -2.30 Mpc
Chi squared: 1645.55 (delta chi2 13.42)
Degrees of freedom: 1743
"""
