from numba import njit
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.linalg import cho_factor, cho_solve
from y2024DES.data import get_data, effective_sample_size as sn_size
from y2025BAO.data import get_data as get_bao_data

sn_legend, z_cmb, z_hel, mu_values, cov_matrix_sn = get_data()
cho_sn = cho_factor(cov_matrix_sn)
bao_legend, bao_data, cov_matrix_bao = get_bao_data()
cho_bao = cho_factor(cov_matrix_bao)

c = 299792.458  # Speed of light in km/s

grid = np.linspace(0, np.max(z_cmb), num=1000)
zhel_plus1 = 1 + z_hel


@njit
def Ez(z, params):
    Om, w0 = params[3], params[4]
    z_plus_1 = 1 + z
    cubed = z_plus_1**3
    rho_de = (2 * cubed / (1 + cubed)) ** (2 * (1 + w0))
    return np.sqrt(Om * cubed + (1 - Om) * rho_de)


def theory_mu(params):
    y = 1 / Ez(grid, params)
    I = np.interp(z_cmb, grid, cumulative_trapezoid(y=y, x=grid, initial=0))
    return params[0] + 25 + 5 * np.log10(zhel_plus1 * c * I / params[2])


@njit
def H_z(z, params):
    return params[2] * Ez(z, params)


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
    DV_mask = qty == 0
    DM_mask = qty == 1
    DH_mask = qty == 2
    results = np.empty(z.size, dtype=np.float64)
    results[DH_mask] = DH_z(z[DH_mask], params)
    results[DM_mask] = DM_z(z[DM_mask], params)
    results[DV_mask] = DV_z(z[DV_mask], params)
    return results / params[1]


# Planck prior
Omh2_planck = 0.1430
Omh2_planck_sigma = 0.0011


def chi_squared(params):
    Omh2 = params[3] * (params[2] / 100) ** 2
    chi2_prior = ((Omh2_planck - Omh2) / Omh2_planck_sigma) ** 2

    delta_sn = mu_values - theory_mu(params)
    chi_sn = delta_sn.dot(cho_solve(cho_sn, delta_sn, check_finite=False))

    delta_bao = bao_data["value"] - bao_theory(bao_data["z"], quantities, params)
    chi_bao = delta_bao.dot(cho_solve(cho_bao, delta_bao, check_finite=False))
    return chi_sn + chi_bao + chi2_prior


bounds = np.array(
    [
        (-0.6, 0.6),  # ΔM
        (120.0, 160.0),  # r_d
        (60.0, 80.0),  # H0
        (0.1, 0.7),  # Ωm
        (-2.0, 0.0),  # w0
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
    import emcee, corner
    import matplotlib.pyplot as plt
    from multiprocessing import Pool
    from sn.plotting import plot_predictions as plot_sn_predictions
    from .plot_predictions import plot_bao_predictions

    np.random.seed(42)
    ndim = len(bounds)
    nwalkers = 150
    burn_in = 200
    nsteps = 2000 + burn_in
    initial_pos = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(nwalkers, ndim))

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
        print("acceptance fraction:", np.mean(sampler.acceptance_fraction))
        print("effective samples", ndim * nwalkers * nsteps / np.max(tau))
    except emcee.autocorr.AutocorrError as e:
        print("Autocorrelation time could not be computed", e)

    chains_samples = sampler.get_chain(discard=burn_in, flat=False)
    samples = sampler.get_chain(discard=burn_in, flat=True)

    [
        [dM_16, dM_50, dM_84],
        [rd_16, rd_50, rd_84],
        [H0_16, H0_50, H0_84],
        [Om_16, Om_50, Om_84],
        [w0_16, w0_50, w0_84],
    ] = np.percentile(samples, [15.9, 50, 84.1], axis=0).T

    best_fit = np.array([dM_50, rd_50, H0_50, Om_50, w0_50])

    print(f"ΔM: {dM_50:.3f} +{(dM_84 - dM_50):.3f} -{(dM_50 - dM_16):.3f} mag")
    print(f"r_d: {rd_50:.2f} +{(rd_84 - rd_50):.2f} -{(rd_50 - rd_16):.2f} Mpc")
    print(f"H0: {H0_50:.2f} +{(H0_84 - H0_50):.2f} -{(H0_50 - H0_16):.2f} km/s/Mpc")
    print(f"Ωm: {Om_50:.3f} +{(Om_84 - Om_50):.3f} -{(Om_50 - Om_16):.3f}")
    print(f"w0: {w0_50:.3f} +{(w0_84 - w0_50):.3f} -{(w0_50 - w0_16):.3f}")
    print(f"Chi squared: {chi_squared(best_fit):.2f}")
    print(f"Degrees of freedom: {1 + bao_data['value'].size + sn_size - len(best_fit)}")

    plot_bao_predictions(
        theory_predictions=lambda z, qty: bao_theory(z, qty, best_fit),
        data=bao_data,
        errors=np.sqrt(np.diag(cov_matrix_bao)),
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

    labels = ["$Δ_M$", "$r_d$", "$H_0$", "$Ω_M$", "$w_0$"]
    corner.corner(
        samples,
        labels=labels,
        quantiles=[0.159, 0.5, 0.841],
        show_titles=True,
        title_fmt=".4f",
        smooth=1.5,
        smooth1d=1.5,
        bins=100,
        levels=(0.393, 0.864),  # 1 and 2 sigmas in 2D
        fill_contours=False,
        plot_datapoints=False,
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
Flat ΛCDM
ΔM: -0.071 +0.024 -0.024 mag
r_d: 148.13 +1.22 -1.21 Mpc
H0: 67.87 +0.89 -0.89 km/s/Mpc
Ωm: 0.310 +0.008 -0.008
w0: -1
Chi squared: 1658.97
Degrees of freedom: 1745

===============================

Flat wCDM
ΔM: 0.002 +0.036 -0.034 mag
r_d: 142.75 +2.12 -2.26 Mpc
H0: 69.27 +1.09 -1.04 km/s/Mpc
Ωm: 0.298 +0.009 -0.009
w0: -0.871 +0.038 -0.038
Chi squared: 1648.10 (Δ chi2 10.87)
Degrees of freedom: 1744

===============================

Flat w0 - (1 + w0) * (((1 + z)**3 - 1) / ((1 + z)**3 + 1))
ΔM: -0.025 +0.028 -0.028 mag
r_d: 144.64 +1.59 -1.57 Mpc
H0: 68.19 +0.92 -0.90 km/s/Mpc
Ωm: 0.308 +0.008 -0.008
w0: -0.834 +0.045 -0.046
Chi squared: 1646.49 (Δ chi2 12.48)
Degrees of freedom: 1744

===============================

Flat w(z) = w0 + wa * z / (1 + z)
ΔM: -0.065 +0.046 -0.038 mag
r_d: 147.68 +2.54 -3.20 Mpc
H0: 66.73 +1.64 -1.33 km/s/Mpc
Ωm: 0.321 +0.013 -0.015
w0: -0.784 +0.073 -0.067
wa: -0.718 +0.450 -0.460
Chi squared: 1645.45 (Δ chi2 13.52)
Degrees of freedom: 1743
"""
