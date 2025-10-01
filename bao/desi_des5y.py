from numba import njit
import numpy as np
import emcee
import corner
from scipy.integrate import cumulative_trapezoid
from scipy.linalg import cho_factor, cho_solve
import matplotlib.pyplot as plt
from multiprocessing import Pool
from y2024DES.data import get_data, effective_sample_size as sn_size
from y2025BAO.data import get_data as get_bao_data
from sn.plotting import plot_predictions as plot_sn_predictions
from .plot_predictions import plot_bao_predictions

sn_legend, z_cmb, z_hel, mu_values, cov_matrix_sn = get_data()
cho_sn = cho_factor(cov_matrix_sn)
bao_legend, bao_data, cov_matrix_bao = get_bao_data()
cho_bao = cho_factor(cov_matrix_bao)

c = 299792.458  # Speed of light in km/s

grid = np.linspace(0, np.max(z_cmb), num=1000)
zhel_plus1 = 1 + z_hel


@njit
def Ez(z, params):
    Om, w0 = params[2], params[3]
    z_plus_1 = 1 + z
    cubed = z_plus_1**3
    rho_de = (2 * cubed / (1 + cubed)) ** (2 * (1 + w0))
    return np.sqrt(Om * cubed + (1 - Om) * rho_de)


def theory_mu(params):
    y = 1 / Ez(grid, params)
    I = np.interp(z_cmb, grid, cumulative_trapezoid(y=y, x=grid, initial=0))
    return params[0] + 25 + 5 * np.log10(zhel_plus1 * c * I)


@njit
def H_z(z, params):
    return Ez(z, params)


@njit
def DH_z(z, params):
    return c / H_z(z, params)


@njit
def DM_z(z, params):
    x = np.linspace(0, z, num=max(250, int(250 * z)))
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
    rd_h = params[1] * 100
    results = np.empty(z.size, dtype=np.float64)
    for i in range(z.size):
        q = qty[i]
        if q == 0:
            results[i] = DV_z(z[i], params) / rd_h
        elif q == 1:
            results[i] = DM_z(z[i], params) / rd_h
        elif q == 2:
            results[i] = DH_z(z[i], params) / rd_h
    return results


def chi_squared(params):
    delta_sn = mu_values - theory_mu(params)
    chi_sn = np.dot(delta_sn, cho_solve(cho_sn, delta_sn, check_finite=False))

    delta_bao = bao_data["value"] - bao_theory(bao_data["z"], quantities, params)
    chi_bao = np.dot(delta_bao, cho_solve(cho_bao, delta_bao, check_finite=False))
    return chi_sn + chi_bao


bounds = np.array(
    [
        (-10, -8.5),  # ΔM
        (90, 110),  # r_d * h
        (0.1, 0.7),  # Ωm
        (-2, 0),  # w0
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
    nwalkers = 500
    burn_in = 100
    nsteps = 1000 + burn_in
    initial_pos = np.zeros((nwalkers, ndim))

    for dim, (lower, upper) in enumerate(bounds):
        initial_pos[:, dim] = np.random.uniform(lower, upper, nwalkers)

    with Pool(6) as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers,
            ndim,
            log_probability,
            pool=pool,
            moves=[(emcee.moves.KDEMove(), 0.5), (emcee.moves.StretchMove(), 0.5)],
        )
        sampler.run_mcmc(initial_pos, nsteps, progress=True)

    try:
        tau = sampler.get_autocorr_time()
        print("auto-correlation time", tau)
    except emcee.autocorr.AutocorrError as e:
        print("Autocorrelation time could not be computed", e)

    chains_samples = sampler.get_chain(discard=0, flat=False)
    samples = sampler.get_chain(discard=burn_in, flat=True)

    [
        [dM_16, dM_50, dM_84],
        [rd_16, rd_50, rd_84],
        [Om_16, Om_50, Om_84],
        [w0_16, w0_50, w0_84],
    ] = np.percentile(samples, [15.9, 50, 84.1], axis=0).T

    best_fit = np.array([dM_50, rd_50, Om_50, w0_50])

    print(f"ΔM: {dM_50:.3f} +{(dM_84 - dM_50):.3f} -{(dM_50 - dM_16):.3f}")
    print(f"r_d * h: {rd_50:.2f} +{(rd_84 - rd_50):.2f} -{(rd_50 - rd_16):.2f}")
    print(f"Ωm: {Om_50:.3f} +{(Om_84 - Om_50):.3f} -{(Om_50 - Om_16):.3f}")
    print(f"w0: {w0_50:.3f} +{(w0_84 - w0_50):.3f} -{(w0_50 - w0_16):.3f}")
    print(f"Chi squared: {chi_squared(best_fit):.2f}")
    print(f"Degrees of freedom: {bao_data['value'].size + sn_size - len(best_fit)}")

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

    labels = ["$Δ_M$", "$r_d x h$", "$Ω_M$", "$w_0$"]
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

    _, axes = plt.subplots(ndim, figsize=(10, 7))
    for i in range(ndim):
        axes[i].plot(chains_samples[:, :, i], color="black", alpha=0.3, lw=0.4)
        axes[i].set_ylabel(labels[i])
        axes[i].axvline(x=burn_in, color="red", linestyle="--", alpha=0.5)
        axes[i].axhline(y=best_fit[i], color="white", linestyle="--", alpha=0.5)
    axes[ndim - 1].set_xlabel("chain step")
    plt.show()


if __name__ == "__main__":
    main()


"""
Flat ΛCDM
ΔM: -9.230 +0.006 -0.006 mag
r_d * h: 100.54 +0.66 -0.65 Mpc
Ωm: 0.310 +0.008 -0.008
w0: -1
wa: 0
Chi squared: 1658.97
Degrees of freedom: 1755

===============================

Flat wCDM
ΔM: -9.200 +0.011 -0.011 mag
r_d * h: 98.85 +0.82 -0.81 Mpc
Ωm: 0.298 +0.009 -0.009
w0: -0.871 +0.038 -0.038
wa: 0
Chi squared: 1648.10 (Δ chi2 10.87)
Degrees of freedom: 1754

===============================

Flat w0 - (1 + w0) * (((1 + z)**3 - 1) / ((1 + z)**3 + 1))
ΔM: -9.193 +0.012 -0.012 mag
r_d * h: 98.63 +0.83 -0.82 Mpc
Ωm: 0.307 +0.008 -0.008
w0: -0.834 +0.045 -0.046
wa: -(1 + w0)
Chi squared: 1646.49 (Δ chi2 12.48)
Degrees of freedom: 1754

===============================

Flat w(z) = w0 + wa * z / (1 + z)
ΔM: -9.187 +0.014 -0.014 mag
r_d * h: 98.52 +0.85 -0.83 Mpc
Ωm: 0.321 +0.013 -0.016
w0: -0.784 +0.074 -0.069
wa: -0.719 +0.467 -0.463
Chi squared: 1645.46 (Δ chi2 13.51)
Degrees of freedom: 1753
"""
