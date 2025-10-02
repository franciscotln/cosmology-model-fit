from numba import njit
import numpy as np
import emcee
import corner
from scipy.integrate import cumulative_trapezoid
from scipy.linalg import cho_factor, cho_solve
import matplotlib.pyplot as plt
from multiprocessing import Pool
import y2024BBN.data as bbn
from y2023union3.data import get_data
from y2025BAO.data import get_data as get_bao_data
from sn.plotting import plot_predictions as plot_sn_predictions
from .plot_predictions import plot_bao_predictions

sn_legend, z_sn_vals, mu_vals, cov_matrix_sn = get_data()
bao_legend, bao_data, bao_cov_matrix = get_bao_data()
cho_sn = cho_factor(cov_matrix_sn)
cho_bao = cho_factor(bao_cov_matrix)
cho_bbn = cho_factor(bbn.cov_matrix)

c = 299792.458  # Speed of light in km/s

grid = np.linspace(0, np.max(z_sn_vals), num=1000)


@njit
def Ez(z, params):
    Om, w0 = params[1], params[4]
    one_plus_z = 1 + z
    cubic = one_plus_z**3
    rho_de = (2 * cubic / (1 + cubic)) ** (2 * (1 + w0))
    return np.sqrt(Om * cubic + (1 - Om) * rho_de)


def integral_Ez(params):
    integral_values = cumulative_trapezoid(1 / Ez(grid, params), grid, initial=0)
    return np.interp(z_sn_vals, grid, integral_values)


def distance_modulus(params):
    H0, mag_offset = params[0], params[-1]
    dL = (1 + z_sn_vals) * c * integral_Ez(params) / H0
    return mag_offset + 25 + 5 * np.log10(dL)


@njit
def H_z(z, params):
    return params[0] * Ez(z, params)


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

quantities = np.array([qty_map[q] for q in bao_data["quantity"]], dtype=np.int32)


@njit
def r_drag(wb, wm, n_eff=3.04):  # arXiv:2503.14738v2 (eq 2)
    wb_term = (0.02236 / wb) ** 0.13
    wm_term = (0.1432 / wm) ** 0.23
    n_eff_term = (3.04 / n_eff) ** 0.1
    return 147.05 * wb_term * wm_term * n_eff_term


@njit
def theory_predictions(z, qty, params):
    rd = r_drag(wb=params[2], wm=params[1] * (params[0] / 100) ** 2, n_eff=params[3])
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
    delta_sn = mu_vals - distance_modulus(params)
    chi_sn = np.dot(delta_sn, cho_solve(cho_sn, delta_sn, check_finite=False))

    delta_bao = bao_data["value"] - theory_predictions(
        bao_data["z"], quantities, params
    )
    chi_bao = np.dot(delta_bao, cho_solve(cho_bao, delta_bao, check_finite=False))

    delta_bbn = bbn.data - params[2:4]
    chi_bbn = np.dot(delta_bbn, cho_solve(cho_bbn, delta_bbn, check_finite=False))

    return chi_sn + chi_bao + chi_bbn


bounds = np.array(
    [
        (55, 75),  # H0
        (0.1, 0.6),  # Ωm
        (0.019, 0.025),  # Ωb * h^2
        (2, 4),  # N_eff
        (-2, 0),  # w0
        (-0.7, 0.7),  # ΔM
    ]
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
    nwalkers = 500
    burn_in = 100
    nsteps = 1500 + burn_in
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
                (emcee.moves.KDEMove(), 0.5),
                (emcee.moves.DEMove(), 0.4),
                (emcee.moves.DESnookerMove(), 0.1),
            ],
        )
        sampler.run_mcmc(initial_pos, nsteps, progress=True)

    try:
        tau = sampler.get_autocorr_time()
        print("auto-correlation time", tau)
    except emcee.autocorr.AutocorrError as e:
        print("Autocorrelation time could not be computed", e)

    chains_samples = sampler.get_chain(discard=0, flat=False)
    samples = sampler.get_chain(discard=burn_in, flat=True)

    Omh2_samples = samples[:, 1] * (samples[:, 0] / 100) ** 2
    Omh2_16, Omh2_50, Omh2_84 = np.percentile(Omh2_samples, [15.9, 50, 84.1])

    [
        [H0_16, H0_50, H0_84],
        [Om_16, Om_50, Om_84],
        [Obh2_16, Obh2_50, Obh2_84],
        [Neff_16, Neff_50, Neff_84],
        [w0_16, w0_50, w0_84],
        [dM_16, dM_50, dM_84],
    ] = np.percentile(samples, [15.9, 50, 84.1], axis=0).T

    best_fit = np.array(
        [H0_50, Om_50, Obh2_50, Neff_50, w0_50, dM_50], dtype=np.float64
    )

    print(f"ΔM: {dM_50:.3f} +{(dM_84 - dM_50):.3f} -{(dM_50 - dM_16):.3f} mag")
    print(f"N_eff: {Neff_50:.3f} +{(Neff_84 - Neff_50):.3f} -{(Neff_50 - Neff_16):.3f}")
    print(f"H0: {H0_50:.2f} +{(H0_84 - H0_50):.2f} -{(H0_50 - H0_16):.2f} km/s/Mpc")
    print(f"Ωm: {Om_50:.3f} +{(Om_84 - Om_50):.3f} -{(Om_50 - Om_16):.3f}")
    print(f"Ωm h²: {Omh2_50:.5f} +{(Omh2_84 - Omh2_50):.5f} -{(Omh2_50 - Omh2_16):.5f}")
    print(f"Ωb h²: {Obh2_50:.5f} +{(Obh2_84 - Obh2_50):.5f} -{(Obh2_50 - Obh2_16):.5f}")
    print(f"w0: {w0_50:.3f} +{(w0_84 - w0_50):.3f} -{(w0_50 - w0_16):.3f}")
    print(f"Chi squared: {chi_squared(best_fit):.2f}")
    print(
        f"Degs of freedom: {bbn.data.size + bao_data['value'].size + z_sn_vals.size - len(best_fit)}"
    )

    plot_bao_predictions(
        theory_predictions=lambda z, qty: theory_predictions(z, qty, best_fit),
        data=bao_data,
        errors=np.sqrt(np.diag(bao_cov_matrix)),
        title=bao_legend,
    )
    plot_sn_predictions(
        legend=sn_legend,
        x=z_sn_vals,
        y=mu_vals,
        y_err=np.sqrt(np.diag(cov_matrix_sn)),
        y_model=distance_modulus(best_fit),
        label=f"Best fit: $Ω_m$={Om_50:.3f}",
        x_scale="log",
    )

    labels = ["$H_0$", "$Ω_m$", "$Ω_b h^2$", "$N_{eff}$", "$w_0$", "$Δ_M$"]
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
        levels=(0.393, 0.864),  # 1 and 2 sigmas in 2D
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
*******************************
DESI BAO DR2 2024 + BBN 2024 + Union3
*******************************

Flat ΛCDM
ΔM: -0.139 ± 0.097 mag
N_eff: 2.948 +0.211 -0.212
H0: 68.08 +1.26 -1.30 km/s/Mpc
Ωm: 0.304 ± 0.008
Ωm h²: 0.14086 +0.00656 -0.00645
Ωb h²: 0.02197 ± 0.00063
w0: -1
Chi squared: 38.82
Degs of freedom: 32

===============================

Flat wCDM
ΔM: -0.238 ± 0.105 mag
N_eff: 2.947 +0.212 -0.210
H0: 64.67 +1.84 -1.85 km/s/Mpc
Ωm: 0.298 ± 0.009
Ωm h²: 0.12462 +0.00906 -0.00893
Ωb h²: 0.02197 ± 0.00063
w0: -0.866 +0.051 -0.052
Chi squared: 32.16
Degs of freedom: 31

===============================

Flat -1 + 2 * (1 + w0) / (1 + (1 + z)**3)
ΔM: -0.222 ± 0.101 mag
N_eff: 2.947 ± 0.211
H0: 64.92 ± 1.63 km/s/Mpc
Ωm: 0.310 ± 0.009
Ωm h²: 0.13055 +0.00717 -0.00695
Ωb h²: 0.02197 ± 0.00063
w0: -0.803 +0.066 -0.067
Chi squared: 30.37
Degs of freedom: 31
"""
