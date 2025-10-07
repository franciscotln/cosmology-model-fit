from numba import njit
import numpy as np
import emcee
import corner
from scipy.integrate import cumulative_trapezoid
from scipy.linalg import cho_factor, cho_solve
import matplotlib.pyplot as plt
from multiprocessing import Pool
import y2024BBN.prior_lcdm_shonberg as bbn
from cmb.data_chen_compression import r_drag
from y2023union3.data import get_data
from y2025BAO.data import get_data as get_bao_data
from sn.plotting import plot_predictions as plot_sn_predictions
from .plot_predictions import plot_bao_predictions

sn_legend, z_sn_vals, mu_vals, cov_matrix_sn = get_data()
bao_legend, bao_data, bao_cov_matrix = get_bao_data()
cho_sn = cho_factor(cov_matrix_sn)
cho_bao = cho_factor(bao_cov_matrix)

c = 299792.458  # Speed of light in km/s

grid = np.linspace(0, np.max(z_sn_vals), num=1000)


@njit
def Ez(z, params):
    Om, w0 = params[1], params[3]
    one_plus_z = 1 + z
    cubic = one_plus_z**3
    rho_de = (2 * cubic / (1 + cubic)) ** (2 * (1 + w0))
    return np.sqrt(Om * cubic + (1 - Om) * rho_de)


def integral_Ez(params):
    integral_values = cumulative_trapezoid(1 / Ez(grid, params), grid, initial=0)
    return np.interp(z_sn_vals, grid, integral_values)


def mu_theory(params):
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
    rd = r_drag(wb=params[2], wm=params[1] * (params[0] / 100) ** 2)
    DV_mask = qty == 0
    DM_mask = qty == 1
    DH_mask = qty == 2
    results = np.empty(z.size, dtype=np.float64)
    results[DH_mask] = DH_z(z[DH_mask], params)
    results[DM_mask] = DM_z(z[DM_mask], params)
    results[DV_mask] = DV_z(z[DV_mask], params)
    return results / rd


def chi_squared(params):
    delta_sn = mu_vals - mu_theory(params)
    chi_sn = np.dot(delta_sn, cho_solve(cho_sn, delta_sn, check_finite=False))

    delta_bao = bao_data["value"] - bao_theory(bao_data["z"], quantities, params)
    chi_bao = np.dot(delta_bao, cho_solve(cho_bao, delta_bao, check_finite=False))

    delta_bbn = bbn.Obh2 - params[2]
    chi_bbn = (delta_bbn / bbn.Obh2_sigma) ** 2

    return chi_sn + chi_bao + chi_bbn


bounds = np.array(
    [
        (55, 75),  # H0
        (0.1, 0.6),  # Ωm
        (0.019, 0.025),  # Ωb * h^2
        (-2, 0),  # w0
        (-0.7, 0.7),  # ΔM
    ]
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
        print("effective samples", ndim * nwalkers * (nsteps - burn_in) / np.max(tau))
    except emcee.autocorr.AutocorrError as e:
        print("Autocorrelation time could not be computed", e)

    chains_samples = sampler.get_chain(discard=burn_in, flat=False)
    samples = sampler.get_chain(discard=burn_in, flat=True)

    one_sigma_percentiles = [15.9, 50, 84.1]
    Omh2_samples = samples[:, 1] * (samples[:, 0] / 100) ** 2
    Omh2_16, Omh2_50, Omh2_84 = np.percentile(Omh2_samples, one_sigma_percentiles)
    r_d_samples = r_drag(wb=samples[:, 2], wm=Omh2_samples)
    r_d_16, r_d_50, r_d_84 = np.percentile(r_d_samples, one_sigma_percentiles)

    [
        [H0_16, H0_50, H0_84],
        [Om_16, Om_50, Om_84],
        [Obh2_16, Obh2_50, Obh2_84],
        [w0_16, w0_50, w0_84],
        [dM_16, dM_50, dM_84],
    ] = np.percentile(samples, one_sigma_percentiles, axis=0).T

    best_fit = np.array([H0_50, Om_50, Obh2_50, w0_50, dM_50], dtype=np.float64)

    print(f"ΔM: {dM_50:.3f} +{(dM_84 - dM_50):.3f} -{(dM_50 - dM_16):.3f} mag")
    print(f"H0: {H0_50:.2f} +{(H0_84 - H0_50):.2f} -{(H0_50 - H0_16):.2f} km/s/Mpc")
    print(f"Ωm: {Om_50:.3f} +{(Om_84 - Om_50):.3f} -{(Om_50 - Om_16):.3f}")
    print(f"ωm: {Omh2_50:.4f} +{(Omh2_84 - Omh2_50):.4f} -{(Omh2_50 - Omh2_16):.4f}")
    print(f"ωb: {Obh2_50:.5f} +{(Obh2_84 - Obh2_50):.5f} -{(Obh2_50 - Obh2_16):.5f}")
    print(f"w0: {w0_50:.3f} +{(w0_84 - w0_50):.3f} -{(w0_50 - w0_16):.3f}")
    print(f"r_d: {r_d_50:.2f} +{(r_d_84 - r_d_50):.2f} -{(r_d_50 - r_d_16):.2f} Mpc")
    print(f"Chi squared: {chi_squared(best_fit):.2f}")
    print(
        f"Degs of freedom: {1 + bao_data['value'].size + z_sn_vals.size - len(best_fit)}"
    )

    plot_bao_predictions(
        theory_predictions=lambda z, qty: bao_theory(z, qty, best_fit),
        data=bao_data,
        errors=np.sqrt(np.diag(bao_cov_matrix)),
        title=bao_legend,
    )
    plot_sn_predictions(
        legend=sn_legend,
        x=z_sn_vals,
        y=mu_vals,
        y_err=np.sqrt(np.diag(cov_matrix_sn)),
        y_model=mu_theory(best_fit),
        label=f"Best fit: $Ω_m$={Om_50:.3f}",
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
        levels=(0.393, 0.864),  # 1 and 2 sigmas in 2D
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
DESI BAO DR2 2024 + BBN 2024 + Union3
*******************************

Flat ΛCDM
ΔM: -0.116 +0.089 -0.088 mag
H0: 68.79 +0.60 -0.59 km/s/Mpc
Ωm: 0.304 +0.008 -0.008
ωm: 0.1438 +0.0050 -0.0048
ωb: 0.02218 +0.00053 -0.00054
w0: -0.994 +0.667 -0.676
r_d: 146.89 +1.50 -1.52 Mpc
Chi squared: 38.8
Degs of freedom: 32

===============================

Flat wCDM
ΔM: -0.223 +0.100 -0.101 mag
H0: 65.12 +1.55 -1.58 km/s/Mpc
Ωm: 0.298 +0.009 -0.009
ωm: 0.1265 +0.0084 -0.0084
ωb: 0.02218 +0.00054 -0.00055
w0: -0.868 +0.050 -0.051
r_d: 151.69 +2.68 -2.50 Mpc
Chi squared: 32.2
Degs of freedom: 31

===============================

Flat -1 + 2 * (1 + w0) / (1 + (1 + z)**3)
ΔM: -0.205 +0.094 -0.095 mag
H0: 65.40 +1.29 -1.25 km/s/Mpc
Ωm: 0.310 +0.009 -0.009
ωm: 0.1326 +0.0061 -0.0058
ωb: 0.02219 +0.00054 -0.00055
w0: -0.803 +0.065 -0.067
r_d: 149.94 +1.85 -1.87 Mpc
Chi squared: 30.4
Degs of freedom: 31
"""
