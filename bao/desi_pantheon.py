from numba import njit
import numpy as np
import emcee
import corner
from scipy.integrate import cumulative_trapezoid, quad
from scipy.linalg import cho_factor, cho_solve
import matplotlib.pyplot as plt
from multiprocessing import Pool
from y2025BAO.data import get_data as get_bao_data
from y2022pantheonSHOES.data import get_data
from sn.plotting import plot_predictions as plot_sn_predictions
from .plot_predictions import plot_bao_predictions

legend, z_cmb, z_hel, mb_vals, cov_matrix_sn = get_data()
bao_legend, data, bao_cov_matrix = get_bao_data()
cho_sn = cho_factor(cov_matrix_sn)
cho_bao = cho_factor(bao_cov_matrix)

c = 299792.458  # Speed of light in km/s
rd = 147.09  # Mpc, fixed

grid = np.linspace(0, np.max(z_cmb), num=1000)
one_plus_z_hel = 1 + z_hel


@njit
def Ez(z, O_m, w0):
    one_plus_z = 1 + z
    cubed = one_plus_z**3
    rho_de = (2 * cubed / (1 + cubed)) ** (2 * (1 + w0))
    return np.sqrt(O_m * cubed + (1 - O_m) * rho_de)


def integral_Ez(params):
    x = grid
    y = 1 / Ez(grid, *params[2:])
    return np.interp(z_cmb, x, cumulative_trapezoid(y=y, x=x, initial=0))


def apparent_mag(params):
    M, H0 = params[0], params[1]
    dL = one_plus_z_hel * (c / H0) * integral_Ez(params)
    return M + 25 + 5 * np.log10(dL)


@njit
def H_z(z, params):
    H0, Om, w0 = params[1], params[2], params[3]
    return H0 * Ez(z, Om, w0)


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

quantities = np.array([qty_map[q] for q in data["quantity"]], dtype=np.int32)


@njit
def bao_theory(z, qty, params):
    DV_mask = qty == 0
    DM_mask = qty == 1
    DH_mask = qty == 2
    results = np.empty(z.size, dtype=np.float64)
    results[DH_mask] = DH_z(z[DH_mask], params)
    results[DM_mask] = DM_z(z[DM_mask], params)
    results[DV_mask] = DV_z(z[DV_mask], params)
    return results / rd


bounds = np.array(
    [
        (-20, -19),  # M
        (50, 80),  # H0
        (0.2, 0.7),  # Ωm
        (-2, 0),  # w0
    ],
    dtype=np.float64,
)


def chi_squared(params):
    delta_sn = mb_vals - apparent_mag(params)
    chi_sn = delta_sn.dot(cho_solve(cho_sn, delta_sn, check_finite=False))

    delta_bao = data["value"] - bao_theory(data["z"], quantities, params)
    chi_bao = delta_bao.dot(cho_solve(cho_bao, delta_bao, check_finite=False))
    return chi_sn + chi_bao


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
        print("acceptance fraction:", np.mean(sampler.acceptance_fraction))
        print("effective samples", ndim * nwalkers * nsteps / np.max(tau))
    except emcee.autocorr.AutocorrError as e:
        print("Autocorrelation time could not be computed", e)

    chains_samples = sampler.get_chain(discard=burn_in, flat=False)
    samples = sampler.get_chain(discard=burn_in, flat=True)

    [
        [M_16, M_50, M_84],
        [H0_16, H0_50, H0_84],
        [Om_16, Om_50, Om_84],
        [w0_16, w0_50, w0_84],
    ] = np.percentile(samples, [15.9, 50, 84.1], axis=0).T

    best_fit = np.array([M_50, H0_50, Om_50, w0_50], dtype=np.float64)

    print(f"M0: {M_50:.3f} +{(M_84 - M_50):.3f} -{(M_50 - M_16):.3f}")
    print(f"H0: {H0_50:.2f} +{(H0_84 - H0_50):.2f} -{(H0_50 - H0_16):.2f}")
    print(f"Ωm: {Om_50:.3f} +{(Om_84 - Om_50):.3f} -{(Om_50 - Om_16):.3f}")
    print(f"w0: {w0_50:.3f} +{(w0_84 - w0_50):.3f} -{(w0_50 - w0_16):.3f}")
    print(f"Chi squared: {chi_squared(best_fit):.2f}")
    print(f"Degrees of freedom: {data['z'].size + z_cmb.size - len(best_fit)}")

    plot_bao_predictions(
        theory_predictions=lambda z, qty: bao_theory(z, qty, best_fit),
        data=data,
        errors=np.sqrt(np.diag(bao_cov_matrix)),
        title=bao_legend,
    )
    plot_sn_predictions(
        legend=legend,
        x=z_cmb,
        y=mb_vals,
        y_err=np.sqrt(np.diag(cov_matrix_sn)),
        y_model=apparent_mag(best_fit),
        label=f"Best fit: $\Omega_m$={Om_50:.3f}",
        x_scale="log",
    )

    labels = ["$M_0$", "$H_0$", "$Ω_m$", "$w_0$"]
    corner.corner(
        samples,
        labels=labels,
        quantiles=[0.159, 0.5, 0.841],
        show_titles=True,
        title_fmt=".4f",
        smooth=2.0,
        smooth1d=2.0,
        bins=100,
        levels=(0.393, 0.864),
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
r_d: 147.09 Mpc (fixed)
M0: -19.402 +0.013 -0.013 mag
H0: 68.67 +0.46 -0.45 km/s/Mpc
Ωm: 0.304 +0.008 -0.008
w0: -1
Chi squared: 1416.14
Degrees of freedom: 1600

===============================

Flat wCDM
r_d: 147.09 Mpc (fixed)
M0: -19.416 +0.014 -0.014 mag
H0: 67.83 +0.60 -0.59 km/s/Mpc
Ωm: 0.298 +0.009 -0.009
w0: -0.913 +0.040 -0.040
Chi squared: 1411.54 (Δ chi2 4.60)
Degrees of freedom: 1599

===============================

Flat w0 - (1 + w0) * ((1 + z)**3 - 1) / ((1 + z)**3 + 1)
r_d: 147.09 Mpc (fixed)
M0: -19.415 +0.014 -0.014 mag
H0: 67.78 +0.59 -0.59 km/s/Mpc
Ωm: 0.304 +0.008 -0.008
w0: -0.894 +0.046 -0.048
Chi squared: 1411.30 (Δ chi2 4.84)
Degrees of freedom: 1599
"""
