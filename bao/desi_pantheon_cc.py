from numba import njit
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.linalg import cho_factor, cho_solve
from y2022pantheonSHOES.data import get_data
from y2005cc.data import get_data as get_cc_data
from y2025BAO.data import get_data as get_bao_data

cc_legend, z_cc_vals, H_cc_vals, cov_matrix_cc = get_cc_data()
legend, z_sn_vals, z_sn_hel_vals, apparent_mag_values, cov_matrix_sn = get_data()
bao_legend, bao_data, cov_matrix_bao = get_bao_data()
cho_cc = cho_factor(cov_matrix_cc)
cho_bao = cho_factor(cov_matrix_bao)
cho_sn = cho_factor(cov_matrix_sn)
logdet_cc = np.linalg.slogdet(cov_matrix_cc)[1]
N_cc = len(z_cc_vals)

c = 299792.458  # Speed of light in km/s

z_grid = np.linspace(0, np.max(z_sn_vals), num=1000)
one_plus_z_hel = 1 + z_sn_hel_vals


@njit
def Ez(z, params):
    O_m, w0 = params[3], params[4]
    one_plus_z = 1 + z
    cubed = one_plus_z**3
    rho_de = (2 * cubed / (1 + cubed)) ** (2 * (1 + w0))
    return np.sqrt(O_m * cubed + (1 - O_m) * rho_de)


def integral_Ez(params):
    integral_values = cumulative_trapezoid(1 / Ez(z_grid, params), z_grid, initial=0)
    return np.interp(z_sn_vals, z_grid, integral_values)


def sn_apparent_mag(params):
    H0, M = params[0], params[1]
    comoving_distance = (c / H0) * integral_Ez(params)
    return M + 25 + 5 * np.log10(one_plus_z_hel * comoving_distance)


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
def bao_theory(z, qty, params):
    results = np.empty(z.size, dtype=np.float64)
    for i in range(z.size):
        q = qty[i]
        if q == 0:
            results[i] = DV_z(z[i], params) / params[2]
        elif q == 1:
            results[i] = DM_z(z[i], params) / params[2]
        elif q == 2:
            results[i] = DH_z(z[i], params) / params[2]
    return results


bounds = np.array(
    [
        (40.0, 90.0),  # H0
        (-20.0, -19.0),  # M
        (115.0, 170.0),  # r_d
        (0.0, 1.0),  # Ωm
        (-2.0, 0.0),  # w0
        (0.4, 2.5),  # f_cc
    ],
    dtype=np.float64,
)


def chi_squared(params):
    f_cc = params[-1]
    delta_sn = apparent_mag_values - sn_apparent_mag(params)
    chi_sn = np.dot(delta_sn, cho_solve(cho_sn, delta_sn, check_finite=False))

    delta_bao = bao_data["value"] - bao_theory(bao_data["z"], quantities, params)
    chi_bao = np.dot(delta_bao, cho_solve(cho_bao, delta_bao, check_finite=False))

    delta_cc = H_cc_vals - H_z(z_cc_vals, params)
    chi_cc = f_cc**2 * np.dot(delta_cc, cho_solve(cho_cc, delta_cc, check_finite=False))

    return chi_sn + chi_bao + chi_cc


@njit
def log_prior(params):
    if np.all((bounds[:, 0] < params) & (params < bounds[:, 1])):
        return 0.0
    return -np.inf


def log_likelihood(params):
    f_cc = params[-1]
    normalization_cc = N_cc * np.log(2 * np.pi) + logdet_cc - 2 * N_cc * np.log(f_cc)
    return -0.5 * chi_squared(params) - 0.5 * normalization_cc


def log_probability(params):
    lp = log_prior(params)
    if np.isinf(lp):
        return -np.inf
    return lp + log_likelihood(params)


def main():
    import emcee, corner
    import matplotlib.pyplot as plt
    from multiprocessing import Pool
    from gelman_rubin import gelman_rubin
    from log_evidence import log_evidence
    from sn.plotting import plot_predictions as plot_sn_predictions
    from .plot_predictions import plot_bao_predictions

    ndim = len(bounds)
    nwalkers = 150
    burn_in = 200
    nsteps = 2400 + burn_in
    np.random.seed(42)
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
        print("acceptance fraction", np.mean(sampler.acceptance_fraction))
        print("effective samples", ndim * nwalkers * (nsteps - burn_in) / np.max(tau))
    except emcee.autocorr.AutocorrError as e:
        print("Autocorrelation time could not be computed", e)

    samples = sampler.get_chain(discard=burn_in, flat=True)
    chains_samples = sampler.get_chain(discard=burn_in, flat=False)
    log_probs = sampler.get_log_prob(discard=burn_in, flat=True)

    print("Gelman-Rubin R-hat:", gelman_rubin(chains_samples))

    [
        [h0_16, h0_50, h0_84],
        [M_16, M_50, M_84],
        [rd_16, rd_50, rd_84],
        [Om_16, Om_50, Om_84],
        [w0_16, w0_50, w0_84],
        [f_cc_16, f_cc_50, f_cc_84],
    ] = np.percentile(samples, [15.9, 50, 84.1], axis=0).T

    best_fit = [h0_50, M_50, rd_50, Om_50, w0_50, f_cc_50]

    deg_of_freedom = (
        z_sn_vals.size + bao_data["value"].size + z_cc_vals.size - len(best_fit)
    )

    print(f"f_cc: {f_cc_50:.2f} +{(f_cc_84 - f_cc_50):.2f} -{(f_cc_50 - f_cc_16):.2f}")
    print(f"H0: {h0_50:.2f} +{(h0_84 - h0_50):.2f} -{(h0_50 - h0_16):.2f}")
    print(f"M: {M_50:.3f} +{(M_84 - M_50):.3f} -{(M_50 - M_16):.3f}")
    print(f"r_d: {rd_50:.2f} +{(rd_84 - rd_50):.2f} -{(rd_50 - rd_16):.2f}")
    print(f"Ωm: {Om_50:.3f} +{(Om_84 - Om_50):.3f} -{(Om_50 - Om_16):.3f}")
    print(f"w0: {w0_50:.3f} +{(w0_84 - w0_50):.3f} -{(w0_50 - w0_16):.3f}")
    print(f"Chi squared: {chi_squared(best_fit):.2f}")
    print(f"Log Evidence: {log_evidence(samples, log_probs, log_probability):.1f}")
    print(f"Degrees of freedom: {deg_of_freedom}")

    plot_bao_predictions(
        theory_predictions=lambda z, qty: bao_theory(z, qty, best_fit),
        data=bao_data,
        errors=np.sqrt(np.diag(cov_matrix_bao)),
        title=bao_legend,
    )
    plot_sn_predictions(
        legend=legend,
        x=z_sn_vals,
        y=apparent_mag_values,
        y_err=np.sqrt(np.diag(cov_matrix_sn)),
        y_model=sn_apparent_mag(best_fit),
        label=f"Best fit: $\Omega_m$={Om_50:.3f}, $H_0$={h0_50:.2f} km/s/Mpc",
        x_scale="log",
    )

    labels = ["$H_0$", "M", "$r_d$", "Ωm", "$w_0$", "$f_{CC}$"]
    corner.corner(
        samples,
        labels=labels,
        quantiles=[0.159, 0.5, 0.841],
        show_titles=True,
        title_fmt=".3f",
        bins=100,
        fill_contours=False,
        plot_datapoints=False,
        smooth=1.5,
        smooth1d=1.5,
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
Flat ΛCDM: w(z) = -1
f_cc: 1.48 +0.18 -0.18
H0: 68.64 +2.27 -2.27 km/s/Mpc
M: -19.403 +0.070 -0.072 mag
r_d: 147.1 +4.9 -4.6 Mpc
Ωm: 0.305 +0.008 -0.008
w0: -1
Chi squared: 1448.50
Log Evidence: -846.1
Degrees of freedom: 1631

===============================

Flat wCDM: w(z) = w0
f_cc: 1.47 +0.18 -0.18
H0: 67.86 +2.30 -2.24 km/s/Mpc
M: -19.416 +0.072 -0.072 mag
r_d: 147.1 +4.8 -4.7 Mpc
Ωm: 0.298 +0.009 -0.008
w0: -0.917 +0.040 -0.039
Chi squared: 1443.91 (Δ chi2 = 4.59 from ΛCDM)
Log Evidence: -846.2
Degrees of freedom: 1630

===============================

Flat: w(z) = -1 + 2 * (1 + w0) / (1 + (1 + z)**3)
f_cc: 1.46 +0.18 -0.18
H0: 67.85 +2.31 -2.29 km/s/Mpc
M: -19.414 +0.072 -0.073 mag
r_d: 147.0 +5.0 -4.7 Mpc
Ωm: 0.304 +0.008 -0.008
w0: -0.899 +0.046 -0.047
Chi squared: 1443.57 (Δ chi2 = 4.93 from ΛCDM)
Log Evidence: -846.0
Degrees of freedom: 1630
"""
