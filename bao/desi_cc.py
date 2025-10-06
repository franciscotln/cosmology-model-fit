from numba import njit
import numpy as np
import emcee
import corner
from scipy.linalg import cho_factor, cho_solve
import matplotlib.pyplot as plt
from multiprocessing import Pool
from y2005cc.data import get_data as get_cc_data
from y2025BAO.data import get_data as get_bao_data
from .plot_predictions import plot_bao_predictions
from cosmic_chronometers.plot_predictions import plot_cc_predictions

cc_legend, z_cc_vals, H_cc_vals, cc_cov_matrix = get_cc_data()
bao_legend, data, bao_cov_matrix = get_bao_data()
cho_bao = cho_factor(bao_cov_matrix)
cho_cc = cho_factor(cc_cov_matrix)
logdet_cc = np.linalg.slogdet(cc_cov_matrix)[1]
N_cc = len(z_cc_vals)

c = 299792.458  # Speed of light in km/s


@njit
def Ez(z, params):
    O_m, w0 = params[3], params[4]
    one_plus_z = 1 + z
    cubic = one_plus_z**3
    rho_de = (2 * cubic / (1 + cubic)) ** (2 * (1 + w0))
    return np.sqrt(O_m * cubic + (1 - O_m) * rho_de)


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

quantities = np.array([qty_map[q] for q in data["quantity"]], dtype=np.int32)


@njit
def theory_bao(z, qty, params):
    results = np.empty(z.size, dtype=np.float64)
    DV_mask = qty == 0
    DM_mask = qty == 1
    DH_mask = qty == 2
    results[DH_mask] = DH_z(z[DH_mask], params)
    results[DM_mask] = DM_z(z[DM_mask], params)
    results[DV_mask] = DV_z(z[DV_mask], params)
    return results / params[2]


bounds = np.array(
    [
        (0.4, 2.5),  # f_cc
        (45, 90),  # H0
        (120, 175),  # r_d
        (0.2, 0.7),  # Ωm
        (-2.0, 0.0),  # w0
    ],
    dtype=np.float64,
)


def chi_squared(params):
    f_cc = params[0]
    delta_cc = H_cc_vals - H_z(z_cc_vals, params)
    chi_cc = f_cc**2 * np.dot(delta_cc, cho_solve(cho_cc, delta_cc, check_finite=False))

    delta_bao = data["value"] - theory_bao(data["z"], quantities, params)
    chi_bao = np.dot(delta_bao, cho_solve(cho_bao, delta_bao, check_finite=False))
    return chi_cc + chi_bao


@njit
def log_prior(params):
    if np.all((bounds[:, 0] < params) & (params < bounds[:, 1])):
        return 0.0
    return -np.inf


def log_likelihood(params):
    f_cc = params[0]
    normalization_cc = N_cc * np.log(2 * np.pi) + logdet_cc - 2 * N_cc * np.log(f_cc)
    return -0.5 * chi_squared(params) - 0.5 * normalization_cc


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

    [
        [f_cc_16, f_cc_50, f_cc_84],
        [h0_16, h0_50, h0_84],
        [rd_16, rd_50, rd_84],
        [Om_16, Om_50, Om_84],
        [w0_16, w0_50, w0_84],
    ] = np.percentile(samples, [15.9, 50, 84.1], axis=0).T

    best_fit = np.array([f_cc_50, h0_50, rd_50, Om_50, w0_50], dtype=np.float64)

    Omh2_samples = samples[:, 1] ** 2 * samples[:, 3] / 100**2
    Omh2_16, Omh2_50, Omh2_84 = np.percentile(Omh2_samples, [15.9, 50, 84.1])

    print(f"f_cc: {f_cc_50:.2f} +{(f_cc_84 - f_cc_50):.2f} -{(f_cc_50 - f_cc_16):.2f}")
    print(f"H0: {h0_50:.1f} +{(h0_84 - h0_50):.1f} -{(h0_50 - h0_16):.1f}")
    print(f"r_d: {rd_50:.1f} +{(rd_84 - rd_50):.1f} -{(rd_50 - rd_16):.1f}")
    print(f"Ωm: {Om_50:.3f} +{(Om_84 - Om_50):.3f} -{(Om_50 - Om_16):.3f}")
    print(
        f"Ωm h^2: {Omh2_50:.4f} +{(Omh2_84 - Omh2_50):.4f} -{(Omh2_50 - Omh2_16):.4f}"
    )
    print(f"w0: {w0_50:.3f} +{(w0_84 - w0_50):.3f} -{(w0_50 - w0_16):.3f}")
    print(f"Chi squared: {chi_squared(best_fit):.2f}")
    print(f"log likelihood: {log_likelihood(best_fit):.2f}")
    print(f"Degrees of freedom: {data['value'].size + z_cc_vals.size - len(best_fit)}")

    plot_bao_predictions(
        theory_predictions=lambda z, qty: theory_bao(z, qty, best_fit),
        data=data,
        errors=np.sqrt(np.diag(bao_cov_matrix)),
        title=f"{bao_legend}: $H_0$={h0_50:.2f}, $r_d$={rd_50:.2f}",
    )
    plot_cc_predictions(
        H_z=lambda z: H_z(z, best_fit),
        z=z_cc_vals,
        H=H_cc_vals,
        H_err=np.sqrt(np.diag(cc_cov_matrix)) / f_cc_50,
        label=f"{cc_legend}: $H_0$={h0_50:.1f} km/s/Mpc",
    )

    labels = ["$f_{CCH}$", "$H_0$", "$r_d$", "$Ω_m$", "$w_0$"]
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
Dataset: DESI 2025
*******************************

Flat ΛCDM
f_cc: 1.47 +0.19 -0.18
H0: 69.1 +2.3 -2.3 km/s/Mpc
r_d: 146.9 +4.9 -4.6 Mpc
Ωm: 0.299 +0.009 -0.008
Ωm h^2: 0.1425 +0.0094 -0.0092
w0: -1
Chi squared: 42.59
log likelihood: -135.81
Degrees of freedom: 42

===============================

Flat wCDM
f_cc: 1.47 +0.18 -0.18
H0: 67.9 +2.6 -2.5 km/s/Mpc
r_d: 147.1 +5.0 -4.6 Mpc
Ωm: 0.298 +0.009 -0.009
Ωm h^2: 0.1375 +0.0106 -0.0104
w0: -0.922 +0.075 -0.079
Chi squared: 41.45
log likelihood: -135.29
Degrees of freedom: 41

===============================

Flat -1 + 2 * (1 + w0) / (1 + (1 + z)**3)
f_cc: 1.46 +0.19 -0.18
H0: 67.3 +2.7 -2.7 km/s/Mpc
r_d: 147.1 +5.0 -4.7 Mpc
Ωm: 0.307 +0.011 -0.011
Ωm h^2: 0.1390 +0.0099 -0.0097
w0: -0.857 +0.118 -0.124
Chi squared: 40.72
log likelihood: -135.10
Degrees of freedom: 41

===============================

Flat w0waCDM
f_cc: 1.43 +0.19 -0.18
H0: 64.8 +3.8 -3.7
r_d: 147.2 +5.1 -4.8
Ωm: 0.348 +0.043 -0.049
w0: -0.553 +0.394 -0.359
wa: -1.458 +1.419 -1.405
Chi squared: 38.55
log likelihood: -134.57
Degrees of freedom: 40

*******************************
Dataset: SDSS 2020 compilation
*******************************

Flat ΛCDM
f_cc: 1.46 +0.19 -0.18
H0: 69.1 +2.5 -2.5 km/s/Mpc
r_d: 146.1 +5.1 -4.7 Mpc
Ωm: 0.298 +0.015 -0.015
w0: -1
Chi squared: 43.04
log likelihood: -132.16
Degrees of freedom: 45

===============================

Flat wCDM
f_cc: 1.45 +0.19 -0.18
H0: 67.0 +3.0 -2.9 km/s/Mpc
r_d: 146.7 +5.2 -4.8 Mpc
Ωm: 0.289 +0.018 -0.019
w0: -0.836 +0.122 -0.128
Chi squared: 40.96
log likelihood: -131.33
Degrees of freedom: 44

===============================

Flat -1 + 2 * (1 + w0) / (1 + (1 + z)**3)
f_cc: 1.44 +0.19 -0.18
H0: 67.1 +3.2 -3.1 km/s/Mpc
r_d: 146.3 +5.2 -4.8 Mpc
Ωm: 0.305 +0.017 -0.016
w0: -0.820 +0.165 -0.176
Chi squared: 41.30
log likelihood: -131.58
Degrees of freedom: 44
"""
