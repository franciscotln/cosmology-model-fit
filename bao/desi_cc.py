import numpy as np
import emcee
import corner
from scipy.integrate import quad
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
inv_cov_cc = np.linalg.inv(cc_cov_matrix)
logdet_cc = np.linalg.slogdet(cc_cov_matrix)[1]
N_cc = len(z_cc_vals)

c = 299792.458  # Speed of light in km/s


def Ez(z, O_m, w0):
    one_plus_z = 1 + z
    rho_de = (2 * one_plus_z**3 / (1 + one_plus_z**3)) ** (2 * (1 + w0))
    return np.sqrt(O_m * one_plus_z**3 + (1 - O_m) * rho_de)


def H_z(z, params):
    return params[1] * Ez(z, *params[3:])


def DH_z(z, params):
    return c / H_z(z, params)


def DM_z(z, params):
    return quad(lambda zp: DH_z(zp, params), 0, z)[0]


def DV_z(z, params):
    DH = DH_z(z, params)
    DM = DM_z(z, params)
    return (z * DH * DM**2) ** (1 / 3)


quantity_funcs = {
    "DV_over_rs": lambda z, params: DV_z(z, params) / params[2],
    "DM_over_rs": lambda z, params: DM_z(z, params) / params[2],
    "DH_over_rs": lambda z, params: DH_z(z, params) / params[2],
}


def theory_bao(z, qty, params):
    return np.array([(quantity_funcs[qty](z, params)) for z, qty in zip(z, qty)])


bounds = np.array(
    [
        (0.4, 2.5),  # f_cc
        (50, 100),  # H0
        (120, 180),  # r_d
        (0.2, 0.7),  # Ωm
        (-2, 0.5),  # w0
    ]
)


def chi_squared(params):
    f_cc = params[0]
    delta_cc = H_cc_vals - H_z(z_cc_vals, params)
    chi_cc = np.dot(delta_cc, np.dot(inv_cov_cc * f_cc**2, delta_cc))

    delta_bao = data["value"] - theory_bao(data["z"], data["quantity"], params)
    chi_bao = np.dot(delta_bao, cho_solve(cho_bao, delta_bao))
    return chi_cc + chi_bao


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
    nwalkers = 16 * ndim
    burn_in = 500
    nsteps = 15000 + burn_in
    initial_pos = np.zeros((nwalkers, ndim))

    for dim, (lower, upper) in enumerate(bounds):
        initial_pos[:, dim] = np.random.uniform(lower, upper, nwalkers)

    with Pool(10) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool=pool)
        sampler.run_mcmc(initial_pos, nsteps, progress=True)

    try:
        tau = sampler.get_autocorr_time()
        print("auto-correlation time", tau)
    except emcee.autocorr.AutocorrError as e:
        print("Autocorrelation time could not be computed", e)

    samples = sampler.get_chain(discard=burn_in, flat=True)

    [
        [f_cc_16, f_cc_50, f_cc_84],
        [h0_16, h0_50, h0_84],
        [rd_16, rd_50, rd_84],
        [Om_16, Om_50, Om_84],
        [w0_16, w0_50, w0_84],
    ] = np.percentile(samples, [15.9, 50, 84.1], axis=0).T

    best_fit = [f_cc_50, h0_50, rd_50, Om_50, w0_50]

    print(f"f_cc: {f_cc_50:.2f} +{(f_cc_84 - f_cc_50):.2f} -{(f_cc_50 - f_cc_16):.2f}")
    print(f"H0: {h0_50:.1f} +{(h0_84 - h0_50):.1f} -{(h0_50 - h0_16):.1f}")
    print(f"r_d: {rd_50:.1f} +{(rd_84 - rd_50):.1f} -{(rd_50 - rd_16):.1f}")
    print(f"Ωm: {Om_50:.3f} +{(Om_84 - Om_50):.3f} -{(Om_50 - Om_16):.3f}")
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

    labels = ["$f_{CCH}$", "$H_0$", "$r_d$", "$\Omega_m$", "$w_0$"]
    corner.corner(
        samples,
        labels=labels,
        quantiles=[0.159, 0.5, 0.841],
        show_titles=True,
        title_fmt=".3f",
        smooth=1.5,
        smooth1d=1.5,
        bins=100,
        levels=(0.393, 0.864),  # 1 and 2 sigmas in 2D
        fill_contours=False,
        plot_datapoints=False,
    )
    plt.show()


if __name__ == "__main__":
    main()

"""
******************************
Dataset: DESI 2025
******************************

Flat ΛCDM model
f_cc: 1.46 +0.19 -0.18
H0: 69.1 +2.3 -2.3 km/s/Mpc
r_d: 146.8 +5.0 -4.6 Mpc
Ωm: 0.299 +0.009 -0.008
w0: -1
Chi squared: 41.61
log likelihood: -131.35
Degrees of freedom: 41

=============================

Flat wCDM model
f_cc: 1.46 +0.19 -0.18
H0: 67.9 +2.6 -2.5 km/s/Mpc
r_d: 147.0 +5.0 -4.7 Mpc
Ωm: 0.298 +0.009 -0.009
w0: -0.921 +0.075 -0.078
Chi squared: 40.43
log likelihood: -130.81
Degrees of freedom: 40

==============================

Flat w0 - (1 + w0) * (((1 + z)**3 - 1) / ((1 + z)**3 + 1))
f_cc: 1.45 +0.19 -0.18
H0: 67.3 +2.8 -2.7 km/s/Mpc
r_d: 147.0 +5.0 -4.6 Mpc
Ωm: 0.307 +0.011 -0.011
w0: -0.855 +0.119 -0.125
Chi squared: 39.81
log likelihood: -130.62
Degrees of freedom: 40

=============================

Flat w0waCDM model
f_cc: 1.43 +0.19 -0.18
H0: 64.7 +3.8 -3.7 km/s/Mpc
r_d: 147.1 +5.2 -4.8 Mpc
Ωm: 0.350 +0.043 -0.049
w0: -0.535 +0.397 -0.362
wa: -1.520 +1.411 -1.409
Chi squared: 37.56
log likelihood: -130.14
Degrees of freedom: 39

******************************
Dataset: SDSS 2020 compilation
******************************

Flat ΛCDM model
f_cc: 1.46 +0.19 -0.18
H0: 69.1 +2.5 -2.5 km/s/Mpc
r_d: 146.1 +5.1 -4.7 Mpc
Ωm: 0.298 +0.015 -0.015
w0: -1
Chi squared: 43.04
log likelihood: -132.16
Degrees of freedom: 45

=============================

Flat wCDM model
f_cc: 1.45 +0.19 -0.18
H0: 67.0 +3.0 -2.9 km/s/Mpc
r_d: 146.7 +5.2 -4.8 Mpc
Ωm: 0.289 +0.018 -0.019
w0: -0.836 +0.122 -0.128
Chi squared: 40.96
log likelihood: -131.33
Degrees of freedom: 44

=============================

Flat w0 - (1 + w0) * (((1 + z)**3 - 1) / ((1 + z)**3 + 1))
f_cc: 1.44 +0.19 -0.18
H0: 67.1 +3.2 -3.1 km/s/Mpc
r_d: 146.3 +5.2 -4.8 Mpc
Ωm: 0.305 +0.017 -0.016
w0: -0.820 +0.165 -0.176
Chi squared: 41.30
log likelihood: -131.58
Degrees of freedom: 44
"""
