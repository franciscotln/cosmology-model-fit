import numpy as np
import emcee
import corner
from scipy.integrate import quad
from scipy.constants import c as c0
from scipy.linalg import cho_factor, cho_solve
import matplotlib.pyplot as plt
from multiprocessing import Pool
from y2025BAO.data import get_data
from .plot_predictions import plot_bao_predictions

c = c0 / 1000  # Speed of light in km/s
legend, data, cov_matrix = get_data()
cho = cho_factor(cov_matrix)


def H_z(z, H0, Om, w0):
    one_plus_z = 1 + z
    evolving_de = ((2 * one_plus_z**2) / (1 + one_plus_z**2)) ** (3 * (1 + w0))
    return H0 * np.sqrt(Om * one_plus_z**3 + (1 - Om) * evolving_de)


def DH_z(z, params):
    return c / H_z(z, *params[1:])


def DM_z(z, params):
    return quad(lambda zp: DH_z(zp, params), 0, z)[0]


def DV_z(z, params):
    DH = DH_z(z, params)
    DM = DM_z(z, params)
    return (z * DH * DM**2) ** (1 / 3)


quantity_funcs = {
    "DV_over_rs": lambda z, params: DV_z(z, params) / params[0],
    "DM_over_rs": lambda z, params: DM_z(z, params) / params[0],
    "DH_over_rs": lambda z, params: DH_z(z, params) / params[0],
}


def theory_predictions(z, qty, params):
    return np.array([(quantity_funcs[qty](z, params)) for z, qty in zip(z, qty)])


bounds = np.array(
    [
        (130, 160),  # r_d
        (50, 80),  # H0
        (0.1, 0.7),  # Ωm
        (-1.5, 0),  # w0
    ]
)


def chi_squared(params):
    delta = data["value"] - theory_predictions(data["z"], data["quantity"], params)
    return np.dot(delta, cho_solve(cho, delta))


# Prior from Planck 2018 https://arxiv.org/abs/1807.06209
# Ωm x ​h^2 = 0.14237 ± 0.00135. Prior width increased by 70% to 0.0023
def log_prior(params):
    if np.all((bounds[:, 0] < params) & (params < bounds[:, 1])):
        Om_x_h2 = params[2] * (params[1] / 100) ** 2
        return -0.5 * ((0.14237 - Om_x_h2) / 0.0023) ** 2
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
    nwalkers = 10 * ndim
    burn_in = 500
    nsteps = 10000 + burn_in
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
        [rd_16, rd_50, rd_84],
        [H0_16, H0_50, H0_84],
        [Om_16, Om_50, Om_84],
        [w0_16, w0_50, w0_84],
    ] = np.percentile(samples, [15.9, 50, 84.1], axis=0).T

    best_fit = [rd_50, H0_50, Om_50, w0_50]

    residuals = data["value"] - theory_predictions(data["z"], data["quantity"])
    SS_res = np.sum(residuals**2)
    SS_tot = np.sum((data["value"] - np.mean(data["value"])) ** 2)
    r2 = 1 - SS_res / SS_tot

    print(f"r_d: {rd_50:.2f} +{(rd_84 - rd_50):.2f} -{(rd_50 - rd_16):.2f}")
    print(f"H0: {H0_50:.2f} +{(H0_84 - H0_50):.2f} -{(H0_50 - H0_16):.2f}")
    print(f"Ωm: {Om_50:.3f} +{(Om_84 - Om_50):.3f} -{(Om_50 - Om_16):.3f}")
    print(f"w0: {w0_50:.3f} +{(w0_84 - w0_50):.3f} -{(w0_50 - w0_16):.3f}")
    print(f"Chi squared: {chi_squared(best_fit):.2f}")
    print(f"Degs of freedom: {data['value'].size  - len(best_fit)}")
    print(f"R^2: {r2:.4f}")
    print(f"RMSD: {np.sqrt(np.mean(residuals**2)):.3f}")

    plot_bao_predictions(
        theory_predictions=lambda z, qty: theory_predictions(z, qty, best_fit),
        data=data,
        errors=np.sqrt(np.diag(cov_matrix)),
        title=f"{legend}: $r_d$={rd_50:.2f} Mpc, $\Omega_M$={Om_50:.3f}",
    )
    corner.corner(
        samples,
        labels=["$r_d$", "$H_0$", "$Ω_m$", "$w_0$"],
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


if __name__ == "__main__":
    main()


"""
******************************
Dataset: DESI 2025
******************************

Flat ΛCDM:
r_d: 146.82 +1.70 -1.68 Mpc
H0: 69.14 +1.15 -1.13 km/s/Mpc
Ωm: 0.298 +0.009 -0.008
w0: -1
wa: 0
Chi squared: 10.27
Degs of freedom: 10
R^2: 0.9987
RMSD: 0.305

===============================

Flat wCDM:
r_d: 144.38 +2.85 -3.08 Mpc
H0: 69.20 +1.20 -1.16 km/s/Mpc
Ωm: 0.297 +0.009 -0.009
w0: -0.915 +0.075 -0.079
wa: 0
Chi squared: 9.09
Degs of freedom: 9
R^2: 0.9989
RMSD: 0.281

==============================

Flat wzCDM:
r_d: 144.93 +2.36 -2.35
H0: 68.42 +1.30 -1.26
Ωm: 0.304 +0.010 -0.010
w0: -0.873 +0.099 -0.105
wa: 0
Chi squared: 8.68
Degs of freedom: 9
R^2: 0.9990
RMSD: 0.272

******************************
Dataset: SDSS 2020
******************************

Flat ΛCDM
r_d: 145.48 +2.88 -2.79 Mpc
H0: 69.13 +2.02 -1.99 km/s/Mpc
Ωm: 0.298 +0.017 -0.016
w0: -1
wa: 0
Chi squared: 10.81
Degs of freedom: 11
R^2: 0.9947
RMSD: 0.739

================================

Flat wCDM
r_d: 138.13 +5.43 -5.04 Mpc
H0: 69.99 +2.02 -2.15 km/s/Mpc
Ωm: 0.290 +0.018 -0.016
w0: -0.776 +0.096 -0.127
wa: 0
Chi squared: 7.54
Degs of freedom: 10
R^2: 0.9956
RMSD: 0.671

================================

Flat wzCDM
r_d: 140.56 +4.16 -4.25 Mpc
H0: 67.83 +2.08 -2.00 km/s/Mpc
Ωm: 0.309 +0.018 -0.018
w0: -0.689 +0.159 -0.176
wa: 0
Chi squared: 7.57
Degs of freedom: 10
R^2: 0.9954
RMSD: 0.686
"""
