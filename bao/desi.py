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


def rd(omega_b_h2, omega_c_h2, N_eff=3.04):  # Mpc
    # https://arxiv.org/abs/2212.04522
    return (
        147.05
        * (omega_b_h2 / 0.02236) ** -0.13
        * ((omega_b_h2 + omega_c_h2) / 0.1432) ** -0.23
        * (N_eff / 3.04) ** -0.1
    )


def H_z(z, H0, omega_b_h2, omega_c_h2, w0=-1):
    h = H0 / 100
    Om = (omega_b_h2 + omega_c_h2) / h**2
    OL = 1 - Om

    one_plus_z = 1 + z
    rho_de = (2 * one_plus_z**2 / (1 + one_plus_z**2)) ** (3 * (1 + w0))
    return H0 * np.sqrt(Om * one_plus_z**3 + OL * rho_de)


def DH_z(z, params):
    return c / H_z(z, *params)


def DM_z(z, params):
    return quad(lambda zp: DH_z(zp, params), 0, z)[0]


def DV_z(z, params):
    DH = DH_z(z, params)
    DM = DM_z(z, params)
    return (z * DH * DM**2) ** (1 / 3)


quantity_funcs = {
    "DV_over_rs": lambda z, params: DV_z(z, params) / rd(*params[1:3]),
    "DM_over_rs": lambda z, params: DM_z(z, params) / rd(*params[1:3]),
    "DH_over_rs": lambda z, params: DH_z(z, params) / rd(*params[1:3]),
}


def theory_predictions(z, qty, params):
    return np.array([(quantity_funcs[qty](z, params)) for z, qty in zip(z, qty)])


bounds = np.array(
    [
        (50, 80),  # H0
        (0.001, 0.04),  # Ωb h^2
        (0.01, 0.2),  # Ωc h^2
        (-1.5, 0),  # w0
    ]
)


def chi_squared(params):
    delta = data["value"] - theory_predictions(data["z"], data["quantity"], params)
    return np.dot(delta, cho_solve(cho, delta))


# Prior from BBN: https://arxiv.org/abs/2401.15054
omega_b_h2_prior = 0.02196
omega_b_h2_prior_sigma = 0.00063


def log_prior(params):
    if not np.all((bounds[:, 0] < params) & (params < bounds[:, 1])):
        return -np.inf
    return -0.5 * ((omega_b_h2_prior - params[1]) / omega_b_h2_prior_sigma) ** 2


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
    nsteps = 20000 + burn_in
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
        [H0_16, H0_50, H0_84],
        [Ombh2_16, Ombh2_50, Ombh2_84],
        [Omch2_16, Omch2_50, Omch2_84],
        [w0_16, w0_50, w0_84],
    ] = np.percentile(samples, [15.9, 50, 84.1], axis=0).T

    best_fit = [H0_50, Ombh2_50, Omch2_50, w0_50]

    # Om and rd are derived from H0, Ombh2, and Omch2
    h_samples = samples[:, 0] / 100
    Om_samples = (samples[:, 1] + samples[:, 2]) / h_samples**2
    Om_16, Om_50, Om_84 = np.percentile(Om_samples, [15.9, 50, 84.1])
    rd_samples = rd(samples[:, 1], samples[:, 2])
    rd_16, rd_50, rd_84 = np.percentile(rd_samples, [15.9, 50, 84.1])

    residuals = data["value"] - theory_predictions(
        data["z"], data["quantity"], best_fit
    )
    SS_res = np.sum(residuals**2)
    SS_tot = np.sum((data["value"] - np.mean(data["value"])) ** 2)
    r2 = 1 - SS_res / SS_tot

    print(f"H0: {H0_50:.2f} +{(H0_84 - H0_50):.2f} -{(H0_50 - H0_16):.2f} km/s/Mpc")
    print(
        f"Ωb h^2: {Ombh2_50:.4f} +{(Ombh2_84 - Ombh2_50):.4f} -{(Ombh2_50 - Ombh2_16):.4f}"
    )
    print(
        f"Ωc h^2: {Omch2_50:.4f} +{(Omch2_84 - Omch2_50):.4f} -{(Omch2_50 - Omch2_16):.4f}"
    )
    print(f"w0: {w0_50:.3f} +{(w0_84 - w0_50):.3f} -{(w0_50 - w0_16):.3f}")
    print(f"Ωm: {Om_50:.4f} +{Om_84-Om_50:.4f} -{Om_50-Om_16:.4f}")
    print(f"r_d: {rd_50:.2f} +{(rd_84 - rd_50):.2f} -{(rd_50 - rd_16):.2f} Mpc")
    print(f"Chi squared: {chi_squared(best_fit):.2f}")
    print(f"Degs of freedom: {data['value'].size  - len(best_fit)}")
    print(f"R^2: {r2:.4f}")
    print(f"RMSD: {np.sqrt(np.mean(residuals**2)):.3f}")

    plot_bao_predictions(
        theory_predictions=lambda z, qty: theory_predictions(z, qty, best_fit),
        data=data,
        errors=np.sqrt(np.diag(cov_matrix)),
        title=f"{legend}: $H_0$={H0_50:.2f} km/s/Mpc, $\\Omega_m$={Om_50:.4f}",
    )
    corner.corner(
        samples,
        labels=["$H_0$", "$Ω_b x h^2$", "$Ω_c x h^2$", "$w_0$"],
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
H0: 68.49 +0.59 -0.59 km/s/Mpc
Ωb h^2: 0.0220 +0.0006 -0.0006
Ωc h^2: 0.1177 +0.0045 -0.0044
w0: -1
Ωm: 0.2978 +0.0087 -0.0085
r_d: 148.23 +1.46 -1.45 Mpc
Chi squared: 10.27
Degs of freedom: 9
R^2: 0.9987
RMSD: 0.305

===============================

Flat wCDM:
H0: 66.52 +2.08 -2.10 km/s/Mpc
Ωb h^2: 0.0220 +0.0006 -0.0006
Ωc h^2: 0.1099 +0.0091 -0.0096
w0: -0.921 +0.077 -0.081
Ωm: 0.2972 +0.0090 -0.0087
r_d: 150.21 +2.83 -2.48 Mpc
Chi squared: 9.06
Degs of freedom: 9
R^2: 0.9989
RMSD: 0.281

==============================

Flat wzCDM:
H0: 66.28 +2.07 -1.95 km/s/Mpc
Ωb h^2: 0.0220 +0.0006 -0.0006
Ωc h^2: 0.1116 +0.0071 -0.0070
w0: -0.881 +0.100 -0.107
Ωm: 0.3037 +0.0103 -0.0101
r_d: 149.78 +2.10 -2.04 Mpc
Chi squared: 8.69
Degs of freedom: 9
R^2: 0.9990
RMSD: 0.273

******************************
Dataset: SDSS 2020
******************************

Flat ΛCDM
H0: 67.43 +0.93 -0.93 km/s/Mpc
Ωb h^2: 0.0220 +0.0006 -0.0006
Ωc h^2: 0.1150 +0.0099 -0.0092
w0: -1
Ωm: 0.3012 +0.0180 -0.0169
Derived r_d: 148.91 Mpc
Chi squared: 10.48
Degs of freedom: 10
R^2: 0.9941
RMSD: 0.770

================================

Flat wCDM
H0: 61.67 +3.89 -4.37 km/s/Mpc
Ωb h^2: 0.0220 +0.0006 -0.0006
Ωc h^2: 0.0893 +0.0200 -0.0224
w0: -0.767 +0.134 -0.146
Ωm: 0.2901 +0.0211 -0.0241
Derived r_d: 156.18 Mpc
Chi squared: 7.43
Degs of freedom: 10
R^2: 0.9949
RMSD: 0.717

================================

Flat wzCDM
H0: 62.58 +3.30 -3.17 km/s/Mpc
Ωb h^2: 0.0220 +0.0006 -0.0006
Ωc h^2: 0.1000 +0.0139 -0.0131
w0: -0.721 +0.167 -0.181
Ωm: 0.3106 +0.0195 -0.0182
Derived r_d: 152.90 Mpc
Chi squared: 7.57
Degs of freedom: 10
R^2: 0.9947
RMSD: 0.732
"""
