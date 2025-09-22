import numpy as np
import emcee
import corner
from scipy.integrate import quad
from scipy.linalg import cho_factor, cho_solve
import matplotlib.pyplot as plt
from multiprocessing import Pool
from y2025BAO.data import get_data
from cmb.data import r_drag, Omega_r_h2, c
from .plot_predictions import plot_bao_predictions

legend, data, cov_matrix = get_data()
cho = cho_factor(cov_matrix)


def rd(params):  # Mpc
    H0, Om, Obh2 = params[0], params[1], params[2]
    Omh2 = Om * (H0 / 100) ** 2
    return r_drag(wb=Obh2, wm=Omh2)


def Ez(z, H0, Om, w0):
    h = H0 / 100
    Or = Omega_r_h2() / h**2
    OL = 1 - Om - Or
    one_plus_z = 1 + z
    cubic = one_plus_z**3
    rho_de = (2 * cubic / (1 + cubic)) ** (2 * (1 + w0))
    return np.sqrt(Or * one_plus_z**4 + Om * one_plus_z**3 + OL * rho_de)


def H_z(z, H0, Om, w0):
    return H0 * Ez(z, H0, Om, w0)


def DH_z(z, params):
    H0, Om, w0 = params[0], params[1], params[3]
    return c / H_z(z, H0, Om, w0)


def DM_z(z, params):
    return quad(lambda zp: DH_z(zp, params), 0, z)[0]


def DV_z(z, params):
    DH = DH_z(z, params)
    DM = DM_z(z, params)
    return (z * DH * DM**2) ** (1 / 3)


quantity_funcs = {
    "DV_over_rs": lambda z, params: DV_z(z, params) / rd(params),
    "DM_over_rs": lambda z, params: DM_z(z, params) / rd(params),
    "DH_over_rs": lambda z, params: DH_z(z, params) / rd(params),
}


def theory_predictions(z, qty, params):
    return np.array([(quantity_funcs[qty](z, params)) for z, qty in zip(z, qty)])


bounds = np.array(
    [
        (50, 80),  # H0
        (0.15, 0.55),  # Ωm
        (0.015, 0.030),  # Ωb h^2
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
    return -0.5 * ((omega_b_h2_prior - params[2]) / omega_b_h2_prior_sigma) ** 2


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
        [Om_16, Om_50, Om_84],
        [Obh2_16, Obh2_50, Obh2_84],
        [w0_16, w0_50, w0_84],
    ] = np.percentile(samples, [15.9, 50, 84.1], axis=0).T

    best_fit = [H0_50, Om_50, Obh2_50, w0_50]

    h_samples = samples[:, 0] / 100
    Omh2_samples = samples[:, 1] * h_samples**2
    rd_samples = r_drag(wb=samples[:, 2], wm=Omh2_samples)
    Omh2_16, Omh2_50, Omh2_84 = np.percentile(Omh2_samples, [15.9, 50, 84.1])
    rd_16, rd_50, rd_84 = np.percentile(rd_samples, [15.9, 50, 84.1])

    residuals = data["value"] - theory_predictions(
        data["z"], data["quantity"], best_fit
    )
    SS_res = np.sum(residuals**2)
    SS_tot = np.sum((data["value"] - np.mean(data["value"])) ** 2)
    r2 = 1 - SS_res / SS_tot

    print(f"H0: {H0_50:.2f} +{(H0_84 - H0_50):.2f} -{(H0_50 - H0_16):.2f} km/s/Mpc")
    print(
        f"Ωb h^2: {Obh2_50:.4f} +{(Obh2_84 - Obh2_50):.4f} -{(Obh2_50 - Obh2_16):.4f}"
    )
    print(
        f"Ωm h^2: {Omh2_50:.4f} +{(Omh2_84 - Omh2_50):.4f} -{(Omh2_50 - Omh2_16):.4f}"
    )
    print(f"Ωm: {Om_50:.4f} +{Om_84-Om_50:.4f} -{Om_50-Om_16:.4f}")
    print(f"w0: {w0_50:.3f} +{(w0_84 - w0_50):.3f} -{(w0_50 - w0_16):.3f}")
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
        labels=["$H_0$", "$Ω_m$", "$Ω_b x h^2$", "$w_0$"],
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
*******************************
Dataset: DESI 2025
*******************************

Flat ΛCDM:
H0: 68.52 +0.65 -0.64 km/s/Mpc
Ωb h^2: 0.0220 ± 0.0006
Ωm h^2: 0.1397 +0.0053 -0.0051
Ωm: 0.2975 +0.0088 -0.0085
w0: -1
r_d: 148.18 +1.68 -1.69 Mpc
Chi squared: 10.29
Degs of freedom: 10
R^2: 0.9987
RMSD: 0.305

===============================

Flat wCDM:
H0: 66.29 +2.23 -2.25 km/s/Mpc
Ωb h^2: 0.0220 ± 0.0006
Ωm h^2: 0.1308 +0.0100 -0.0102
Ωm: 0.2968 +0.0091 -0.0088
w0: -0.917 +0.076 -0.079
r_d: 150.66 +3.20 -2.92 Mpc
Chi squared: 9.05
Degs of freedom: 9
R^2: 0.9989
RMSD: 0.280

===============================

Flat alternative: w(z) = -1 + 2 * (1 + w0) / (1 + (1 + z)**3)
H0: 65.73 +2.25 -2.12 km/s/Mpc
Ωb h^2: 0.0220 ± 0.0006
Ωm h^2: 0.1329 +0.0073 -0.0070
Ωm: 0.3075 +0.0119 -0.0117
w0: -0.834 +0.122 -0.129
r_d: 150.03 +2.26 -2.22 Mpc
Chi squared: 8.43
Degs of freedom: 9
R^2: 0.9990
RMSD: 0.266

*******************************
Dataset: SDSS 2020
*******************************

Flat ΛCDM
H0: 67.40 +1.10 -1.05 km/s/Mpc
Ωb h^2: 0.0220 ± 0.0006
Ωm h^2: 0.1368 +0.0111 -0.0100
Ωm: 0.3012 +0.0183 -0.0170
w0: -1
r_d: 148.93 +3.02 -3.08 Mpc
Chi squared: 10.50
Degs of freedom: 11
R^2: 0.9941
RMSD: 0.770

===============================

Flat wCDM
H0: 60.88 +4.22 -4.65 km/s/Mpc
Ωb h^2: 0.0220 ± 0.0006
Ωm h^2: 0.1077 +0.0215 -0.0235
Ωm: 0.2882 +0.0218 -0.0256
w0: -0.755 +0.137 -0.147
r_d: 157.95 +9.27 -6.92 Mpc
Chi squared: 7.43
Degs of freedom: 10
R^2: 0.9950
RMSD: 0.714

===============================

Flat alternative: w(z) = -1 + 2 * (1 + w0) / (1 + (1 + z)**3)
H0: 62.11 +3.49 -3.19 km/s/Mpc
Ωb h^2: 0.0220 ± 0.0006
Ωm h^2: 0.1228 +0.0137 -0.0125
Ωm: 0.3177 +0.0212 -0.0203
w0: -0.663 +0.195 -0.211
r_d: 153.01 +4.14 -4.11 Mpc
Chi squared: 7.66
Degs of freedom: 10
R^2: 0.9946
RMSD: 0.740
"""
