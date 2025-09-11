import numpy as np
import emcee
import corner
from scipy.integrate import quad
from scipy.linalg import cho_factor, cho_solve
import matplotlib.pyplot as plt
from multiprocessing import Pool
from y2025BAO.data import get_data as get_bao_data
from .plot_predictions import plot_bao_predictions

bao_legend, data, bao_cov_matrix = get_bao_data()
cho_bao = cho_factor(bao_cov_matrix)

c = 299792.458  # Speed of light in km/s


def Ez(z, O_m, w0):
    one_plus_z = 1 + z
    cubic = one_plus_z**3
    rho_de = (2 * cubic / (1 + cubic)) ** (2 * (1 + w0))
    return np.sqrt(O_m * cubic + (1 - O_m))


def H_z(z, params):
    return params[1] * Ez(z, *params[2:])


def DH_z(z, params):
    return c / H_z(z, params)


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
        (120, 160),  # r_d
        (50, 80),  # H0
        (0.1, 0.5),  # Ωm
        (-2, 0),  # w0
    ]
)


def chi_squared(params):
    delta_bao = data["value"] - theory_predictions(data["z"], data["quantity"], params)
    chi_bao = np.dot(delta_bao, cho_solve(cho_bao, delta_bao))
    return chi_bao


# Prior from Planck 2018 https://arxiv.org/abs/1807.06209 table 1 (Combined column)
# Ωm x ​h^2 = 0.1428 ± 0.0011. Prior width increased by 70% to 0.00187
def log_prior(params):
    if np.all((bounds[:, 0] < params) & (params < bounds[:, 1])):
        Om_x_h2 = params[2] * (params[1] / 100) ** 2
        return -0.5 * ((0.1428 - Om_x_h2) / 0.00187) ** 2
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
    nwalkers = 50 * ndim
    burn_in = 200
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

    chains_samples = sampler.get_chain(discard=0, flat=False)
    samples = sampler.get_chain(discard=burn_in, flat=True)

    [
        [rd_16, rd_50, rd_84],
        [H0_16, H0_50, H0_84],
        [Om_16, Om_50, Om_84],
        [w0_16, w0_50, w0_84],
    ] = np.percentile(samples, [15.9, 50, 84.1], axis=0).T

    best_fit = [rd_50, H0_50, Om_50, w0_50]

    print(f"r_d: {rd_50:.2f} +{(rd_84 - rd_50):.2f} -{(rd_50 - rd_16):.2f}")
    print(f"H0: {H0_50:.2f} +{(H0_84 - H0_50):.2f} -{(H0_50 - H0_16):.2f}")
    print(f"Ωm: {Om_50:.3f} +{(Om_84 - Om_50):.3f} -{(Om_50 - Om_16):.3f}")
    print(f"w0: {w0_50:.3f} +{(w0_84 - w0_50):.3f} -{(w0_50 - w0_16):.3f}")
    print(f"Chi squared: {chi_squared(best_fit):.2f}")
    print(f"Degrees of freedom: {data['z'].size - len(best_fit)}")

    plot_bao_predictions(
        theory_predictions=lambda z, qty: theory_predictions(z, qty, best_fit),
        data=data,
        errors=np.sqrt(np.diag(bao_cov_matrix)),
        title=bao_legend,
    )

    labels = ["$r_d$", "$H_0$", "$\Omega_m$", "$w_0$"]
    corner.corner(
        samples,
        labels=labels,
        quantiles=[0.159, 0.5, 0.841],
        show_titles=True,
        title_fmt=".4f",
        smooth=1.5,
        smooth1d=1.5,
        bins=100,
        levels=(0.393, 0.864),
        fill_contours=False,
        plot_datapoints=False,
    )
    plt.show()

    _, axes = plt.subplots(ndim, figsize=(10, 7))
    for i in range(ndim):
        axes[i].plot(chains_samples[:, :, i], color="black", alpha=0.3)
        axes[i].set_ylabel(labels[i])
        axes[i].set_xlabel("chain step")
        axes[i].axvline(x=burn_in, color="red", linestyle="--", alpha=0.5)
        axes[i].axhline(y=best_fit[i], color="white", linestyle="--", alpha=0.5)
    plt.show()


if __name__ == "__main__":
    main()

"""
Flat ΛCDM
r_d: 146.58 +1.55 -1.53 Mpc
H0: 69.27 +1.11 -1.09 km/s/Mpc
Ωm: 0.298 +0.009 -0.008
w0: -1
Chi squared: 10.54
Degrees of freedom: 10

====================

Flat wCDM w(z) = w0
r_d: 144.18 +2.75 -3.00 Mpc
H0: 69.35 +1.13 -1.11 km/s/Mpc
Ωm: 0.297 +0.009 -0.009
w0: -0.918 +0.075 -0.078
Chi squared: 9.40
Degrees of freedom: 9

====================

Flat w(z) = -1 + 2 * (1 + w0) / (1 + (1 + z)**3)
r_d: 144.83 +2.08 -2.07 Mpc
H0: 68.14 +1.38 -1.32 km/s/Mpc
Ωm: 0.307 +0.012 -0.011
w0: -0.837 +0.119 -0.126
Chi squared: 8.72
Degrees of freedom: 9
"""
