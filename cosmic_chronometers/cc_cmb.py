import numba
import numpy as np
import emcee
import corner
import matplotlib.pyplot as plt
from multiprocessing import Pool
import cmb.data_cmb_act_compression as cmb
from y2005cc.data import get_data
from .plot_predictions import plot_cc_predictions

c = cmb.c  # Speed of light in km/s
Or_h2 = cmb.Omega_r_h2()

legend, z_values, H_values, cov_matrix_cc = get_data()
cc_err_factor = 1.5  # Error reduction factor
cov_matrix_cc = cov_matrix_cc / cc_err_factor**2
inv_cov_cc = np.linalg.inv(cov_matrix_cc)
logdet = np.linalg.slogdet(cov_matrix_cc)[1]


@numba.njit
def H_z(z, params):
    H0, Om = params[0], params[1]
    h = H0 / 100
    Or = Or_h2 / h**2
    cubed = (1 + z) ** 3
    rho_de = 1
    return H0 * np.sqrt(Or * (1 + z) ** 4 + Om * cubed + (1 - Om - Or) * rho_de)


bounds = np.array(
    [
        (55, 80),  # H0
        (0.1, 0.5),  # Om
        (0.0210, 0.0235),  # Ωb * h^2
        (0.4, 3),  # f_cc
    ]
)


def chi_squared(params):
    f_cc = params[-1]
    delta_cc = H_values - H_z(z_values, params)
    chi2_cc = np.dot(delta_cc, np.dot(inv_cov_cc * f_cc**2, delta_cc))

    delta_cm = cmb.DISTANCE_PRIORS - cmb.cmb_distances(
        lambda z: H_z(z, params) / params[0], *params[0:3]
    )
    chi2_cmb = np.dot(delta_cm, np.dot(cmb.inv_cov_mat, delta_cm))
    return chi2_cc + chi2_cmb


def log_likelihood(params):
    N = len(z_values)
    normalization = N * np.log(2 * np.pi) + logdet - 2 * N * np.log(params[-1])
    return -0.5 * (chi_squared(params) + normalization)


def log_prior(params):
    if np.all((bounds[:, 0] < params) & (params < bounds[:, 1])):
        return 0.0
    return -np.inf


def log_probability(params):
    lp = log_prior(params)
    if np.isinf(lp):
        return -np.inf
    return lp + log_likelihood(params)


def main():
    ndim = len(bounds)
    nwalkers = 8 * ndim
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
    print("correlation matrix:")
    print(np.array2string(np.corrcoef(samples, rowvar=False), precision=5))

    [
        [H0_16, H0_50, H0_84],
        [Om_16, Om_50, Om_84],
        [Obh2_16, Obh2_50, Obh2_84],
        [f_16, f_50, f_84],
    ] = np.percentile(samples, [15.9, 50, 84.1], axis=0).T

    best_fit = [H0_50, Om_50, Obh2_50, f_50]

    print(f"H0: {H0_50:.2f} +{(H0_84 - H0_50):.2f} -{(H0_50 - H0_16):.2f}")
    print(f"Ωm: {Om_50:.4f} +{(Om_84 - Om_50):.4f} -{(Om_50 - Om_16):.4f}")
    print(
        f"Ωb x h^2: {Obh2_50:.5f} +{(Obh2_84 - Obh2_50):.5f} -{(Obh2_50 - Obh2_16):.5f}"
    )
    print(f"f: {f_50:.2f} +{(f_84 - f_50):.2f} -{(f_50 - f_16):.2f}")
    print(f"Chi squared: {chi_squared(best_fit):.2f}")
    print(f"Log likelihood: {log_likelihood(best_fit):.2f}")
    print(f"Degs of freedom: {3 + z_values.size  - len(best_fit)}")

    plot_cc_predictions(
        H_z=lambda z: H_z(z, best_fit),
        z=z_values,
        H=H_values,
        H_err=np.sqrt(np.diag(cov_matrix_cc)) / f_50,
        label=f"{legend} $H_0$: {H0_50:.2f} ± {(H0_84 - H0_50):.2f} km/s/Mpc",
    )
    corner.corner(
        samples,
        labels=["$H_0$", "$Ω_m$", "$Ω_b h^2$", "f"],
        quantiles=[0.159, 0.5, 0.841],
        show_titles=True,
        title_fmt=".4f",
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
*******************************
Results for data from
https://arxiv.org/pdf/2307.09501
and one data point from
https://arxiv.org/pdf/2506.03836
*******************************

Flat ΛCDM
H0: 67.25 ± 0.50 km/s/Mpc
Ωm: 0.317 ± 0.007
Ωb x h^2: 0.02237 ± 0.00014
f_cc: 1.00 +0.12 -0.12
Chi squared: 33.31
Log likelihood: -130.47
Degs of freedom: 32

correlation matrix:
[[ 1.      -0.99291  0.703   -0.00839]
 [-0.99291  1.      -0.63719  0.00822]
 [ 0.703   -0.63719  1.      -0.00837]
 [-0.00839  0.00822 -0.00837  1.     ]]
"""
