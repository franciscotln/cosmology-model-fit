from numba import njit
import numpy as np
from scipy.constants import c as c0
from scipy.linalg import cho_factor
from solve_triangular import solve_triangular
from y2005cc.data import get_data

c = c0 / 1000  # Speed of light in km/s

legend, z_values, H_values, cov_matrix = get_data()

L = cho_factor(cov_matrix, lower=True)[0]
logdet_base = 2.0 * np.log(np.diag(L)).sum()
N = z_values.size


@njit
def H_z(z, params):
    H0, Om = params[0], params[1]
    return H0 * np.sqrt(Om * (1.0 + z) ** 3 + (1.0 - Om))


@njit
def log_likelihood_jit(params):
    f_array = params[2] + params[3] * z_values
    if np.any(f_array <= 1e-4):
        return -np.inf

    delta = H_values - H_z(z_values, params)
    chi2 = solve_triangular(L, f_array * delta)

    logdet = logdet_base - 2.0 * np.log(f_array).sum()
    normalization = N * np.log(2 * np.pi) + logdet

    return -0.5 * (chi2 + normalization)


def log_likelihood(params):
    return log_likelihood_jit(params)


def main():
    from multiprocessing import Pool
    from nautilus import Sampler, Prior
    from getdist import plots, MCSamples
    import matplotlib.pyplot as plt
    from ohd.plot_predictions import plot_cc_predictions

    prior = Prior()
    prior.add_parameter("H0", dist=(30, 100))
    prior.add_parameter("Om", dist=(0.0, 1.0))
    prior.add_parameter("f0", dist=(0.5, 4.0))
    prior.add_parameter("fa", dist=(-2.0, 2.0))

    with Pool(5) as pool:
        sampler = Sampler(
            prior, log_likelihood, n_live=6_000, pool=pool, seed=42, pass_dict=False
        )
        sampler.run(verbose=True)

    samples, log_w, log_l = sampler.posterior()
    weights = np.exp(log_w)
    labels=["H_0", "\\Omega_m", "f_0", "f_a"]

    gd_samples = MCSamples(
        samples=samples,
        weights=weights,
        loglikes=log_l,
        names=prior.keys,
        labels=labels,
    )
    gd_samples.addDerived(gd_samples["Om"] * (gd_samples["H0"] / 100) ** 2, name="Omh2")
    gd_samples.updateBaseStatistics()

    for par in gd_samples.getParamNames().names:
        print(f"{par}: {gd_samples.mean(par):.4f} ± {gd_samples.std(par):.4f}")

    best_fit = samples[np.argmax(log_l)]
    f_array = best_fit[2] + best_fit[3] * z_values
    delta = H_values - H_z(z_values, best_fit)
    chi2 = solve_triangular(L, f_array * delta)

    print(f"Log likelihood (MAP): {log_likelihood(best_fit):.2f}")
    print(f"Log evidence: {sampler.log_z:.2f}")
    print(f"χ2 (MAP): {chi2:.2f}")
    print(f"DOF: {N - len(best_fit)}")

    plots.getSubplotPlotter().triangle_plot(
        gd_samples, filled=True, title_limit=1, contour_colors=["C0"], color=["C0"],
    )
    plt.show()

    plot_cc_predictions(
        H_z=lambda z: H_z(z, best_fit),
        z=z_values,
        H=H_values,
        H_err=np.sqrt(np.diag(cov_matrix)) / f_array,
        label=f"{legend} $H_0$: {best_fit[0]:.1f} km/s/Mpc",
    )


if __name__ == "__main__":
    main()


# *********************************
# Results for data from
# https://arxiv.org/pdf/2307.09501
#
# 4 data points from
# https://arxiv.org/pdf/2506.03836
# https://arxiv.org/pdf/2511.02730v1
#
# 1 data point from
# https://arxiv.org/pdf/2606.07298v1
#
# 1 data point from
# https://arxiv.org/pdf/2608.13178
# *********************************


# Model: Flat ΛCDM
# ---------------------------------

# Redshift dependent covariance error scaling f(z) = f0 + fa * z:
# cov[i, j] = base_cov[i, j] / (f(z_i) * f(z_j))
#
# H0: 67.6 +- 2.6
# Ωm: 0.318 +0.035 -0.044
# f0: 2.23 +- 0.35  (prior ~U[0.5, 4.0])
# fa: -0.79 +0.27 -0.32 (prior ~U[-2.0, 2.0])
# Ωm h^2: 0.144 +- 0.015
# Log likelihood (MAP): -145.70
# Log evidence: -154.46 (diff: 3.80 strong evidence favouring the model with f0, fa)
# χ2 (MAP): 38.53
# DOF: 34
# χ2/DOF: 1.13
# ---------------------------------

# Without error scaling (fixed f0 = 1, fa = 0):
# H0: 66.4 +5.4 -5.4 km/s/Mpc
# Ωm: 0.34 +0.08 -0.06
# Ωm h^2: 0.148 +0.018 -0.017
# Log likelihood (MAP): -154.31
# Log evidence: -158.26
# χ2 (MAP): 16.39
# DOF: 36
# χ2/DOF: 0.46 
# ---------------------------------

# Log likelihood ratio test:
# -2 * log(L0/L1) = -2 * log(L0) + 2 * log(L1)
# -2 * (-154.31) + 2 * (-145.70) = 17.22
#
# DOF = 2
# p-value = 0.00018
# We are 99.98% confident that the model with f is better than the one without f.
# So the uncertainties in the H(z) dataset are overestimated and redshift dependent
# 3.74 sigma significance.
