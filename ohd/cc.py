from numba import njit
import numpy as np
from scipy.linalg import cho_factor
from solve_triangular import solve_triangular
from y2005cc.data import get_data

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
    f_array = params[2] + params[3] * z_values / (1.0 + z_values)
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
    prior.add_parameter("f0", dist=(0.1, 6.0))
    prior.add_parameter("fa", dist=(-9.0, 9.0))

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
    f_array = best_fit[2] + best_fit[3] * z_values / (1. + z_values)
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


# Model: Flat ΛCDM
# ---------------------------------

# Redshift dependent covariance error scaling f(z) = f0 + fa * z/(1+z):
# cov[i, j] = base_cov[i, j] / (f(z_i) * f(z_j))
#
# H0: 68.1 +- 2.2 km/s/Mpc
# Ωm: 0.306 +0.037 -0.041
# Ωm h^2: 0.141 +- 0.016
# f0: 2.98 +- 0.57  (prior ~U[0.1, 6.0])
# fa: -3.3 +- 1.2 (prior ~U[-9.0, 9.0])
# Log likelihood (MAP): -149.48
# Log evidence: -159.03 (diff: 4.29 strong evidence favouring the model with f0, fa)
# χ2 (MAP): 39.07
# DOF: 35
# χ2/DOF: 1.12
# ---------------------------------

# Without error scaling (fixed f0 = 1, fa = 0):
# H0: 66.3 ± 5.4 km/s/Mpc
# Om: 0.344 +0.054 -0.081
# Ωm h^2: 0.148 ± 0.018
# Log likelihood (MAP): -159.38
# Log evidence: -163.32
# χ2 (MAP): 16.61
# DOF: 37
# χ2/DOF: 0.45
# ---------------------------------

# Log likelihood ratio test:
# -2 * log(L0/L1) = -2 * log(L0) + 2 * log(L1)
# -2 * (-159.38) + 2 * (-149.48) = 19.80
#
# DOF = 2
#
# Parametric-bootstrap likelihood-ratio test:
# The full covariance-rescaling model improves the maximum log likelihood by
# Delta log L = 9.9 relative to the fixed published-covariance model:
#
# Lambda = 2 * (log L_scaling - log L_fixed) = 19.80.
#
# A parametric bootstrap with 200k data sets simulated under the
# fixed-covariance null produced 11 values of Lambda >= 19.80, giving
# p_bootstrap = 6.0e-05 (Monte Carlo SE = 1.7e-05).
#
# Conditional on flat LCDM, Gaussian errors, the published covariance,
# the chosen parameter bounds, and f(z) = f0 + fa * z/(1+z), the data favor
# the covariance-rescaling extension over the fixed-covariance model.
#
# This result does not by itself establish that the published uncertainties
# are overestimated or that their discrepancy is intrinsically
# redshift-dependent; unmodeled systematics or mean-model inadequacy could
# also produce this preference.