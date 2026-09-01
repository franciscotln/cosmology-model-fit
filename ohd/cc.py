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
    f_array = np.exp(params[2]) * (1.0 + z_values) ** params[3]
    if np.any(f_array <= 1e-4):
        return -np.inf

    delta = H_values - H_z(z_values, params)
    y = solve_triangular(L, f_array * delta)
    chi2 = np.dot(y, y)

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
    prior.add_parameter("Om", dist=(0.0, 2.0))
    prior.add_parameter("ln_f0", dist=(-0.5, 2.5))
    prior.add_parameter("n", dist=(-4.0, 4.0))

    with Pool(8) as pool:
        sampler = Sampler(
            prior, log_likelihood, n_live=8_000, pool=pool, seed=42, pass_dict=False
        )
        sampler.run(verbose=True)

    samples, log_w, log_l = sampler.posterior()
    weights = np.exp(log_w)
    labels=["H_0", "\\Omega_m", "\\ln f_0", "n"]

    gd_samples = MCSamples(
        samples=samples,
        weights=weights,
        loglikes=log_l,
        names=prior.keys,
        labels=labels,
    )
    gd_samples.addDerived(gd_samples["Om"] * (gd_samples["H0"] / 100) ** 2, name="Omh2")
    gd_samples.updateBaseStatistics()

    for name in gd_samples.getParamNames().names:
        print(gd_samples.getInlineLatex(name, limit=1))

    best_fit = samples[np.argmax(log_l)]
    f_array = np.exp(best_fit[2]) * (1.0 + z_values) ** best_fit[3]
    delta = H_values - H_z(z_values, best_fit)
    y = solve_triangular(L, f_array * delta)
    chi2 = np.dot(y, y)

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

# Redshift dependent covariance error scaling f(z) = f0 * (1+z)^n:
# cov[i, j] = base_cov[i, j] / (f(z_i) * f(z_j))
#
# H0 = 68.2 +2.1 -1.9 km/s/Mpc
# Ωm = 0.303 +- 0.042
# Ωm h^2 = 0.141 +- 0.017
# ln(f0) = 1.14 +0.27 -0.24 (prior ~U[-0.5, 2.5])
# n = -1.36 +- 0.48 (prior ~U[-4, 4])
# Log likelihood (MAP): -149.38
# Log evidence: -158.75 (diff: 4.57 strong evidence favouring the model with f0, n)
# χ2 (MAP): 38.88
# DOF: 35
# χ2/DOF: 1.11
# ---------------------------------

# Without error scaling (fixed f0 = 1, n = 0):
# H0: 66.3 ± 5.4 km/s/Mpc
# Om: 0.344 +0.053 -0.083
# Ωm h^2: 0.148 ± 0.018
# Log likelihood (MAP): -159.38
# Log evidence: -163.32
# χ2 (MAP): 16.61
# DOF: 37
# χ2/DOF: 0.45
# ---------------------------------

# Log likelihood ratio test:
# -2 * log(L0/L1) = -2 * log(L0) + 2 * log(L1)
# -2 * (-159.38) + 2 * (-149.38) = 20.00
#
# DOF = 2
#
# Parametric-bootstrap likelihood-ratio test:
# The full covariance-rescaling model improves the maximum log likelihood by
# Delta log L = 10.0 relative to the fixed published-covariance model:
#
# Lambda = 2 * (log L_scaling - log L_fixed) = 20.0.
#
# A parametric bootstrap with 100k data sets simulated under the
# fixed-covariance null produced 12 values of Lambda >= 20.0, giving
# p_bootstrap = 1.3e-4 (Monte Carlo SE = 3.6e-05).
#
# Conditional on flat LCDM, Gaussian errors, the published covariance,
# the chosen parameter bounds, and f(z) = f0 * (1+z)^n, the data favor
# the covariance-rescaling extension over the fixed-covariance model.
#
# This result does not by itself establish that the published uncertainties
# are overestimated or that their discrepancy is intrinsically
# redshift-dependent; unmodeled systematics or mean-model inadequacy could
# also produce this preference.