from numba import njit
import numpy as np
from solve_triangular import solve_triangular
from y2005cc.data_no_loubser import get_data, method

legend, z_values, H_values, diag_stat, cov_matrix_sys = get_data(split_sys=True)


@njit
def H_z(z, params):
    h, om = params[0], params[1]
    return 100 * h * np.sqrt(om * (1.0 + z) ** 3 + (1.0 - om))


@njit
def get_fz(params):
    z_pivot = 0.982
    fp, n = params[2], params[3]
    return fp * ((1.0 + z_values) / (1.0 + z_pivot)) ** n


@njit
def log_likelihood_jit(params):
    cov_mat = cov_matrix_sys + np.diag(diag_stat**2 * get_fz(params)**2)
    L = np.linalg.cholesky(cov_mat)
    logdet = 2.0 * np.sum(np.log(np.diag(L)))

    diff = H_values - H_z(z_values, params)
    y = solve_triangular(L, diff)
    chi2 = np.dot(y, y)
    normalization = z_values.size * np.log(2 * np.pi) + logdet

    return -0.5 * (chi2 + normalization)


def log_likelihood(params):
    return log_likelihood_jit(params)


def main():
    from multiprocessing import Pool
    from scipy.stats import lognorm
    from nautilus import Sampler, Prior
    from getdist import plots, MCSamples
    import matplotlib.pyplot as plt
    from ohd.plot_predictions import plot_cc_predictions

    prior = Prior()
    prior.add_parameter("h", dist=(0.5, 1.0))
    prior.add_parameter("om", dist=(0.01, 1.0))
    prior.add_parameter("fp", dist=lognorm(s=0.4, scale=0.7))
    prior.add_parameter("n", dist=(-5.0, 10.0))

    with Pool(8) as pool:
        sampler = Sampler(prior, log_likelihood, n_live=10_000, pool=pool, seed=42, pass_dict=False)
        sampler.run(verbose=True)

    samples, log_w, log_l = sampler.posterior()
    MAP_PARAMS = samples[np.argmax(log_l)]
    weights = np.exp(log_w)
    labels = ["h", "Ω_m", "f_{pivot}", "n"]

    gd_samples = MCSamples(samples=samples, weights=weights, names=prior.keys, labels=labels)
    gd_samples.addDerived(gd_samples["om"] * gd_samples["h"]**2, name="omh2", label="Ω_m h^2")
    gd_samples.updateBaseStatistics()

    plot_params = ["omh2"] + prior.keys
    correlation = gd_samples.corr(plot_params)
    print("Correlation matrix:\n", correlation)

    for name in gd_samples.getParamNames().names:
        print(gd_samples.getInlineLatex(name, limit=1))

    f_z = get_fz(MAP_PARAMS)
    cov = cov_matrix_sys + np.diag(diag_stat**2 * f_z**2)
    L = np.linalg.cholesky(cov)
    y = solve_triangular(L, H_values - H_z(z_values, MAP_PARAMS))
    chi2 = np.dot(y, y)
    DOF = z_values.size - len(MAP_PARAMS)
    chi2_red = chi2 / DOF

    print(f"Log likelihood (MAP): {np.max(log_l):.2f}")
    print(f"Log evidence: {sampler.log_z:.2f}")
    print(f"χ2 (MAP): {chi2:.2f}")
    print(f"DOF: {DOF}")
    print(f"χ2/DOF: {chi2_red:.2f}")

    g = plots.getSubplotPlotter()
    g.triangle_plot(
        gd_samples,
        params=plot_params,
        filled=True,
        title_limit=1,
        contour_colors=["C0"],
        color=["C0"],
    )
    for i in range(1, len(plot_params)):
        for j in range(i):
            ax = g.subplots[i, j]
            ax.text(
                0.05,
                0.93,
                f"$\\rho = {correlation[i, j]:+.2f}$",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=6,
            )

    plt.show()

    plot_cc_predictions(
        H_z=lambda z: H_z(z, MAP_PARAMS),
        z=z_values,
        H=H_values,
        H_err=np.sqrt(np.diag(cov_matrix_sys) + diag_stat**2),
        label=legend,
        method=method,
        err_scaling=1 / f_z,
    )


if __name__ == "__main__":
    main()


# ---------------------------------
# The Loubser et al. (2025) 3 data points
# have been excluded from this analysis below
# they have their own highly correlated
# systematics covariance
# ---------------------------------


# Model: Flat ΛCDM
# ---------------------------------

# Redshift dependent covariance diagonal scaling
# f(z) = f_pivot * [(1 + z) / (1 + z_pivot)]^n
# with z_pivot = 1.034, corr(ln(fp), n) = 0:
# cov[i, i] = cov_sys[i, i] + cov_diag_stat[i, i] * f(z_i)^2
# cov[i, j] = cov_sys[i, j]
#
# h = 0.673 ± 0.034
# Ωm = 0.322 +0.036 -0.043
# Ωm h^2 = 0.145 ± 0.015
# n = 2.99 +0.84 -1.2 (prior ~ U[-5, 10])
# fp = 0.63 +0.10 -0.16 (prior ~ lognormal(μ=ln(0.7), σ=0.4))
# Log likelihood (MAP): -137.27
# Log evidence: -144.12
# χ2 (MAP): 34.67
# DOF: 32
# χ2/DOF: 1.08
# Correlation matrix:
#           Ωm h^2      h           Ωm          fp          n
# Ωm h^2 [[ 1.          0.25245567  0.62680432 -0.04181495 -0.07417154]
# h       [ 0.25245567  1.         -0.58950626  0.07712055 -0.05113214]
# Ωm      [ 0.62680432 -0.58950626  1.         -0.09230017 -0.02847623]
# fp      [-0.04181495  0.07712055 -0.09230017  1.          0.00428507]
# n       [-0.07417154 -0.05113214 -0.02847623  0.00428507  1.        ]]
# ---------------------------------


# Constant covariance diagonal scaling f(z) = f:
# cov[i, i] = cov_sys[i, i] + cov_diag_stat[i, i] * f^2
# cov[i, j] = cov_sys[i, j]
#
# h = 0.669 ± 0.043
# Ωm = 0.329 +0.038 -0.050
# Ωm h^2 = 0.146 ± 0.012
# f = 0.513 +0.089 -0.120 (prior ~ lognormal(μ=ln(0.5), σ=0.4))
# Log likelihood (MAP): -142.60
# Log evidence: -147.52
# χ2 (MAP): 36.61
# DOF: 33
# χ2/DOF: 1.11
# ---------------------------------


# Without error scaling (fixed f0 = 1, n = 0):
# H0 = 0.669 ± 0.053 km/s/Mpc
# Ωm = 0.335 +0.052 -0.080
# Ωm h^2 = 0.147 0.017 -0.019
# Log likelihood (MAP): -147.59
# Log evidence: -151.21
# χ2 (MAP): 15.57
# DOF: 34
# χ2/DOF: 0.46
# ---------------------------------


# Log likelihood ratio test f(z) vs no scaling:
# -2 * log(L0/L1) = -2 * log(L0) + 2 * log(L1)
# -2 * (-147.59) + 2 * (-137.27) = 20.64
# corresponding to a p-value of approximately 3.264e-05
# 4.0 sigma significance
