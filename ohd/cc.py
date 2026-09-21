from numba import njit
import numpy as np
from solve_triangular import solve_triangular
from y2005cc.data_no_loubser import get_data, method

legend, z_values, H_values, diag_stat, cov_matrix_sys = get_data(split_sys=True)


@njit
def H_z(z, params):
    omh2, om = params[0], params[1]
    h2 = omh2 / om
    return 100 * np.sqrt(omh2 * (1.0 + z) ** 3 + (h2 - omh2))


@njit
def get_fz(params):
    z_pivot = 1.035
    fp, n = np.exp(params[2]), params[3]
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
    from nautilus import Sampler, Prior
    from getdist import plots, MCSamples
    import matplotlib.pyplot as plt
    from ohd.plot_predictions import plot_cc_predictions

    prior = Prior()
    prior.add_parameter("omh2", dist=(0.01, 0.25))
    prior.add_parameter("om", dist=(0.01, 1.0))
    prior.add_parameter("ln_fp", dist=(-2.0, 1.0))
    prior.add_parameter("n", dist=(-3.0, 7.0))

    with Pool(8) as pool:
        sampler = Sampler(
            prior, log_likelihood, n_live=10_000, pool=pool, seed=42, pass_dict=False
        )
        sampler.run(verbose=True)

    samples, log_w, log_l = sampler.posterior()
    MAP_PARAMS = samples[np.argmax(log_l)]
    weights = np.exp(log_w)
    labels = ["Ω_m h^2", "Ω_m", "ln(f_{pivot})", "n"]

    gd_samples = MCSamples(samples=samples, weights=weights, names=prior.keys, labels=labels)
    gd_samples.addDerived(100 * np.sqrt(gd_samples["omh2"] / gd_samples["om"]), name="H0", label="H_0")
    gd_samples.addDerived(np.exp(gd_samples["ln_fp"]), name="fp", label="f_{pivot}")
    gd_samples.updateBaseStatistics()

    plot_params = ["H0"] + prior.keys
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
# with z_pivot = 1.035, corr(ln(fp), n) = 0:
# cov[i, i] = cov_sys[i, i] + cov_diag_stat[i, i] * f(z_i)^2
# cov[i, j] = cov_sys[i, j]
#
# H0 = 67.1 ± 3.3 km/s/Mpc
# Ωm = 0.327 +0.036 -0.042
# Ωm h^2 = 0.147 ± 0.015
# ln(fp) = -0.46 ± 0.27
# n = 3.03 +0.87 -1.2
# fp = 0.65 +0.12 -0.20
# Log likelihood (MAP): -137.26
# Log evidence: -144.66
# χ2 (MAP): 34.96
# DOF: 32
# χ2/DOF: 1.09
# Correlation matrix:
#    H0.          Ωm h^2       Ωm        ln(fp)          n
#  [[ 1.          0.26287809 -0.59727902  0.09146906 -0.04243834]
#  [ 0.26287809  1.          0.61082046 -0.02179484 -0.06077785]
#  [-0.59727902  0.61082046  1.         -0.08862343 -0.02366998]
#  [ 0.09146906 -0.02179484 -0.08862343  1.          0.00487898]
#  [-0.04243834 -0.06077785 -0.02366998  0.00487898  1.        ]]
# ---------------------------------


# Constant covariance diagonal scaling f(z) = f:
# cov[i, i] = cov_sys[i, i] + cov_diag_stat[i, i] * f^2
# cov[i, j] = cov_sys[i, j]
#
# H0 = 66.6 ± 4.3 km/s/Mpc
# Ωm = 0.334 +0.037 -0.051
# Ωm h^2 = 0.147 ± 0.012
# ln(f) = -0.70 +0.27 -0.23
# f = 0.51 +0.11 -0.14
# Log likelihood (MAP): -142.60
# Log evidence: -148.54
# χ2 (MAP): 36.65
# DOF: 33
# χ2/DOF: 1.11
# ---------------------------------


# Without error scaling (fixed f0 = 1, n = 0):
# H0 = 66.4 ± 5.4 km/s/Mpc
# Ωm = 0.346 +0.053 -0.083
# Ωm h^2 = 0.149 ± 0.018
# Log likelihood (MAP): -147.59
# Log evidence: -151.28
# χ2 (MAP): 15.57
# DOF: 34
# χ2/DOF: 0.46
# ---------------------------------


# Log likelihood ratio test f(z) vs no scaling:
# -2 * log(L0/L1) = -2 * log(L0) + 2 * log(L1)
# -2 * (-147.59) + 2 * (-137.26) = 20.66
# corresponding to a p-value of approximately 3.264e-05
# 4.0 sigma significance
