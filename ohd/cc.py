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
    z_pivot = 0.98
    f_piv, n = np.exp(params[2]), params[3]

    Hz = H_z(z_values, params)
    Hz_pivot = H_z(z_pivot, params)

    shape = ((1.0 + z_values) * Hz) / ((1.0 + z_pivot) * Hz_pivot)
    return f_piv * shape**n


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
    prior.add_parameter("Omh2", dist=(0.01, 0.25))
    prior.add_parameter("Om", dist=(0.01, 1.0))
    prior.add_parameter("ln_fp", dist=(-1.7, 0.7))
    prior.add_parameter("n", dist=(-2.5, 2.5))

    with Pool(8) as pool:
        sampler = Sampler(
            prior, log_likelihood, n_live=10_000, pool=pool, seed=42, pass_dict=False
        )
        sampler.run(verbose=True)

    samples, log_w, log_l = sampler.posterior()
    MAP_PARAMS = samples[np.argmax(log_l)]
    weights = np.exp(log_w)
    labels = ["Ω_m h^2", "Ω_m", "ln(f_{pivot})", "n"]

    gd_samples = MCSamples(
        samples=samples,
        weights=weights,
        loglikes=-log_l,
        names=prior.keys,
        labels=labels,
    )
    gd_samples.addDerived(
        100 * np.sqrt(gd_samples["Omh2"] / gd_samples["Om"]), name="H0", label="H_0",
    )
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
# f(z) = fp * [(1 + z) / (1 + z_pivot)]^n
# with z_pivot = 0.98, corr(ln(fp), n) = 0:
# cov[i, j] = cov_sys[i, j] + cov_diag_stat[i, j] * f(z_i) * f(z_j)
#
# H0 = 67.5 ± 3.7 km/s/Mpc
# Ωm h^2 = 0.147 ± 0.015
# Ωm = 0.324 +0.038 -0.047
# ln(fp) = -0.40 ± 0.23
# n = 0.98 +0.31 -0.43
# fp = 0.69 +0.11 -0.17
# Log likelihood (MAP): -139.78
# Log evidence: -147.35 (diff = 3.94)
# χ2 (MAP): 32.31
# DOF: 32
# χ2/DOF: 1.01
# Correlation matrix:
#     H0      Ωm h^2  Ωm      ln(fp)  n
#  [[ 1.      0.2045 -0.6487  0.0788  0.0014]
#   [ 0.2045  1.      0.6041 -0.0244 -0.0882]
#   [-0.6487  0.6041  1.     -0.0751 -0.0791]
#   [ 0.0788 -0.0244 -0.0751  1.     -0.0030]
#   [ 0.0014 -0.0882 -0.0791 -0.0030  1.    ]]
# ---------------------------------


# Constant covariance diagonal scaling f(z) = f0:
# cov[i, j] = cov_sys[i, j] + cov_diag_stat[i, j] * f0^2
#
# H0 = 66.8 ± 4.6 km/s/Mpc
# Ωm h^2 = 0.147 ± 0.013
# Ωm = 0.334 +0.040 -0.058
# ln(f0) = -0.45 ± 0.18
# f0 = 0.645 +0.090 -0.130
# Log likelihood (MAP): -143.90
# Log evidence: -149.83 (diff = 1.46)
# χ2 (MAP): 31.46
# DOF: 33
# χ2/DOF: 0.95
# ---------------------------------


# Without error scaling (fixed f0 = 1, n = 0):
# H0 = 66.4 ± 5.4 km/s/Mpc
# Ωm = 0.346 +0.053 -0.083
# Ωm h^2 = 0.149 ± 0.018
# Log likelihood (MAP): -147.59
# Log evidence: -151.29
# χ2 (MAP): 15.57
# DOF: 34
# χ2/DOF: 0.46
# ---------------------------------


# Log likelihood ratio test f(z) vs no scaling:
# -2 * log(L0/L1) = -2 * log(L0) + 2 * log(L1)
# -2 * (-147.59) + 2 * (-139.78) = 15.62
# corresponding to a p-value of approximately 4.06e-04
# 3.27 sigma significance


# ---------------------------------
# Including the Loubser et al. (2025) 3 data points
# Model: Flat ΛCDM
# ---------------------------------

# Redshift dependent covariance diagonal scaling
# f(z) = fp * [(1 + z) / (1 + z_pivot)]^n
# with z_pivot = 1.019, corr(ln(fp), n) = 0:
# cov[i, j] = cov_sys[i, j] + cov_diag_stat[i, j] * f(z_i) * f(z_j)
#
# H0 = 65.8 ± 3.6 km/s/Mpc
# Ωm h^2 = 0.147 ± 0.015
# Ωm = 0.341^{+0.039}_{-0.049}
# n = 1.00^{+0.33}_{-0.47}
# ln(fp) = -0.36 ± 0.24
# fp = 0.72^{+0.13}_{-0.20}
# Log likelihood (MAP): -152.86
# Log evidence: -160.23
# χ2 (MAP): 38.28
# DOF: 35
# χ2/DOF: 1.09
# Correlation matrix:
#  [[ 1.      0.2347 -0.6464  0.0721  0.0388]
#   [ 0.2347  1.      0.5818  0.0063 -0.0889]
#   [-0.6464  0.5818  1.     -0.0474 -0.1126]
#   [ 0.0721  0.0063 -0.0474  1.     -0.0026]
#   [ 0.0388 -0.0889 -0.1126 -0.0026  1.    ]]
# ---------------------------------


# Constant covariance diagonal scaling f(z) = f0:
# cov[i, j] = cov_sys[i, j] + cov_diag_stat[i, j] * f0^2
#
# H0 = 64.6 ± 4.5 km/s/Mpc
# Ωm h^2 = 0.147 +0.012 -0.014
# Ωm = 0.357 +0.043 -0.063
# ln(f0) = -0.44 ± 0.19
# f0 = 0.659 +0.097 -0.140
# Log likelihood (MAP): -156.59
# Log evidence: -162.36
# χ2 (MAP): 36.61
# DOF: 36
# χ2/DOF: 1.02
# ---------------------------------


# Without error scaling (fixed f0 = 1, n = 0):
# H0 = 64.3 ± 5.3 km/s/Mpc
# Ωm h^2 = 0.151 ± 0.018
# Ωm = 0.372^{+0.056}_{-0.088}
# Log likelihood (MAP): -159.71
# Log evidence: -163.34
# χ2 (MAP): 19.95
# DOF: 37
# χ2/DOF: 0.54
# ---------------------------------


# Log likelihood ratio test f(z) vs no scaling:
# -2 * log(L0/L1) = -2 * log(L0) + 2 * log(L1)
# -2 * (-159.71) + 2 * (-152.86) = 13.70
# corresponding to a p-value of approximately 1.06e-03
# 3.19 sigma significance
