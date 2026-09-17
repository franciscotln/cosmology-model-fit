from numba import njit
import numpy as np
from solve_triangular import solve_triangular
from y2005cc.data_no_loubser import get_data, method

legend, z_values, H_values, diag_stat, cov_matrix_sys = get_data(split_sys=True)
H_err = np.sqrt(np.diag(cov_matrix_sys) + diag_stat**2)


@njit
def H_z(z, params):
    omh2, om = params[0], params[1]
    h2 = omh2 / om
    return 100 * np.sqrt(omh2 * (1.0 + z) ** 3 + (h2 - omh2))


@njit
def get_fz(params):
    z_pivot = 0.728 # corr(ln(fp), n) = -8.53e-04
    f_piv, n = np.exp(params[2]), params[3]
    return f_piv * ((1.0 + z_values) / (1.0 + z_pivot))**n


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
    prior.add_parameter("ln_fp", dist=(-1.4, 0.4))
    prior.add_parameter("n", dist=(-4.0, 4.0))

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
        H_err=H_err,
        label=legend,
        method=method,
        err_scaling=1 / f_z,
    )


if __name__ == "__main__":
    main()


# Model: Flat ΛCDM
# ---------------------------------

# Redshift dependent covariance diagonal scaling
# f(z) = fp * [(1 + z) / (1 + z_pivot)]^n
# with z_pivot = 0.728, corr(ln(fp), n) = 0:
# cov[i, j] = cov_sys[i, j] + cov_diag_stat[i, j] * f(z_i) * f(z_j)
#
# H0 = 66.2 ± 3.9 km/s/Mpc
# Ωm h^2 = 0.148 ± 0.016
# Ωm = 0.341 +0.041 -0.053
# n = 1.48 ± 0.57
# ln(fp) = -0.47 ± 0.18
# fp = 0.636 +0.088 -0.13
# Log likelihood (MAP): -153.11
# Log evidence: -160.48 (diff = 2.89)
# χ2 (MAP): 36.97
# DOF: 35
# χ2/DOF: 1.06
# Correlation matrix:
#     H0          Ωm h^2      Ωm          ln(fp)      n
#  [[ 1.          1.8122e-01 -6.8829e-01  2.3849e-02  9.9548e-02]
#   [ 1.8122e-01  1.          5.7853e-01  1.4337e-02  1.0764e-02]
#   [-6.8829e-01  5.7853e-01  1.          1.2396e-03 -8.1496e-02]
#   [ 2.3849e-02  1.4337e-02  1.2396e-03  1.         -8.5361e-04]
#   [ 9.9548e-02  1.0764e-02 -8.1496e-02 -8.5361e-04  1.        ]]
# ---------------------------------


# Constant covariance diagonal scaling f(z) = f0:
# cov[i, j] = cov_sys[i, j] + cov_diag_stat[i, j] * f0^2
#
# H0 = 64.6 ± 4.6 km/s/Mpc
# Ωm h^2 = 0.147 +0.013 -0.014
# Ωm = 0.358 +0.044 -0.065
# ln(f0) = -0.38 +0.15 -0.17
# f0 = 0.693 +0.086 -0.13
# Log likelihood (MAP): -156.76
# Log evidence: -162.34
# χ2 (MAP): 35.82
# DOF: 36
# χ2/DOF: 1.00
# ---------------------------------


# Without error scaling (fixed f0 = 1, n = 0):
# H0 = 64.3 ± 5.3 km/s/Mpc
# Ωm h^2 = 0.151 ± 0.018
# Ωm = 0.372 +0.056 -0.088
# Log likelihood (MAP): -159.74
# Log evidence: -163.37
# χ2 (MAP): 19.93
# DOF: 37
# χ2/DOF: 0.54
# ---------------------------------


# Log likelihood ratio test f(z) vs no scaling:
# -2 * log(L0/L1) = -2 * log(L0) + 2 * log(L1)
# -2 * (-159.74) + 2 * (-153.11) = 13.26
# corresponding to a p-value of approximately 1.41e-03
# 3.19 sigma significance


# ---------------------------------
# The Loubser et al. (2025) 3 data points
# have been excluded from this analysis below
# ---------------------------------


# Model: Flat ΛCDM
# ---------------------------------

# Redshift dependent covariance diagonal scaling
# f(z) = fp * [(1 + z) / (1 + z_pivot)]^n
# with z_pivot = 0.728, corr(ln(fp), n) = 0:
# cov[i, j] = cov_sys[i, j] + cov_diag_stat[i, j] * f(z_i) * f(z_j)
#
# H0 = 67.8 ± 3.9 km/s/Mpc
# Ωm h^2 = 0.147 ± 0.016
# Ωm = 0.323 +0.039 -0.050
# n = 1.52 ± 0.56
# ln(fp) = -0.50 ± 0.18
# fp = 0.614 +0.083 -0.13
# Log likelihood (MAP): -140.08
# Log evidence: -147.56 (diff = 3.75)
# χ2 (MAP): 31.15
# DOF: 32
# χ2/DOF: 0.97
# Correlation matrix:
#     H0      Ωm h^2  Ωm      ln(fp)  n
#  [[ 1.      0.1567 -0.6793  0.0440  0.0559]
#   [ 0.1567  1.      0.6090 -0.0135 -0.0065]
#   [-0.6793  0.6090  1.     -0.0358 -0.0554]
#   [ 0.0440 -0.0135 -0.0358  1.     -0.0014]
#   [ 0.0559 -0.0065 -0.0554 -0.0014  1.    ]]
# ---------------------------------


# Constant covariance diagonal scaling f(z) = f0:
# cov[i, j] = cov_sys[i, j] + cov_diag_stat[i, j] * f0^2
#
# H0 = 66.8 ± 4.6 km/s/Mpc
# Ωm h^2 = 0.147 +0.012 -0.014
# Ωm = 0.334 +0.041 -0.059
# ln(f0) = -0.41 +0.15 -0.17
# f0 = 0.670 +0.082 -0.13
# Log likelihood (MAP): -144.07
# Log evidence: -149.78 (diff = 1.53)
# χ2 (MAP): 30.90
# DOF: 33
# χ2/DOF: 0.94
# ---------------------------------


# Without error scaling (fixed f0 = 1, n = 0):
# H0 = 66.4 ± 5.4 km/s/Mpc
# Ωm = 0.346 +0.054 -0.083
# Ωm h^2 = 0.149 ± 0.018
# Log likelihood (MAP): -147.61
# Log evidence: -151.31
# χ2 (MAP): 15.56
# DOF: 34
# χ2/DOF: 0.46
# ---------------------------------


# Log likelihood ratio test f(z) vs no scaling:
# -2 * log(L0/L1) = -2 * log(L0) + 2 * log(L1)
# -2 * (-147.61) + 2 * (-140.08) = 15.06
# corresponding to a p-value of approximately 5.37e-04
# 3.27 sigma significance
