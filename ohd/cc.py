from numba import njit
import numpy as np
from solve_triangular import solve_triangular
from y2005cc.data import get_data, method

legend, z_values, H_values, diag_stat, cov_matrix_sys = get_data(split_sys=True)

non_d = method != "D"


@njit
def H_z(z, params):
    h, om = params[0], params[1]
    return 100 * h * np.sqrt(om * (1.0 + z) ** 3 + (1.0 - om))


@njit
def get_fz(params):
    z_pivot = 0.62
    fp, n = np.exp(params[2]), params[3]
    fz = np.full_like(z_values, fp)
    fz[non_d] *= ((1.0 + z_values[non_d]) / (1.0 + z_pivot)) ** n
    return fz


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
    prior.add_parameter("h", dist=(0.5, 1.0))
    prior.add_parameter("om", dist=(0.01, 1.0))
    prior.add_parameter("ln_fp", dist=(-1.5, 0.5))
    prior.add_parameter("n", dist=(-2.0, 4.0))

    with Pool(8) as pool:
        sampler = Sampler(prior, log_likelihood, n_live=10_000, pool=pool, seed=42, pass_dict=False)
        sampler.run(verbose=True)

    samples, log_w, log_l = sampler.posterior()
    MAP_PARAMS = samples[np.argmax(log_l)]
    weights = np.exp(log_w)
    labels = ["h", "Ω_m", "ln(f_{pivot})", "n"]

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


# Model: Flat ΛCDM
# ---------------------------------

# Redshift dependent covariance diagonal scaling
# f(z) = f_pivot * [(1 + z) / (1 + z_pivot)]^n for method != D4000A
# f(z) = f_pivot for method == D4000A
# with z_pivot = 0.62, corr(ln(fp), n) = 0:
# cov[i, i] = cov_sys[i, i] + cov_diag_stat[i, i] * f(z_i)^2
# cov[i, j] = cov_sys[i, j]
#
# h = 0.702 +0.021 -0.018
# Ωm = 0.314 +0.034 -0.042
# Ωm h^2 = 0.154 ± 0.015
# ln(fp) = -0.55 +0.15 -0.18
# n = 1.51 ± 0.52
#
# Log likelihood (MAP): -151.80
# Log evidence: -160.06
# χ2 (MAP): 39.19
# DOF: 35
# χ2/DOF: 1.12
# Correlation matrix:
#   Ωm h^2       h           Ωm          ln(fp)      n
# [[ 1.         -0.29823828  0.89535712  0.00695859  0.13307379]
#  [-0.29823828  1.         -0.68719743 -0.16102157  0.11920848]
#  [ 0.89535712 -0.68719743  1.          0.08910272  0.04168552]
#  [ 0.00695859 -0.16102157  0.08910272  1.          0.00629284]
#  [ 0.13307379  0.11920848  0.04168552  0.00629284  1.        ]]
# ---------------------------------


# Constant covariance diagonal scaling f(z) = f:
# cov[i, i] = cov_sys[i, i] + cov_diag_stat[i, i] * f^2
# cov[i, j] = cov_sys[i, j]
#
# h = 0.686 ± 0.031
# Ωm = 0.322 +0.036 -0.050
# Ωm h^2 = 0.150 ± 0.012
# ln(fp) = -0.37 +0.14 -0.16
#
# Log likelihood (MAP): -156.39
# Log evidence: -162.76
# χ2 (MAP): 36.92
# DOF: 36
# χ2/DOF: 1.03
# ---------------------------------


# Without error scaling (fixed f0 = 1, n = 0):
# h = 0.673\pm 0.040
# Ωm = 0.340^{+0.051}_{-0.074}
# Ωm h^2 = 0.152\pm 0.017
# Log likelihood (MAP): -159.67
# Log evidence: -163.71
# χ2 (MAP): 20.04
# DOF: 37
# χ2/DOF: 0.54
# ---------------------------------


# Log likelihood ratio test f(z) vs no scaling:
# -2 * log(L0/L1) = -2 * log(L0) + 2 * log(L1)
# -2 * (-159.67) + 2 * (-151.80) = 15.74
# corresponding to a p-value of approximately 3.8e-04
