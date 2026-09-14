from numba import njit
import numpy as np
from solve_triangular import solve_triangular
from y2005cc.data import get_data

legend, z_values, H_values, H_err, cov_matrix_sys = get_data(split_sys=True)


@njit
def H_z(z, params):
    omh2, om = params[0], params[1]
    h2 = omh2 / om
    return 100 * np.sqrt(omh2 * (1.0 + z) ** 3 + (h2 - omh2))


z_pivot = 0.6145  # corr(ln(fp), n) = 3.8e-03


@njit
def log_likelihood_jit(params):
    f_pivot, n = np.exp(params[2]), params[3]
    f_z = f_pivot * ((1.0 + z_values) / (1.0 + z_pivot))**n
    cov_mat = np.diag(H_err**2 * f_z**2) + cov_matrix_sys
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
    prior.add_parameter("ln_fp", dist=(np.log(0.3), np.log(1.1)))
    prior.add_parameter("n", dist=(-4.0, 4.0))

    with Pool(8) as pool:
        sampler = Sampler(
            prior, log_likelihood, n_live=10_000, pool=pool, seed=42, pass_dict=False
        )
        sampler.run(verbose=True)

    samples, log_w, log_l = sampler.posterior()
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

    best_fit = samples[np.argmax(log_l)]

    f_z = np.exp(best_fit[2]) * ((1.0 + z_values) / (1.0 + z_pivot))**best_fit[3]
    cov = np.diag(H_err**2 * f_z**2) + cov_matrix_sys
    L = np.linalg.cholesky(cov)
    y = solve_triangular(L, H_values - H_z(z_values, best_fit))

    chi2 = np.dot(y, y)
    DOF = z_values.size - len(best_fit)
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
        H_z=lambda z: H_z(z, best_fit),
        z=z_values,
        H=H_values,
        H_err=H_err,
        label=legend,
        err_scaling=1 / f_z,
    )


if __name__ == "__main__":
    main()


# Model: Flat ΛCDM
# ---------------------------------

# Redshift dependent covariance diagonal scaling f(z) = f0 * [(1+z) / (1+z_piv)]^n
# with z_piv = 0.6142, corr(ln(fp), n) = 0:
# cov[i, j] = cov_sys[i, j] + cov_diag[i, j] * f(z_i) * f(z_j)
#
# H0 = 67.1 +- 3.8 km/s/Mpc
# Ωm = 0.315 +0.038 -0.049
# ln(fp) = -0.49 +0.11 -0.13 (prior ~U[ln(0.3), ln(1.1)])
# n = 1.37 +- 0.48 (prior ~U[-4, 4])
# Ωm h^2 = 0.141 +- 0.015
# fp = 0.619 +0.058 -0.085
# Log likelihood (MAP): -149.86
# Log evidence: -157.63 (diff: 5.69 strong evidence favouring the model with fp, n)
# χ2 (MAP): 37.61
# DOF: 35
# χ2/DOF: 1.07
# ---------------------------------

# Constant covariance diagonal scaling f(z) = f0:
# cov[i, j] = cov_sys[i, j] + cov_diag[i, j] * f0^2
#
# H0 = 65.6 ± 4.6 km/s/Mpc
# Ωm = 0.345 +0.042 -0.060
# ln(f0) = -0.39 +0.11 -0.13
# Ωm h^2 = 0.147 ± 0.013
# f0 = 0.684 +0.063 -0.093
# Log likelihood (MAP): -154.21
# Log evidence: -159.91 (diff: 3.41 moderate evidence favouring the model with f0)
# χ2 (MAP): 38.00
# DOF: 36
# χ2/DOF: 1.06
# ---------------------------------

# Without error scaling (fixed f0 = 1, n = 0):
# H0: 65.8 ± 5.4 km/s/Mpc
# Om: 0.353 +0.054 -0.085
# Ωm h^2: 0.150 ± 0.018
# Log likelihood (MAP): -159.36
# Log evidence: -163.03
# χ2 (MAP): 16.57
# DOF: 37
# χ2/DOF: 0.45
# ---------------------------------

# Log likelihood ratio test f(z) vs no scaling:
# -2 * log(L0/L1) = -2 * log(L0) + 2 * log(L1)
# -2 * (-159.36) + 2 * (-149.86) = 19.00
# corresponding to a p-value of approximately 7.485x10^-5,
# indicating strong evidence in favor of the model with f0 and n.


# ------ without systematics ------

# scaling diagonal elements f(z) = f0 * [(1+z) / (1+z_piv)]^n
# cov[i, j] = cov_diag[i, j] * f(z_i) * f(z_j)

# H0 = 68.7 +1.5 -1.4 km/s/Mpc
# Ωm = 0.306 +0.035 -0.041
# ln(fp) = -0.49 +0.11 -0.13 (prior ~U[ln(0.3), ln(1.2)])
# n = 1.40 ± 0.49 (prior ~U[-4, 4])
# Ωm h^2 = 0.144 ± 0.014
# fp = 0.617 +0.058 -0.084
# Log likelihood (MAP): -148.25
# Log evidence: -157.47 (diff: 5.54 strong evidence favouring the model with f0, n)
# χ2 (MAP): 39.31
# DOF: 35
# χ2/DOF: 1.12
# ---------------------------------

# Constant covariance diagonal scaling f(z) = f0:
# cov[i, j] = cov_diag[i, j] * f0^2

# H0 = 67.8 ± 2.1 km/s/Mpc
# Ωm = 0.327 +0.035 -0.045
# ln(f0) = -0.39 +0.11 -0.13
# Ωm h^2 = 0.150 ± 0.012
# f0 = 0.685 +0.063 -0.092
# Log likelihood (MAP): -152.98
# Log evidence: -159.90 (diff: 3.11 moderate evidence favouring the model with f0)
# χ2 (MAP): 38.99
# DOF: 36
# χ2/DOF: 1.08
# ---------------------------------

# No scaling (f0 = 1, n = 0)
# H0 = 67.4 ± 3.1 km/s/Mpc
# Ωm = 0.339 +0.050 -0.070
# Ωm h^2 = 0.152 ± 0.017
# Log likelihood (MAP): -158.43
# Log evidence: -163.01
# χ2 (MAP): 16.62
# DOF: 37
# χ2/DOF: 0.45
# ---------------------------------


# Log likelihood ratio test f(z) vs no scaling:
# -2 * log(L0/L1) = -2 * log(L0) + 2 * log(L1)
# -2 * (-158.43) + 2 * (-148.25) = 20.36
# corresponding to a p-value of approximately 3.792x10^-5,
# indicating strong evidence in favor of the model with f0 and n.