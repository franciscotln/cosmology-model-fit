from numba import njit
import numpy as np
from solve_triangular import solve_triangular
from y2005cc.data import get_data

legend, z_values, H_values, H_err, cov_matrix_sys = get_data(split_sys=True)


@njit
def H_z(z, params):
    H0, Om = params[0], params[1]
    return H0 * np.sqrt(Om * (1.0 + z) ** 3 + (1.0 - Om))


@njit
def log_likelihood_jit(params):
    f0 = np.exp(params[2])
    f_array = f0 * (1.0 + z_values)**params[3]
    if np.any(f_array <= 1e-4):
        return -np.inf

    cov_mat = np.diag(H_err**2 * f_array**2) + cov_matrix_sys
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
    prior.add_parameter("H0", dist=(30, 100))
    prior.add_parameter("Om", dist=(0.0, 1.0))
    prior.add_parameter("ln_f0", dist=(-2.5, 0.1))
    prior.add_parameter("n", dist=(-4.0, 4.0))

    with Pool(8) as pool:
        sampler = Sampler(
            prior, log_likelihood, n_live=10_000, pool=pool, seed=42, pass_dict=False
        )
        sampler.run(verbose=True)

    samples, log_w, log_l = sampler.posterior()
    weights = np.exp(log_w)
    labels = ["H_0", "Ω_m", "ln(f_0)", "n"]

    gd_samples = MCSamples(
        samples=samples,
        weights=weights,
        loglikes=log_l,
        names=prior.keys,
        labels=labels,
    )
    gd_samples.addDerived(
        gd_samples["Om"] * (gd_samples["H0"] / 100) ** 2, name="Omh2", label="Ω_m h^2",
    )
    gd_samples.addDerived(np.exp(gd_samples["ln_f0"]), name="f0", label="f_0")
    gd_samples.updateBaseStatistics()

    for name in gd_samples.getParamNames().names:
        print(gd_samples.getInlineLatex(name, limit=1))

    best_fit = samples[np.argmax(log_l)]

    f_array = np.exp(best_fit[2]) * (1.0 + z_values) ** best_fit[3]
    cov = np.diag(H_err**2 * f_array**2) + cov_matrix_sys
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

    plots.getSubplotPlotter().triangle_plot(
        gd_samples,
        params=prior.keys,
        filled=True,
        title_limit=1,
        contour_colors=["C0"],
        color=["C0"],
    )
    plt.show()

    plot_cc_predictions(
        H_z=lambda z: H_z(z, best_fit),
        z=z_values,
        H=H_values,
        H_err=H_err,
        label=f"{legend} $H_0$: {best_fit[0]:.1f} km/s/Mpc",
        err_scaling=1 / f_array,
    )


if __name__ == "__main__":
    main()


# Model: Flat ΛCDM
# ---------------------------------

# Redshift dependent covariance diagonal scaling f(z) = f0 * (1+z)^n:
# cov[i, j] = cov_sys[i, j] + cov_diag[i, j] * f(z_i) * f(z_j)
#
# H0 = 67.5 +- 3.8 km/s/Mpc
# Ωm = 0.313 +0.038 -0.048
# ln(f0) = -1.12 +0.24 -0.27 (prior ~U[-2.5, 0.1])
# n = 1.33 +- 0.48 (prior ~U[-4, 4])
# Ωm h^2 = 0.141 +- 0.015
# f0 = 0.338 +0.055 -0.010
# Log likelihood (MAP): -150.11
# Log evidence: -158.77 (diff: 4.55 strong evidence favouring the model with f0, n)
# χ2 (MAP): 37.74
# DOF: 35
# χ2/DOF: 1.08
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
# -2 * (-159.38) + 2 * (-150.11) = 18.54
# corresponding to a p-value of approximately 9.42x10^-5,
# indicating strong evidence in favor of the model with f0 and n.


# ------ without systematics ------

# scaling diagonal elements f(z) = f0 * (1+z)^n
# H0 = 68.6 +1.5 -1.4 km/s/Mpc
# Ωm = 0.306 +0.035 -0.042
# ln(f0) = -1.14 +0.24 -0.27 (prior ~U[-2.5, 0.1])
# n = 1.36 ± 0.48 (prior ~U[-4, 4])
# Ωm h^2 = 0.143 ± 0.014
# f0 = 0.332 +0.053 -0.10
# Log likelihood (MAP): -148.47
# Log evidence: -158.57 (diff: 4.72 strong evidence favouring the model with f0, n)
# χ2 (MAP): 39.06
# DOF: 35
# χ2/DOF: 1.12
# ---------------------------------

# No scaling (f0 = 1, n = 0)
# H0 = 67.6 ± 3.1 km/s/Mpc
# Ωm = 0.332 +0.050 -0.068
# Ωm h^2 = 0.150 ± 0.017
# Log likelihood (MAP): -158.45
# Log evidence: -163.29
# χ2 (MAP): 16.65
# DOF: 37
# χ2/DOF: 0.45
# ---------------------------------

# Log likelihood ratio test:
# -2 * log(L0/L1) = -2 * log(L0) + 2 * log(L1)
# -2 * (-158.45) + 2 * (-148.47) = 19.96
# corresponding to a p-value of approximately 4.57x10^-5,
# indicating strong evidence in favor of the model with f0 and n.