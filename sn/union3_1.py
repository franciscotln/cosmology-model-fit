from numba import njit
import numpy as np
from scipy.constants import c as c0
from interpolator import interp_hermite
from solve_triangular import solve_triangular
from y2026union3_1.data import get_data

legend, z_cmb, z_hel, mu_vals, cov_matrix = get_data()
L_cho = np.linalg.cholesky(cov_matrix)
logdet = 2 * np.sum(np.log(np.diag(L_cho)))
N = z_cmb.size

c = c0 / 1000  # Speed of light (km/s)
H0 = 70.0  # Hubble constant (km/s/Mpc)

z_grid = np.linspace(0, np.max(z_cmb) + 0.1, num=2000)
dz = z_grid[1] - z_grid[0]


@njit
def Hz(z, params):
    Om = params[1]
    return H0 * np.sqrt(Om * (1.0 + z) ** 3 + (1.0 - Om))


@njit
def DM_z(z, params):
    dh_grid = c / Hz(z_grid, params)
    dh = (dh_grid[:-1] + dh_grid[1:]) / 2
    cum_dm = np.zeros(z_grid.size, dtype=np.float64)
    cum_dm[1:] = np.cumsum(dh * dz)
    return interp_hermite(z, z_grid, cum_dm, dh_grid)


@njit
def get_z_cosmo(params):
    # Heaviside step at z = 0.2
    offset = 1e-3 * params[2] * np.where(z_cmb <= 0.2, 1, -1)
    return z_cmb + offset


def mu_corr(params, DM_obs):
    z_cosmo = get_z_cosmo(params)
    return 5 * np.log10(DM_z(z_cosmo, params) / DM_obs)


@njit
def mu_theory(params, DM):
    return params[0] + 25.0 + 5 * np.log10((1.0 + z_hel) * DM)


@njit
def chi_squared(params):
    z_cosmo = get_z_cosmo(params)
    DM = DM_z(z_cosmo, params)
    delta = mu_vals - mu_theory(params, DM)
    y = solve_triangular(L_cho, delta)
    return np.dot(y, y)


@njit
def log_likelihood(params):
    return -0.5 * (chi_squared(params) + logdet + N * np.log(2 * np.pi))


def main():
    from getdist import plots, MCSamples
    import matplotlib.pyplot as plt
    from nautilus import Sampler, Prior
    from multiprocessing import Pool
    from sn.plotting import plot_predictions, plot_residuals

    prior = Prior()
    prior.add_parameter("dM", dist=(-1, +1))  # mag
    prior.add_parameter("om", dist=(0.1, 0.7))
    prior.add_parameter("dz_1000", dist=(-3.5, 3.5)) # 1000 x Δz

    with Pool(7) as pool:
        sampler = Sampler(
            prior, log_likelihood, n_live=7_000, pool=pool, seed=42, pass_dict=False,
        )
        sampler.run(verbose=True)

    samples, log_w, log_l = sampler.posterior()

    labels = ["ΔM", "Ω_m", "1000 Δz"]
    gd_samples = MCSamples(
        samples=samples,
        weights=np.exp(log_w),
        loglikes=-log_l,
        names=prior.keys,
        labels=labels,
    )
    gd_samples.updateBaseStatistics()

    for name in gd_samples.getParamNames().names:
        print(gd_samples.getInlineLatex(name, limit=1))

    index_MAP = np.argmax(log_l)
    params_MAP = samples[index_MAP]
    print(f"χ2 (MAP): {chi_squared(params_MAP):.2f}")
    print(f"log likelihood (MAP): {log_likelihood(params_MAP):.2f}")
    print(f"Log evidence: {sampler.log_z:.1f}")
    print(f"DOF: {len(z_cmb) - len(prior.keys)}")

    best_fit = gd_samples.mean(prior.keys)
    DM_best = DM_z(z_cmb, best_fit)
    mu_pred = mu_theory(best_fit, DM_best)
    mu_corrected = mu_vals - mu_corr(best_fit, DM_best)
    residuals = mu_corrected - mu_pred
    mu_std = np.sqrt(np.diag(cov_matrix))

    plots.get_subplot_plotter().triangle_plot(
        roots=gd_samples,
        params=["dM", "om", "dz_1000"],
        title_limit=1,
        contour_colors=["C0"],
        filled=True,
    )
    plt.show()

    plot_predictions(
        legend=legend,
        x=z_cmb,
        y=mu_corrected,
        y_err=mu_std,
        y_model=mu_pred,
        label=f"$Ω_m$={params_MAP[1]:.3f}",
        x_scale="log",
    )
    plot_residuals(z_values=z_cmb, residuals=residuals, y_err=mu_std, bins=7)


if __name__ == "__main__":
    main()

# *******************************
# Dataset: Union 3.1 (2026)
# z range: 0.050 - 2.262
# Sample size: 22
# *******************************


# ----------- Flat ΛCDM -----------
# ΔM: 0.028 ± 0.020 mag
# Ωm: 0.336 ± 0.024
# χ2 (MAP): 28.18
# log likelihood (MAP): 43.62
# Log evidence: 36.0
# DOF: 20
# ---------------------------------


# ----------- Flat ΛCDM -----------
# Z offset step correction in SNe observed redshifts
# turning point z <= 0.2 positive z > 0.2 negative
# z_cosmo = z_cmb ± Δz

# 1000 Δz = 1.09 ± 0.45
# Ωm: 0.303 ± 0.026
# ΔM: -0.001 ± 0.023 mag
# χ2 (MAP): 22.20 (2.56 sigma significance)
# log likelihood (MAP): 46.61
# Log evidence: 37.1 (Δ logZ = 1.1 in favour of step correction)
# DOF: 19
# ---------------------------------


# ----------- Flat wCDM -----------
# w0: -0.83 +0.19 -0.11 (prior ~ U[-1.5, 0])
# Ωm: 0.260 +0.084 -0.070
# ΔM: 0.035 ± 0.020 mag
# χ2 (MAP): 26.86 (1.26 sigma away from ΛCDM)
# log likelihood (MAP): 44.28
# Log evidence: 35.5 (Δ logZ = -0.5 in favour of ΛCDM)
# DOF: 19
# ---------------------------------


# ----------- Flat w0waCDM -----------
# w0 + wa < 0 enforced in the likelihood
#
# w0: -0.44 +0.26 0.40 (prior ~ U[-3, 1])
# wa: -6.3 +4.8 -2.9 (prior ~ U[-16, 3])
# Ωm: 0.445 +0.082 -0.038
# ΔM: 0.083 ± 0.033 mag
# χ2 (MAP): 24.40 (1.59 sigma away from ΛCDM)
# log likelihood (MAP): 45.51
# Log evidence: 34.8 (Δ logZ = -1.2 in favour of ΛCDM)
# DOF: 18
# ---------------------------------
