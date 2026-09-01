from numba import njit
import numpy as np
from scipy.constants import c as c0
from scipy.linalg import cho_factor
from interpolator import interp_hermite
from solve_triangular import solve_triangular
from y2025DESdovekie.data import get_data, effective_sample_size

legend, z_cmb, z_hel, mu_vals, covmat = get_data()

cho = cho_factor(covmat, lower=True)[0]

c = c0 / 1000  # Speed of light (km/s)

z_grid = np.linspace(0, np.max(z_cmb) + 0.1, num=3000)
dz = z_grid[1] - z_grid[0]


@njit
def Ode_z(z, w0):
    # Thawing quintessence with w(z) ranging from -1 to 1
    zp1 = 1.0 + z
    return (2 * zp1**3 / ((1.0 + w0) + (1.0 - w0) * zp1**3)) ** 2


@njit
def Hz(z, params):
    H0, Om = params[1], params[2]
    Ol = 1.0 - Om
    return H0 * np.sqrt(Om * (1.0 + z)**3 + Ol)


@njit
def DM_z(z, params):
    dh_grid = c / Hz(z_grid, params)
    n = z_grid.size
    cum_dm = np.zeros(n, dtype=np.float64)

    # Compute local derivatives d(dh)/dz using central differences
    d_dh = np.empty(n, dtype=np.float64)

    # Central difference for internal points
    d_dh[1:-1] = (dh_grid[2:] - dh_grid[:-2]) / (2 * dz)
    # Forward/Backward difference at boundaries
    d_dh[0] = (dh_grid[1] - dh_grid[0]) / dz
    d_dh[-1] = (dh_grid[-1] - dh_grid[-2]) / dz

    # Integrate with 4th-order cubic correction per interval
    dz_sq_over_12 = (dz ** 2) / 12
    acc = 0.0

    for i in range(n - 1):
        # Trapezoidal area + 1st-derivative endpoint correction
        trap = 0.5 * dz * (dh_grid[i] + dh_grid[i + 1])
        corr = dz_sq_over_12 * (d_dh[i] - d_dh[i + 1])
        acc += trap + corr
        cum_dm[i + 1] = acc

    return interp_hermite(z, z_grid, cum_dm, dh_grid)


@njit
def get_z_cosmo(params):
    # z_turn = 0.10563
    v_km_s = 100 * params[3] * np.where(z_cmb <= 0.11, 1, -1)
    z_pec = v_km_s / c
    return -1.0 + (1.0 + z_cmb) / (1.0 + z_pec)


def mu_corr(params, DM_obs):
    # For plotting purposes only
    z_cosmo = get_z_cosmo(params)
    return 5.0 * np.log10(DM_z(z_cosmo, params) / DM_obs)


@njit
def theory_mu(offset, DM):
    return offset + 25.0 + 5 * np.log10((1.0 + z_hel) * DM)


@njit
def chi_squared(params):
    z_cosmo = get_z_cosmo(params)
    DM = DM_z(z_cosmo, params)
    offset = params[0]

    delta = mu_vals - theory_mu(offset, DM)
    y = solve_triangular(cho, delta)
    return np.dot(y, y)


def log_likelihood(params):
    return -0.5 * chi_squared(params)


def main():
    from scipy.stats import norm
    from getdist import plots, MCSamples
    import matplotlib.pyplot as plt
    from nautilus import Sampler, Prior
    from multiprocessing import Pool
    from sn.plotting import plot_predictions, plot_residuals

    prior = Prior()
    prior.add_parameter("dM", dist=(-1.0, +1.0))
    # TRGB Freedman et al. 2025
    prior.add_parameter("H0", dist=norm(loc=70.39, scale=1.80))
    prior.add_parameter("om", dist=(0.0, 0.8))
    prior.add_parameter("v", dist=(-5.0, 5.0)) # x 100 km/s

    with Pool(6) as pool:
        sampler = Sampler(
            prior, log_likelihood, n_live=6_000, pool=pool, seed=42, pass_dict=False,
        )
        sampler.run(verbose=True)

    samples, log_w, log_l = sampler.posterior()

    labels = ["ΔM", "H_0", "Ω_m", "v_{100}"]
    gd_samples = MCSamples(
        samples=samples,
        weights=np.exp(log_w),
        loglikes=log_l,
        names=prior.keys,
        labels=labels,
    )
    gd_samples.addDerived(
        100 * gd_samples["v"], name="v_km_s", label="v_{km/s}"
    )

    for par in gd_samples.getParamNames().names:
        print(f"{par}: {gd_samples.mean(par):.5f} ± {gd_samples.std(par):.5f}")

    index_MAP = np.argmax(log_l)
    print(f"χ2 (MAP): {chi_squared(samples[index_MAP]):.2f}")
    print(f"Log evidence: {sampler.log_z:.1f}")
    print(f"DOF: {effective_sample_size - len(prior.keys)}")

    best_fit = gd_samples.mean(prior.keys)
    DM_best = DM_z(z_cmb, best_fit)
    mu_pred = theory_mu(offset=best_fit[0], DM=DM_best)
    mu_corrected = mu_vals - mu_corr(best_fit, DM_best)
    residuals = mu_corrected - mu_pred
    mu_std = np.sqrt(np.diag(covmat))

    plots.get_subplot_plotter().triangle_plot(
        roots=gd_samples,
        params=["dM", "H0", "om", "v_km_s"],
        title_limit=1,
        contour_colors=["C0"],
    )
    plt.show()

    plot_predictions(
        legend=legend,
        x=z_cmb,
        y=mu_corrected,
        y_err=mu_std,
        y_model=mu_pred,
        label=f"$Ω_m$={gd_samples.mean('om'):.3f}",
        x_scale="log",
    )
    plot_residuals(z_values=z_cmb, residuals=residuals, y_err=mu_std, bins=60)


if __name__ == "__main__":
    main()


# ********************************
# Data set: DES-SN5YR Dovekie 
# z range: 0.025 - 1.144
# Sample size: 1820 (effective: 1714 SNe)
# ********************************


# ----------- Flat ΛCDM -----------
# ΔM: 0.020 ± 0.057 mag
# H0: 70.4 ± 1.8 km/s/Mpc
# Ωm: 0.331 ± 0.015
# χ2 (MAP): 1631.42
# Log evidence: -823.9
# DOF: 1711
# ---------------------------------


# ----------- Flat ΛCDM -----------
# Velocity step correction in SNe observed redshifts
# turning point z <= 0.10563 inflow z > 0.10563 outflow
# z_cosmo = -1 + (1 + z) / (1 + v/c)

# v: -141 ± 67 km/s (prior ~ U[-5, 5] x 100 km/s)
# v / z_turn: -1325 ± 634 km/s

# ΔM: 0.004 ± 0.056 mag
# H0: 70.4 ± 1.8 km/s/Mpc
# Ωm: 0.309 ± 0.018
# χ2 (MAP): 1626.90 (2.13 sigma significance)
# Log evidence: -823.4 (Δ logZ = 0.5 in favour of velocity step correction)
# DOF: 1710
# ---------------------------------


# ----------- Flat wCDM -----------
# w0: -0.84 +0.15 -0.13 (prior ~ U[-1.5, 0])

# ΔM: 0.028 ± 0.057 mag
# H0: 70.4 ± 1.8 km/s/Mpc
# Ωm: 0.251 +0.089 -0.055
# χ2 (MAP): 1630.18 (1.11 sigma significance)
# Log evidence: -824.6 (ΛCDM preferred)
# DOF: 1710
# ---------------------------------


# ----------- Flat wzCDM ----------
# w(z) = -1 + 2 * (1 + w0) / (1 + w0 + (1 - w0) * (1 + z)^3)

# w0: -0.806 +0.098 -0.120 (prior ~ U[-1.0, 0])
# wa: d w(z)/dz at z=0 = -1.5 * (1 - w0^2)

# ΔM: 0.033 ± 0.056 mag
# H0: 70.4 ± 1.8 km/s/Mpc
# Ωm: 0.284 +0.037 -0.027
# χ2 (MAP): 1629.53 (1.37 sigma significance)
# Log evidence: -824.1 (ΛCDM preferred)
# DOF: 1710
# ---------------------------------


# ---------- Flat w0waCDM ---------
# w0: -0.46 +0.23 -0.35 (prior ~ U[-10, 5])
# wa: -7.8 +4.5 -3.2 (prior ~ U[-20, 10])

# ΔM: 0.057 ± 0.056 mag
# H0: 70.4 ± 1.7 km/s/Mpc
# Ωm: 0.465 +0.053 -0.026
# χ2 (MAP): 1625.31 (1.99 sigma significance)
# Log evidence: -826.1 (ΛCDM preferred)
# DOF: 1709
# ---------------------------------
