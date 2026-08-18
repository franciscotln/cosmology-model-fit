from numba import njit
import numpy as np
from scipy.constants import c as c0
from interpolator import interp_hermite
from y2026union3_1.data import get_data

legend, z_cmb, z_hel, mu_vals, cov_matrix = get_data()
inv_cov = np.linalg.inv(cov_matrix)

c = c0 / 1000  # Speed of light (km/s)

z_grid = np.linspace(0, np.max(z_cmb) + 0.1, num=4000)
dz = np.diff(z_grid)


@njit
def Ode(z, w0):
    # Thawing quintessence
    a3 = (1.0 + z) ** -3
    return 4 / ((1.0 + w0) * a3 + (1.0 - w0)) ** 2


@njit
def Hz(z, params):
    H0, Om = params[1], params[2]
    return H0 * np.sqrt(Om * (1.0 + z) ** 3 + (1.0 - Om))


@njit
def DM_z(z, params):
    dh_grid = c / Hz(z_grid, params)
    dh = (dh_grid[:-1] + dh_grid[1:]) / 2
    cum_dm = np.zeros(z_grid.size, dtype=np.float64)
    cum_dm[1:] = np.cumsum(dh * dz)
    return interp_hermite(z, z_grid, cum_dm, dh_grid)


@njit
def mu_corr(params, DM_obs):
    # Heaviside step at z = 0.2
    v_km_s = 100 * params[3] * np.where(z_cmb <= 0.2, 1, -1)
    z_pec = v_km_s / c
    z_cosmo = -1.0 + (1.0 + z_cmb) / (1.0 + z_pec)
    return 5 * np.log10(DM_z(z_cosmo, params) / DM_obs)


@njit
def mu_theory(params, DM):
    return params[0] + 25.0 + 5 * np.log10((1.0 + z_hel) * DM)


@njit
def chi_squared(params):
    DM = DM_z(z_cmb, params)
    delta = mu_vals - mu_corr(params, DM) - mu_theory(params, DM)
    return delta @ inv_cov @ delta


@njit
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
    prior.add_parameter("om", dist=(0.1, 0.7))
    prior.add_parameter("v", dist=(-9.0, 9.0)) # x 100 km/s

    with Pool(7) as pool:
        sampler = Sampler(
            prior, log_likelihood, n_live=7_000, pool=pool, seed=42, pass_dict=False,
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
        gd_samples["om"] * (gd_samples["H0"] / 100) ** 2, name="omh2", label="Ω_m h^2"
    )
    gd_samples.addDerived(
        100 * gd_samples["v"], name="v_km_s", label="v_{km/s}"
    )

    for par in gd_samples.getParamNames().names:
        print(f"{par}: {gd_samples.mean(par):.5f} ± {gd_samples.std(par):.5f}")

    index_MAP = np.argmax(log_l)
    print(f"χ2 (MAP): {chi_squared(samples[index_MAP]):.2f}")
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
        params=["dM", "om", "omh2", "v_km_s"],
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
    plot_residuals(z_values=z_cmb, residuals=residuals, y_err=mu_std, bins=7)


if __name__ == "__main__":
    main()

# *******************************
# Dataset: Union 3.1 (2026)
# z range: 0.050 - 2.262
# Sample size: 22
# *******************************


# ----------- Flat ΛCDM -----------
# ΔM: 0.039 ± 0.059
# H0 (km/s/Mpc): 70.4 ± 1.8
# Ωm: 0.336 ± 0.025
# Ωm h^2: 0.166 +0.014/-0.016
# χ2 (MAP): 28.76
# Log evidence: -22.0
# Degs of freedom: 19
# ---------------------------------


# ----------- Flat ΛCDM -----------
# Velocity step correction SNe observed redshifts
# (turning point z <= 0.2 inflow z > 0.2 outflow)
# z_cosmo = -1 + (1 + z) / (1 + v/c)

# v: -308 ± 120 km/s (prior ~ U[-9, 9] x 100 km/s)
# v / (z_cut=0.2): -1540 ± 600 km/s

# ΔM: 0.008 ± 0.059 mag
# H0: 70.4 ± 1.8 km/s/Mpc
# Ωm: 0.299 ± 0.027 (0.1 sigma agreement with ΛCDM from BAO)
# Ωm h^2: 0.148 +0.014/-0.016
# χ2 (MAP): 22.15 (2.57 sigma significance)
# Log evidence: -20.5 (Δ logZ = 1.5 in favour of step correction)
# Degs of freedom: 18
# ---------------------------------


# ----------- Flat wCDM -----------
# w0: -0.82 +0.18/-0.10 (prior ~ U[-1.5, 0.0])

# ΔM: 0.044 ± 0.059
# H0 (km/s/Mpc): 70.3 ± 1.8
# Ωm: 0.253 +0.083/-0.074
# Ωm h^2: 0.125 ± 0.037
# χ2 (MAP): 27.2 (1.26 sigma away from ΛCDM)
# Log evidence: -22.5 (Δ logZ = -0.5 in favour of ΛCDM)
# Degs of freedom: 18
# ---------------------------------


# ----------- Flat wzCDM -----------
# w(z) = -1 + 2 * (1 + w0) / (1 + w0 + (1 - w0) * (1 + z)**3)
# w0: -0.75 ± 0.13 (prior ~ U[-1.0, -1/3])

# ΔM: 0.054 ± 0.059
# H0 (km/s/Mpc): 70.4 ± 1.8
# Ωm: 0.278 +0.045/-0.038
# Ωm h^2: 0.138 +0.023/-0.020
# χ2 (MAP): 26.5 (1.52 sigma away from ΛCDM)
# Log evidence: -21.4 (Δ logZ = 0.6 in favour of wzCDM)
# Degs of freedom: 18
# ---------------------------------


# ----------- Flat w0waCDM -----------
# w0: -0.40 +0.27 -0.40 (prior ~ U[-2.0, 0.5])
# wa: -6.6 +4.8 -3.1 (prior ~ U[-16.0, 3.0])

# ΔM: 0.096 ± 0.063
# H0 (km/s/Mpc): 70.4 ± 1.7
# Ωm: 0.447 +0.080/-0.038
# Ωm h^2: 0.222 +0.040/-0.023
# χ2 (MAP): 24.43 (1.59 sigma away from ΛCDM)
# Log evidence: -22.5 (Δ logZ = -0.5 in favour of ΛCDM)
# Degrees of freedom: 17
# ---------------------------------
