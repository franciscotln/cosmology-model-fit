from numba import njit
import numpy as np
from scipy.constants import c as c0
from interpolator import interp_hermite
from y2026union3_1.data import get_data

legend, z_cmb, z_hel, mu_vals, cov_matrix = get_data()
inv_cov = np.linalg.inv(cov_matrix)

c = c0 / 1000  # Speed of light (km/s)

z_grid = np.linspace(0, np.max(z_cmb) + 0.1, num=3000)
dz = np.diff(z_grid)


@njit
def Ode(z, w0):
    # Thawing quintessence
    a3 = (1.0 + z) ** -3
    return 4 / ((1.0 + w0) * a3 + (1.0 - w0)) ** 2


@njit
def Ez(z, params):
    Om = params[2]
    return np.sqrt(Om * (1.0 + z) ** 3 + (1.0 - Om))


@njit
def DM_z(z, params):
    Hz = params[1] * Ez(z_grid, params)
    dh_grid = c / Hz
    dh = (dh_grid[:-1] + dh_grid[1:]) / 2
    cum_dm = np.zeros(z_grid.size, dtype=np.float64)
    cum_dm[1:] = np.cumsum(dh * dz)
    return interp_hermite(z, z_grid, cum_dm, dh_grid)


@njit
def mu_theory(params):
    zc = params[3]
    Mz = params[0] + (5 / np.log(10)) * zc**2 / (zc + z_cmb)
    return Mz + 25.0 + 5 * np.log10((1.0 + z_hel) * DM_z(z_cmb, params))


@njit
def chi_squared(params):
    delta = mu_vals - mu_theory(params)
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
    prior.add_parameter(
        "H0", dist=norm(loc=70.39, scale=1.80)
    )  # TRGB Freedman et al. 2025
    prior.add_parameter("om", dist=(0.1, 0.7))
    prior.add_parameter("z_c", dist=(0.0, 0.25))

    with Pool(6) as pool:
        sampler = Sampler(
            prior, log_likelihood, n_live=6_000, pool=pool, seed=42, pass_dict=False
        )
        sampler.run(verbose=True)

    samples, log_w, log_l = sampler.posterior()

    labels = ["ΔM_0", "H_0", "Ω_m", "z_c"]
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

    plots.get_subplot_plotter().triangle_plot(
        gd_samples, title_limit=1, contour_colors=["C0"]
    )
    plt.show()

    best_fit = gd_samples.mean(prior.keys)
    degs_of_freedom = len(z_cmb) - len(best_fit)

    for par in gd_samples.getParamNames().names:
        print(f"{par}: {gd_samples.mean(par):.5f} ± {gd_samples.std(par):.5f}")

    index_MAP = np.argmax(log_l)
    print(f"χ2 (MAP): {chi_squared(samples[index_MAP]):.2f}")
    print(f"Log evidence: {sampler.log_z:.1f}")
    print(f"Degrees of freedom: {degs_of_freedom}")

    predicted_distances = mu_theory(best_fit)
    residuals = mu_vals - predicted_distances
    sigma_mu = np.sqrt(np.diag(cov_matrix))

    plot_predictions(
        legend=legend,
        x=z_cmb,
        y=mu_vals,
        y_err=sigma_mu,
        y_model=mu_theory(best_fit),
        label=f"$Ω_m$={gd_samples.mean('om'):.3f}",
        x_scale="log",
    )
    plot_residuals(z_values=z_cmb, residuals=residuals, y_err=sigma_mu, bins=7)


if __name__ == "__main__":
    main()

"""
*******************************
Dataset: Union 3 Bins
z range: 0.050 - 2.262
Sample size: 22
*******************************

Flat ΛCDM: w(z) = -1

ΔM: 0.037 ± 0.059
H0 (km/s/Mpc): 70.3 ± 1.8
Ωm: 0.336 ± 0.025
Ωm h^2: 0.166 +0.014/-0.016
χ2 (MAP): 28.8
Log evidence: -22.0
Degs of freedom: 19

===============================

Flat ΛCDM: w(z) = -1
Outflow mag correction of SNe M(z) = M_inf + (5 / ln(10)) * z_c^2 / (z_c + z)

ΔM_inf: -0.030 +0.077 -0.068 mag
z_c: 0.064 ± 0.030 (prior ~ U(0, 0.25))
H0: 70.4 ± 1.8 km/s/Mpc
Ωm: 0.283 +0.038/-0.038 (agreement with ΛCDM from BAO)
Ωm h^2: 0.140 +0.020/-0.020
χ2 (MAP): 25.5 (1.82 sigma away from constant M)
Log evidence: -21.7
Degs of freedom: 18

===============================

Flat wCDM: w(z) = w0

ΔM: 0.044 ± 0.059
H0 (km/s/Mpc): 70.3 ± 1.8
Ωm: 0.253 +0.083/-0.074
w0: -0.82 +0.18/-0.10 (prior ~ U(-1.5, 0.0))
Ωm h^2: 0.125 ± 0.037
χ2 (MAP): 27.2 (1.26 sigma away from ΛCDM)
Log evidence: -22.5
Degs of freedom: 18

===============================

Flat wzCDM: w(z) = -1 + 2 * (1 + w0) / (1 + w0 + (1 - w0) * (1 + z)**3)

ΔM: 0.054 ± 0.059
H0 (km/s/Mpc): 70.4 ± 1.8
Ωm: 0.278 +0.045/-0.038
w0: -0.75 ± 0.13 (prior ~ U(-1.0, -1/3))
Ωm h^2: 0.138 +0.023/-0.020
χ2 (MAP): 26.5 (1.52 sigma away from ΛCDM)
Log evidence: -21.4
Degs of freedom: 18

===============================

Flat w0waCDM: w(z) = w0 + wa * z / (1 + z)

ΔM: 0.096 ± 0.063
H0 (km/s/Mpc): 70.4 ± 1.7
Ωm: 0.447 +0.080/-0.038
w0: -0.40 +0.27 -0.40 (prior ~ U(-2.0, 0.5))
wa: -6.6 +4.8 -3.1 (prior ~ U(-16.0, 3.0))
Ωm h^2: 0.222 +0.040/-0.023
χ2 (MAP): 24.43 (1.59 sigma away from ΛCDM)
Log evidence: -22.5
Degrees of freedom: 17
"""
