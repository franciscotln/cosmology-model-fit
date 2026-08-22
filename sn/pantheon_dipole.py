from numba import njit
import numpy as np
from scipy.linalg import cho_factor
from scipy.constants import c as c0
from interpolator import interp_hermite
from solve_triangular import solve_triangular
from y2022pantheonSHOES.data import get_data_with_position

legend, z_cmb, z_hel, mb_vals, ra, dec, cov_matrix = get_data_with_position()
cho = cho_factor(cov_matrix, lower=True)[0]

c = c0 / 1000  # Speed of light (km/s)

ra_rad = np.deg2rad(ra)
dec_rad = np.deg2rad(dec)
nx = np.cos(dec_rad) * np.cos(ra_rad)
ny = np.cos(dec_rad) * np.sin(ra_rad)
nz = np.sin(dec_rad)

# Target direction coordinates
ra_fixed_deg = 217  # deg
dec_fixed_deg = -29  # deg
# Convert direction to unit vector (dx, dy, dz)
ra_f_rad = np.deg2rad(ra_fixed_deg)
dec_f_rad = np.deg2rad(dec_fixed_deg)
d1 = np.cos(dec_f_rad) * np.cos(ra_f_rad)
d2 = np.cos(dec_f_rad) * np.sin(ra_f_rad)
d3 = np.sin(dec_f_rad)
cos_angle = nx * d1 + ny * d2 + nz * d3

z_grid = np.linspace(0, np.max(z_cmb) + 0.1, num=4000)
dz = np.diff(z_grid)

cubed = (1.0 + z_grid) ** 3


@njit
def H_z(params):
    H0, Om = params[1], params[2]
    return H0 * np.sqrt(Om * cubed + (1.0 - Om))


@njit
def DM_z(params, z):
    dh_grid = c / H_z(params)
    dh = (dh_grid[:-1] + dh_grid[1:]) / 2
    cum_dm = np.zeros(z_grid.size, dtype=np.float64)
    cum_dm[1:] = np.cumsum(dh * dz)
    return interp_hermite(z, z_grid, cum_dm, dh_grid)


@njit
def mu_theory(DM):
    return 25.0 + 5 * np.log10((1.0 + z_hel) * DM)


@njit
def mu_corr(params, DM_obs):
    DZ = 0.02  # sharpness of the transition
    Z_C = 0.10  # redshift where the velocity drops by 50%
    attenuation = 0.5 * (1.0 - np.tanh((z_cmb - Z_C) / DZ))
    v_km_s = 100 * params[3] * cos_angle * attenuation
    z_pec = v_km_s / c
    z_cosmo = -1.0 + (1.0 + z_cmb) / (1.0 + z_pec)
    return 5.0 * np.log10(DM_z(params, z_cosmo) / DM_obs)


@njit
def chi_squared(params):
    DM = DM_z(params, z_cmb)
    delta = mb_vals - params[0] - mu_corr(params, DM) - mu_theory(DM)
    return solve_triangular(cho, delta)


def log_likelihood(params):
    return -0.5 * chi_squared(params)


def main():
    from multiprocessing import Pool
    from getdist import plots, MCSamples
    import matplotlib.pyplot as plt
    from nautilus import Sampler, Prior
    from scipy.stats import norm
    from sn.plotting import plot_predictions, plot_residuals

    prior = Prior()
    prior.add_parameter("M", dist=(-20, -19))  # mag
    prior.add_parameter("H0", dist=norm(loc=70.39, scale=1.80))  # km/s/Mpc
    prior.add_parameter("om", dist=(0.1, 0.7))
    prior.add_parameter("v", dist=(-1, 4))  # v (x 100 km/s) dipole towards the Shapley supercluster (v > 0)

    with Pool(6) as pool:
        sampler = Sampler(
            prior, log_likelihood, n_live=6_000, pool=pool, seed=42, pass_dict=False,
        )
        sampler.run(verbose=True)

    samples, log_w, log_l = sampler.posterior()
    gd_samples = MCSamples(
        samples=samples,
        weights=np.exp(log_w),
        loglikes=log_l,
        names=prior.keys,
        labels=["M", "H_0", "Ω_m", "v_{100}"],
    )
    gd_samples.addDerived(100 * gd_samples["v"], name="v_km_s", label="v_{km/s}")
    gd_samples.updateBaseStatistics()

    for par in gd_samples.getParamNames().names:
        print(f"{par}: {gd_samples.mean(par):.5f} ± {gd_samples.std(par):.5f}")

    index_MAP = np.argmax(log_l)
    print(f"χ2 (MAP): {chi_squared(samples[index_MAP]):.2f}")
    print(f"Log evidence: {sampler.log_z:.1f}")
    print(f"DOF: {len(z_cmb) - len(prior.keys)}")

    best_fit = gd_samples.mean(prior.keys)
    DM_best = DM_z(best_fit, z_cmb)
    mu_pred = mu_theory(DM_best)
    mb_corrected = mb_vals - mu_corr(best_fit, DM_best)
    residuals = mb_corrected - gd_samples.mean('M') - mu_pred
    mu_std = np.sqrt(np.diag(cov_matrix))

    plots.get_subplot_plotter().triangle_plot(
        roots=gd_samples,
        params=["M", "H0", "om", "v_km_s"],
        title_limit=1,
        contour_colors=["C0"],
        filled=True,
    )
    plt.show()

    plot_predictions(
        legend=legend,
        x=z_cmb,
        y=mb_corrected - gd_samples.mean('M'),
        y_err=mu_std,
        y_model=mu_pred,
        label=f"$Ω_m$={gd_samples.mean('om'):.3f}",
        x_scale="log",
    )
    plot_residuals(z_values=z_cmb, residuals=residuals, y_err=mu_std, bins=50)


if __name__ == "__main__":
    main()


# ----------- Flat ΛCDM (v=0) -----
# M: -19.339 +- 0.055 mag
# H0: 70.4 +- 1.8 km/s/Mpc
# Ωm: 0.332 +- 0.018
# DOF: 1587
# χ2 (MAP): 1402.92
# Log Evidence: -708.7
# ---------------------------------


# ----------- Flat ΛCDM -----------
# M: -19.341 +- 0.055
# H0: 70.4 +- 1.8 km/s/Mpc
# Ωm: 0.332 +- 0.018
# v (dipole): 129 +- 40 km/s
# DOF: 1586
# χ2 (MAP): 1391.22 (delta χ2 = 11.7)
# Log Evidence: -704.5 (delta logZ = 4.2)
# ---------------------------------
