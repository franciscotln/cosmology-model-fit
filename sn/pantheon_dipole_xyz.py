from numba import njit
import numpy as np
from scipy.linalg import cho_factor
from scipy.constants import c as c0
from interpolator import interp_hermite
from solve_triangular import solve_triangular
from y2022pantheonSHOES.data import get_data_with_position

legend, z_cmb, z_hel, mb_vals, ra, dec, survey_id, cov_matrix = get_data_with_position()
cho = cho_factor(cov_matrix, lower=True)[0]

c = c0 / 1000  # Speed of light (km/s)

ra_rad = np.deg2rad(ra)
dec_rad = np.deg2rad(dec)
nx = np.cos(dec_rad) * np.cos(ra_rad)
ny = np.cos(dec_rad) * np.sin(ra_rad)
nz = np.sin(dec_rad)

target_ids = [1, 5, 15, 50, 51, 56, 63, 150]
survey_mask = np.isin(survey_id, target_ids).astype(int)

z_grid = np.linspace(0, np.max(z_cmb) + 0.1, num=4000)
dz = z_grid[1] - z_grid[0]


@njit
def H_z(z, params):
    H0, Om = params[1], params[2]
    return H0 * np.sqrt(Om * (1.0 + z) ** 3 + (1.0 - Om))


@njit
def DM_z(params, z):
    dh_grid = c / H_z(z_grid, params)

    n = z_grid.size
    cum_dm = np.zeros(n, dtype=np.float64)

    acc = 0.0
    for i in range(n - 1):
        dh_avg = 0.5 * (dh_grid[i] + dh_grid[i + 1])
        acc += dz * dh_avg
        cum_dm[i + 1] = acc

    return interp_hermite(z, x=z_grid, y=cum_dm, y_prime=dh_grid)


@njit
def mu_theory(DM):
    return 25.0 + 5 * np.log10((1.0 + z_hel) * DM)


@njit
def get_z_cosmo(params):
    DZ = 0.02  # sharpness of the transition
    Z_C = 0.10  # redshift where the velocity drops by 50%
    attenuation = 0.5 * (1.0 - np.tanh((z_cmb - Z_C) / DZ))
    v_los = nx * params[3] + ny * params[4] + nz * params[5]
    v_km_s = 100 * v_los * attenuation * survey_mask
    z_pec = v_km_s / c
    return -1.0 + (1.0 + z_cmb) / (1.0 + z_pec)


def mu_corr(params, DM_obs):
    z_cosmo = get_z_cosmo(params)
    return 5.0 * np.log10(DM_z(params, z_cosmo) / DM_obs)


@njit
def chi_squared(params):
    z_cosmo = get_z_cosmo(params)
    if np.any(z_cosmo <= 0.0):
        return 1e8
    M = params[0]
    delta = mb_vals - M - mu_theory(DM_z(params, z_cosmo))
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
    # Bulk flow velocity components (x 100 km/s) in equatorial cartesian coordinates
    prior.add_parameter("vx", dist=(-4, 1))
    prior.add_parameter("vy", dist=(-4, 1))
    prior.add_parameter("vz", dist=(-4, 2))

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
        labels=["M", "H_0", "Ω_m", "v_x", "v_y", "v_z"],
    )
    vx, vy, vz = gd_samples["vx"], gd_samples["vy"], gd_samples["vz"]
    v_amp = np.sqrt(vx**2 + vy**2 + vz**2)
    gd_samples.addDerived(100 * v_amp, name="v_km_s", label="v_{km/s}")
    gd_samples.addDerived(np.rad2deg(np.arctan2(vy, vx)) % 360, name="ra_dip", label="RA")
    gd_samples.addDerived(np.rad2deg(np.arcsin(vz / v_amp)), name="dec_dip", label="DEC")
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
        params=["M", "H0", "om", "v_km_s", "ra_dip", "dec_dip", "vx", "vy", "vz"],
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
    plot_residuals(z_values=z_cmb, residuals=residuals, y_err=mu_std, bins=40)


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
# Selecting surveys with IDs (1, 5, 15, 50, 51, 56, 63, 150)
#
# M: -19.341 +- 0.054 mag
# H0: 70.4 +- 1.7 km/s/Mpc
# Ωm: 0.331 +- 0.018
# Vx: -1.08 +- 0.38 (x 100 km/s)
# Vy: -1.35 +- 0.50 (x 100 km/s)
# Vz: -1.03 +- 0.56 (x 100 km/s)
#
# v (dipole): 213 +- 50 km/s
# RA (dipole): 230 +16 -13 deg
# DEC (dipole): -29 +12 -16 deg
#
# DOF: 1584
# χ2 (MAP): 1384.31 (delta χ2 = 18.61 -> 3.6 sigma preference for the dipole)
# Log Evidence: -703.9 (delta logZ = 4.8 -> strong preference for the dipole)
# ---------------------------------
