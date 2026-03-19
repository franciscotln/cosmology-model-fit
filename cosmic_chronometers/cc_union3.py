from numba import njit
import numpy as np
from scipy.constants import c as c0
from interpolator import interp_hermite
from y2026union3_1.data import get_data as get_sn_data
from y2005cc.data import get_data as get_cc_data

legend_sn, z_cmb, z_hel, mu_vals, cov_matrix_sn = get_sn_data()
legend_cc, z_cc_vals, H_cc_vals, cov_matrix_cc = get_cc_data()

logdet_cc = np.linalg.slogdet(cov_matrix_cc)[1]
N_cc = len(z_cc_vals)

inv_cov_sn = np.linalg.inv(cov_matrix_sn)
inv_cov_cc = np.linalg.inv(cov_matrix_cc)

c = c0 / 1000  # Speed of light in km/s

z_grid = np.linspace(0, np.max(z_cmb), num=4000)
dz = np.diff(z_grid)


@njit
def Ode_z(z, w0):
    # Thawing quintessence
    zp1 = 1.0 + z
    return (2 * zp1**3 / (1.0 + w0 + (1.0 - w0) * zp1**3)) ** 2


@njit
def H_z(z, params):
    H0, Om = params[2], params[3]
    return H0 * np.sqrt(Om * (1.0 + z) ** 3 + (1.0 - Om))


@njit
def DM_z(z, params):
    dh_grid = c / H_z(z_grid, params)
    dh = (dh_grid[:-1] + dh_grid[1:]) / 2
    cum_dm = np.zeros(z_grid.size, dtype=np.float64)
    cum_dm[1:] = np.cumsum(dh * dz)
    return interp_hermite(z, x=z_grid, y=cum_dm, y_prime=dh_grid)


@njit
def mu_corr(params, DM_obs):
    # Heaviside step at z = 0.2
    v_km_s = 100 * params[4] * np.where(z_cmb <= 0.2, 1, -1)
    z_pec = v_km_s / c
    z_cosmo = -1.0 + (1.0 + z_cmb) / (1.0 + z_pec)

    return 5 * np.log10(DM_z(z_cosmo, params) / DM_obs)


@njit
def mu_theory(offset, DM):
    return offset + 25.0 + 5 * np.log10((1.0 + z_hel) * DM)


@njit
def chi_squared(params):
    f_cc, offset = params[0], params[1]

    DM_obs = DM_z(z_cmb, params)
    delta_sn = mu_vals - mu_theory(offset, DM_obs) - mu_corr(params, DM_obs)
    chi_sn = delta_sn @ inv_cov_sn @ delta_sn

    cc_delta = H_cc_vals - H_z(z_cc_vals, params)
    chi_cc = f_cc**2 * cc_delta @ inv_cov_cc @ cc_delta

    return chi_sn + chi_cc


@njit
def log_likelihood(params):
    f_cc = params[0]
    normalization_cc = N_cc * np.log(2 * np.pi) + logdet_cc - 2 * N_cc * np.log(f_cc)
    return -0.5 * chi_squared(params) - 0.5 * normalization_cc


def main():
    from nautilus import Sampler, Prior
    from corner import corner, quantile
    import matplotlib.pyplot as plt
    from multiprocessing import Pool
    from sn.plotting import plot_predictions as plot_sn_predictions
    from .plot_predictions import plot_cc_predictions

    prior = Prior()
    prior.add_parameter("f_cc", dist=(0.05, 3.35))
    prior.add_parameter("dM", dist=(-1.0, 1.0))
    prior.add_parameter("H0", dist=(40.0, 95.0))
    prior.add_parameter("Om", dist=(0.1, 0.7))
    prior.add_parameter("v", dist=(-11.0, 5.5))

    with Pool(6) as pool:
        sampler = Sampler(
            prior, log_likelihood, n_live=6_000, pool=pool, seed=42, pass_dict=False
        )
        sampler.run(verbose=True)

    samples, log_w, log_l = sampler.posterior()
    w = np.exp(log_w)
    log_evd = sampler.log_z
    one_sigma_ci = [0.159, 0.5, 0.841]
    labels = ["$f_{cc}$", "$ΔM$", "$H_0$", "$Ω_m$", "$v_{100}$"]

    corner(
        samples,
        weights=w,
        labels=labels,
        quantiles=one_sigma_ci,
        show_titles=True,
        title_fmt=".4f",
        bins=100,
        fill_contours=False,
        plot_datapoints=False,
        smooth=2.0,
        smooth1d=2.0,
        levels=(0.393, 0.864),
        range=np.repeat(0.9999, len(labels)),
    )
    plt.show()

    fcc_16, fcc_50, fcc_84 = quantile(samples[:, 0], one_sigma_ci, weights=w)
    dM_16, dM_50, dM_84 = quantile(samples[:, 1], one_sigma_ci, weights=w)
    h0_16, h0_50, h0_84 = quantile(samples[:, 2], one_sigma_ci, weights=w)
    Om_16, Om_50, Om_84 = quantile(samples[:, 3], one_sigma_ci, weights=w)
    v_16, v_50, v_84 = quantile(samples[:, 4], one_sigma_ci, weights=w)

    Omh2_samples = samples[:, 3] * (samples[:, 2] / 100) ** 2
    Omh2_16, Omh2_50, Omh2_84 = quantile(Omh2_samples, one_sigma_ci, weights=w)

    best_fit = [fcc_50, dM_50, h0_50, Om_50, v_50]
    deg_of_freedom = len(z_cmb) + N_cc - len(labels)

    print(f"f_cc: {fcc_50:.2f} +{(fcc_84 - fcc_50):.2f} -{(fcc_50 - fcc_16):.2f}")
    print(f"ΔM: {dM_50:.3f} +{(dM_84 - dM_50):.3f} -{(dM_50 - dM_16):.3f} mag")
    print(f"H0: {h0_50:.1f} +{(h0_84 - h0_50):.1f} -{(h0_50 - h0_16):.1f} km/s/Mpc")
    print(f"Ωm: {Om_50:.3f} +{(Om_84 - Om_50):.3f} -{(Om_50 - Om_16):.3f}")
    print(f"ωm: {Omh2_50:.4f} +{(Omh2_84 - Omh2_50):.4f} -{(Omh2_50 - Omh2_16):.4f}")
    print(f"v: {v_50:.2f} +{(v_84 - v_50):.2f} -{(v_50 - v_16):.2f} x 100 km/s")
    print(f"Chi squared: {chi_squared(best_fit):.2f}")
    print(f"Log evidence: {log_evd:.1f}")
    print(f"Degrees of freedom: {deg_of_freedom}")

    plot_cc_predictions(
        H_z=lambda z: H_z(z, best_fit),
        z=z_cc_vals,
        H=H_cc_vals,
        H_err=np.sqrt(np.diag(cov_matrix_cc)) / fcc_50,
        label=legend_cc,
    )
    plot_sn_predictions(
        legend=legend_sn,
        x=z_cmb,
        y=mu_vals - mu_corr(best_fit, DM_z(z_cmb, best_fit)),
        y_err=np.sqrt(np.diag(cov_matrix_sn)),
        y_model=mu_theory(dM_50, DM_z(z_cmb, best_fit)),
        label=f"$Ω_m$={Om_50:.4f}",
        x_scale="log",
    )


if __name__ == "__main__":
    main()


"""
Flat ΛCDM
ΔM: -0.076 +0.073 -0.076 mag
H0: 66.8 +2.5 -2.5 km/s/Mpc
Ωm: 0.333 +0.022 -0.021
ωm: 0.149 +0.010 -0.010
f_cc: 1.48 +0.18 -0.17
Chi squared: 63.80
Log evidence: -168.0
Degrees of freedom: 54
"""

"""
Flat ΛCDM
Isotropic velocity SNe observed redshifts (turning point z <= 0.2 inflow z > 0.2 outflow)
z_cosmo = -1 + (1 + z) / (1 + v/c)

v: -290 +117 -116 km/s (prior U(-11.0, 5.5) x 100 km/s)
ΔM: -0.044 +0.075 -0.076 mag
H0: 68.5 +2.7 -2.6 km/s/Mpc
Ωm: 0.307 +0.023 -0.022
ωm: 0.144 +0.010 -0.010
f_cc: 1.48 +0.18 -0.17
Chi squared: 57.36 (2.54 sigma significance)
Log evidence: -166.7 (ΔlogZ = 1.3 in favour of v corrections)
Degrees of freedom: 53
"""

"""
Flat wCDM: w(z) = w0
ΔM: -0.057 +0.080 -0.082 mag
H0: 67.1 +2.6 -2.6 km/s/Mpc
Ωm: 0.307 +0.042 -0.048
ωm: 0.1379 +0.0172 -0.0198
w0: -0.91 +0.12 -0.13 (prior U(-1.5, -0.5))
f_cc: 1.47 +0.18 -0.17
Chi squared: 62.49 (1.14 sigma significance)
Log evidence: -168.9 (ΔlogZ = -0.9 in favour of ΛCDM)
Degrees of freedom: 53

==============================

Flat CPL: w(z) = w0 + wa * z / (1 + z)
TODO
"""
