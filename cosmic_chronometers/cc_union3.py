from numba import njit
import numpy as np
from scipy.constants import c as c0
from scipy.linalg import cho_factor, solve_triangular
from interpolator import interp_hermite
from y2026union3_1.data import get_data as get_sn_data
from y2005cc.data import get_data as get_cc_data

legend_sn, z_cmb, z_hel, mu_vals, cov_matrix_sn = get_sn_data()
legend_cc, z_cc_vals, H_cc_vals, cov_matrix_cc = get_cc_data()

logdet_cc = np.linalg.slogdet(cov_matrix_cc)[1]
N_cc = len(z_cc_vals)

cho_sn = cho_factor(cov_matrix_sn, lower=True)[0]
cho_cc = cho_factor(cov_matrix_cc, lower=True)[0]

c = c0 / 1000  # Speed of light in km/s

z_grid = np.linspace(0, np.max(z_cmb), num=2000)
dx = np.diff(z_grid)


@njit
def Ode_z(z, w0):
    zp1 = 1.0 + z
    return (2 * zp1**3 / (1.0 + w0 + (1.0 - w0) * zp1**3)) ** 2


@njit
def Ez(z, Om, w0):
    zp1 = 1.0 + z
    return np.sqrt(Om * zp1**3 + (1.0 - Om) * Ode_z(z, w0))


@njit
def H_z(z, H0, Om, w0):
    return H0 * Ez(z, Om, w0)


@njit
def DM_z(z, H0, Om, w0):
    dh_grid = c / H_z(z_grid, H0, Om, w0)
    dy = (dh_grid[:-1] + dh_grid[1:]) / 2
    cum_dm = np.zeros(z_grid.size, dtype=np.float64)
    cum_dm[1:] = np.cumsum(dx * dy)
    return interp_hermite(z, z_grid, cum_dm, dh_grid)


@njit
def mu_theory(dM, H0, Om, w0):
    dL = (1.0 + z_hel) * DM_z(z_cmb, H0, Om, w0)
    return dM + 25.0 + 5 * np.log10(dL)


def solve_triang(cho_L, delta):
    y = solve_triangular(cho_L, delta, lower=True, check_finite=False)
    return np.dot(y, y)


def chi_squared(params):
    delta_sn = mu_vals - mu_theory(
        params["dM"], params["H0"], params["Om"], params["w0"]
    )
    chi_sn = solve_triang(cho_sn, delta_sn)

    cc_delta = H_cc_vals - H_z(z_cc_vals, params["H0"], params["Om"], params["w0"])
    chi_cc = params["f_cc"] ** 2 * solve_triang(cho_cc, cc_delta)
    return chi_sn + chi_cc


def log_likelihood(params):
    f_cc = params["f_cc"]
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
    prior.add_parameter("f_cc", dist=(0.1, 3.0))
    prior.add_parameter("dM", dist=(-1.0, 1.0))
    prior.add_parameter("H0", dist=(55.0, 85.0))
    prior.add_parameter("Om", dist=(0.1, 0.7))
    prior.add_parameter("w0", dist=(-1.0, 0.0))

    with Pool(6) as pool:
        sampler = Sampler(prior, log_likelihood, n_live=8_000, pool=pool, seed=42)
        sampler.run(verbose=True)

    samples, log_w, log_l = sampler.posterior()
    w = np.exp(log_w)
    log_evd = sampler.log_z
    one_sigma_ci = [0.159, 0.5, 0.841]
    labels = ["$f_{cc}$", "$ΔM$", "$H_0$", "$Ω_m$", "$w_0$"]

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
    w0_16, w0_50, w0_84 = quantile(samples[:, 4], one_sigma_ci, weights=w)

    Omh2_samples = samples[:, 3] * (samples[:, 2] / 100) ** 2
    Omh2_16, Omh2_50, Omh2_84 = quantile(Omh2_samples, one_sigma_ci, weights=w)

    best_fit = {"f_cc": fcc_50, "dM": dM_50, "H0": h0_50, "Om": Om_50, "w0": w0_50}
    deg_of_freedom = z_cmb.size + z_cc_vals.size - len(labels)

    print(f"f_cc: {fcc_50:.2f} +{(fcc_84 - fcc_50):.2f} -{(fcc_50 - fcc_16):.2f}")
    print(f"ΔM: {dM_50:.3f} +{(dM_84 - dM_50):.3f} -{(dM_50 - dM_16):.3f} mag")
    print(f"H0: {h0_50:.1f} +{(h0_84 - h0_50):.1f} -{(h0_50 - h0_16):.1f} km/s/Mpc")
    print(f"Ωm: {Om_50:.3f} +{(Om_84 - Om_50):.3f} -{(Om_50 - Om_16):.3f}")
    print(f"ωm: {Omh2_50:.4f} +{(Omh2_84 - Omh2_50):.4f} -{(Omh2_50 - Omh2_16):.4f}")
    print(f"w0: {w0_50:.2f} +{(w0_84 - w0_50):.2f} -{(w0_50 - w0_16):.2f}")
    print(f"Chi squared: {chi_squared(best_fit):.2f}")
    print(f"Log evidence: {log_evd:.1f}")
    print(f"Degrees of freedom: {deg_of_freedom}")

    plot_cc_predictions(
        H_z=lambda z: H_z(z, best_fit["H0"], best_fit["Om"], best_fit["w0"]),
        z=z_cc_vals,
        H=H_cc_vals,
        H_err=np.sqrt(np.diag(cov_matrix_cc)) / fcc_50,
        label=legend_cc,
    )
    plot_sn_predictions(
        legend=legend_sn,
        x=z_cmb,
        y=mu_vals,
        y_err=np.sqrt(np.diag(cov_matrix_sn)),
        y_model=mu_theory(dM_50, h0_50, Om_50, w0_50),
        label=f"$Ω_m$={Om_50:.4f}",
        x_scale="log",
    )


if __name__ == "__main__":
    main()


"""
Flat ΛCDM
f_cc: 1.49 +0.18 -0.17
ΔM: -0.077 +0.074 -0.076 mag
H0: 66.7 +2.6 -2.5 km/s/Mpc
Ωm: 0.333 +0.022 -0.021
ωm: 0.1486 +0.0103 -0.0100
Chi squared: 63.94
Log evidence: -167.3
Degrees of freedom: 54
"""

"""
Flat ΛCDM
Outflow mag correction of SNe M(z) = M_inf - M'0 * z_c^2 / (z_c + z), z_c=0.043

f_cc: 1.48 +0.18 -0.17
ΔM_inf: -0.056 +0.075 -0.077 mag
M'0: -2.8 +1.8 -1.7 (prior ~ U(-13, 7))
H0: 68.5 +2.9 -2.8 km/s/Mpc
Ωm: 0.306 +0.027 -0.025
ωm: 0.1436 +0.0107 -0.0102
Chi squared: 60.86 (1.75 significance)
Log evidence: -167.6
Degrees of freedom: 53
"""

"""
Flat wCDM: w(z) = w0
f_cc: 1.47 +0.18 -0.17
ΔM: -0.059 +0.081 -0.081 mag
H0: 67.1 +2.7 -2.6 km/s/Mpc
Ωm: 0.307 +0.042 -0.048
ωm: 0.1377 +0.0172 -0.0197
w0: -0.91 +0.12 -0.13 prior U(-1.5, -0.5)
Chi squared: 62.44
Log evidence: -168.1
Degrees of freedom: 53

==============================

Flat wzCDM: w(z) = -1 + 2 * (1 + w0) / (1 + w0 + (1 - w0) * (1 + z)^3)
f_cc: 1.47 +0.18 -0.17
ΔM: -0.049 +0.077 -0.078 mag
H0: 67.1 +2.6 -2.6 km/s/Mpc
Ωm: 0.310 +0.026 -0.027
ωm: 0.1393 +0.0117 -0.0118
w0: -0.84 +0.11 -0.10 prior U(-1, 0) - truncated at -1, 1.45 sigma to the left of the mean
wa: d w(z)/dz at z=0 = -1.5 * (1 - w0^2)
Chi squared: 62.13
Log evidence: -168.0
Degrees of freedom: 53

==============================

Flat CPL: w(z) = w0 + wa * z / (1 + z)
TODO
"""
