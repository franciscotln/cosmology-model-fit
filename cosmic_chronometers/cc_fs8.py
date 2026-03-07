from numba import njit
import numpy as np
from scipy.constants import c as c0
from scipy.integrate import solve_ivp
from interpolator import interp_hermite, interp_pchip
from y2005cc.data import get_data
import y2018fs8.data as fs8

c = c0 / 1000  # Speed of light in km/s

legend, z_cc, H_values, cov_matrix = get_data()

z_fs8, fs8_values = fs8.data["z"], fs8.data["fs8"]

inv_cov_fs8 = np.linalg.inv(fs8.cov_mat)
logdet_fs8 = np.linalg.slogdet(fs8.cov_mat)[1]

inv_cov_cc = np.linalg.inv(cov_matrix)
logdet_cc = np.linalg.slogdet(cov_matrix)[1]

z_grid = np.linspace(0, np.max(z_fs8) + 0.1, num=4000)
dz = np.diff(z_grid)


@njit
def Ode_z(z, w0):
    cubic = (1 + z) ** 3
    return (2 * cubic / (1 + w0 + (1 - w0) * cubic)) ** 2


@njit
def H_z(z, params):
    H0, Om, w0 = params[0], params[1], params[-1]
    return H0 * np.sqrt(Om * (1.0 + z) ** 3 + (1.0 - Om) * Ode_z(z, w0))


@njit
def dH_da(z, params):
    a = 1 / (1 + z)
    dz = 1e-06
    H_plus = H_z(z + dz, params)
    H_minus = H_z(z - dz, params)
    dH_dz = (H_plus - H_minus) / (2 * dz)
    return -dH_dz / a**2


@njit
def DM(z, params):
    dh_grid = c / H_z(z_grid, params)
    dh = (dh_grid[:-1] + dh_grid[1:]) / 2
    cum_dm = np.zeros(len(z_grid), dtype=np.float64)
    cum_dm[1:] = np.cumsum(dh * dz)
    return interp_hermite(z, z_grid, cum_dm, dh_grid)


denominator_fiducial = np.zeros(len(z_fs8), dtype=np.float64)

for i in range(len(z_fs8)):
    zi = z_fs8[i]
    Om_fid = fs8.data["omega_fid"][i]
    params = [67.5, Om_fid, 0.8, 1.0, 1.0, -1.0]
    DM_i = DM(np.array([zi]), params)[0]
    denominator_fiducial[i] = H_z(zi, params) * DM_i


@njit
def growth_ode(a, y, *params):
    H0, Om = params[0], params[1]
    delta, d_delta_da = y

    z = 1 / a - 1
    H_val = H_z(z, params)
    dH_da_val = dH_da(z, params)

    source = (3 / 2) * (Om / a**5) * delta * (H0 / H_val) ** 2
    friction = -(3 / a + dH_da_val / H_val) * d_delta_da
    d2_delta_da = friction + source

    return [d_delta_da, d2_delta_da]


max_z = 1000
a_vals = np.logspace(np.log10(1 / (1 + max_z)), 0, 10_000)


def fs8_theory(z, params):
    sol = solve_ivp(
        growth_ode,
        t_span=(a_vals[0], a_vals[-1]),
        y0=(a_vals[0], 1.0),
        t_eval=a_vals,
        rtol=1e-8,
        atol=1e-10,
        args=params,
    )
    delta, d_delta_da = sol.y

    delta0 = interp_hermite(np.array([1.0]), a_vals, delta, d_delta_da)[0]
    # f = d(ln delta)/d(ln a) = (a / delta) * d(delta)/da
    # sigma8(z) = sigma8 * delta(z) / delta(z=0)
    a = 1 / (1 + z)
    return params[2] * a * interp_pchip(a, a_vals, d_delta_da) / delta0


def chi_squared(params):
    f_cc, f_fs8 = params[3], params[4]

    q = H_z(z_fs8, params) * DM(z_fs8, params) / denominator_fiducial
    delta_fs8 = fs8_values - fs8_theory(z_fs8, params) / q
    chi2_fs8 = f_fs8**2 * np.dot(delta_fs8, np.dot(inv_cov_fs8, delta_fs8))

    delta_cc = H_values - H_z(z_cc, params)
    chi2_cc = f_cc**2 * np.dot(delta_cc, np.dot(inv_cov_cc, delta_cc))

    return chi2_cc + chi2_fs8


def log_likelihood(params):
    N_cc = z_cc.size
    normalization_cc = (
        N_cc * np.log(2 * np.pi) + logdet_cc - 2 * N_cc * np.log(params[3])
    )

    N_fs8 = z_fs8.size
    normalization_fs8 = (
        N_fs8 * np.log(2 * np.pi) + logdet_fs8 - 2 * N_fs8 * np.log(params[4])
    )

    return -0.5 * (chi_squared(params) + normalization_cc + normalization_fs8)


def main():
    from nautilus import Sampler, Prior
    from corner import corner, quantile
    import matplotlib.pyplot as plt
    from multiprocessing import Pool
    from .plot_predictions import plot_cc_predictions

    prior = Prior()
    prior.add_parameter("H0", dist=(40.0, 100.0))
    prior.add_parameter("Om", dist=(0.1, 0.6))
    prior.add_parameter("sig8", dist=(0.2, 1.2))
    prior.add_parameter("f_cc", dist=(0.3, 2.6))
    prior.add_parameter("f_fs8", dist=(0.5, 2.2))
    prior.add_parameter("w0", dist=(-1.0, 0.0))

    with Pool(6) as pool:
        sampler = Sampler(
            prior, log_likelihood, n_live=8_000, pool=pool, seed=42, pass_dict=False
        )
        sampler.run(verbose=True)

    samples, log_w, log_l = sampler.posterior()
    w = np.exp(log_w)
    log_evd = sampler.log_z
    one_sigma_ci = [0.159, 0.5, 0.841]

    corner(
        samples,
        weights=w,
        labels=prior.keys,
        quantiles=one_sigma_ci,
        show_titles=True,
        title_fmt=".4f",
        bins=100,
        fill_contours=False,
        plot_datapoints=False,
        smooth=2.0,
        smooth1d=2.0,
        levels=(0.393, 0.864),
        range=np.repeat(0.9999, len(prior.keys)),
    )
    plt.show()

    H0_16, H0_50, H0_84 = quantile(samples[:, 0], one_sigma_ci, weights=w)
    Om_16, Om_50, Om_84 = quantile(samples[:, 1], one_sigma_ci, weights=w)
    sig8_16, sig8_50, sig8_84 = quantile(samples[:, 2], one_sigma_ci, weights=w)
    fcc_16, fcc_50, fcc_84 = quantile(samples[:, 3], one_sigma_ci, weights=w)
    fs_16, fs_50, fs_84 = quantile(samples[:, 4], one_sigma_ci, weights=w)
    w0_16, w0_50, w0_84 = quantile(samples[:, 5], one_sigma_ci, weights=w)

    S8_samples = samples[:, 2] * np.sqrt(samples[:, 1] / 0.3)
    S8_16, S8_50, S8_84 = quantile(S8_samples, one_sigma_ci, weights=w)

    best_fit = [H0_50, Om_50, sig8_50, fcc_50, fs_50, w0_50]

    print(f"H0: {H0_50:.1f} +{(H0_84 - H0_50):.1f} -{(H0_50 - H0_16):.1f} km/s/Mpc")
    print(f"Ωm: {Om_50:.3f} +{(Om_84 - Om_50):.3f} -{(Om_50 - Om_16):.3f}")
    print(f"σ8: {sig8_50:.3f} +{(sig8_84 - sig8_50):.3f} -{(sig8_50 - sig8_16):.3f}")
    print(f"S8: {S8_50:.3f} +{(S8_84 - S8_50):.3f} -{(S8_50 - S8_16):.3f}")
    print(f"f_cc: {fcc_50:.2f} +{(fcc_84 - fcc_50):.2f} -{(fcc_50 - fcc_16):.2f}")
    print(f"f_fs8: {fs_50:.2f} +{(fs_84 - fs_50):.2f} -{(fs_50 - fs_16):.2f}")
    print(f"w0: {w0_50:.3f} +{(w0_84 - w0_50):.3f} -{(w0_50 - w0_16):.3f}")
    print(f"Chi squared: {chi_squared(best_fit):.2f}")
    print(f"Log likelihood: {log_likelihood(best_fit):.2f}")
    print(f"Log evidence: {log_evd:.1f}")
    print(f"Degs of freedom: {len(z_cc) + len(z_fs8) - len(best_fit)}")

    plot_cc_predictions(
        H_z=lambda z: H_z(z, best_fit),
        z=z_cc,
        H=H_values,
        H_err=np.sqrt(np.diag(cov_matrix)) / fcc_50,
        label=f"{legend} $H_0$: {H0_50:.1f} ± {(H0_84 - H0_50):.1f} km/s/Mpc",
    )

    z_plot = np.linspace(0, np.max(z_fs8), 200)
    fs8_plot = fs8_theory(z_plot, best_fit)
    q = H_z(z_fs8, best_fit) * DM(z_fs8, best_fit) / denominator_fiducial
    plt.errorbar(
        z_fs8,
        fs8_values * q,
        yerr=fs8.data["fs8_err"] * q / fs_50,
        fmt=".",
        label="data",
    )
    plt.plot(z_plot, fs8_plot, label="best-fit", color="C1")
    plt.xlabel("z")
    plt.ylabel(r"$f\sigma_8(z)$")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()

"""
Flat ΛCDM: w(z) = -1
H0: 70.0 +2.6 -2.6 km/s/Mpc
Ωm: 0.285 +0.019 -0.018
σ8: 0.7827 +0.0132 -0.0132
S8: 0.762 +0.020 -0.019
f_cc: 1.46 +0.18 -0.17
f_fs8: 1.32 +0.12 -0.12
Chi squared: 96.63
Log likelihood: -40.89
Log evidence: -52.7
Degs of freedom: 94
"""

"""
Flat wCDM: w(z) = w0
H0: 66.8 +3.0 -3.0 km/s/Mpc
Ωm: 0.261 +0.022 -0.024
σ8: 0.866 +0.065 -0.049
S8: 0.808 +0.033 -0.031
f_cc: 1.46 +0.18 -0.17
f_fs8: 1.35 +0.12 -0.12
w0: -0.752 +0.115 -0.120 (prior -1.6 to 0.0)
Chi squared: 95.67
Log likelihood: -38.84
Log evidence: -52.3
Degs of freedom: 93
"""

"""
Flat wzCDM: w(z) = -1 + 2 * (1 + w0) / (1 + w0 + (1 - w0) * (1 + z)^3)
H0: 67.4 +2.9 -2.8 km/s/Mpc
Ωm: 0.289 +0.019 -0.018
σ8: 0.816 +0.027 -0.023
S8: 0.799 +0.031 -0.028
f_cc: 1.45 +0.18 -0.17
f_fs8: 1.34 +0.12 -0.12
w0: -0.727 +0.145 -0.148 (prior -1.0 to 0.0)
Chi squared: 95.90
Log likelihood: -39.73
Log evidence: -52.5
Degs of freedom: 90
"""

"""
Flat w0waCDM: w(z) = w0 + wa * z / (1 + z)
TODO
"""
