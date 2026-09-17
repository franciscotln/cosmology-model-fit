from numba import njit
import numpy as np
from scipy.constants import c as c0
from interpolator import interp_hermite, interp_pchip
from solve_ivp import solve_ivp
from solve_triangular import solve_triangular
from y2005cc.data import get_data
import y2018fs8.data as fs8

c = c0 / 1000  # Speed of light in km/s

legend, z_cc, H_values, diag_stat_cc, cov_matrix_sys = get_data(split_sys=True)
cho_fs8 = np.linalg.cholesky(fs8.cov_mat)
logdet_fs8 = 2 * np.sum(np.log(np.diag(cho_fs8)))

z_fs8, fs8_values = fs8.data["z"], fs8.data["fs8"]
a_vals_fs8 = 1 / (1.0 + z_fs8)

H_err = np.sqrt(np.diag(cov_matrix_sys) + diag_stat_cc**2)
N_cc = z_cc.size
N_fs8 = z_fs8.size

z_max = max(np.max(z_fs8), np.max(z_cc))
z_grid = np.linspace(0, z_max + 0.1, num=4000)
dz = z_grid[1] - z_grid[0]


@njit
def w_de_z(z, w0):
    # Thawing quintessence wzCDM
    return -1.0 + 2 * (1.0 + w0) / (1.0 + w0 + (1.0 - w0) * (1.0 + z) ** 3)


@njit
def Ode_z(z, w0):
    # Thawing quintessence wzCDM
    cubic = (1 + z) ** 3
    return (2 * cubic / (1 + w0 + (1 - w0) * cubic)) ** 2


@njit
def d_Ode_dz(z, Ode, w0):
    return Ode * Ode_z(z, w0) * 3 * (1.0 + w_de_z(z, w0)) / (1.0 + z)


@njit
def H_z(z, params):
    H0, Om, w0 = params[0], params[1], params[-1]
    return H0 * np.sqrt(Om * (1.0 + z) ** 3 + (1.0 - Om) * Ode_z(z, w0))


@njit
def dH_da(z, H_val, params):
    H0, Om, w0 = params[0], params[1], params[-1]
    a = 1 / (1.0 + z)
    numerator = 3 * Om * (1.0 + z) ** 2 + d_Ode_dz(z=z, Ode=1 - Om, w0=w0)
    denominator = 2 * a**2 * H_val / H0**2
    return -numerator / denominator


@njit
def DM(z, params):
    dh_grid = c / H_z(z_grid, params)
    dh = (dh_grid[:-1] + dh_grid[1:]) / 2
    cum_dm = np.zeros(z_grid.size, dtype=np.float64)
    cum_dm[1:] = np.cumsum(dh * dz)
    return interp_hermite(z, z_grid, cum_dm, dh_grid)


@njit
def growth_ODE(a, y, params):
    H0, Om = params[0], params[1]
    delta, d_delta_da = y

    z = 1 / a - 1
    H_val = H_z(z, params)
    dH_da_val = dH_da(z, H_val, params)

    source = (3 / 2) * (Om / a**5) * delta * (H0 / H_val) ** 2
    friction = -(3 / a + dH_da_val / H_val) * d_delta_da
    d2_delta_da = friction + source

    return np.array([d_delta_da, d2_delta_da])


max_z = 200
a_span = np.logspace(np.log10(1 / (1 + max_z)), 0, 1_000)


@njit
def fs8_theory(a, params):
    sol = solve_ivp(
        growth_ODE,
        t_span=(a_span[0], a_span[-1]),
        y0=(a_span[0], 1.0),  # δ(a_init) = a_init, dδ/da(a_init) = 1.0
        t_eval=a_span,
        rtol=1e-6,
        atol=1e-8,
        args=(params,),
    )
    delta, d_delta_da = sol.y
    sigma8_0 = params[2]
    delta_0 = delta[-1]
    # f = d(ln delta)/d(ln a) = (a / delta) * d(delta)/da
    # sigma8(z) = sigma8 * delta(z) / delta(z=0)
    return (sigma8_0 / delta_0) * a * interp_pchip(a, a_span, d_delta_da)


Hz_DMz_fid = np.zeros(N_fs8, dtype=np.float64)
for i in range(N_fs8):
    zi = z_fs8[i]
    Om_fid = fs8.data["omega_fid"][i]
    s8_fid = fs8.data["s8_fid"][i]
    H0_fid = fs8.data["H0_fid"][i]
    w0_fid = -1.0
    params = [H0_fid, Om_fid, s8_fid, 1.0, 1.0, w0_fid]
    DM_i = DM(np.array([zi]), params)[0]
    Hz_DMz_fid[i] = H_z(zi, params) * DM_i


@njit
def chi2_fs8(params):
    q = H_z(z_fs8, params) * DM(z_fs8, params) / Hz_DMz_fid

    delta = fs8_values - fs8_theory(a_vals_fs8, params) / q
    y = solve_triangular(cho_fs8, delta)
    return params[5] ** 2 * np.dot(y, y)


@njit
def chi2_cc(params, cho_cc):
    delta = H_values - H_z(z_cc, params)
    y = solve_triangular(cho_cc, delta)
    return np.dot(y, y)


@njit
def chi_squared(params, cho_cc):
    return chi2_cc(params, cho_cc) + chi2_fs8(params)


z_pivot_cc = 0.698

@njit
def log_likelihood_jit(params):
    fp_cc, n_cc = np.exp(params[3]), params[4]
    fz_cc = fp_cc * ((1 + z_cc) / (1 + z_pivot_cc)) ** n_cc

    cov_cc = np.diag(diag_stat_cc**2 * fz_cc**2) + cov_matrix_sys
    cho_cc = np.linalg.cholesky(cov_cc)
    logdet_cc = 2.0 * np.sum(np.log(np.diag(cho_cc)))

    normalization_cc =  N_cc * np.log(2 * np.pi) + logdet_cc
    normalization_fs8 = N_fs8 * np.log(2 * np.pi) - 2 * N_fs8 * np.log(params[5]) + logdet_fs8
    return -0.5 * (chi_squared(params, cho_cc) + normalization_cc + normalization_fs8)


def log_likelihood(params):
    return log_likelihood_jit(params)


def main():
    from nautilus import Sampler, Prior
    from corner import corner, quantile
    import matplotlib.pyplot as plt
    from multiprocessing import Pool
    from fs8.plot_predictions import plot_predictions as plot_fs8_predictions
    from ohd.plot_predictions import plot_cc_predictions

    prior = Prior()
    prior.add_parameter("H0", dist=(35, 100))
    prior.add_parameter("Om", dist=(0.01, 0.6))
    prior.add_parameter("sig8", dist=(0.2, 1.5))
    prior.add_parameter("ln_f_cc", dist=(np.log(0.25), np.log(1.40)))
    prior.add_parameter("n_cc", dist=(-4.0, 4.0))
    prior.add_parameter("f_fs8", dist=(0.05, 3))
    prior.add_parameter("w0", dist=(-1.5, 0))

    with Pool(6) as pool:
        sampler = Sampler(
            prior, log_likelihood, n_live=6_000, pool=pool, seed=42, pass_dict=False,
        )
        sampler.run(verbose=True)

    samples, log_w, log_l = sampler.posterior()
    Omh2_samples = samples[:, 1] * (samples[:, 0] / 100) ** 2
    w = np.exp(log_w)
    log_evd = sampler.log_z
    one_sigma_ci = [0.159, 0.5, 0.841]

    H0_16, H0_50, H0_84 = quantile(samples[:, 0], one_sigma_ci, weights=w)
    Om_16, Om_50, Om_84 = quantile(samples[:, 1], one_sigma_ci, weights=w)
    sig8_16, sig8_50, sig8_84 = quantile(samples[:, 2], one_sigma_ci, weights=w)
    ln_fcc_16, ln_fcc_50, ln_fcc_84 = quantile(samples[:, 3], one_sigma_ci, weights=w)
    ncc_16, ncc_50, ncc_84 = quantile(samples[:, 4], one_sigma_ci, weights=w)
    fs_16, fs_50, fs_84 = quantile(samples[:, 5], one_sigma_ci, weights=w)
    w0_16, w0_50, w0_84 = quantile(samples[:, 6], one_sigma_ci, weights=w)
    Omh2_16, Omh2_50, Omh2_84 = quantile(Omh2_samples, one_sigma_ci, weights=w)

    S8_samples = samples[:, 2] * np.sqrt(samples[:, 1] / 0.3)
    S8_16, S8_50, S8_84 = quantile(S8_samples, one_sigma_ci, weights=w)

    best_fit = samples[np.argmax(log_l)]
    fz_cc = np.exp(best_fit[3]) * ((1.0 + z_cc) / (1.0 + z_pivot_cc))**best_fit[4]
    cov_cc = np.diag(diag_stat_cc**2 * fz_cc**2) + cov_matrix_sys
    cho_cc = np.linalg.cholesky(cov_cc)

    print(f"H0: {H0_50:.1f} +{(H0_84 - H0_50):.1f} -{(H0_50 - H0_16):.1f} km/s/Mpc")
    print(f"Ωm: {Om_50:.3f} +{(Om_84 - Om_50):.3f} -{(Om_50 - Om_16):.3f}")
    print(f"σ8: {sig8_50:.3f} +{(sig8_84 - sig8_50):.3f} -{(sig8_50 - sig8_16):.3f}")
    print(f"S8: {S8_50:.3f} +{(S8_84 - S8_50):.3f} -{(S8_50 - S8_16):.3f}")
    print(f"w0: {w0_50:.3f} +{(w0_84 - w0_50):.3f} -{(w0_50 - w0_16):.3f}")
    print(f"Ωm h^2: {Omh2_50:.3f} +{(Omh2_84 - Omh2_50):.3f} -{(Omh2_50 - Omh2_16):.3f}")
    print(f"ln(f_cc): {ln_fcc_50:.2f} +{(ln_fcc_84 - ln_fcc_50):.2f} -{(ln_fcc_50 - ln_fcc_16):.2f}")
    print(f"n_cc: {ncc_50:.2f} +{(ncc_84 - ncc_50):.2f} -{(ncc_50 - ncc_16):.2f}")
    print(f"f_fs8: {fs_50:.2f} +{(fs_84 - fs_50):.2f} -{(fs_50 - fs_16):.2f}")
    print(f"Chi squared: {chi_squared(best_fit, cho_cc):.2f}")
    print(f"Log likelihood: {np.max(log_l):.2f}")
    print(f"Log evidence: {log_evd:.1f}")
    print(f"Degs of freedom: {len(z_cc) + len(z_fs8) - len(best_fit)}")

    labels = ["$H_0$", "$\\Omega_m$", "$\\sigma_8$", "$ln(f_{cc})$", "$n_{cc}$", "$f_{fs8}$", "$w_0$"]
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

    plot_cc_predictions(
        H_z=lambda z: H_z(z, best_fit),
        z=z_cc,
        H=H_values,
        H_err=H_err,
        label=f"{legend} $H_0$: {H0_50:.1f} ± {(H0_84 - H0_50):.1f} km/s/Mpc",
        err_scaling=1 / fz_cc,
    )
    plot_fs8_predictions(
        fs8_theory=lambda z: fs8_theory(1 / (1 + z), best_fit),
        data=fs8.data,
        q=H_z(z_fs8, best_fit) * DM(z_fs8, best_fit) / Hz_DMz_fid,
        f_err=fs_50,
    )


if __name__ == "__main__":
    main()


# ----------- Flat ΛCDM -----------
# H0: 67.4 +3.0 -3.0 km/s/Mpc
# Ωm: 0.315 +0.018 -0.017
# σ8: 0.786 +0.011 -0.010
# S8: 0.806 +0.019 -0.018
# Ωm h^2: 0.143 +0.013 -0.012
# ln(fp_cc): -0.51 +0.18 -0.17
# n_cc: 1.34 +0.53 -0.51
# f_fs8: 1.78 +0.17 -0.17
# Chi squared: 91.04
# Log likelihood: -46.57
# Log evidence: -60.6
# Degs of freedom: 89
# ---------------------------------


# ----------- Flat wCDM -----------
# H0: 64.2 +3.1 -3.1 km/s/Mpc
# Ωm: 0.284 +0.021 -0.022
# σ8: 0.881 +0.050 -0.041
# S8: 0.860 +0.027 -0.026
# w0: -0.719 +0.090 -0.093 (prior U[-1.5, 0])
# Ωm h^2: 0.117 +0.015 -0.015
# ln(fp_cc): -0.50 +0.18 -0.17
# n_cc: 1.31 +0.54 -0.52
# f_fs8: 1.93 +0.19 -0.18
# Chi squared: 94.65
# Log likelihood: -42.43
# Log evidence: -58.3
# Degs of freedom: 88
# ---------------------------------


# ----------- Flat wzCDM ----------
# w(z) = -1 + 2 * (1 + w0) / (1 + w0 + (1 - w0) * (1 + z)^3)
#
# H0: 64.1 +3.0 -3.0 km/s/Mpc
# Ωm: 0.317 +0.017 -0.016
# σ8: 0.838 +0.023 -0.022
# S8: 0.861 +0.026 -0.026
# w0: -0.616 +0.108 -0.123 (prior U[-1.5, 0])
# Ωm h^2: 0.130 +0.013 -0.012
# ln(fp_cc): -0.50 +0.18 -0.17
# n_cc: 1.36 +0.55 -0.52
# f_fs8: 1.93 +0.19 -0.18
# Chi squared: 95.17
# Log likelihood: -42.51
# Log evidence: -58.2
# Degs of freedom: 88
# ---------------------------------


# ---------- Flat w0waCDM ---------
# w0 (prior U[-3, 1])
# wa (prior U[-3, 2])
# w0 + wa < 0 enforced in the likelihood
# TODO
# ---------------------------------
