from numba import njit
import numpy as np
from scipy.constants import c as c0
from interpolator import interp_hermite, interp_pchip
from solve_ivp import solve_ivp
from solve_triangular import solve_triangular
from y2005cc.data import method, get_data
import y2018fs8.data as fs8

c = c0 / 1000  # Speed of light in km/s

legend, z_cc, H_values, diag_stat_cc, cov_matrix_sys = get_data(split_sys=True)
cho_fs8 = np.linalg.cholesky(fs8.cov_mat)
logdet_fs8 = 2 * np.sum(np.log(np.diag(cho_fs8)))

z_fs8, fs8_values = fs8.data["z"], fs8.data["fs8"]
a_vals_fs8 = 1 / (1.0 + z_fs8)

N_cc = z_cc.size
N_fs8 = z_fs8.size
non_d = method != "D"

z_max = max(np.max(z_fs8), np.max(z_cc))
z_grid = np.linspace(0, z_max + 0.1, num=4000)
dz = z_grid[1] - z_grid[0]


@njit
def w_de_z(z, w0, wa):
    return w0 + wa * z / (1.0 + z)


@njit
def Ode_z(z, w0, wa):
    return (1. + z)**(3 * (1. + w0 + wa)) * np.exp(-3 * wa * z / (1. + z))


@njit
def d_Ode_dz(z, Ode, w0, wa):
    return Ode * Ode_z(z, w0, wa) * 3 * (1.0 + w_de_z(z, w0, wa)) / (1.0 + z)


@njit
def H_z(z, params):
    H0, Om, w0, wa = params[0], params[1], params[6], params[7]
    return H0 * np.sqrt(Om * (1.0 + z) ** 3 + (1.0 - Om) * Ode_z(z, w0, wa))


@njit
def dH_da(z, H_val, params):
    H0, Om, w0, wa = params[0], params[1], params[6], params[7]
    a = 1 / (1.0 + z)
    numerator = 3 * Om * (1.0 + z) ** 2 + d_Ode_dz(z, Ode=1 - Om, w0=w0, wa=wa)
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
    wa_fid = 0.0
    params = [H0_fid, Om_fid, s8_fid, 0.0, 0.0, 0.0, w0_fid, wa_fid]
    DM_i, = DM(np.array([zi]), params)
    Hz_DMz_fid[i] = H_z(zi, params) * DM_i


@njit
def chi2_fs8(params):
    q = H_z(z_fs8, params) * DM(z_fs8, params) / Hz_DMz_fid

    delta = fs8_values - fs8_theory(a_vals_fs8, params) / q
    y = solve_triangular(cho_fs8, delta)
    return np.exp(-2 * params[5]) * np.dot(y, y)


@njit
def chi2_cc(params, cho_cc):
    delta = H_values - H_z(z_cc, params)
    y = solve_triangular(cho_cc, delta)
    return np.dot(y, y)


@njit
def chi_squared(params, cho_cc):
    return chi2_cc(params, cho_cc) + chi2_fs8(params)


@njit
def get_fz(params):
    z_pivot = 0.62
    fp, n = np.exp(params[3]), params[4]
    fz = np.full_like(z_cc, fp)
    fz[non_d] *= ((1.0 + z_cc[non_d]) / (1.0 + z_pivot)) ** n
    return fz


@njit
def log_likelihood_jit(params):
    if params[6] + params[7] >= -1/3:
        return -np.inf

    cov_cc = cov_matrix_sys + np.diag(diag_stat_cc**2 * get_fz(params)**2)
    cho_cc = np.linalg.cholesky(cov_cc)
    logdet_cc = 2 * np.sum(np.log(np.diag(cho_cc)))

    normalization_cc =  N_cc * np.log(2 * np.pi) + logdet_cc
    normalization_fs8 = N_fs8 * np.log(2 * np.pi) + logdet_fs8 + 2 * N_fs8 * params[5]
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
    prior.add_parameter("ln_f_cc", dist=(-1.5, 0.5))
    prior.add_parameter("n_cc", dist=(-2, 4))
    prior.add_parameter("ln_f_fs8", dist=(-1.15, 0.15))
    prior.add_parameter("w0", dist=(-3, 2))
    prior.add_parameter("wa", dist=(-3, 3))

    with Pool(6) as pool:
        sampler = Sampler(prior, log_likelihood, n_live=5_000, pool=pool, seed=42, pass_dict=False)
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
    ln_fs_16, ln_fs_50, ln_fs_84 = quantile(samples[:, 5], one_sigma_ci, weights=w)
    w0_16, w0_50, w0_84 = quantile(samples[:, 6], one_sigma_ci, weights=w)
    wa_16, wa_50, wa_84 = quantile(samples[:, 7], one_sigma_ci, weights=w)
    Omh2_16, Omh2_50, Omh2_84 = quantile(Omh2_samples, one_sigma_ci, weights=w)

    S8_samples = samples[:, 2] * np.sqrt(samples[:, 1] / 0.3)
    S8_16, S8_50, S8_84 = quantile(S8_samples, one_sigma_ci, weights=w)

    best_fit = samples[np.argmax(log_l)]
    fz_cc = get_fz(best_fit)
    cov_cc = cov_matrix_sys + np.diag(diag_stat_cc**2 * fz_cc**2)
    cho_cc = np.linalg.cholesky(cov_cc)

    print(f"H0: {H0_50:.1f} +{(H0_84 - H0_50):.1f} -{(H0_50 - H0_16):.1f} km/s/Mpc")
    print(f"Ωm: {Om_50:.3f} +{(Om_84 - Om_50):.3f} -{(Om_50 - Om_16):.3f}")
    print(f"σ8: {sig8_50:.3f} +{(sig8_84 - sig8_50):.3f} -{(sig8_50 - sig8_16):.3f}")
    print(f"S8: {S8_50:.3f} +{(S8_84 - S8_50):.3f} -{(S8_50 - S8_16):.3f}")
    print(f"w0: {w0_50:.3f} +{(w0_84 - w0_50):.3f} -{(w0_50 - w0_16):.3f}")
    print(f"wa: {wa_50:.3f} +{(wa_84 - wa_50):.3f} -{(wa_50 - wa_16):.3f}")
    print(f"Ωm h^2: {Omh2_50:.3f} +{(Omh2_84 - Omh2_50):.3f} -{(Omh2_50 - Omh2_16):.3f}")
    print(f"n_cc: {ncc_50:.2f} +{(ncc_84 - ncc_50):.2f} -{(ncc_50 - ncc_16):.2f}")
    print(f"ln(f_cc): {ln_fcc_50:.2f} +{(ln_fcc_84 - ln_fcc_50):.2f} -{(ln_fcc_50 - ln_fcc_16):.2f}")
    print(f"ln(f_fs8): {ln_fs_50:.2f} +{(ln_fs_84 - ln_fs_50):.2f} -{(ln_fs_50 - ln_fs_16):.2f}")
    print(f"Chi2 (MAP): {chi_squared(best_fit, cho_cc):.2f}")
    print(f"Log likelihood (MAP): {np.max(log_l):.2f}")
    print(f"Log evidence: {log_evd:.1f}")
    print(f"DOF: {len(z_cc) + len(z_fs8) - len(best_fit)}")

    labels = ["$H_0$", "$\\Omega_m$", "$\\sigma_8$", "$ln(f_{cc})$", "$n_{cc}$", "$ln(f_{fs8})$", "$w_0$", "$w_a$"]
    corner(
        samples,
        weights=w,
        labels=labels,
        quantiles=one_sigma_ci,
        show_titles=True,
        title_fmt=".4f",
        bins=100,
        fill_contours=True,
        plot_datapoints=False,
        plot_density=False,
        color="#1f77b4",
        hist_kwargs={"linewidth": 1.5},
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
        H_err=np.sqrt(np.diag(cov_matrix_sys) + diag_stat_cc**2),
        label=f"{legend} $H_0$: {H0_50:.1f} ± {(H0_84 - H0_50):.1f} km/s/Mpc",
        method=method,
        err_scaling=1 / fz_cc,
    )
    plot_fs8_predictions(
        fs8_theory=lambda z: fs8_theory(1 / (1 + z), best_fit),
        data=fs8.data,
        q=H_z(z_fs8, best_fit) * DM(z_fs8, best_fit) / Hz_DMz_fid,
        f_err=1 / np.exp(best_fit[5]),
    )


if __name__ == "__main__":
    main()


# ----------- Flat ΛCDM -----------
# H0: 70.3 +1.5 -1.6 km/s/Mpc
# Ωm: 0.312 +0.017 -0.017
# Ωm h^2: 0.154 +0.009 -0.008
# σ8: 0.787 +0.011 -0.010
# S8: 0.802 +0.019 -0.018
#
# n_cc: 1.49 +0.51 -0.50
# ln(f_cc): -0.57 +0.17 -0.15
# ln(f_fs8): -0.57 +0.10 -0.09
#
# Chi2 (MAP): 97.27
# Log likelihood (MAP): -45.15
# Log evidence: -59.7
# DOF: 89
# ---------------------------------


# ----------- Flat wCDM -----------
# H0: 67.8 +1.6 -1.8 km/s/Mpc
# Ωm: 0.282 +0.020 -0.021
# Ωm h^2: 0.130 +0.012 -0.012
# σ8: 0.881 +0.049 -0.040
# S8: 0.856 +0.026 -0.025
# w: -0.724 +0.086 -0.091 (prior ~ U[-2.0, 0])
#
# n_cc: 1.52 +0.54 -0.52
# ln(f_cc): -0.57 +0.17 -0.16
# ln(f_fs8): -0.65 +0.10 -0.09
#
# Chi2 (MAP): 97.24
# Log likelihood (MAP): -40.85
# Log evidence: -57.6
# DOF: 88
# ---------------------------------


# ---------- Flat w0waCDM ---------
# w0 + wa < -1 / 3 enforced in the likelihood
#
# H0: 67.3 +1.7 -1.7 km/s/Mpc
# Ωm: 0.313 +0.034 -0.036
# Ωm h^2: 0.142 +0.014 -0.016
# σ8: 0.840 +0.054 -0.039
# S8: 0.861 +0.026 -0.026
# w0: -0.638 +0.143 -0.128 (prior ~ U[-3, 2])
# wa: -0.619 +0.666 -0.874 (prior ~ U[-3, 3])
#
# n_cc: 1.56 +0.54 -0.52
# ln(f_cc): -0.58 +0.17 -0.16
# ln(f_fs8): -0.64 +0.10 -0.10
#
# Chi2 (MAP): 98.19
# Log likelihood (MAP): -40.66
# Log evidence: -59.6
# DOF: 87
# ---------------------------------
