from numba import njit
import numpy as np
from scipy.constants import c as c0
from scipy.linalg import cho_factor
from interpolator import interp_hermite, interp_pchip
from solve_ivp import solve_ivp
from solve_triangular import solve_triangular
from y2005cc.data import get_data
import y2018fs8.data as fs8

c = c0 / 1000  # Speed of light in km/s

legend, z_cc, H_values, cov_matrix = get_data()
cho_cc = cho_factor(cov_matrix, lower=True)[0]
cho_fs8 = cho_factor(fs8.cov_mat, lower=True)[0]

z_fs8, fs8_values = fs8.data["z"], fs8.data["fs8"]
a_vals_fs8 = 1 / (1.0 + z_fs8)

N_cc = z_cc.size
N_fs8 = z_fs8.size

z_max = max(np.max(z_fs8), np.max(z_cc))
z_grid = np.linspace(0, z_max + 0.1, num=4000)
dz = np.diff(z_grid)


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
    cum_dm = np.zeros(len(z_grid), dtype=np.float64)
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
    return params[4] ** 2 * solve_triangular(cho_fs8, delta)


@njit
def chi2_cc(params):
    delta = H_values - H_z(z_cc, params)
    return  params[3] ** 2 * solve_triangular(cho_cc, delta)


@njit
def chi_squared(params):
    return chi2_cc(params) + chi2_fs8(params)


@njit
def log_likelihood_jit(params):
    normalization_cc = -2 * N_cc * np.log(params[3])
    normalization_fs8 = -2 * N_fs8 * np.log(params[4])
    return -0.5 * (chi_squared(params) + normalization_cc + normalization_fs8)


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
    prior.add_parameter("f_cc", dist=(0.05, 3))
    prior.add_parameter("f_fs8", dist=(0.05, 3))
    prior.add_parameter("w0", dist=(-1, 0))

    with Pool(6) as pool:
        sampler = Sampler(
            prior, log_likelihood, n_live=6_000, pool=pool, seed=42, pass_dict=False,
        )
        sampler.run(verbose=True)

    samples, log_w, log_l = sampler.posterior()
    w = np.exp(log_w)
    log_evd = sampler.log_z
    one_sigma_ci = [0.159, 0.5, 0.841]

    labels = ["$H_0$", "$\\Omega_m$", "$\\sigma_8$", "$f_{cc}$", "$f_{fs8}$", "$w_0$"]
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
    plot_fs8_predictions(
        fs8_theory=lambda z: fs8_theory(1 / (1 + z), best_fit),
        data=fs8.data,
        q=H_z(z_fs8, best_fit) * DM(z_fs8, best_fit) / Hz_DMz_fid,
        f_err=fs_50,
    )


if __name__ == "__main__":
    main()


# ----------- Flat ΛCDM -----------
# H0: 67.9 +2.5 -2.4 km/s/Mpc
# Ωm: 0.315 +0.018 -0.017
# σ8: 0.786 +0.011 -0.010
# S8: 0.805 +0.019 -0.018
# f_cc: 1.50 +0.18 -0.17
# f_fs8: 1.78 +0.17 -0.17
# Chi squared: 91.63
# Log likelihood: 2.09
# Log evidence: -10.9
# Degs of freedom: 89
# ---------------------------------


# ----------- Flat wCDM -----------
# H0: 64.7 +2.6 -2.6 km/s/Mpc
# Ωm: 0.285 +0.021 -0.022
# σ8: 0.882 +0.052 -0.042
# S8: 0.862 +0.028 -0.027
# f_cc: 1.48 +0.17 -0.17
# f_fs8: 1.93 +0.19 -0.18
# w0: -0.715 +0.092 -0.095 (prior U[-1.5, 0])
# Chi squared: 90.73
# Log likelihood: 6.25 (2.88 sigma significance)
# Log evidence: -8.6 (Δ logZ = 2.3 against ΛCDM)
# Degs of freedom: 88
# ---------------------------------


# ----------- Flat wzCDM ----------
# w(z) = -1 + 2 * (1 + w0) / (1 + w0 + (1 - w0) * (1 + z)^3)
#
# H0: 64.8 +2.5 -2.5 km/s/Mpc
# Ωm: 0.318 +0.017 -0.016
# σ8: 0.837 +0.023 -0.022
# S8: 0.862 +0.026 -0.027
# f_cc: 1.47 +0.17 -0.17
# f_fs8: 1.93 +0.19 -0.18
# w0: -0.62 +0.11 -0.12 (prior U[-1, 0])
# Chi squared: 90.76
# Log likelihood: 6.08 (2.82 sigma significance)
# Log evidence: -8.2 (Δ logZ = 2.7 against ΛCDM)
# Degs of freedom: 88
# ---------------------------------


# ---------- Flat w0waCDM ---------
# TODO
# ---------------------------------
