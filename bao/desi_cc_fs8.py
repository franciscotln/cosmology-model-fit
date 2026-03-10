from numba import njit
import numpy as np
from scipy.constants import c as c0
from scipy.integrate import solve_ivp
from interpolator import interp_hermite, interp_pchip
from y2025BAO.data import get_data as get_bao_data
from y2005cc.data import get_data as get_cc_data
import y2018fs8.data as fs8

c = c0 / 1000  # Speed of light in km/s

bao_legend, data, bao_cov_matrix = get_bao_data()
cc_legend, z_cc, H_values, cov_matrix = get_cc_data()

z_fs8, fs8_values = fs8.data["z"], fs8.data["fs8"]

inv_cov_bao = np.linalg.inv(bao_cov_matrix)

inv_cov_fs8 = np.linalg.inv(fs8.cov_mat)
logdet_fs8 = np.linalg.slogdet(fs8.cov_mat)[1]

inv_cov_cc = np.linalg.inv(cov_matrix)
logdet_cc = np.linalg.slogdet(cov_matrix)[1]

N_cc = z_cc.size
N_fs8 = z_fs8.size

z_max = max(np.max(z_fs8), np.max(z_cc), np.max(data["z"]))
z_grid = np.linspace(0, z_max + 0.1, num=4000)
dz = np.diff(z_grid)


@njit
def w_de_z(z, w0):
    return -1 + 2 * (1 + w0) / (1 + w0 + (1 - w0) * (1 + z) ** 3)


@njit
def Ode_z(z, w0):
    cubic = (1.0 + z) ** 3
    return (2 * cubic / (1.0 + w0 + (1.0 - w0) * cubic)) ** 2


@njit
def H_z(z, params):
    H0, Om, w0 = params[0], params[1], params[-1]
    return H0 * np.sqrt(Om * (1.0 + z) ** 3 + (1.0 - Om) * Ode_z(z, w0))


@njit
def DH_z(z, params):
    return c / H_z(z, params)


@njit
def DM_z(z, params):
    dh_grid = c / H_z(z_grid, params)
    dh = (dh_grid[:-1] + dh_grid[1:]) / 2
    cum_dm = np.zeros(len(z_grid))
    cum_dm[1:] = np.cumsum(dh * dz)
    return np.interp(z, z_grid, cum_dm)


@njit
def DV_z(z, params):
    DH = DH_z(z, params)
    DM = DM_z(z, params)
    return (z * DH * DM**2) ** (1 / 3)


@njit
def dH_da(z, params):
    H0, Om, w0 = params[0], params[1], params[-1]
    Ode = 1.0 - Om
    a = 1 / (1 + z)

    mat = 3 * Om * (1.0 + 0.0) * (1.0 + z) ** 3
    de = 3 * Ode * (1.0 + w_de_z(z, w0)) * Ode_z(z, w0)
    numerator = -0.5 * H0**2 * (mat + de)
    denominator = a * H_z(z, params)
    return numerator / denominator


denominator_fiducial = np.zeros(N_fs8, dtype=np.float64)
for i in range(N_fs8):
    zi = z_fs8[i]
    Om_fid = fs8.data["omega_fid"][i]
    s8_fid = fs8.data["s8_fid"][i]
    w0_fid = -1.0
    S8_fid = s8_fid * (Om_fid / 0.3) ** 0.5
    params = [67.5, Om_fid, s8_fid, S8_fid, 1.0, 1.0, 147.0, w0_fid]
    denominator_fiducial[i] = H_z(zi, params) * DM_z(zi, params)


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


max_z = 500
a_vals = np.logspace(np.log10(1 / (1 + max_z)), 0, 5000)


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
    sig8 = params[2]
    a = 1 / (1.0 + z)
    # f = d(ln delta)/d(ln a) = (a / delta) * d(delta)/da
    # sigma8(z) = sigma8 * delta(z) / delta(z=0)
    return sig8 * a * interp_pchip(a, a_vals, d_delta_da) / delta0


PLANCK_MASK = (fs8.data["omega_fid"] >= 0.3) & (fs8.data["s8_fid"] >= 0.8)
S8_fid = fs8.data["s8_fid"] * (fs8.data["omega_fid"] / 0.3) ** 0.5


def chi2_fs8(params):
    g8, f_fs8 = params[3], params[5]

    alpha = np.where(PLANCK_MASK, 1.0, S8_fid / g8)
    q = H_z(z_fs8, params) * DM_z(z_fs8, params) / denominator_fiducial

    delta = fs8_values - fs8_theory(z_fs8, params) * alpha / q
    return delta @ (inv_cov_fs8 * f_fs8**2) @ delta


qty_map = {"DV_over_rs": 0, "DM_over_rs": 1, "DH_over_rs": 2}
desi_qty = np.array([qty_map[q] for q in data["quantity"]], dtype=np.int32)


@njit
def bao_theory(z, qty, params):
    results = np.empty(z.size, dtype=np.float64)
    DV_mask = qty == 0
    DM_mask = qty == 1
    DH_mask = qty == 2
    results[DH_mask] = DH_z(z[DH_mask], params)
    results[DM_mask] = DM_z(z[DM_mask], params)
    results[DV_mask] = DV_z(z[DV_mask], params)
    return results / params[6]


@njit
def chi2_bao(params):
    delta = data["value"] - bao_theory(data["z"], desi_qty, params)
    return delta @ inv_cov_bao @ delta


@njit
def chi2_cc(params):
    f_cc = params[4]
    delta = H_values - H_z(z_cc, params)
    return delta @ (inv_cov_cc * f_cc**2) @ delta


def chi_squared(params):
    return chi2_cc(params) + chi2_fs8(params) + chi2_bao(params)


def log_likelihood(params):
    normalization_cc = (
        N_cc * np.log(2 * np.pi) + logdet_cc - 2 * N_cc * np.log(params[4])
    )

    normalization_fs8 = (
        N_fs8 * np.log(2 * np.pi) + logdet_fs8 - 2 * N_fs8 * np.log(params[5])
    )

    return -0.5 * (chi_squared(params) + normalization_cc + normalization_fs8)


def main():
    from nautilus import Sampler, Prior
    from corner import corner, quantile
    import matplotlib.pyplot as plt
    from multiprocessing import Pool
    from cosmic_chronometers.plot_predictions import plot_cc_predictions
    from fs8.plot_predictions import plot_predictions as plot_fs8_predictions
    from bao.plot_predictions import plot_bao_predictions

    prior = Prior()
    prior.add_parameter("H0", dist=(40, 100))
    prior.add_parameter("Om", dist=(0.1, 0.6))
    prior.add_parameter("sig8", dist=(0.1, 1.5))
    prior.add_parameter("g8", dist=(0.5, 1.5))
    prior.add_parameter("f_cc", dist=(0.03, 3.0))
    prior.add_parameter("f_fs8", dist=(0.2, 3.0))
    prior.add_parameter("rd", dist=(110, 180))
    prior.add_parameter("w0", dist=(-1, 0))

    with Pool(6) as pool:
        sampler = Sampler(
            prior, log_likelihood, n_live=5_000, pool=pool, seed=42, pass_dict=False
        )
        sampler.run(verbose=True)

    samples, log_w, log_l = sampler.posterior()
    w = np.exp(log_w)
    log_evd = sampler.log_z
    one_sigma_ci = [0.159, 0.5, 0.841]

    labels = [
        "$H_0$",
        "$Ω_m$",
        "$\sigma_8$",
        "$g_8$",
        "$f_{cc}$",
        "$f_{fs8}$",
        "$r_{drag}$",
        "$w_0$",
    ]
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
    g8_16, g8_50, g8_84 = quantile(samples[:, 3], one_sigma_ci, weights=w)
    fcc_16, fcc_50, fcc_84 = quantile(samples[:, 4], one_sigma_ci, weights=w)
    fs_16, fs_50, fs_84 = quantile(samples[:, 5], one_sigma_ci, weights=w)
    rd_16, rd_50, rd_84 = quantile(samples[:, 6], one_sigma_ci, weights=w)
    w0_16, w0_50, w0_84 = quantile(samples[:, 7], one_sigma_ci, weights=w)

    S8_samples = samples[:, 2] * np.sqrt(samples[:, 1] / 0.3)
    S8_16, S8_50, S8_84 = quantile(S8_samples, one_sigma_ci, weights=w)

    best_fit = [H0_50, Om_50, sig8_50, g8_50, fcc_50, fs_50, rd_50, w0_50]

    print(f"H0: {H0_50:.1f} +{(H0_84 - H0_50):.1f} -{(H0_50 - H0_16):.1f} km/s/Mpc")
    print(f"Ωm: {Om_50:.3f} +{(Om_84 - Om_50):.3f} -{(Om_50 - Om_16):.3f}")
    print(f"σ8: {sig8_50:.3f} +{(sig8_84 - sig8_50):.3f} -{(sig8_50 - sig8_16):.3f}")
    print(f"S8: {S8_50:.3f} +{(S8_84 - S8_50):.3f} -{(S8_50 - S8_16):.3f}")
    print(f"g8: {g8_50:.3f} +{(g8_84 - g8_50):.3f} -{(g8_50 - g8_16):.3f}")
    print(f"f_cc: {fcc_50:.2f} +{(fcc_84 - fcc_50):.2f} -{(fcc_50 - fcc_16):.2f}")
    print(f"f_fs8: {fs_50:.2f} +{(fs_84 - fs_50):.2f} -{(fs_50 - fs_16):.2f}")
    print(f"rd: {rd_50:.1f} +{(rd_84 - rd_50):.1f} -{(rd_50 - rd_16):.1f} Mpc")
    print(f"w0: {w0_50:.3f} +{(w0_84 - w0_50):.3f} -{(w0_50 - w0_16):.3f}")
    print(f"Chi squared: {chi_squared(best_fit):.2f}")
    print(f"Log likelihood: {log_likelihood(best_fit):.2f}")
    print(f"Log evidence: {log_evd:.1f}")
    print(f"Degs of freedom: {len(z_cc) + len(z_fs8) + len(data) - len(best_fit)}")

    plot_bao_predictions(
        theory_predictions=lambda z, qty: bao_theory(z, qty, best_fit),
        data=data,
        errors=np.sqrt(np.diag(bao_cov_matrix)),
        title=bao_legend,
    )
    plot_cc_predictions(
        H_z=lambda z: H_z(z, best_fit),
        z=z_cc,
        H=H_values,
        H_err=np.sqrt(np.diag(cov_matrix)) / fcc_50,
        label=f"{cc_legend} $H_0$: {H0_50:.1f} ± {(H0_84 - H0_50):.1f} km/s/Mpc",
    )
    plot_fs8_predictions(
        fs8_theory=lambda z: fs8_theory(z, best_fit),
        data=fs8.data,
        q=H_z(z_fs8, best_fit)
        * DM_z(z_fs8, best_fit)
        / (denominator_fiducial * np.where(PLANCK_MASK, 1.0, S8_fid / g8_50)),
        f_err=fs_50,
    )


if __name__ == "__main__":
    main()


"""
Flat ΛCDM: w(z) = -1

H0: 68.8 +2.3 -2.3 km/s/Mpc
Ωm: 0.301 +0.008 -0.008
σ8: 0.794 +0.012 -0.012
S8: 0.796 +0.015 -0.014 (1.68 sigma from Planck+ACT)
g8: 0.803 +0.021 -0.020
f_cc: 1.48 +0.18 -0.17 (error overestimation factor in CCH data)
f_fs8: 1.50 +0.14 -0.13 (error overestimation factor in FS8 data)
rd: 147.1 +5.0 -4.6 Mpc
Chi squared: 106.00
Log likelihood: -39.42
Log evidence: -60.3
Degs of freedom: 104

---

without overestimation factors f_cc and f_fs8:

H0: 68.9 +3.3 -3.3 km/s/Mpc
Ωm: 0.299 +0.008 -0.008
σ8: 0.795 +0.018 -0.018
S8: 0.794 +0.020 -0.020 (1.47 sigma from Planck+ACT)
g8: 0.804 +0.031 -0.030
rd: 147.2 +7.3 -6.7 Mpc
Chi squared: 53.23
Log likelihood: -52.36
Degs of freedom: 106
"""

"""
Flat wCDM: w(z) = w0

H0: 67.4 +2.4 -2.4 km/s/Mpc
Ωm: 0.299 +0.008 -0.008
σ8: 0.818 +0.022 -0.021
S8: 0.817 +0.021 -0.020
g8: 0.802 +0.020 -0.020
f_cc: 1.48 +0.18 -0.17
f_fs8: 1.51 +0.14 -0.14
rd: 147.4 +4.9 -4.5 Mpc
w0: -0.902 +0.064 -0.066 (prior -1.4 to -0.4)
Chi squared: 104.38
Log likelihood: -38.31 (1.49 sigma significance)
Log evidence: -61.0 (Δ logZ = -0.7 in favour of ΛCDM)
Degs of freedom: 103

---

without overestimation factors f_cc and f_fs8:

H0: 67.5 +3.4 -3.4 km/s/Mpc
Ωm: 0.298 +0.009 -0.009
σ8: 0.817 +0.027 -0.025
S8: 0.815 +0.026 -0.026
g8: 0.803 +0.031 -0.030
rd: 147.5 +7.3 -6.7 Mpc
w0: -0.906 +0.069 -0.073 (prior -1.4 to -0.4)
Chi squared: 51.54
Log likelihood: -51.51 (1.30 sigma significance)
Degs of freedom: 105
"""

"""
Flat wzCDM: w(z) = -1 + 2 * (1 + w0) / (1 + w0 + (1 - w0) * (1 + z)^3)

H0: 66.7 +2.4 -2.5 km/s/Mpc
Ωm: 0.311 +0.010 -0.010
σ8: 0.813 +0.017 -0.016
S8: 0.828 +0.024 -0.022
g8: 0.801 +0.020 -0.020
f_cc: 1.47 +0.18 -0.17
f_fs8: 1.51 +0.14 -0.14
rd: 147.4 +5.0 -4.6 Mpc
w0: -0.804 +0.104 -0.104 (prior -1.0 to 0.0)
Chi squared: 103.62
Log likelihood: -38.11 (1.62 sigma significance)
Log evidence: -60.3 (Δ logZ = 0.0 equal to ΛCDM)
Degs of freedom: 103

---

without overestimation factors f_cc and f_fs8:

H0: 66.5 +3.4 -3.4 km/s/Mpc
Ωm: 0.312 +0.011 -0.010
σ8: 0.815 +0.022 -0.021
S8: 0.831 +0.030 -0.028
g8: 0.802 +0.031 -0.030
rd: 147.6 +7.3 -6.7 Mpc
w0: -0.787 +0.117 -0.115 (prior -1.0 to 0.0)
Chi squared: 51.02 (1.64 sigma significance)
Log likelihood: -51.25
Degs of freedom: 105
"""

"""
Flat w0waCDM: w(z) = w0 + wa * z / (1 + z)
TODO
"""
