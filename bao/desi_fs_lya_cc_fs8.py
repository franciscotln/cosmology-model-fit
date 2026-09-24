from numba import njit
import numpy as np
from scipy.constants import c as c0
from interpolator import interp_hermite, interp_pchip
from solve_ivp import solve_ivp
from solve_triangular import solve_triangular
from y2025BAO.data_fs_lya import get_data as get_bao_data
from y2005cc.data import get_data as get_cc_data
import y2018fs8.data as fs8

c = c0 / 1000  # Speed of light in km/s

bao_legend, data, bao_cov_matrix = get_bao_data()
cc_legend, z_cc, H_values, cov_matrix = get_cc_data()

z_fs8, fs8_values = fs8.data["z"], fs8.data["fs8"]
a_fs8 = 1 / (1.0 + z_fs8)

L_bao = np.linalg.cholesky(bao_cov_matrix)
logdet_bao = 2 * np.sum(np.log(np.diag(L_bao)))

L_cc = np.linalg.cholesky(cov_matrix)
logdet_cc = 2 * np.sum(np.log(np.diag(L_cc)))

L_fs8 = np.linalg.cholesky(fs8.cov_mat)
logdet_fs8 = 2 * np.sum(np.log(np.diag(L_fs8)))

N_bao = len(data)
N_cc = len(z_cc)
N_fs8 = len(z_fs8)

z_max = max(np.max(z_fs8), np.max(z_cc), np.max(data["z"]))
z_grid = np.linspace(0, z_max + 0.1, num=4000)
dz = np.diff(z_grid)


@njit
def w_de_z(z, w0, wa):
    return w0 + wa * z / (1 + z)


@njit
def Ode_z(z, w0, wa):
    return (1. + z)**(3 * (1. + w0 + wa)) * np.exp(-3 * wa * z / (1 + z))


@njit
def H_z(z, params):
    H0, Om, w0, wa = params[0], params[1], params[6], params[7]
    return H0 * np.sqrt(Om * (1. + z) ** 3 + (1. - Om) * Ode_z(z, w0, wa))


@njit
def DM_DH_grid(params):
    dh_grid = c / H_z(z_grid, params)
    dh = (dh_grid[:-1] + dh_grid[1:]) / 2
    cum_dm = np.zeros(len(z_grid))
    cum_dm[1:] = np.cumsum(dh * dz)
    return (cum_dm, dh_grid)


@njit
def DH_z(z, dm_dh_grid):
    return interp_pchip(z, z_grid, dm_dh_grid[1])


@njit
def DM_z(z, dm_dh_grid):
    return interp_hermite(z, x=z_grid, y=dm_dh_grid[0], y_prime=dm_dh_grid[1])


@njit
def DV_z(z, DM, DH):
    return (z * DH * DM**2) ** (1 / 3)


@njit
def dH_da(z, H_val, params):
    H0, Om, w0, wa = params[0], params[1], params[6], params[7]
    Ode = 1.0 - Om
    a = 1 / (1 + z)

    matter = Om * (1.0 + 0.0) * (1.0 + z) ** 3
    dark_eng = Ode * (1.0 + w_de_z(z, w0, wa)) * Ode_z(z, w0, wa)
    numerator = -1.5 * H0**2 * (matter + dark_eng)
    denominator = a * H_val
    return numerator / denominator


Hz_DMz_fid = np.zeros(N_fs8, dtype=np.float64)
for i in range(N_fs8):
    zi = z_fs8[i]
    Om_fid = fs8.data["omega_fid"][i]
    s8_fid = fs8.data["s8_fid"][i]
    H0_fid = fs8.data["H0_fid"][i]
    w0_fid = -1.0
    wa_fid = 0.0
    params = [H0_fid, Om_fid, s8_fid, 1.0, 1.0, 147.0, w0_fid, wa_fid]
    Hz_DMz_fid[i] = H_z(zi, params) * DM_z(np.array([zi]), DM_DH_grid(params))[0]


@njit
def growth_ODE(a, y, params):
    H0, om = params[0], params[1]
    delta, d_delta_da = y

    z = 1 / a - 1
    H_val = H_z(z, params)
    dH_da_val = dH_da(z, H_val, params)

    source = (3 / 2) * (om / a**5) * delta * (H0 / H_val) ** 2
    friction = -(3 / a + dH_da_val / H_val) * d_delta_da
    d2_delta_da = friction + source

    return np.array([d_delta_da, d2_delta_da])


max_z = 200
a_span = np.logspace(np.log10(1 / (1 + max_z)), 0, 2000)


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
    delta_0 = delta[-1]
    sigma8_0 = params[2]
    # f = d(ln delta)/d(ln a) = (a / delta) * d(delta)/da
    # sigma8(z) = sigma8 * delta(z) / delta(z=0)
    return (sigma8_0 / delta_0) * a * interp_pchip(a, a_span, d_delta_da)


@njit
def chi2_fs8(params, dm_dh_grid):
    q = H_z(z_fs8, params) * DM_z(z_fs8, dm_dh_grid) / Hz_DMz_fid
    delta = fs8_values - fs8_theory(a_fs8, params) / q
    y = solve_triangular(L_fs8, delta)
    return np.exp(params[4])**-2 * np.dot(y, y)


@njit
def chi2_cc(params):
    delta = H_values - H_z(z_cc, params)
    y = solve_triangular(L_cc, delta)
    return np.exp(params[3])**-2 * np.dot(y, y)


qty_map = {"DV_over_rs": 0, "DM_over_rs": 1, "DH_over_rs": 2, "F_AP": 3}
desi_qty = np.array([qty_map[q] for q in data["quantity"]], dtype=np.int32)


@njit
def bao_theory(z, qty, rd, dm_dh_grid):
    DM = DM_z(z, dm_dh_grid)
    DH = DH_z(z, dm_dh_grid)

    results = np.empty(z.size, dtype=np.float64)
    DV_mask = qty == 0
    DM_mask = qty == 1
    DH_mask = qty == 2
    FAP_mask = qty == 3
    results[DH_mask] = DH[DH_mask] / rd
    results[DM_mask] = DM[DM_mask] / rd
    results[DV_mask] = DV_z(z[DV_mask], DM[DV_mask], DH[DV_mask]) / rd
    results[FAP_mask] = DM[FAP_mask] / DH[FAP_mask]
    return results


@njit
def chi2_bao(params, dm_dh_grid):
    delta = data["value"] - bao_theory(data["z"], desi_qty, params[5], dm_dh_grid)
    y = solve_triangular(L_bao, delta)
    return np.dot(y, y)


@njit
def chi_squared(params):
    dm_dh_grid = DM_DH_grid(params)
    return chi2_cc(params) + chi2_fs8(params, dm_dh_grid) + chi2_bao(params, dm_dh_grid)


@njit
def log_likelihood_jit(params):
    if params[6] + params[7] >= 0:
        return -np.inf

    norm_cc = N_cc * np.log(2 * np.pi) + logdet_cc + 2 * N_cc * params[3]
    norm_fs8 = N_fs8 * np.log(2 * np.pi) + logdet_fs8 + 2 * N_fs8 * params[4]
    norm_bao = N_bao * np.log(2 * np.pi) + logdet_bao
    return -0.5 * (chi_squared(params) + norm_cc + norm_fs8 + norm_bao)


def log_likelihood(params):
    return log_likelihood_jit(params)


def main():
    from nautilus import Sampler, Prior
    from corner import corner, quantile
    import matplotlib.pyplot as plt
    from multiprocessing import Pool
    from ohd.plot_predictions import plot_cc_predictions
    from fs8.plot_predictions import plot_predictions as plot_fs8_predictions
    from bao.plot_predictions import plot_bao_predictions

    prior = Prior()
    prior.add_parameter("H0", dist=(40, 100))
    prior.add_parameter("Om", dist=(0.1, 0.6))
    prior.add_parameter("sig8", dist=(0.1, 1.5))
    prior.add_parameter("ln_f_cc", dist=(-1.5, 0.5))
    prior.add_parameter("ln_f_fs8", dist=(-1.15, 0.15))
    prior.add_parameter("rd", dist=(110, 180))
    prior.add_parameter("w0", dist=(-3.0, 1.0))
    prior.add_parameter("wa", dist=(-4.0, 4.0))

    with Pool(6) as pool:
        sampler = Sampler(prior, log_likelihood, n_live=5_000, pool=pool, seed=42, pass_dict=False)
        sampler.run(verbose=True)

    samples, log_w, log_l = sampler.posterior()
    w = np.exp(log_w)
    log_evd = sampler.log_z
    one_sigma_ci = [0.159, 0.5, 0.841]

    labels = [
        "$H_0$",
        "$\\Omega_m$",
        "$\\sigma_8$",
        "$ln(f_{cc})$",
        "$ln(f_{fs8})$",
        "$r_{drag}$",
        "$w_0$",
        "$w_a$",
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
    ln_fcc_16, ln_fcc_50, ln_fcc_84 = quantile(samples[:, 3], one_sigma_ci, weights=w)
    ln_fs_16, ln_fs_50, ln_fs_84 = quantile(samples[:, 4], one_sigma_ci, weights=w)
    rd_16, rd_50, rd_84 = quantile(samples[:, 5], one_sigma_ci, weights=w)
    w0_16, w0_50, w0_84 = quantile(samples[:, 6], one_sigma_ci, weights=w)
    wa_16, wa_50, wa_84 = quantile(samples[:, 7], one_sigma_ci, weights=w)

    S8_samples = samples[:, 2] * np.sqrt(samples[:, 1] / 0.3)
    S8_16, S8_50, S8_84 = quantile(S8_samples, one_sigma_ci, weights=w)

    best_fit = samples[np.argmax(log_l)]

    print(f"H0: {H0_50:.1f} +{(H0_84 - H0_50):.1f} -{(H0_50 - H0_16):.1f} km/s/Mpc")
    print(f"Ωm: {Om_50:.3f} +{(Om_84 - Om_50):.3f} -{(Om_50 - Om_16):.3f}")
    print(f"σ8: {sig8_50:.3f} +{(sig8_84 - sig8_50):.3f} -{(sig8_50 - sig8_16):.3f}")
    print(f"S8: {S8_50:.3f} +{(S8_84 - S8_50):.3f} -{(S8_50 - S8_16):.3f}")
    print(f"ln f_cc: {ln_fcc_50:.2f} +{(ln_fcc_84 - ln_fcc_50):.2f} -{(ln_fcc_50 - ln_fcc_16):.2f}")
    print(f"ln f_fs8: {ln_fs_50:.2f} +{(ln_fs_84 - ln_fs_50):.2f} -{(ln_fs_50 - ln_fs_16):.2f}")
    print(f"rd: {rd_50:.1f} +{(rd_84 - rd_50):.1f} -{(rd_50 - rd_16):.1f} Mpc")
    print(f"w0: {w0_50:.3f} +{(w0_84 - w0_50):.3f} -{(w0_50 - w0_16):.3f}")
    print(f"wa: {wa_50:.3f} +{(wa_84 - wa_50):.3f} -{(wa_50 - wa_16):.3f}")
    print(f"Chi squared: {chi_squared(best_fit):.2f}")
    print(f"Log likelihood: {log_likelihood(best_fit):.2f}")
    print(f"Log evidence: {log_evd:.1f}")
    print(f"Degs of freedom: {N_cc + N_fs8 + N_bao - len(best_fit)}")

    dm_dh_grid = DM_DH_grid(best_fit)

    plot_bao_predictions(
        theory_predictions=lambda z, qty: bao_theory(z, qty, best_fit[5], dm_dh_grid),
        data=data,
        errors=np.sqrt(np.diag(bao_cov_matrix)),
        title=f"{bao_legend}: $r_d$={best_fit[5]:.1f} Mpc",
    )
    plot_cc_predictions(
        H_z=lambda z: H_z(z, best_fit),
        z=z_cc,
        H=H_values,
        H_err=np.sqrt(np.diag(cov_matrix)) * np.exp(ln_fcc_50),
        label=f"{cc_legend} $H_0$: {H0_50:.1f} ± {(H0_84 - H0_50):.1f} km/s/Mpc",
    )
    plot_fs8_predictions(
        fs8_theory=lambda z: fs8_theory(1 / (1.0 + z), best_fit),
        data=fs8.data,
        q=H_z(z_fs8, best_fit) * DM_z(z_fs8, dm_dh_grid) / Hz_DMz_fid,
        f_err=1 / np.exp(ln_fs_50),
    )


if __name__ == "__main__":
    main()


# ----------- Flat ΛCDM -----------
# H0: 69.0 +1.7 -1.7 km/s/Mpc
# Ωm: 0.304 +0.007 -0.007
# σ8: 0.790 +0.009 -0.009
# S8: 0.795 +0.011 -0.011
# rd: 146.3 +3.6 -3.5 Mpc
#
# ln(f_cc): -0.31 +0.12 -0.11
# ln(f_fs8): -0.57 +0.10 -0.09
#
# Chi squared: 108.42
# Log likelihood: -46.63
# Log evidence: -64.7
# Degs of freedom: 103
# ---------------------------------


# ----------- Flat wCDM -----------
# H0: 67.3 +1.8 -1.8 km/s/Mpc
# Ωm: 0.304 +0.007 -0.007
# σ8: 0.820 +0.019 -0.017
# S8: 0.826 +0.018 -0.018
# rd: 146.4 +3.7 -3.5 Mpc
# w: -0.878 +0.055 -0.056 (prior U[-1.5, -0.5])
#
# ln(f_cc): -0.30 +0.12 -0.11
# ln(f_fs8): -0.62 +0.10 -0.10
#
# Chi squared: 110.25
# Log likelihood: -44.34
# Log evidence: -64.3
# Degs of freedom: 102
# ---------------------------------


# ---------- Flat w0waCDM ---------
# w0 + wa < 0 enforced in the likelihood
#
# H0: 65.1 +2.1 -2.0 km/s/Mpc
# Ωm: 0.351 +0.018 -0.019
# σ8: 0.801 +0.015 -0.013
# S8: 0.865 +0.025 -0.025
# rd: 146.4 +3.8 -3.6 Mpc
# w0: -0.549 +0.157 -0.157 (prior U[-3, 1])
# wa: -1.50 +0.64 -0.63 (prior U[-4, 4])
#
# ln(f_cc): -0.28 +0.12 -0.11
# ln(f_fs8): -0.64 +0.10 -0.09
#
# Chi squared: 100.52
# Log likelihood: -41.53
# Log evidence: -64.7
# Degs of freedom: 101
# ---------------------------------
