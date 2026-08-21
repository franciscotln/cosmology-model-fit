from numba import njit
import numpy as np
from scipy.constants import c as c0
from scipy.integrate import solve_ivp
from interpolator import interp_hermite, interp_pchip
from y2025BAO.data_fs_lya import get_data as get_bao_data
from y2005cc.data import get_data as get_cc_data
import y2018fs8.data as fs8

c = c0 / 1000  # Speed of light in km/s

bao_legend, data, bao_cov_matrix = get_bao_data()
cc_legend, z_cc, H_values, cov_matrix = get_cc_data()

z_fs8, fs8_values = fs8.data["z"], fs8.data["fs8"]
a_fs8 = 1 / (1.0 + z_fs8)

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
    return interp_hermite(z, z_grid, cum_dm, dh_grid)


@njit
def DV_z(z, params):
    DH = DH_z(z, params)
    DM = DM_z(z, params)
    return (z * DH * DM**2) ** (1 / 3)


@njit
def dH_da(z, H_val, params):
    H0, Om, w0 = params[0], params[1], params[-1]
    Ode = 1.0 - Om
    a = 1 / (1 + z)

    matter = 3 * Om * (1.0 + 0.0) * (1.0 + z) ** 3
    dark_eng = 3 * Ode * (1.0 + w_de_z(z, w0)) * Ode_z(z, w0)
    numerator = -0.5 * H0**2 * (matter + dark_eng)
    denominator = a * H_val
    return numerator / denominator


Hz_DMz_fid = np.zeros(N_fs8, dtype=np.float64)
for i in range(N_fs8):
    zi = z_fs8[i]
    Om_fid = fs8.data["omega_fid"][i]
    s8_fid = fs8.data["s8_fid"][i]
    H0_fid = fs8.data["H0_fid"][i]
    w0_fid = -1.0
    params = [H0_fid, Om_fid, s8_fid, 1.0, 1.0, 147.0, w0_fid]
    Hz_DMz_fid[i] = H_z(zi, params) * DM_z(np.array([zi]), params)[0]


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

    return [d_delta_da, d2_delta_da]


max_z = 200
a_span = np.logspace(np.log10(1 / (1 + max_z)), 0, 2000)


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


def chi2_fs8(params):
    q = H_z(z_fs8, params) * DM_z(z_fs8, params) / Hz_DMz_fid
    delta = fs8_values - fs8_theory(a_fs8, params) / q
    return params[4] ** 2 * delta @ inv_cov_fs8 @ delta


@njit
def chi2_cc(params):
    delta = H_values - H_z(z_cc, params)
    return params[3] ** 2 * delta @ inv_cov_cc @ delta


qty_map = {"DV_over_rs": 0, "DM_over_rs": 1, "DH_over_rs": 2, "F_AP": 3}
desi_qty = np.array([qty_map[q] for q in data["quantity"]], dtype=np.int32)


@njit
def bao_theory(z, qty, params):
    results = np.empty(z.size, dtype=np.float64)
    DV_mask = qty == 0
    DM_mask = qty == 1
    DH_mask = qty == 2
    FAP_mask = qty == 3
    rd = params[5]
    results[DH_mask] = DH_z(z[DH_mask], params) / rd
    results[DM_mask] = DM_z(z[DM_mask], params) / rd
    results[DV_mask] = DV_z(z[DV_mask], params) / rd
    results[FAP_mask] = DM_z(z[FAP_mask], params) / DH_z(z[FAP_mask], params)
    return results


@njit
def chi2_bao(params):
    delta = data["value"] - bao_theory(data["z"], desi_qty, params)
    return delta @ inv_cov_bao @ delta


def chi_squared(params):
    return chi2_cc(params) + chi2_fs8(params) + chi2_bao(params)


def log_likelihood(params):
    normalization_cc = (
        N_cc * np.log(2 * np.pi) + logdet_cc - 2 * N_cc * np.log(params[3])
    )

    normalization_fs8 = (
        N_fs8 * np.log(2 * np.pi) + logdet_fs8 - 2 * N_fs8 * np.log(params[4])
    )

    return -0.5 * (chi_squared(params) + normalization_cc + normalization_fs8)


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
    prior.add_parameter("f_cc", dist=(0.03, 3.0))
    prior.add_parameter("f_fs8", dist=(0.2, 3.0))
    prior.add_parameter("rd", dist=(110, 180))
    prior.add_parameter("w0", dist=(-1.0, 0.0))

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
        "$\\Omega_m$",
        "$\\sigma_8$",
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
    fcc_16, fcc_50, fcc_84 = quantile(samples[:, 3], one_sigma_ci, weights=w)
    fs_16, fs_50, fs_84 = quantile(samples[:, 4], one_sigma_ci, weights=w)
    rd_16, rd_50, rd_84 = quantile(samples[:, 5], one_sigma_ci, weights=w)
    w0_16, w0_50, w0_84 = quantile(samples[:, 6], one_sigma_ci, weights=w)

    S8_samples = samples[:, 2] * np.sqrt(samples[:, 1] / 0.3)
    S8_16, S8_50, S8_84 = quantile(S8_samples, one_sigma_ci, weights=w)

    best_fit = [H0_50, Om_50, sig8_50, fcc_50, fs_50, rd_50, w0_50]

    print(f"H0: {H0_50:.1f} +{(H0_84 - H0_50):.1f} -{(H0_50 - H0_16):.1f} km/s/Mpc")
    print(f"Ωm: {Om_50:.3f} +{(Om_84 - Om_50):.3f} -{(Om_50 - Om_16):.3f}")
    print(f"σ8: {sig8_50:.3f} +{(sig8_84 - sig8_50):.3f} -{(sig8_50 - sig8_16):.3f}")
    print(f"S8: {S8_50:.3f} +{(S8_84 - S8_50):.3f} -{(S8_50 - S8_16):.3f}")
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
        fs8_theory=lambda z: fs8_theory(1 / (1.0 + z), best_fit),
        data=fs8.data,
        q=H_z(z_fs8, best_fit) * DM_z(z_fs8, best_fit) / Hz_DMz_fid,
        f_err=fs_50,
    )


if __name__ == "__main__":
    main()


# ----------- Flat ΛCDM -----------
# H0: 68.6 +2.2 -2.2 km/s/Mpc
# Ωm: 0.304 +0.007 -0.007
# σ8: 0.790 +0.009 -0.009
# S8: 0.795 +0.011 -0.011
# f_cc: 1.50 +0.18 -0.17
# f_fs8: 1.79 +0.17 -0.17
# rd: 147.3 +4.9 -4.5 Mpc
# Chi squared: 105.38
# Log likelihood: -49.13
# Log evidence: -67.1
# Degs of freedom: 102
# ---------------------------------


# ----------- Flat wCDM -----------
# H0: 66.7 +2.3 -2.3 km/s/Mpc
# Ωm: 0.304 +0.007 -0.007
# σ8: 0.820 +0.019 -0.018
# S8: 0.826 +0.018 -0.018
# f_cc: 1.50 +0.18 -0.17
# f_fs8: 1.88 +0.19 -0.18
# rd: 147.7 +4.9 -4.5 Mpc
# w0: -0.878 +0.055 -0.056 (prior U[-1.5, -0.5])
# Chi squared: 106.05
# Log likelihood: -46.83 (2.14 sigma significance)
# Log evidence: -66.7 (Δ logZ = 0.4 against ΛCDM)
# Degs of freedom: 101
# ---------------------------------


# ----------- Flat wzCDM ----------
# w(z) = -1 + 2 * (1 + w0) / (1 + w0 + (1 - w0) * (1 + z)^3)
# H0: 65.6 +2.4 -2.4 km/s/Mpc
# Ωm: 0.320 +0.009 -0.009
# σ8: 0.818 +0.014 -0.014
# S8: 0.845 +0.021 -0.021
# f_cc: 1.49 +0.18 -0.17
# f_fs8: 1.91 +0.19 -0.18
# rd: 147.7 +4.9 -4.6 Mpc
# w0: -0.731 +0.085 -0.094 (prior U[-1, 0])
# Chi squared: 104.70
# Log likelihood: -45.45 (2.71 sigma significance)
# Log evidence: -64.9 (Δ logZ = 2.2 against ΛCDM)
# Degs of freedom: 101
# ---------------------------------


# ---------- Flat w0waCDM ---------
# TODO
# ---------------------------------
