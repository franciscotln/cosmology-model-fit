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

z_max = max(np.max(z_fs8), np.max(z_cc), np.max(data["z"]))
z_grid = np.linspace(0, z_max + 0.1, num=4000)
dz = np.diff(z_grid)


@njit
def H_z(z, params):
    H0, Om, w0 = params[0], params[1], params[-1]
    cubic = (1 + z) ** 3
    rho_de = (2 * cubic / (1 + w0 + (1 - w0) * cubic)) ** 2
    return H0 * np.sqrt(Om * (1 + z) ** 3 + (1 - Om) * rho_de)


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
    a = 1 / (1 + z)
    dz = 1e-06
    H_plus = H_z(z + dz, params)
    H_minus = H_z(z - dz, params)
    dH_dz = (H_plus - H_minus) / (2 * dz)
    return -dH_dz / a**2


denominator_fiducial = np.zeros(len(z_fs8), dtype=np.float64)

for i in range(len(z_fs8)):
    zi = z_fs8[i]
    Om_fid = fs8.data["omega_fid"][i]
    params = [67.5, Om_fid, 0.8, 1.0, 1.0, 147.0, -1.0]
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


qty_map = {"DV_over_rs": 0, "DM_over_rs": 1, "DH_over_rs": 2}
quantities = np.array([qty_map[q] for q in data["quantity"]], dtype=np.int32)


@njit
def theory_bao(z, qty, params):
    results = np.empty(z.size, dtype=np.float64)
    DV_mask = qty == 0
    DM_mask = qty == 1
    DH_mask = qty == 2
    results[DH_mask] = DH_z(z[DH_mask], params)
    results[DM_mask] = DM_z(z[DM_mask], params)
    results[DV_mask] = DV_z(z[DV_mask], params)
    return results / params[5]


def chi_squared(params):
    f_cc, f_fs8 = params[3], params[4]

    q = H_z(z_fs8, params) * DM_z(z_fs8, params) / denominator_fiducial
    delta_fs8 = fs8_values - fs8_theory(z_fs8, params) / q
    chi2_fs8 = f_fs8**2 * np.dot(delta_fs8, np.dot(inv_cov_fs8, delta_fs8))

    delta_cc = H_values - H_z(z_cc, params)
    chi2_cc = f_cc**2 * np.dot(delta_cc, np.dot(inv_cov_cc, delta_cc))

    delta_bao = data["value"] - theory_bao(data["z"], quantities, params)
    chi_bao = np.dot(delta_bao, np.dot(inv_cov_bao, delta_bao))

    return chi2_cc + chi2_fs8 + chi_bao


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
    from cosmic_chronometers.plot_predictions import plot_cc_predictions
    from bao.plot_predictions import plot_bao_predictions

    prior = Prior()
    prior.add_parameter("H0", dist=(40.0, 100.0))
    prior.add_parameter("Om", dist=(0.1, 0.6))
    prior.add_parameter("sig8", dist=(0.2, 1.2))
    prior.add_parameter("f_cc", dist=(0.3, 2.6))
    prior.add_parameter("f_fs8", dist=(0.5, 2.2))
    prior.add_parameter("rd", dist=(110.0, 180.0))
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
    rd_16, rd_50, rd_84 = quantile(samples[:, 5], one_sigma_ci, weights=w)
    w0_16, w0_50, w0_84 = quantile(samples[:, 6], one_sigma_ci, weights=w)

    S8_samples = samples[:, 2] * np.sqrt(samples[:, 1] / 0.3)
    S8_16, S8_50, S8_84 = quantile(S8_samples, one_sigma_ci, weights=w)

    best_fit = [H0_50, Om_50, sig8_50, fcc_50, fs_50, rd_50 , w0_50]

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
    print(f"Degs of freedom: {len(z_cc) + len(z_fs8) - len(best_fit)}")

    plot_bao_predictions(
        theory_predictions=lambda z, qty: theory_bao(z, qty, best_fit),
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

    z_plot = np.linspace(0, np.max(z_fs8), 200)
    fs8_plot = fs8_theory(z_plot, best_fit)
    q = H_z(z_fs8, best_fit) * DM_z(z_fs8, best_fit) / denominator_fiducial
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
H0: 69.2 +2.3 -2.3 km/s/Mpc
Ωm: 0.295 +0.008 -0.008
σ8: 0.778 +0.011 -0.011
S8: 0.772 +0.013 -0.013
f_cc: 1.48 +0.18 -0.17
f_fs8: 1.31 +0.12 -0.12
rd: 146.9 +4.9 -4.6 Mpc
Chi squared: 107.83
Log likelihood: -46.24
Log evidence: -63.1
Degs of freedom: 93
"""

"""
Flat wCDM: w(z) = w0
H0: 67.5 +2.5 -2.4 km/s/Mpc
Ωm: 0.293 +0.008 -0.008
σ8: 0.808 +0.022 -0.021
S8: 0.798 +0.020 -0.019
f_cc: 1.48 +0.18 -0.17
f_fs8: 1.34 +0.12 -0.12
rd: 147.3 +5.0 -4.6 Mpc
w0: -0.880 +0.064 -0.066 (prior -1.4 to -0.4)
Chi squared: 106.65
Log likelihood: -44.62
Log evidence: -63.3
Degs of freedom: 92
"""

"""
Flat wzCDM: w(z) = -1 + 2 * (1 + w0) / (1 + w0 + (1 - w0) * (1 + z)^3)
H0: 66.8 +2.5 -2.5 km/s/Mpc
Ωm: 0.307 +0.010 -0.010
σ8: 0.799 +0.016 -0.015
S8: 0.807 +0.023 -0.022
f_cc: 1.47 +0.18 -0.17
f_fs8: 1.33 +0.12 -0.12
rd: 147.2 +5.0 -4.7 Mpc
w0: -0.782 +0.104 -0.107 (prior -1.0 to 0.0)
wa: -1.5 * (1 - w0^2)
Chi squared: 105.64
Log likelihood: -44.60
Log evidence: -62.8
Degs of freedom: 92
"""

"""
Flat w0waCDM: w(z) = w0 + wa * z / (1 + z)
TODO
"""
