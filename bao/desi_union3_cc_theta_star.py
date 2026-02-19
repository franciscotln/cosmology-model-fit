from numba import njit
import numpy as np
import cmb.data_planck_act_compression as cmb
from interpolator import interp_hermite
from y2026union3_1.data import get_data as get_sn_data
from y2005cc.data import get_data as get_cc_data
from y2025BAO.data import get_data as get_bao_data

cc_legend, z_cc_vals, H_cc_vals, cov_matrix_cc = get_cc_data()
sn_legend, z_cmb, z_hel, mu_values, cov_matrix_sn = get_sn_data()
bao_legend, bao_data, cov_matrix_bao = get_bao_data()

inv_cov_sn = np.linalg.inv(cov_matrix_sn)
inv_cov_bao = np.linalg.inv(cov_matrix_bao)
inv_cov_cc = np.linalg.inv(cov_matrix_cc)

logdet_cc = np.linalg.slogdet(cov_matrix_cc)[1]
N_cc = len(z_cc_vals)

c = cmb.c  # km/s
Or_h2 = cmb.Or_h2
Omnu_h2 = cmb.Omnu_h2

z_max = max(np.max(z_cmb), np.max(bao_data["z"])) + 0.1
z_grid = np.linspace(0, z_max, num=4000)
dz = np.diff(z_grid)


@njit
def Ode_z(z, w0):
    zp1 = 1.0 + z
    cubed = zp1**3
    # return 1.0  # ΛCDM
    # return cubed ** (1.0 + w0)  # wCDM
    return (2 * cubed / (1.0 + w0 + (1.0 - w0) * cubed)) ** 2  # wzCDM
    # return cubed ** (1.0 + w0 + wa) * np.exp(-3 * wa * z / zp1)  # w0waCDM


@njit
def Ez(z, H0, Obh2, Och2, w0):
    h = H0 / 100
    Omnu = Omnu_h2 / h**2
    Or = Or_h2 / h**2
    Ombc = (Obh2 + Och2) / h**2
    Ode = 1.0 - Ombc - Or - Omnu

    radiation_term = Or * (1.0 + z) ** 4
    matter_term = Ombc * (1.0 + z) ** 3
    neutrino_term = Omnu * cmb.Omnu_z(z)
    dark_energy_term = Ode * Ode_z(z, w0)

    return np.sqrt(radiation_term + matter_term + dark_energy_term + neutrino_term)


@njit
def H_z(z, theta):
    H0, Obh2, Och2, w0 = theta[2:]
    return H0 * Ez(z, H0, Obh2, Och2, w0)


cmb.set_HZ(H_z)


@njit
def DH_z(z, theta):
    return c / H_z(z, theta)


@njit
def DM_z(z, theta):
    dh_grid = DH_z(z_grid, theta)
    dh = (dh_grid[:-1] + dh_grid[1:]) / 2
    cum_dm = np.zeros(z_grid.size, dtype=np.float64)
    cum_dm[1:] = np.cumsum(dh * dz)
    return interp_hermite(z, z_grid, cum_dm, dh_grid)


@njit
def DV_z(z, theta):
    DH = DH_z(z, theta)
    DM = DM_z(z, theta)
    return (z * DH * DM**2) ** (1 / 3)


qty_map = {"DV_over_rs": 0, "DM_over_rs": 1, "DH_over_rs": 2}
quantities = np.array([qty_map[q] for q in bao_data["quantity"]], dtype=np.int32)


@njit
def bao_theory(z, qty, theta):
    Obh2, Och2 = theta[3], theta[4]
    rd = cmb.r_drag(wb=Obh2, wm=Obh2 + Och2 + Omnu_h2)

    DV_mask = qty == 0
    DM_mask = qty == 1
    DH_mask = qty == 2
    results = np.empty(z.size, dtype=np.float64)
    results[DH_mask] = DH_z(z[DH_mask], theta)
    results[DM_mask] = DM_z(z[DM_mask], theta)
    results[DV_mask] = DV_z(z[DV_mask], theta)
    return results / rd


@njit
def mu_theory(theta):
    dL = (1.0 + z_hel) * DM_z(z_cmb, theta)
    return theta[1] + 25.0 + 5 * np.log10(dL)


def chi_squared(theta):
    delta = (cmb.DISTANCE_PRIORS - cmb.cmb_distances(theta[3], theta[4], theta))[1]
    thetastar_cov = cmb.covariance[1, 1]
    chi_theta_star = delta**2 / thetastar_cov

    delta_sn = mu_values - mu_theory(theta)
    chi_sn = delta_sn @ inv_cov_sn @ delta_sn

    delta_bao = bao_data["value"] - bao_theory(bao_data["z"], quantities, theta)
    chi_bao = delta_bao @ inv_cov_bao @ delta_bao

    delta_cc = H_cc_vals - H_z(z_cc_vals, theta)
    chi_cc = delta_cc @ inv_cov_cc @ delta_cc * theta[0] ** 2

    return chi_theta_star + chi_sn + chi_bao + chi_cc


def log_likelihood(theta):
    f_cc = theta[0]
    normalization_cc = N_cc * np.log(2 * np.pi) + logdet_cc - 2 * N_cc * np.log(f_cc)
    return -0.5 * chi_squared(theta) - 0.5 * normalization_cc


def main():
    from corner import corner, quantile
    import matplotlib.pyplot as plt
    from nautilus import Sampler, Prior
    from multiprocessing import Pool
    from sn.plotting import plot_predictions as plot_sn_predictions
    from cosmic_chronometers.plot_predictions import plot_cc_predictions
    from bao.plot_predictions import plot_bao_predictions

    prior = Prior()
    # f_cc: CC error rescaling (overestimated)
    prior.add_parameter("f_cc", dist=(0.2, 3.0))
    # ΔM: magnitude offset
    prior.add_parameter("ΔM", dist=(-1.0, 1.0))
    # H0: Hubble constant at present
    prior.add_parameter("H0", dist=(50.0, 85.0))
    # Ωb x h^2: baryon density parameter
    prior.add_parameter("ωb", dist=(0.003, 0.050))
    # Ωc x h^2: cold dark matter density param today
    prior.add_parameter("ωc", dist=(0.05, 0.30))
    # w0: dark energy equation of state today
    prior.add_parameter("w0", dist=(-1.0, -1 / 3))

    with Pool(8) as pool:
        sampler = Sampler(
            prior, log_likelihood, n_live=8_000, pool=pool, seed=42, pass_dict=False
        )
        sampler.run(verbose=True)

    samples, log_w, log_l = sampler.posterior()
    w = np.exp(log_w)

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

    fcc_16, fcc_50, fcc_84 = quantile(samples[:, 0], one_sigma_ci, weights=w)
    dM_16, dM_50, dM_84 = quantile(samples[:, 1], one_sigma_ci, weights=w)
    h0_16, h0_50, h0_84 = quantile(samples[:, 2], one_sigma_ci, weights=w)
    wb_16, wb_50, wb_84 = quantile(samples[:, 3], one_sigma_ci, weights=w)
    wc_16, wc_50, wc_84 = quantile(samples[:, 4], one_sigma_ci, weights=w)
    w0_16, w0_50, w0_84 = quantile(samples[:, 5], one_sigma_ci, weights=w)

    best_fit = [fcc_50, dM_50, h0_50, wb_50, wc_50, w0_50]

    deg_of_freedom = (
        1 + len(z_cmb) + len(bao_data["z"]) + len(z_cc_vals) - len(prior.keys)
    )

    Omh2_samples = samples[:, 3] + samples[:, 4] + Omnu_h2
    Om_samples = Omh2_samples / (samples[:, 2] / 100) ** 2
    r_d_samples = cmb.r_drag(samples[:, 3], Omh2_samples)
    rd_16, rd_50, rd_84 = quantile(r_d_samples, one_sigma_ci, weights=w)
    Omh2_16, Omh2_50, Omh2_84 = quantile(Omh2_samples, one_sigma_ci, weights=w)
    Om_16, Om_50, Om_84 = quantile(Om_samples, one_sigma_ci, weights=w)
    print(f"f_cc: {fcc_50:.2f} +{(fcc_84 - fcc_50):.2f} -{(fcc_50 - fcc_16):.2f}")
    print(f"ΔM: {dM_50:.3f} +{(dM_84 - dM_50):.3f} -{(dM_50 - dM_16):.3f} mag")
    print(f"H0: {h0_50:.1f} +{(h0_84 - h0_50):.1f} -{(h0_50 - h0_16):.1f} km/s/Mpc")
    print(f"ωb: {wb_50:.4f} +{(wb_84 - wb_50):.4f} -{(wb_50 - wb_16):.4f} Mpc")
    print(f"ωc: {wc_50:.4f} +{(wc_84 - wc_50):.4f} -{(wc_50 - wc_16):.4f}")
    print(f"ωm: {Omh2_50:.4f} +{(Omh2_84 - Omh2_50):.4f} -{(Omh2_50 - Omh2_16):.4f}")
    print(f"Ωm: {Om_50:.3f} +{(Om_84 - Om_50):.3f} -{(Om_50 - Om_16):.3f}")
    print(f"w0: {w0_50:.3f} +{(w0_84 - w0_50):.3f} -{(w0_50 - w0_16):.3f}")
    print(f"r_d: {rd_50:.1f} +{(rd_84 - rd_50):.1f} -{(rd_50 - rd_16):.1f} Mpc")
    print(f"Chi squared: {chi_squared(best_fit):.2f}")
    print(f"Log evidence: {sampler.log_z:.2f}")
    print(f"Degrees of freedom: {deg_of_freedom}")

    plot_bao_predictions(
        theory_predictions=lambda z, qty: bao_theory(z, qty, best_fit),
        data=bao_data,
        errors=np.sqrt(np.diag(cov_matrix_bao)),
        title=bao_legend,
    )
    plot_cc_predictions(
        H_z=lambda z: H_z(z, best_fit),
        z=z_cc_vals,
        H=H_cc_vals,
        H_err=np.sqrt(np.diag(cov_matrix_cc)) / fcc_50,
        label=f"{cc_legend} $H_0$: {h0_50:.1f} km/s/Mpc",
    )
    plot_sn_predictions(
        legend=sn_legend,
        x=z_cmb,
        y=mu_values,
        y_err=np.sqrt(np.diag(cov_matrix_sn)),
        y_model=mu_theory(best_fit),
        label=rf"$Ω_m$={Om_50:.3f}",
        x_scale="log",
    )


if __name__ == "__main__":
    main()


"""
Priors:
f_cc ~U(0.2, 3.0)
ΔM   ~U(-1.0, +1.0)
H0   ~U(50.0, 85.0)
ωb   ~U(0.003, 0.050)
ωc   ~U(0.05, 0.30)

wCDM:
w0   ~U(-1.5, -0.5)

wzCDM:
w0   ~U(-1.0, -1 / 3)

w0waCDM:
w0   ~U(-1.5, 0.0)
wa   ~U(-3.5, +2.0)
w0 + wa < 0 enforced

M(z):
p    ~U(-1.0, 2.5)
"""

"""
Flat ΛCDM: w(z) = -1
f_cc: 1.49 +0.18 -0.17
ΔM: -0.068 +0.042 -0.039 mag
H0: 67.9 +1.5 -1.4 km/s/Mpc
ωb: 0.0214 +0.0021 -0.0019 Mpc
ωc: 0.1162 +0.0012 -0.0010
ωm: 0.1382 +0.0031 -0.0027
Ωm: 0.300 +0.008 -0.007
r_d: 149.2 +2.5 -2.6 Mpc
Chi squared: 77.40
Log evidence: -181.90
Degrees of freedom: 67
"""

"""
Flat ΛCDM: w(z) = -1
Evolving absolute mag of SNe M(z) = ΔM_max + 0.2 * p / (1 + (z / 0.043))

f_cc: 1.49 +0.18 -0.17
ΔM_max: -0.064 +0.042 -0.039 mag
p: 0.685 +0.297 -0.295
H0: 68.6 +1.5 -1.4 km/s/Mpc
ωb: 0.0224 +0.0022 -0.0020 Mpc
ωc: 0.1163 +0.0013 -0.0011
ωm: 0.1393 +0.0034 -0.0029
Ωm: 0.296 +0.008 -0.007
r_d: 148.1 +2.5 -2.7 Mpc
Chi squared: 72.19 (2.28 sigmas away from constant M)
Log evidence: -180.79 (Δ logZ = 1.11 against constant M)
Degrees of freedom: 66
"""

"""
Flat wCDM: w(z) = w0
f_cc: 1.49 +0.18 -0.17
ΔM: -0.018 +0.052 -0.048 mag
H0: 68.6 +1.7 -1.5 km/s/Mpc
ωb: 0.0253 +0.0031 -0.0028 Mpc
ωc: 0.1150 +0.0016 -0.0015
ωm: 0.1408 +0.0042 -0.0035
Ωm: 0.300 +0.007 -0.007
w0: -0.910 +0.040 -0.041
r_d: 145.3 +3.2 -3.4 Mpc
Chi squared: 72.92 (2.12 sigmas away from ΛCDM)
Log evidence: -181.85 (Δ logZ = 0.05 against ΛCDM)
Degrees of freedom: 66
"""

"""
Flat wzCDM: w(z) = -1 + 2 * (1 + w0) / (1 + w0 + (1 - w0) * (1 + z)^3)
f_cc: 1.49 +0.18 -0.17
ΔM: -0.033 +0.047 -0.043 mag
H0: 67.6 +1.6 -1.4 km/s/Mpc
ωb: 0.0242 +0.0026 -0.0024 Mpc
ωc: 0.1160 +0.0015 -0.0013
ωm: 0.1408 +0.0039 -0.0033
Ωm: 0.308 +0.008 -0.008
w0: -0.827 +0.067 -0.070
r_d: 146.2 +2.9 -3.1 Mpc
Chi squared: 71.44 (2.44 sigmas away from ΛCDM)
Log evidence: -180.31 (Δ logZ = 1.59 against ΛCDM)
Degrees of freedom: 66
"""

"""
Flat w0waCDM: w(z) = w0 + wa * z / (1 + z)
f_cc: 1.48 +0.18 -0.17
ΔM: -0.057 +0.057 -0.051 mag
H0: 66.7 +2.1 -1.8 km/s/Mpc
ωb: 0.0223 +0.0037 -0.0031 Mpc
ωc: 0.1175 +0.0019 -0.0023
ωm: 0.1401 +0.0037 -0.0030
Ωm: 0.315 +0.014 -0.014
w0: -0.810 +0.097 -0.091
wa: -0.530 +0.426 -0.469
r_d: 147.9 +3.5 -3.8 Mpc
Chi squared: 70.91 (2.06 sigmas away from ΛCDM)
Log evidence: -183.16 (Δ logZ = -1.26 in favour of ΛCDM)
Degrees of freedom: 65
"""
