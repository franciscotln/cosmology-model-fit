import numpy as np
import emcee
import corner
from scipy.integrate import cumulative_trapezoid, quad
from scipy.linalg import cho_factor, cho_solve
import matplotlib.pyplot as plt
from multiprocessing import Pool
from y2022pantheonSHOES.data import get_data
from y2005cc.data import get_data as get_cc_data
from y2025BAO.data import get_data as get_bao_data
from hubble.plotting import plot_predictions as plot_sn_predictions

cc_legend, z_cc_vals, H_cc_vals, cov_matrix_cc = get_cc_data()
legend, z_sn_vals, z_sn_hel_vals, apparent_mag_values, cov_matrix_sn = get_data()
bao_legend, bao_data, cov_matrix_bao = get_bao_data()
cho_cc = cho_factor(cov_matrix_cc)
cho_sn = cho_factor(cov_matrix_sn)
cho_bao = cho_factor(cov_matrix_bao)

c = 299792.458  # Speed of light in km/s


def h_over_h0(z, params):
    O_m, w0 = params[3], params[4]
    one_plus_z = 1 + z
    evolving_de = ((2 * one_plus_z**2) / (1 + one_plus_z**2)) ** (3 * (1 + w0))
    return np.sqrt(O_m * one_plus_z**3 + (1 - O_m) * evolving_de)


z_grid = np.linspace(0, np.max(z_sn_vals), num=2500)


def integral_e_z(params):
    integral_values = cumulative_trapezoid(
        1 / h_over_h0(z_grid, params), z_grid, initial=0
    )
    return np.interp(z_sn_vals, z_grid, integral_values)


def sn_apparent_mag(params):
    h0, M = params[0], params[1]
    comoving_distance = (c / h0) * integral_e_z(params)
    return M + 25 + 5 * np.log10((1 + z_sn_hel_vals) * comoving_distance)


def plot_bao_predictions(params):
    z_values, observed_values, quantity_types = (
        bao_data["z"],
        bao_data["value"],
        bao_data["quantity"],
    )
    errors = np.sqrt(np.diag(cov_matrix_bao))

    unique_quantities = set(quantity_types)
    colors = {"DV_over_rs": "red", "DM_over_rs": "blue", "DH_over_rs": "green"}

    h0, r_d, omega_m = params[0], params[2], params[3]
    z_smooth = np.linspace(0, max(z_values), 100)
    plt.figure(figsize=(8, 6))
    for q in unique_quantities:
        mask = quantity_types == q
        plt.errorbar(
            x=z_values[mask],
            y=observed_values[mask],
            yerr=errors[mask],
            fmt=".",
            color=colors[q],
            label=q,
            capsize=2,
            linestyle="None",
        )
        model_smooth = []
        for z in z_smooth:
            if q == "DV_over_rs":
                model_smooth.append(DV_z(z, params) / r_d)
            elif q == "DM_over_rs":
                model_smooth.append(DM_z(z, params) / r_d)
            elif q == "DH_over_rs":
                model_smooth.append(DH_z(z, params) / r_d)
        plt.plot(z_smooth, model_smooth, color=colors[q], alpha=0.5)

    plt.xlabel("Redshift (z)")
    plt.ylabel(r"$O = \frac{D}{r_d}$")
    plt.legend()
    plt.grid(True)
    plt.title(
        f"{bao_legend}: $r_d$={r_d:.2f} Mpc, $\Omega_M$={omega_m:.4f}, $H_0$={h0:.2f} km/s/Mpc"
    )
    plt.show()

    plt.errorbar(
        x=z_cc_vals,
        y=H_cc_vals,
        yerr=np.sqrt(np.diag(cov_matrix_cc)),
        fmt=".",
        color="blue",
        alpha=0.4,
        label="CCH data",
        capsize=2,
        linestyle="None",
    )
    plt.plot(z_smooth, H_z(z_smooth, params), color="red", alpha=0.5, label="Model")
    plt.xlabel("Redshift (z)")
    plt.ylabel(r"$H(z)$")
    plt.xlim(0, np.max(z_cc_vals) + 0.2)
    plt.legend()
    plt.title(f"{cc_legend}: $H_0$={h0:.2f} km/s/Mpc")
    plt.show()


def H_z(z, params):
    return params[0] * h_over_h0(z, params)


def DH_z(z, params):
    return c / H_z(z, params)


def DM_z(z, params):
    return quad(lambda zp: DH_z(zp, params), 0, z)[0]


def DV_z(z, params):
    DH = DH_z(z, params)
    DM = DM_z(z, params)
    return (z * DH * DM**2) ** (1 / 3)


quantity_funcs = {
    "DV_over_rs": DV_z,
    "DM_over_rs": DM_z,
    "DH_over_rs": DH_z,
}


def bao_predictions(params):
    return np.array(
        [(quantity_funcs[qty](z, params) / params[2]) for z, _, qty in bao_data]
    )


bounds = np.array(
    [
        (50, 80),  # H0
        (-20, -19),  # M
        (115, 170),  # r_d
        (0.15, 0.7),  # Ωm
        (-3, 0),  # w0
    ]
)


def chi_squared(params):
    delta_sn = apparent_mag_values - sn_apparent_mag(params)
    chi_sn = np.dot(delta_sn, cho_solve(cho_sn, delta_sn))

    delta_bao = bao_data["value"] - bao_predictions(params)
    chi_bao = np.dot(delta_bao, cho_solve(cho_bao, delta_bao))

    delta_cc = H_cc_vals - H_z(z_cc_vals, params)
    chi_cc = np.dot(delta_cc, cho_solve(cho_cc, delta_cc))

    return chi_sn + chi_bao + chi_cc


def log_prior(params):
    if np.all((bounds[:, 0] < params) & (params < bounds[:, 1])):
        return 0.0
    return -np.inf


def log_likelihood(params):
    return -0.5 * chi_squared(params)


def log_probability(params):
    lp = log_prior(params)
    if np.isinf(lp):
        return -np.inf
    return lp + log_likelihood(params)


def main():
    ndim = len(bounds)
    nwalkers = 10 * ndim
    burn_in = 500
    nsteps = 10000 + burn_in
    initial_pos = np.random.default_rng().uniform(
        bounds[:, 0], bounds[:, 1], size=(nwalkers, ndim)
    )

    with Pool(10) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool=pool)
        sampler.run_mcmc(initial_pos, nsteps, progress=True)

    try:
        tau = sampler.get_autocorr_time()
        print("auto-correlation time", tau)
    except emcee.autocorr.AutocorrError as e:
        print("Autocorrelation time could not be computed", e)

    samples = sampler.get_chain(discard=burn_in, flat=True)

    [
        [h0_16, h0_50, h0_84],
        [M_16, M_50, M_84],
        [rd_16, rd_50, rd_84],
        [omega_16, omega_50, omega_84],
        [w0_16, w0_50, w0_84],
    ] = np.percentile(samples, [15.9, 50, 84.1], axis=0).T

    best_fit = [h0_50, M_50, rd_50, omega_50, w0_50]

    deg_of_freedom = (
        z_sn_vals.size + bao_data["value"].size + z_cc_vals.size - len(best_fit)
    )

    print(f"H0: {h0_50:.2f} +{(h0_84 - h0_50):.2f} -{(h0_50 - h0_16):.2f}")
    print(f"M: {M_50:.3f} +{(M_84 - M_50):.3f} -{(M_50 - M_16):.3f}")
    print(f"r_d: {rd_50:.2f} +{(rd_84 - rd_50):.2f} -{(rd_50 - rd_16):.2f}")
    print(
        f"Ωm: {omega_50:.3f} +{(omega_84 - omega_50):.3f} -{(omega_50 - omega_16):.3f}"
    )
    print(f"w0: {w0_50:.3f} +{(w0_84 - w0_50):.3f} -{(w0_50 - w0_16):.3f}")
    print(f"Chi squared: {chi_squared(best_fit):.2f}")
    print(f"Degrees of freedom: {deg_of_freedom}")

    plot_bao_predictions(best_fit)
    plot_sn_predictions(
        legend=legend,
        x=z_sn_vals,
        y=apparent_mag_values,
        y_err=np.sqrt(np.diag(cov_matrix_sn)),
        y_model=sn_apparent_mag(best_fit),
        label=f"Best fit: $\Omega_m$={omega_50:.3f}, $H_0$={h0_50:.2f} km/s/Mpc",
        x_scale="log",
    )

    labels = ["$H_0$", "M", "$r_d$", "Ωm", "$w_0$"]
    corner.corner(
        samples,
        labels=labels,
        quantiles=[0.159, 0.5, 0.841],
        show_titles=True,
        title_fmt=".3f",
        bins=100,
        fill_contours=False,
        plot_datapoints=False,
        smooth=1.5,
        smooth1d=1.5,
        levels=(0.393, 0.864),  # 1 and 2 sigmas in 2D
    )
    plt.show()


if __name__ == "__main__":
    main()


"""
Flat ΛCDM: w(z) = -1
H0: 68.70 +3.30 -3.26 km/s/Mpc
M: -19.401 +0.101 -0.105 mag
r_d: 147.03 +7.25 -6.67 Mpc
Ωm: 0.304 +0.008 -0.008
w0: -1
wa: 0
Chi squared: 1431.36
Degrees of freedom: 1631

==============================

Flat wCDM: w(z) = w0
H0: 67.87 +3.25 -3.28 km/s/Mpc
M: -19.416 +0.101 -0.106 mag
r_d: 147.11 +7.29 -6.66 Mpc
Ωm: 0.298 +0.009 -0.008
w0: -0.917 +0.040 -0.040 (2.1 sigma)
Chi squared: 1427.13
Degrees of freedom: 1630

==============================

Flat alternative: w(z) = w0 - (1 + w0) * (((1 + z)**2 - 1) / ((1 + z)**2 + 1))
H0: 67.75 +3.25 -3.24 km/s/Mpc
M: -19.418 +0.101 -0.106 mag
r_d: 147.23 +7.29 -6.61 Mpc
Ωm: 0.303 +0.008 -0.008
w0: -0.906 +0.044 -0.045 (2.1 sigma)
wa: 0
Chi squared: 1426.99
Degrees of freedom: 1630

==============================

Flat w0waCDM: w(z) = w0 + wa * z/(1 + z)
H0: 67.95 +3.24 -3.27 km/s/Mpc
M: -19.411 +0.101 -0.106 mag
r_d: 146.86 +7.27 -6.58 Mpc
Ωm: 0.301 +0.016 -0.027
w0: -0.899 +0.061 -0.055
wa: -0.098 +0.516 -0.463
Chi squared: 1427.29
Degrees of freedom: 1629
"""
