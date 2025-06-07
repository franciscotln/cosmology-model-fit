import numpy as np
import emcee
import corner
from scipy.integrate import cumulative_trapezoid, quad
from scipy.linalg import cho_factor, cho_solve
import matplotlib.pyplot as plt
from multiprocessing import Pool
from y2025BAO.data import get_data as get_bao_data
from y2022pantheonSHOES.data import get_data
from hubble.plotting import plot_predictions as plot_sn_predictions

legend, z_sn_vals, z_hel_vals, mb_vals, cov_matrix_sn = get_data()
bao_legend, data, bao_cov_matrix = get_bao_data()
cho_sn = cho_factor(cov_matrix_sn)
cho_bao = cho_factor(bao_cov_matrix)

c = 299792.458  # Speed of light in km/s
H0 = 70.0  # Hubble constant in km/s/Mpc


def h_over_h0(z, params):
    O_m, w0 = params[2], params[3]
    one_plus_z = 1 + z
    evolving_de = ((2 * one_plus_z**2) / (1 + one_plus_z**2)) ** (3 * (1 + w0))
    return np.sqrt(O_m * one_plus_z**3 + (1 - O_m) * evolving_de)


z_grid = np.linspace(0, np.max(z_sn_vals), num=2000)


def integral_e_z(params):
    integral_values = cumulative_trapezoid(
        1 / h_over_h0(z_grid, params), z_grid, initial=0
    )
    return np.interp(z_sn_vals, z_grid, integral_values)


def apparent_mag(params):
    M0 = params[1]
    luminosity_distance = (1 + z_hel_vals) * (c / H0) * integral_e_z(params)
    return M0 + 25 + 5 * np.log10(luminosity_distance)


def plot_bao_predictions(params):
    errors = np.sqrt(np.diag(bao_cov_matrix))
    colors = {"DV_over_rs": "red", "DM_over_rs": "blue", "DH_over_rs": "green"}
    r_d = params[0]
    z_smooth = np.linspace(0, max(data["z"]), 100)

    plt.figure(figsize=(8, 6))
    for q in set(data["quantity"]):
        mask = data["quantity"] == q
        plt.errorbar(
            x=data["z"][mask],
            y=data["value"][mask],
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
                model_smooth.append((DH_z(z, params)) / r_d)
        plt.plot(z_smooth, model_smooth, color=colors[q], alpha=0.5)

    plt.xlabel("Redshift (z)")
    plt.ylabel(r"$O = \frac{D}{r_d}$")
    plt.legend()
    plt.grid(True)
    plt.title(bao_legend)
    plt.show()


def H_z(z, params):
    return H0 * h_over_h0(z, params)


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
    r_d = params[0]
    return np.array([(quantity_funcs[qty](z, params) / r_d) for z, _, qty in data])


bounds = np.array(
    [
        (115, 160),  # r_d
        (-20, -19),  # M
        (0.2, 0.7),  # Ωm
        (-2, 0),  # w0
    ]
)


def chi_squared(params):
    delta_sn = mb_vals - apparent_mag(params)
    chi_sn = np.dot(delta_sn, cho_solve(cho_sn, delta_sn))

    delta_bao = data["value"] - bao_predictions(params)
    chi_bao = np.dot(delta_bao, cho_solve(cho_bao, delta_bao))
    return chi_sn + chi_bao


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
    nwalkers = 16 * ndim
    burn_in = 500
    nsteps = 8000 + burn_in
    initial_pos = np.zeros((nwalkers, ndim))

    for dim, (lower, upper) in enumerate(bounds):
        initial_pos[:, dim] = np.random.uniform(lower, upper, nwalkers)

    with Pool(10) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool=pool)
        sampler.run_mcmc(initial_pos, nsteps, progress=True)

    try:
        tau = sampler.get_autocorr_time()
        print("auto-correlation time", tau)
    except emcee.autocorr.AutocorrError as e:
        print("Autocorrelation time could not be computed", e)

    chains_samples = sampler.get_chain(discard=0, flat=False)
    samples = sampler.get_chain(discard=burn_in, flat=True)

    [
        [rd_16, rd_50, rd_84],
        [M_16, M_50, M_84],
        [omega_16, omega_50, omega_84],
        [w0_16, w0_50, w0_84],
    ] = np.percentile(samples, [15.9, 50, 84.1], axis=0).T

    best_fit = [rd_50, M_50, omega_50, w0_50]

    print(f"r_d: {rd_50:.2f} +{(rd_84 - rd_50):.2f} -{(rd_50 - rd_16):.2f}")
    print(f"M0: {M_50:.3f} +{(M_84 - M_50):.3f} -{(M_50 - M_16):.3f}")
    print(
        f"Ωm: {omega_50:.3f} +{(omega_84 - omega_50):.3f} -{(omega_50 - omega_16):.3f}"
    )
    print(f"w0: {w0_50:.3f} +{(w0_84 - w0_50):.3f} -{(w0_50 - w0_16):.3f}")
    print(f"Chi squared: {chi_squared(best_fit):.2f}")
    print(f"Degrees of freedom: {data['z'].size + z_sn_vals.size - len(best_fit)}")

    plot_bao_predictions(best_fit)
    plot_sn_predictions(
        legend=legend,
        x=z_sn_vals,
        y=mb_vals,
        y_err=np.sqrt(np.diag(cov_matrix_sn)),
        y_model=apparent_mag(best_fit),
        label=f"Best fit: $\Omega_m$={omega_50:.3f}",
        x_scale="log",
    )

    labels = ["$r_d$", "$M_0$", "$\Omega_m$", "$w_0$"]
    corner.corner(
        samples,
        labels=labels,
        quantiles=[0.159, 0.5, 0.841],
        show_titles=True,
        title_fmt=".4f",
        smooth=1.5,
        smooth1d=1.5,
        bins=100,
        levels=(0.393, 0.864),
        fill_contours=False,
        plot_datapoints=False,
    )
    plt.show()

    _, axes = plt.subplots(ndim, figsize=(10, 7))
    if ndim == 1:
        axes = [axes]
    for i in range(ndim):
        axes[i].plot(chains_samples[:, :, i], color="black", alpha=0.3)
        axes[i].set_ylabel(labels[i])
        axes[i].set_xlabel("chain step")
        axes[i].axvline(x=burn_in, color="red", linestyle="--", alpha=0.5)
        axes[i].axhline(y=best_fit[i], color="white", linestyle="--", alpha=0.5)
    plt.show()


if __name__ == "__main__":
    main()

"""
Flat ΛCDM
r_d: 144.29 +0.96 -0.96 Mpc
M0: -19.360 +0.005 -0.005 mag
Ωm: 0.304 +0.008 -0.008
w0: -1
wa: 0
Chi squared: 1416.14
Degrees of freedom: 1600

====================

Flat wCDM
r_d: 142.60 +1.24 -1.25 Mpc
M0: -19.348 +0.007 -0.007 mag
Ωm: 0.2977 +0.0086 -0.0085
w0: -0.916 +0.039 -0.039 (2.13 sigma)
wa: 0
Chi squared: 1412.41
Degrees of freedom: 1599

====================

Flat w0 - (1 + w0) * (((1 + z)**2 - 1) / ((1 + z)**2 + 1))
r_d: 142.44 +1.27 -1.24 Mpc
M0: -19.346 +0.008 -0.008 mag
Ωm: 0.302 +0.008 -0.008
w0: -0.903 +0.044 -0.044 (2.2 sigma)
wa: 0
Chi squared: 1411.32
Degrees of freedom: 1599

====================

Flat w0waCDM
r_d: 142.56 +1.26 -1.27 Mpc
M0: -19.346 +0.009 -0.009 mag
Ωm: 0.3021 +0.0154 -0.0228
w0: -0.8964 +0.0607 -0.0561 (1.77 sigma)
wa: -0.1376 +0.4676 -0.4503 (0.30 sigma)
Chi squared: 1412.36
Degrees of freedom: 1598
"""
