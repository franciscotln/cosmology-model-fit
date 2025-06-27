from multiprocessing import Pool
import numpy as np
import emcee
from scipy.integrate import cumulative_trapezoid
from scipy.linalg import cho_factor, cho_solve
import corner
import matplotlib.pyplot as plt
from y2018quasars.data import get_binned_data as get_quasar_data
from y2024DES.data import get_data as get_des5y_data
from y2025BAO.data import get_data as get_bao_data

legend, z_qsr, mu_qsr, sigma_mu = get_quasar_data(22)
sn_legend, z_sn, z_hel_sn, mu_sn, sn_cov = get_des5y_data()
bao_legend, bao_data, bao_cov = get_bao_data()
cho_sn = cho_factor(sn_cov)
cho_bao = cho_factor(bao_cov)

c = 299792.458  # Speed of light (km/s)
H0 = 70  # Hubble constant (km/s/Mpc)

z_unique = np.sort(np.unique(np.concatenate((z_qsr, z_sn))))


def Ez(z, Om, w0):
    one_plus_z = 1 + z
    rho_DE = ((2 * one_plus_z**2) / (1 + one_plus_z**2)) ** (3 * (1 + w0))
    return np.sqrt(Om * one_plus_z**3 + (1 - Om) * rho_DE)


def integral_Ez(z, params):
    z_grid = np.linspace(0, np.max(z), num=3000)
    y = 1 / Ez(z_grid, *params[4:])
    integral_values = cumulative_trapezoid(y=y, x=z_grid, initial=0)
    return np.interp(z, z_grid, integral_values)


def mu_theory(z, z_int, theta):
    dL = (1 + z) * (c / H0) * integral_Ez(z_int, theta)
    return 25 + 5 * np.log10(dL)


def H_z(z, params):
    return H0 * Ez(z, *params[4:])


def DM_z(zs, params):
    z = np.linspace(0, np.max(zs), num=3000)
    return cumulative_trapezoid(c / H_z(z, params), z, initial=0)[-1]


def DV_z(z, params):
    DH = c / H_z(z, params)
    DM = DM_z(z, params)
    return (z * DH * DM**2) ** (1 / 3)


def bao_predictions(params):
    r_d = params[3]
    predictions = []
    for z, _, quantity in bao_data:
        if quantity == "DV_over_rs":
            predictions.append(DV_z(z, params) / r_d)
        elif quantity == "DM_over_rs":
            predictions.append(DM_z(z, params) / r_d)
        elif quantity == "DH_over_rs":
            predictions.append((c / H_z(z, params)) / r_d)
    return np.array(predictions)


def chi_squared_bao(theta):
    delta_bao = bao_data["value"] - bao_predictions(theta)
    return np.dot(delta_bao, cho_solve(cho_bao, delta_bao))


def chi_squared_sn(theta):
    deltaM = theta[2]
    delta_sn = mu_sn - deltaM - mu_theory(z_hel_sn, z_sn, theta)
    return np.dot(delta_sn, cho_solve(cho_sn, delta_sn))


def chi_squared_quasar(theta):
    deltaM, s = theta[0], theta[1]
    delta_quasars = mu_qsr - deltaM - mu_theory(z_qsr, z_qsr, theta)
    return np.sum(delta_quasars**2 / (sigma_mu**2 + s**2))


def log_likelihood(theta):
    s = theta[1]
    chi_squared_sn_val = chi_squared_sn(theta)
    chi_squared_val = chi_squared_quasar(theta)
    chi_squared_bao_val = chi_squared_bao(theta)
    return -0.5 * (chi_squared_bao_val + chi_squared_sn_val) - 0.5 * (
        chi_squared_val + np.sum(np.log(sigma_mu**2 + s**2))
    )


def log_prior(theta):
    deltaM_qsr, s, deltaM_sn, rd, Om, w0 = theta
    if (
        -1 < deltaM_qsr < 1
        and 0 < s < 2.5
        and -0.6 < deltaM_sn < 0.6
        and 110 < rd < 170
        and 0 < Om < 0.6
        and -1.5 < w0 < 0
    ):
        return 0.0
    return -np.inf


def log_posterior(theta):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta)


def plot_bao_predictions(theta):
    observed_values = bao_data["value"]
    z_values = bao_data["z"]
    quantity_types = bao_data["quantity"]
    errors = np.sqrt(np.diag(bao_cov))

    unique_quantities = set(quantity_types)
    colors = {"DV_over_rs": "red", "DM_over_rs": "blue", "DH_over_rs": "green"}

    r_d, omega_m = theta[3], theta[4]
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
            label=f"Data: {q}",
            capsize=2,
            linestyle="None",
        )
        model_smooth = []
        for z in z_smooth:
            if q == "DV_over_rs":
                model_smooth.append(DV_z(z, theta) / r_d)
            elif q == "DM_over_rs":
                model_smooth.append(DM_z(z, theta) / r_d)
            elif q == "DH_over_rs":
                model_smooth.append((c / H_z(z, theta)) / r_d)
        plt.plot(z_smooth, model_smooth, color=colors[q], alpha=0.5)

    plt.xlabel("Redshift (z)")
    plt.ylabel(r"$O = \frac{D}{r_d}$")
    plt.legend()
    plt.grid(True)
    plt.title(f"BAO model: $\Omega_M$={omega_m:.3f}")
    plt.show()


def main():
    ndim = 6
    nwalkers = 6 * ndim
    nsteps = 10000
    initial_state = np.array([0, 1, 0, 140, 0.3, -1]) + 0.01 * np.random.randn(
        nwalkers, ndim
    )

    with Pool(10) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, pool=pool)
        sampler.run_mcmc(initial_state, nsteps, progress=True)

    try:
        tau = sampler.get_autocorr_time()
        print("auto-correlation time", tau)
    except emcee.autocorr.AutocorrError as e:
        print("Autocorrelation time could not be computed", e)

    samples = sampler.get_chain(discard=1000, flat=True)

    for i, param in enumerate(["ΔM_qsr", "s", "ΔM_sn", "rd", "Ωm", "w0"]):
        mcmc_val = np.percentile(samples[:, i], [16, 50, 84])
        print(
            f"{param}: {mcmc_val[1]:.3f} +{mcmc_val[2]-mcmc_val[1]:.3f} -{mcmc_val[1]-mcmc_val[0]:.3f}"
        )

    deltaM_qsr_50 = np.median(samples[:, 0])
    s_50 = np.median(samples[:, 1])
    deltaM_sn_50 = np.median(samples[:, 2])
    rd_50 = np.median(samples[:, 3])
    Om_50 = np.median(samples[:, 4])
    w0_50 = np.median(samples[:, 5])
    best_fit_params = [deltaM_qsr_50, s_50, deltaM_sn_50, rd_50, Om_50, w0_50]

    chi_squared_sn_val = chi_squared_sn(best_fit_params)
    chi_squared_qsr_val = chi_squared_quasar(best_fit_params)
    print(f"chi squared SN: {chi_squared_sn_val:.2f}")
    print(f"chi squared quasars: {chi_squared_qsr_val:.2f}")
    print(f"chi squared total: {chi_squared_sn_val + chi_squared_qsr_val:.2f}")

    labels = ["$Δ_{Mqsr}$", "s", "$Δ_{Msn}$", "$r_d$", "$Ω_m$", "$w_0$"]
    corner.corner(
        samples,
        labels=labels,
        quantiles=[0.159, 0.5, 0.841],
        show_titles=True,
        title_fmt=".3f",
        smooth=1.5,
        smooth1d=1.5,
        bins=100,
        levels=(0.393, 0.864),  # 1 and 2 sigmas in 2D
        fill_contours=False,
        plot_datapoints=False,
    )
    plt.show()

    z_plot = np.linspace(0.02, np.max(z_qsr), num=1000)
    theory = mu_theory(z_plot, z_plot, best_fit_params)

    plt.errorbar(
        z_qsr,
        mu_qsr - deltaM_qsr_50,
        yerr=np.sqrt(sigma_mu**2 + s_50**2),
        fmt=".",
        color="blue",
        label=legend,
        alpha=0.4,
        lw=0.5,
    )
    plt.scatter(
        z_sn, mu_sn - deltaM_sn_50, s=8, color="red", alpha=0.5, label=sn_legend
    )
    plt.ylim(32.5, 54)
    plt.plot(z_plot, theory, color="green", label="$\mu_T$", alpha=0.8)
    plt.xlabel("Redshift ($z$)")
    plt.ylabel("$\mu$")
    plt.xscale("log")
    plt.legend()
    plt.grid(True)
    plt.show()

    plot_bao_predictions(best_fit_params)


if __name__ == "__main__":
    main()


"""
Flat ΛCDM
ΔM_qsr: -0.181 +0.086 -0.087
s: 0.402 +0.075 -0.059
ΔM_sn: -0.003 +0.007 -0.006
rd: 143.478 +0.957 -0.961
Ωm: 0.312 +0.008 -0.008
w0: -1
chi squared SN: 1646.12
chi squared quasars: 19.75
chi squared total: 1665.87

==================================

Flat wCDM
ΔM_qsr: -0.146 +0.089 -0.090 mag
s: 0.405 +0.076 -0.058 mag^2
ΔM_sn: 0.026 +0.011 -0.011 mag
rd: 141.139 +1.162 -1.156 Mpc
Ωm: 0.299 +0.009 -0.009
w0: -0.872 +0.038 -0.039 (3.28 - 3.37 sigma)
chi squared SN: 1638.75
chi squared quasars: 19.82
chi squared total: 1658.57

==================================

Flat wzCDM
ΔM_qsr: -0.139 +0.088 -0.089 mag
s: 0.405 +0.076 -0.059 mag^2
ΔM_sn: 0.030 +0.012 -0.012 mag
rd: 140.930 +1.172 -1.175 Mpc
Ωm: 0.306 +0.008 -0.008
w0: -0.852 +0.041 -0.042 (3.52 - 3.61 sigma)
chi squared SN: 1638.30
chi squared quasars: 19.82
chi squared total: 1658.12
"""
