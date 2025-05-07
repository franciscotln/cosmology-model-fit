from multiprocessing import Pool
import numpy as np
import emcee
from scipy.integrate import cumulative_trapezoid
import corner
import matplotlib.pyplot as plt
from y2018quasars.data import get_binned_data, factor
from y2025BAO.data import get_data as get_bao_data

legend, z, mu, sigma_mu = get_binned_data(30)
bao_legend, bao_data, bao_cov  = get_bao_data()
inv_cov_bao = np.linalg.inv(bao_cov)

c = 299792.458 # Speed of light (km/s)
H0 = 70.0 # Hubble constant (km/s/Mpc)

def Ez(z, params):
    one_plus_z = 1 + z
    Omega_m, w0 = params[3], params[4]
    return np.sqrt(Omega_m * one_plus_z**3 + (1 - Omega_m) * ((2 * one_plus_z**2) / (1 + one_plus_z**2))**(3 * (1 + w0)))

def integral_Ez(zs, params):
    z = np.linspace(0, np.max(zs), num=3000)
    integral_values = cumulative_trapezoid(1/Ez(z, params), z, initial=0)
    return np.interp(zs, z, integral_values)

def mu_model_sn(z, theta):
    return 25 + 5 * np.log10((1 + z) * c / H0 * integral_Ez(z, theta))

def H_z(z, params):
    return H0 * Ez(z, params)

def DM_z(zs, params):
    z = np.linspace(0, np.max(zs), num=3000)
    return cumulative_trapezoid(c / H_z(z, params), z, initial=0)[-1]

def DV_z(z, params):
    DH = c / H_z(z, params)
    DM = DM_z(z, params)
    return (z * DH * DM**2)**(1/3)

def bao_predictions(params):
    r_d = params[2]
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
    delta_bao = bao_data['value'] - bao_predictions(theta)
    return delta_bao.T @ inv_cov_bao @ delta_bao

def chi_squared_quasar(theta):
    beta_prime, s = theta[0], theta[1]
    delta_quasars = mu + factor * beta_prime - mu_model_sn(z, theta)
    return np.sum(delta_quasars**2 / (sigma_mu**2 + s**2))

def log_likelihood(theta):
    s = theta[1]
    chi_squared_bao_val = chi_squared_bao(theta)
    chi_squared_val = chi_squared_quasar(theta)
    return -0.5 * chi_squared_bao_val -0.5 * (chi_squared_val + np.sum(np.log(sigma_mu**2 + s**2)))

def log_prior(theta):
    beta_prime, s, r_d, omega_m, w0 = theta
    if (-8 < beta_prime < -6 and 0 < s < 3 and 110 < r_d < 155 and 0 < omega_m < 0.6 and -1.6 < w0 < 0):
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
    colors = { "DV_over_rs": "red", "DM_over_rs": "blue", "DH_over_rs": "green" }

    r_d, omega_m = theta[2], theta[3]
    z_smooth = np.linspace(0, max(z_values), 100)
    plt.figure(figsize=(8, 6))
    for q in unique_quantities:
        mask = quantity_types == q
        plt.errorbar(
            x=z_values[mask],
            y=observed_values[mask],
            yerr=errors[mask],
            fmt='.',
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
                model_smooth.append(DM_z(z, theta)/ r_d)
            elif q == "DH_over_rs":
                model_smooth.append((c / H_z(z, theta))/ r_d)
        plt.plot(z_smooth, model_smooth, color=colors[q], alpha=0.5)

    plt.xlabel("Redshift (z)")
    plt.ylabel(r"$O = \frac{D}{r_d}$")
    plt.legend()
    plt.grid(True)
    plt.title(f"BAO model: $\Omega_M$={omega_m:.3f}")
    plt.show()

def main():
    ndim = 5
    nwalkers = 8 * ndim
    nsteps = 10000
    p0 = np.array([-7, 1, 140, 0.3, -1]) + 0.01 * np.random.randn(nwalkers, ndim)

    with Pool(10) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, pool=pool)
        sampler.run_mcmc(p0, nsteps, progress=True)

    try:
        tau = sampler.get_autocorr_time()
        print("auto-correlation time", tau)
    except emcee.autocorr.AutocorrError as e:
        print("Autocorrelation time could not be computed", e)

    samples = sampler.get_chain(discard=1000, flat=True)

    for i, param in enumerate(["beta'", "s", "r_d", "Omega_m", "w0"]):
        mcmc_val = np.percentile(samples[:, i], [16, 50, 84])
        print(f"{param}: {mcmc_val[1]:.3f} +{mcmc_val[2]-mcmc_val[1]:.3f} -{mcmc_val[1]-mcmc_val[0]:.3f}")

    beta_50 = np.median(samples[:, 0])
    s_50 = np.median(samples[:, 1])
    r_d_50 = np.median(samples[:, 2])
    omega_m_50 = np.median(samples[:, 3])
    w0_50 = np.median(samples[:, 4])
    best_fit_params = [beta_50, s_50, r_d_50, omega_m_50, w0_50]
    chi_squared_bao_val = chi_squared_bao(best_fit_params)
    chi_squared_qsr_val = chi_squared_quasar(best_fit_params)
    print(f"chi squared BAO: {chi_squared_bao_val:.2f}")
    print(f"chi squared quasars: {chi_squared_qsr_val:.2f}")
    print(f"chi squared total: {chi_squared_bao_val + chi_squared_qsr_val:.2f}")

    corner.corner(
        samples,
        labels=["$\\beta'$", "$s$", "$r_d$", "$\\Omega_m$", "$w_0$"],
        quantiles=[0.159, 0.5, 0.841],
        show_titles=True,
        title_fmt=".3f",
        smooth=1.5,
        smooth1d=1.5,
        bins=100,
        levels=(0.393, 0.864), # 1 and 2 sigmas in 2D
        fill_contours=False,
        plot_datapoints=False,
    )
    plt.show()

    z_plot = np.linspace(0.01, np.max(z), num=1000)

    plt.errorbar(
        z,
        mu + factor * beta_50,
        yerr=np.sqrt(sigma_mu**2 + s_50**2),
        fmt='.',
        color='blue',
        label=legend,
        alpha=0.4,
        lw=0.5,
    )
    plt.plot(z_plot, mu_model_sn(z_plot, best_fit_params), color='orange', label="$\mu_T$", alpha=0.8)
    plt.xlabel('Redshift ($z$)')
    plt.ylabel('$\mu$')
    plt.xscale('log')
    plt.legend()
    plt.grid(True)
    plt.show()

    plot_bao_predictions(best_fit_params)

if __name__ == "__main__":
    main()


"""
Flat ΛCDM
beta': -7.065 +0.011 -0.011
s: 0.608 +0.066 -0.056
r_d: 145.083 +1.052 -1.050
Omega_m: 0.297 +0.009 -0.008
w0: -1
chi squared BAO: 10.27
chi squared quasars: 52.40
chi squared total: 62.67

===============================

Flat wCDM
beta': -7.069 +0.012 -0.012
s: 0.609 +0.066 -0.056
r_d: 142.860 +2.488 -2.367
Omega_m: 0.297 +0.009 -0.009
w0: -0.923 +0.075 -0.080
chi squared BAO: 9.13
chi squared quasars: 52.45
chi squared total: 61.57

==============================

Flat wzCDM
beta': -7.071 +0.013 -0.013
s: 0.611 +0.066 -0.056
r_d: 142.047 +3.049 -2.814
Omega_m: 0.303 +0.010 -0.010
w0: -0.887 +0.099 -0.107
chi squared BAO: 8.72
chi squared quasars: 52.16
chi squared total: 60.89
"""