import numpy as np
import emcee
import corner
from scipy.integrate import cumulative_trapezoid, quad
import matplotlib.pyplot as plt
from multiprocessing import Pool
from y2024DES.data import get_data
from hubble.plotting import plot_predictions as plot_sn_predictions

legend, z_vals, distance_moduli_values, cov_matrix_sn = get_data()
inverse_cov_sn = np.linalg.inv(cov_matrix_sn)

c = 299792.458 # Speed of light in km/s
H0 = 70 # Hubble constant in km/s/Mpc as per DES5Y

# Load BAO data
data = np.genfromtxt(
    "bao/raw-data/data.txt",
    dtype=[("z", float), ("value", float), ("quantity", "U10")],
    delimiter=" ",
    names=True,
)
cov_matrix = np.loadtxt("bao/raw-data/covariance.txt", delimiter=" ", dtype=float)
inv_cov_matrix = np.linalg.inv(cov_matrix)


def w_de(z, params):
    _, _, w0, wa = params
    return w0 + wa * (1 - np.exp(0.5 - 0.5 * (1 + z)**2))


def rho_de(zs, params):
    z = np.linspace(0, np.max(zs), num=2000)
    integral_values = cumulative_trapezoid(3*(1 + w_de(z, params))/(1 + z), z, initial=0)
    return np.exp(np.interp(zs, z, integral_values))


def h_over_h0_model(z, params):
    O_m = params[1]
    sum = 1 + z
    return np.sqrt(O_m * sum**3 + (1 - O_m) * rho_de(z, params))


def wcdm_integral_of_e_z(zs, params):
    z = np.linspace(0, np.max(zs), num=2000)
    integral_values = cumulative_trapezoid(1/h_over_h0_model(z, params), z, initial=0)
    return np.interp(zs, z, integral_values)


def wcdm_distance_modulus(z, params):
    a0_over_ae = 1 + z
    comoving_distance = (c/H0) * wcdm_integral_of_e_z(z, params)
    return 25 + 5 * np.log10(a0_over_ae * comoving_distance)


def plot_bao_predictions(params):
    observed_values = data["value"]
    z_values = data["z"]
    quantity_types = data["quantity"]
    errors = np.sqrt(np.diag(cov_matrix))

    unique_quantities = set(quantity_types)
    colors = { "DV_over_rs": "red", "DM_over_rs": "blue", "DH_over_rs": "green" }

    r_d, omega_m, w0, wa = params
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
                model_smooth.append(DV_z(z, params)/(H0*r_d))
            elif q == "DM_over_rs":
                model_smooth.append(DM_z(z, params)/(H0*r_d))
            elif q == "DH_over_rs":
                model_smooth.append((c / H_z(z, params))/(H0*r_d))
        plt.plot(z_smooth, model_smooth, color=colors[q], alpha=0.5)

    plt.xlabel("Redshift (z)")
    plt.ylabel(r"$O = \frac{D}{r_d}$")
    plt.legend()
    plt.grid(True)
    plt.title(f"BAO model: $r_d * h$={r_d:.2f}, $\Omega_M$={omega_m:.4f}, $w_0$={w0:.4f}, $w_a$={wa:.4f}")
    plt.show()


def H_z(z, params):
    return h_over_h0_model(z, params)
    # return np.sqrt(omega_m * sum**3 + (1 - omega_m)) # LCDM


def DM_z(zs, params):
    return quad(lambda z: c / H_z(z, params), 0, zs)[0]


def DV_z(z, params):
    DH = c / H_z(z, params)
    DM = DM_z(z, params)
    return (z * DH * DM**2)**(1/3)


def model_predictions(params):
    r_d = params[0]
    predictions = []
    for z, _, quantity in data:
        if quantity == "DV_over_rs":
            predictions.append(DV_z(z, params) / (r_d*H0))
        elif quantity == "DM_over_rs":
            predictions.append(DM_z(z, params) / (r_d*H0))
        elif quantity == "DH_over_rs":
            predictions.append((c / H_z(z, params)) / (r_d*H0))
    return np.array(predictions)


bounds = np.array([
    (115, 160), # r_d
    (0.1, 0.7), # omega_m
    (-3, 0), # w0
    (-3, 1.5), # wa
])


def chi_squared(params):
    """ 
    Computes modified likelihood to marginalize over M 
    (Wood-Vasey et al. 2001, Appendix A9-A12)
    """
    delta_sn = distance_moduli_values - wcdm_distance_modulus(z_vals, params)
    deltaT = np.transpose(delta_sn)
    chit2 = np.sum(delta_sn @ inverse_cov_sn @ deltaT)     # First term: (Δ^T C^-1 Δ)
    B = np.sum(delta_sn @ inverse_cov_sn)                  # Second term: B
    C = np.sum(inverse_cov_sn)                             # Third term: C
    chi_sn = chit2 - (B**2 / C) + np.log(C / (2 * np.pi))  # Full modified chi2
    delta_bao = data['value'] - model_predictions(params)
    chi_bao = np.dot(delta_bao, np.dot(inv_cov_matrix, delta_bao))
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
    nwalkers = 50
    burn_in = 500
    nsteps = 2000 + burn_in
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
        [omega_16, omega_50, omega_84],
        [w0_16, w0_50, w0_84],
        [wa_16, wa_50, wa_84],
    ] = np.percentile(samples, [15.9, 50, 84.1], axis=0).T

    best_fit = [rd_50, omega_50, w0_50, wa_50]

    print(f"r_d: {rd_50:.4f} +{(rd_84 - rd_50):.4f} -{(rd_50 - rd_16):.4f}")
    print(f"Ωm: {omega_50:.4f} +{(omega_84 - omega_50):.4f} -{(omega_50 - omega_16):.4f}")
    print(f"w0: {w0_50:.4f} +{(w0_84 - w0_50):.4f} -{(w0_50 - w0_16):.4f}")
    print(f"wa: {wa_50:.4f} +{(wa_84 - wa_50):.4f} -{(wa_50 - wa_16):.4f}")
    print(f"Chi squared: {chi_squared(best_fit):.4f}")
    print(f"Degrees of freedom: {data['value'].size + z_vals.size - len(best_fit)}")

    plot_bao_predictions(best_fit)
    plot_sn_predictions(
        legend=legend,
        x=z_vals,
        y=distance_moduli_values,
        y_err=np.sqrt(np.diag(cov_matrix_sn)),
        y_model=wcdm_distance_modulus(z_vals, best_fit),
        label=f"Best fit: $w_0$={w0_50:.4f}, $\Omega_m$={omega_50:.4f}",
        x_scale="log"
    )

    labels = [r"$r_d$", f"$\Omega_m$", r"$w_0$", r"$w_a$"]
    corner.corner(
        samples,
        labels=labels,
        quantiles=[0.159, 0.5, 0.841],
        show_titles=True,
        title_fmt=".4f",
        smooth=2,
        smooth1d=2,
        bins=50,
    )
    plt.show()

    fig, axes = plt.subplots(ndim, figsize=(10, 7))
    if ndim == 1:
        axes = [axes]
    for i in range(ndim):
        axes[i].plot(chains_samples[:, :, i], color='black', alpha=0.3)
        axes[i].set_ylabel(labels[i])
        axes[i].set_xlabel("chain step")
        axes[i].axvline(x=burn_in, color='red', linestyle='--', alpha=0.5)
        axes[i].axhline(y=best_fit[i], color='white', linestyle='--', alpha=0.5)
    plt.show()


if __name__ == "__main__":
    main()


"""
Flat ΛCDM
r_d: 143.6861 +0.9398 -0.9550
Ωm: 0.3099 +0.0081 -0.0077
w0 = -1
wa = 0
Chi squared: 1667.7203
Degrees of freedom: 1840

==============================

Flat wCDM
r_d: 141.3300 +1.1671 -1.1685
Ωm: 0.2979 +0.0087 -0.0087
w0: -0.8748 +0.0379 -0.0397 (3.23 sigma)
wa: 0
Chi squared: 1657.5935
Degrees of freedom: 1839

===============================

Flat w0 - (1 + w0) * (((1 + z)**2 - 1) / ((1 + z)**2 + 1))
r_d: 141.1310 +1.2121 -1.1808
Ωm: 0.3052 +0.0079 -0.0078
w0: -0.8552 +0.0408 -0.0429 (3.46 sigma)
wa: 0
Chi squared: 1656.5929
Degrees of freedom: 1839

==============================

Flat w0waCDM
r_d: 140.8285 +1.2178 -1.1816
Ωm: 0.3210 +0.0122 -0.0132
w0: -0.7878 +0.0697 -0.0607 (3.25 sigma)
wa: -0.7034 +0.3901 -0.4490 (1.68 sigma)
Chi squared: 1655.2812
Degrees of freedom: 1838

==============================

Flat w0 + wa * z
r_d: 140.9156 +1.2076 -1.2039
Ωm: 0.3267 +0.0118 -0.0132
w0: -0.8214 +0.0539 -0.0506 (3.42 sigma)
wa: -0.4483 +0.2487 -0.2374 (1.84 sigma)
Chi squared: 1655.2542
Degrees of freedom: 1838

============================

Flat w0 + wa * np.tanh(z)
r_d: 140.9847 +1.1919 -1.1959
Ωm: 0.3193 +0.0133 -0.0168
w0: -0.8140 +0.0605 -0.0572 (3.16 sigma)
wa: -0.4294 +0.3130 -0.3112 (1.38 sigma)
Chi squared: 1655.3328
Degrees of freedom: 1838

Flat w0 + wa * np.tanh(0.5 * ((1 + z)**2 - 1))
r_d: 141.0321 +1.1952 -1.2112
Ωm: 0.3188 +0.0128 -0.0150
w0: -0.8230 +0.0573 -0.0527 (3.22 sigma)
wa: -0.3447 +0.2324 -0.2470 (1.44 sigma)
Chi squared: 1655.3989
Degrees of freedom: 1838

Flat w0 + wa * (1 - np.exp(0.5 - 0.5 * (1 + z)**2))
r_d: 140.9714 +1.1877 -1.1869
Ωm: 0.3199 +0.0129 -0.0168
w0: -0.8130 +0.0593 -0.0574 (3.20 sigma)
wa: -0.4398 +0.3094 -0.3021 (1.44 sigma)
Chi squared: 1655.3195
Degrees of freedom: 1838

============================

Flat w0 + wa * np.tanh(0.5 * (1 + z - 1/(1 + z)))
r_d: 140.9350 +1.2293 -1.1756
Ωm: 0.3207 +0.0129 -0.0166
w0: -0.8056 +0.0667 -0.0624 (3.01 sigma)
wa: -0.5517 +0.3900 -0.3664 (1.46 sigma)
Chi squared: 1655.2916
Degrees of freedom: 1838
"""