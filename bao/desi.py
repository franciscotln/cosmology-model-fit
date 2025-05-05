import numpy as np
import emcee
from getdist import MCSamples, plots
from scipy.integrate import cumulative_trapezoid
import matplotlib.pyplot as plt
from multiprocessing import Pool
from y2025BAO.data import get_data

# Speed of light in km/s
c = 299792.458

# Planck rs = 147.18 ± 0.29 Mpc, h0 = 67.37 ± 0.54
legend, data, cov_matrix = get_data()
inv_cov_matrix = np.linalg.inv(cov_matrix)


def H_z(z, params):
    omega_m, w0 = params[1], params[2]
    sum = 1 + z
    return np.sqrt(omega_m * sum**3 + (1 - omega_m) * ((2 * sum**2) / (1 + sum**2))**(3 * (1 + w0)))


def DM_z(zs, params):
    z = np.linspace(0, np.max(zs), num=3000)
    return cumulative_trapezoid(c / H_z(z, params), z, initial=0)[-1]


def DV_z(z, params):
    DH = c / H_z(z, params)
    DM = DM_z(z, params)
    return (z * DH * DM**2)**(1/3)


quantity_funcs = {
    "DV_over_rs": lambda z, params: DV_z(z, params),
    "DM_over_rs": lambda z, params: DM_z(z, params),
    "DH_over_rs": lambda z, params: (c / H_z(z, params)),
}


def model_predictions(params):
    r_d_x_h0 = params[0] * 100
    return np.array([(quantity_funcs[qty](z, params) / r_d_x_h0) for z, _, qty in data])


bounds = np.array([
    (70, 110), # r_d x h
    (0.2, 0.7), # Ωm
    (-3, 1), # w0
    (-4.5, 1.5), # wa
])


def chi_squared(params):
    delta = data['value'] - model_predictions(params)
    return np.dot(delta, np.dot(inv_cov_matrix, delta))


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
    def plot_predictions(params):
        observed_values = data["value"]
        z_values = data["z"]
        quantity_types = data["quantity"]
        errors = np.sqrt(np.diag(cov_matrix))

        residuals = observed_values - model_predictions(params)
        SS_res = np.sum(residuals**2)
        SS_tot = np.sum((observed_values - np.mean(observed_values))**2)
        R_squared = 1 - SS_res/SS_tot

        rmsd = np.sqrt(np.mean(residuals ** 2))

        print(f"R^2: {R_squared:.4f}")
        print(f"RMSD: {rmsd:.3f}")

        unique_quantities = set(quantity_types)
        colors = { "DV_over_rs": "red", "DM_over_rs": "blue", "DH_over_rs": "green" }

        r_d, omega_m = params[0], params[1]
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
            model_smooth = np.array([quantity_funcs[q](z, params)/(100*r_d) for z in z_smooth])
            plt.plot(z_smooth, model_smooth, color=colors[q], alpha=0.5)

        plt.xlabel("Redshift (z)")
        plt.ylabel(r"$O = \frac{D}{r_d}$")
        plt.legend()
        plt.grid(True)
        plt.title(f"BAO model: $r_d x h$={r_d:.2f}, $\Omega_M$={omega_m:.3f}")
        plt.show()


    ndim = len(bounds)
    nwalkers = 10 * ndim
    burn_in = 500
    nsteps = 20000 + burn_in
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

    samples = sampler.get_chain(discard=burn_in, flat=True)

    [
        [rd_16, rd_50, rd_84],
        [omega_16, omega_50, omega_84],
        [w0_16, w0_50, w0_84],
        [wa_16, wa_50, wa_84],
    ] = np.percentile(samples, [15.9, 50, 84.1], axis=0).T

    best_fit = [rd_50, omega_50, w0_50, wa_50]

    print(f"r_d*h: {rd_50:.2f} +{(rd_84 - rd_50):.2f} -{(rd_50 - rd_16):.2f}")
    print(f"Ωm: {omega_50:.3f} +{(omega_84 - omega_50):.3f} -{(omega_50 - omega_16):.3f}")
    print(f"w0: {w0_50:.3f} +{(w0_84 - w0_50):.3f} -{(w0_50 - w0_16):.3f}")
    print(f"wa: {wa_50:.4f} +{(wa_84 - wa_50):.4f} -{(wa_50 - wa_16):.4f}")
    print(f"Chi squared: {chi_squared(best_fit):.2f}")
    print(f"Degs of freedom: {data['value'].size  - len(best_fit)}")

    plot_predictions(best_fit)

    labels = ["r_d x H_0", "Ω_m", "w_0", "w_a"]
    gdsamples = MCSamples(
        samples=samples,
        names=labels,
        labels=labels,
        settings={"fine_bins_2D": 128, "smooth_scale_2D": 0.9}
    )
    g = plots.get_subplot_plotter()
    g.triangle_plot(
        gdsamples,
        Filled=False,
        contour_levels=[0.68, 0.95],
        title_limit=True,
        diag1d_kwargs={"density": True},
    )
    plt.show()

if __name__ == "__main__":
    main()


"""
Flat ΛCDM model
r_d*h: 101.52 +0.74 -0.73 km/s
Ωm: 0.298 +0.009 -0.009
w0: -1
wa: 0
Chi squared: 10.27
Degs of freedom: 11
R^2: 0.9987
RMSD: 0.305

==============================

Flat wCDM model
r_d*h: 99.79 +1.76 -1.65 km/s
Ωm: 0.297 +0.009 -0.009
w0: -0.915 +0.076 -0.079
wa: 0
Chi squared: 9.12
Degs of freedom: 10
R^2: 0.9989
RMSD: 0.279

===============================

Flat w0 - (1 + w0) * (((1 + z)**2 - 1) / ((1 + z)**2 + 1))
r_d*h: 99.05 +2.15 -1.95 km/s
Ωm: 0.304 +0.010 -0.010
w0: -0.869 +0.099 -0.108
wa: 0
Chi squared: 8.69
Degs of freedom: 10
R^2: 0.9990
RMSD: 0.270

==============================

Flat w0waCDM
r_d*h: 92.02 +4.60 -3.53
Ωm: 0.380 +0.036 -0.045
w0: -0.246 +0.344 -0.400
wa: -2.5086 +1.4138 -1.1921
Chi squared: 5.65
Degs of freedom: 9
R^2: 0.9994
RMSD: 0.205
"""
