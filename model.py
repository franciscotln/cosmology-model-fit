import numpy as np
from scipy.optimize import curve_fit
from plotting import plot_predictions, print_color, plot_residuals
from y2022pantheonSHOES.data import get_data

legend, z_values, distance_modulus_values, sigma_distance_moduli = get_data()

# Speed of light (km/s)
C = 299792.458


def a_at_emission(z, p0):
    param_a = (1 + z) * (-1 + (1 + (1 / p0 ** (1/3))) ** 3) - (1/p0)
    multiplier = 3 / (2 * param_a * p0 ** (2/3))
    return (multiplier + multiplier * np.sqrt(1 + (4/3) * param_a * p0)) ** 2


def integral_of_e_z(zs, p0):
    integrals = np.empty((len(zs),), dtype=np.float64)
    i = 0
    for z_item in zs:
        z_axis = np.linspace(0, z_item, 100)
        integ = np.trapz([a_at_emission(z=z, p0=p0)**1.5 for z in z_axis], x=z_axis)
        integrals[i] = integ
        i = i + 1
    return integrals


# Define a theoretical distance modulus:
def model_distance_modulus(z, h0, p0):
    luminosity_distance_model = (C / h0) * (1 / a_at_emission(z=z, p0=p0)) * integral_of_e_z(zs=z, p0=p0)
    return 25 + 5 * np.log10(luminosity_distance_model)


# Fit the curve to the data
[params_opt, params_cov] = curve_fit(
    f=model_distance_modulus,
    xdata=z_values,
    ydata=distance_modulus_values,
    sigma=sigma_distance_moduli,
    absolute_sigma=True,
    p0=[70, 0.01]
)

# Extract the optimal value for H0
[h0, p0] = params_opt
[h0_std, p0_std] = np.sqrt(np.diag(params_cov))

# Calculate residuals
predicted_distance_modulus_values = model_distance_modulus(z=z_values, h0=h0, p0=p0)
residuals = predicted_distance_modulus_values - distance_modulus_values

# Calculate R-squared
average_distance_modulus = np.mean(distance_modulus_values)
ss_res = np.sum(residuals ** 2)
ss_tot = np.sum((distance_modulus_values - average_distance_modulus) ** 2)
r_squared = 1 - (ss_res / ss_tot)

# Calculate root mean square deviation
rmsd = np.sqrt(np.mean(residuals ** 2))

# Print the values in the console
h0_label = f"{h0:.3f} ± {h0_std:.3f}"

print_color("Dataset", legend)
print_color("Sample size", len(z_values))
print_color("Estimated H0 (km/s/Mpc)", h0_label)
print_color("R-squared (%)", f"{100 * r_squared:.5f}")
print_color("RMSD (mag)", f"{rmsd:.5f}")
print_color("p0", f"{p0:.3f} ± {p0_std:.3f}")

# Plot the data and the fit
plot_predictions(
    legend=legend,
    x=z_values,
    y=distance_modulus_values,
    y_err=sigma_distance_moduli,
    y_model=predicted_distance_modulus_values,
    label=f"Model: H0 = {h0_label} km/s/Mpc",
    x_scale='log'
)

# Plot the residual analysis
plot_residuals(z_values = z_values, residuals = residuals, bins = 40)
