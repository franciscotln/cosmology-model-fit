import numpy as np
from scipy.optimize import curve_fit
from plotting import plot_predictions, print_color, plot_residuals
from y2022pantheonSHOES.data import get_data

legend, z_values, distance_modulus_values, sigma_distance_moduli = get_data()

# Speed of light (km/s)
C = 299792.458


# Define the theoretical distance modulus:
def model_distance_modulus(z, h0, n0):
    beta = (-1 + (1 + (1 / n0)) ** 3) * n0 ** 3
    factor_z = -1 + beta * (1 + z)
    sqr_z = np.sqrt(1 + (4 * factor_z / 3))
    sqr_beta = np.sqrt(1 + (4 * (beta - 1) / 3))
    integral_upper_limit = (1 - 2 * sqr_z) / (sqr_z - 1)**2
    integral_lower_limit = (1 - 2 * sqr_beta) / (sqr_beta - 1)**2
    luminosity_distance_model = (8/3) * (C/h0) * (factor_z / (1 + sqr_z)) ** 2 * (n0 / beta) * (integral_upper_limit - integral_lower_limit)
    return 25 + 5 * np.log10(luminosity_distance_model)


# Fit the curve to the data
[params_opt, params_cov] = curve_fit(
    f=model_distance_modulus,
    xdata=z_values,
    ydata=distance_modulus_values,
    sigma=sigma_distance_moduli,
    absolute_sigma=True,
    p0=[72, 0.3]
)

# Extract the optimal value for H0 and n0 - n0 is dimensionless: has unit s^(-1/3) / s^(-1/3)
[h0, n0] = params_opt
[h0_std, n0_std] = np.sqrt(np.diag(params_cov))

# Calculate residuals
predicted_distance_modulus_values = model_distance_modulus(z=z_values, h0=h0, n0=n0)
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
print_color("n0", f"{n0:.5f} ± {n0_std:.5f}")

# Plot the data and the fit
plot_predictions(
    legend=legend,
    x=z_values,
    y=distance_modulus_values,
    y_err=sigma_distance_moduli,
    y_model=predicted_distance_modulus_values,
    label=f"Model: H0 = {h0_label} km/s/Mpc",
    x_scale="log"
)

# Plot the residual analysis
plot_residuals(z_values = z_values, residuals = residuals, bins = 40)
