import numpy as np
from scipy.optimize import curve_fit
from plotting import plot_predictions, print_color, plot_residuals
from y2022pantheonSHOES.data import get_data

legend, z_values, distance_modulus_values, sigma_distance_moduli = get_data()

# Speed of light (km/s)
C = 299792.458


# Theoretical distance modulus for matter-dominated, flat universe:
# max angular distance dA (minimum theta) occurs at z = 0.5832, dA = 631.31 Mpc
def model_distance_modulus(z, h0, dte_He):
    alpha = -1 + (1 + 1.5 * dte_He) ** (1 / 3)
    beta = 1 + (3 / alpha) + (3 / alpha ** 2)
    sqr_z = np.sqrt(1 + 4 * ((1 + z) * beta - 1) / 3)

    a0_over_ae = (0.5 * alpha * (sqr_z - 1)) ** 2
    comoving_distance = (2 * C / h0) * (1 - 1 / np.sqrt(a0_over_ae))
    luminosity_distance = comoving_distance * a0_over_ae
    return 25 + 5 * np.log10(luminosity_distance)


# Fit the curve to the data
[params_opt, params_cov] = curve_fit(
    f=model_distance_modulus,
    xdata=z_values,
    ydata=distance_modulus_values,
    sigma=sigma_distance_moduli,
    absolute_sigma=True,
    p0=[140, 11.7]
)

# Extract the optimal value for H0 and dte_over_te
[h0, dte_He] = params_opt
[h0_std, dte_He_std] = np.sqrt(np.diag(params_cov))

# Calculate residuals
predicted_distance_modulus_values = model_distance_modulus(z=z_values, h0=h0, dte_He=dte_He)
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
print_color("Δte x He", f"{dte_He:.5f} ± {dte_He_std:.5f}")

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
