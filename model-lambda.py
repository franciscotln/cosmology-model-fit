import numpy as np
from scipy.optimize import curve_fit
from plotting import plot_predictions, print_color, plot_residuals
from y2022pantheonSHOES.data import get_data

legend, z_values, distance_modulus_values, sigma_distance_moduli = get_data()

# Speed of light (km/s)
C = 299792.458


# Theoretical distance modulus for cosmological constant-dominated, flat universe:
def model_distance_modulus(z, h0, dte):
    delta_t_emission = dte + 0.000001 #  Units: (km/s/Mpc)^-1
    p = 1 - np.exp(-h0 * delta_t_emission)

    a0_over_ae = (1 / p) * (1 - np.power(1 - p, 1 + z))
    comoving_distance = (C / h0) * (a0_over_ae - 1)
    luminosity_distance = comoving_distance * a0_over_ae
    return 25 + 5 * np.log10(luminosity_distance)


# Fit the curve to the data
[params_opt, params_cov] = curve_fit(
    f=model_distance_modulus,
    xdata=z_values,
    ydata=distance_modulus_values,
    sigma=sigma_distance_moduli,
    absolute_sigma=True,
    p0=[61.30, 0.0052]
)

# Extract the optimal value for H0 and n0
[h0, dte] = params_opt
[h0_std, dte_std] = np.sqrt(np.diag(params_cov))

# Calculate residuals
predicted_distance_modulus_values = model_distance_modulus(z=z_values, h0=h0, dte=dte)
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
print_color("dte", f"{dte:.5f} ± {dte_std:.5f}")

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
