import sys
sys.path.append('/Users/francisco.neto/Documents/private/cosmology-model-fit')

from y2018pantheon.data import get_data
import numpy as np
from scipy.optimize import curve_fit
import scipy.stats as stats
from plotting import plot_predictions, print_color, plot_residuals
from y2018pantheon.data import get_data

legend, z_values, distance_modulus_values, sigma_distance_moduli = get_data()

# Speed of light (km/s)
C = 299792.458


# ΛCDM
def integral_of_e_z(zs, omega_m):
    i = 0
    res = np.empty((len(zs),), dtype=np.float64)
    for z_item in zs:
        z_axis = np.linspace(0, z_item, 100)
        integ = np.trapz([(1 / np.sqrt(omega_m * (1 + z) ** 3 + 1 - omega_m)) for z in z_axis], x=z_axis)
        res[i] = integ
        i = i + 1
    return res


# Define a theoretical distance modulus:
def model_distance_modulus(z, h0, omega_m):
    a0_over_ae = 1 + z
    comoving_distance = (C / h0) * integral_of_e_z(zs = z, omega_m=omega_m)
    luminosity_distance = comoving_distance * a0_over_ae
    return 25 + 5 * np.log10(luminosity_distance)


# Fit the curve to the data
[params_opt, params_cov] = curve_fit(
    f=model_distance_modulus,
    xdata=z_values,
    ydata=distance_modulus_values,
    sigma=sigma_distance_moduli,
    absolute_sigma=True,
    p0=[70, 0.3]
)

# Extract the optimal values for H0 ~ 72.25 and p0 = Ω_m ~ 0.387
[h0, omega_m] = params_opt
[h0_std, omega_m_std] = np.sqrt(np.diag(params_cov))

# Calculate residuals
predicted_distance_modulus_values = model_distance_modulus(z=z_values, h0=h0, omega_m=omega_m)
residuals = predicted_distance_modulus_values - distance_modulus_values

# Compute skewness
skewness = stats.skew(residuals)

# Compute kurtosis
kurtosis = stats.kurtosis(residuals)

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
print_color("z range", f"{z_values[0]:.3f} - {z_values[-1]:.3f}")
print_color("Sample size", len(z_values))
print_color("Estimated H0 (km/s/Mpc)", h0_label)
print_color("R-squared (%)", f"{100 * r_squared:.2f}")
print_color("RMSD (mag)", f"{rmsd:5f}")
print_color("Ω_m", f"{omega_m:.3f} ± {omega_m_std:.4f}")
print_color("Skewness of residuals", f"{skewness:.3f}")
print_color("kurtosis of residuals", f"{kurtosis:.3f}")

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
plot_residuals(z_values = z_values, residuals = residuals, y_err=sigma_distance_moduli, bins = 40)
