import matplotlib.pyplot as plt
import corner
import numpy as np
from scipy.optimize import curve_fit

# Speed of light (km/s)
C = 299792.458

def model_distance_modulus(z, h, p):
    normalized_h0 = 100 * h # (km/s/Mpc)
    a0_over_ae = (1 + z) ** (1 / p)
    comoving_distance = (2 * C * p / normalized_h0) * (1 - 1 / np.sqrt(a0_over_ae))
    luminosity_distance = a0_over_ae * comoving_distance
    return 25 + 5 * np.log10(luminosity_distance)

# Generate mock data
def generate_mock_data(z, h_true, p_true, noise_level):
    mu_true = model_distance_modulus(z, h_true, p_true)
    noise = np.random.normal(0, noise_level, len(z))  # Gaussian noise
    mu_obs = mu_true + noise
    return mu_obs, mu_true

# Fit the data using curve_fit
def fit_data(z, mu_obs):
    # Fit the data to recover h and p.
    popt, pcov = curve_fit(model_distance_modulus, z, mu_obs, p0=[0.4, 0.4], bounds=([0.1, 0.1], [1, 1]))
    return popt, np.sqrt(np.diag(pcov))

# Simulate multiple datasets and assess bias
def simulate_bias(z, h_true, p_true, noise_level, num_simulations=10000):
    recovered_params = []
    for _ in range(num_simulations):
        mu_obs, _ = generate_mock_data(z, h_true, p_true, noise_level)
        popt, _ = fit_data(z, mu_obs)
        recovered_params.append(popt)
    return np.array(recovered_params)

# Parameters
z = np.linspace(0.001, 2.3, 1700)
h_true = 0.722
p_true = 0.675
num_simulations = 10000
noise_level = 0.2

# Run simulations
samples = simulate_bias(z, h_true, p_true, noise_level=noise_level, num_simulations=num_simulations)

# Analyze results
h_recovered = samples[:, 0]
p_recovered = samples[:, 1]

corner.corner(
    samples,
    labels=[r"$h_0$", r"$p$"],
    truths=[h_true, p_true],
    show_titles=True,
    title_fmt=".5f",
    title_kwargs={"fontsize": 12},
    quantiles=[0.16, 0.5, 0.84],
)
plt.show()

h_bias = np.mean(h_recovered) - h_true
p_bias = np.mean(p_recovered) - p_true
h_bias_error = np.std(h_recovered) / np.sqrt(num_simulations)
p_bias_error = np.std(p_recovered) / np.sqrt(num_simulations)

print(f"Bias in h0: {h_bias:.5f} ± {h_bias_error:.6f}")
print(f"Bias in p: {p_bias:.5f} ± {p_bias_error:.6f}")

covariance_matrix = np.cov(h_recovered, p_recovered)
correlation_coefficient = covariance_matrix[0, 1] / (np.sqrt(covariance_matrix[0, 0]) * np.sqrt(covariance_matrix[1, 1]))
print(f"Correlation Coefficient: {correlation_coefficient:.4f}")
