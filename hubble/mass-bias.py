import numpy as np
import matplotlib.pyplot as plt
import corner
import seaborn as sns
from scipy.optimize import minimize
from scipy.stats import pearsonr
from multiprocessing import Pool

# Speed of light (km/s)
C = 299792.458

def model_distance_modulus(z, h, p):
    normalized_h0 = 100 * h  # (km/s/Mpc)
    a0_over_ae = (1 + z) ** (1 / p)
    comoving_distance = (2 * C * p / normalized_h0) * (1 - 1 / np.sqrt(a0_over_ae))
    luminosity_distance = a0_over_ae * comoving_distance
    return 25 + 5 * np.log10(luminosity_distance)

# Generate mock data with full covariance matrix
def generate_mock_data(z, h_true, p_true, cov_matrix):
    mu_true = model_distance_modulus(z, h_true, p_true)
    L = np.linalg.cholesky(cov_matrix)   # Cholesky decomposition
    noise = np.random.randn(len(z)) @ L  # Correlated noise
    mu_obs = mu_true + noise
    return mu_obs

# Define chi-squared function for fitting
def chi_squared(params, z, mu_obs, cov_inv):
    h, p = params
    mu_model = model_distance_modulus(z, h, p)
    residuals = mu_obs - mu_model
    return residuals.T @ cov_inv @ residuals

# Fit data using covariance-aware likelihood
def fit_data(args):
    z, mu_obs, cov_inv = args
    result = minimize(chi_squared, x0=[0.7, 0.3], bounds=[(0.1, 1), (0.1, 1)], args=(z, mu_obs, cov_inv))
    return result.x

# Multiprocessing simulation
def simulate_bias_parallel(z, h_true, p_true, cov_matrix, num_simulations, num_workers):

    with Pool(num_workers) as pool:
        cov_inv = np.linalg.inv(cov_matrix)
        args_list = [(z, generate_mock_data(z, h_true, p_true, cov_matrix), cov_inv) for _ in range(num_simulations)]
        recovered_params = pool.map(fit_data, args_list)

    return np.array(recovered_params)

if __name__ == "__main__":
    # Parameters
    z = np.linspace(0.001, 2.3, 1700)
    h_true = 0.722
    p_true = 0.675

    # Mock covariance matrix
    np.random.seed(42)
    diag_errors = np.random.uniform(0.1, 0.3, size=len(z))
    cov_matrix = np.diag(diag_errors**2)

    for i in range(len(z) - 1):
        cov_matrix[i, i+1] = cov_matrix[i+1, i] = 0.3 * cov_matrix[i, i]  # 30% correlation


    # Run parallel simulations
    samples = simulate_bias_parallel(z, h_true, p_true, cov_matrix, num_simulations=2000, num_workers=20)

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

    # Compute bias
    [h_16, h_50, h_84] = np.percentile(h_recovered, [16, 50, 84])
    [p_16, p_50, p_84] = np.percentile(p_recovered, [16, 50, 84])
    h_bias = h_50 - h_true
    p_bias = p_50 - p_true

    print(f"Bias in h0: {h_bias:.5f} -{(h_50 - h_16):.6f}/+{(h_84 - h_50):.6f}")
    print(f"Bias in p: {p_bias:.5f} -{(p_50 - p_16):.6f}/+{(p_84 - p_50):.6f}")
    print(f"Correlation Coefficient: {pearsonr(h_recovered, p_recovered)}")
