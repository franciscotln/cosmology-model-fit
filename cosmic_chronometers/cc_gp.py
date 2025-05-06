import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from y2005cc.data import get_data

legend, z_values, H_values, cov_matrix = get_data()
dH_values = np.sqrt(np.diag(cov_matrix))  # simple diagonal covariance matrix
z_values = z_values.reshape(-1, 1)

z_pred = np.linspace(0, z_values.max(), 500).reshape(-1, 1)

# Define RBF kernel (Squared Exponential)
kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 10.0))

# Gaussian Process Regressor
gp = GaussianProcessRegressor(
    kernel=kernel,
    alpha=dH_values**2,
    normalize_y=False,
    n_restarts_optimizer=100,
)
gp.fit(z_values, H_values)

H_pred, sigma = gp.predict(z_pred, return_std=True)
print("H0", f"{H_pred[0]:.2f} ± {sigma[0]:.2f}")

plt.errorbar(z_values, H_values, yerr=dH_values, fmt='.', label='CC Data', capsize=2)
plt.plot(z_pred, H_pred, label='GP Mean (RBF)', color='green')
plt.fill_between(z_pred.ravel(), H_pred - sigma, H_pred + sigma, color='green', alpha=0.2, label=r'$1\sigma$')
plt.xlabel('z')
plt.xlim(0, np.max(z_values) + 0.1)
plt.ylabel('H(z) [km/s/Mpc]')
plt.title('Gaussian Process Regression with RBF Kernel')
plt.legend()
plt.grid(True)
plt.show()
