import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, DotProduct, WhiteKernel
from y2005cc.data import get_data

legend, z_values, H_values, cov_matrix = get_data()
dH_values = np.sqrt(np.diag(cov_matrix))  # simple diagonal covariance matrix
z_values = z_values.reshape(-1, 1)

z_pred = np.linspace(0, z_values.max(), 500).reshape(-1, 1)

# Define RBF kernel (Squared Exponential)
kernel = ConstantKernel(constant_value=16**2) * DotProduct(sigma_0=2.0)**2
# Gaussian Process Regressor
gp = GaussianProcessRegressor(
    kernel=kernel,
    alpha=dH_values**2,
    normalize_y=False,
    n_restarts_optimizer=100,
)
gp.fit(z_values, H_values)

H_pred, sigma = gp.predict(z_pred, return_std=True)
print("H0", f"{H_pred[0]:.2f} ± {sigma[0]:.2f}") # H0 65.78 ± 4.96 km/s/Mpc
print("GP score:", gp.score(z_values, H_values)) # 0.899
print("GP marginal likelihood value", gp.log_marginal_likelihood_value_) # -137.57

plt.errorbar(z_values, H_values, yerr=dH_values, fmt='.', label='CC Data', capsize=2)
plt.plot(z_pred, H_pred, label='GP Mean (RBF)', color='gray')
plt.fill_between(z_pred.ravel(), H_pred - sigma, H_pred + sigma, color='gray', alpha=0.4, label=r'$1\sigma$')
plt.fill_between(z_pred.ravel(), H_pred - 2*sigma, H_pred + 2*sigma, color='gray', alpha=0.2, label=r'$2\sigma$')
plt.xlabel('z')
plt.xlim(0, np.max(z_values) + 0.1)
plt.ylabel('H(z) [km/s/Mpc]')
plt.title(f'Gaussian Process Regression $H_0$: {H_pred[0]:.2f} ± {sigma[0]:.2f} km/s/Mpc')
plt.legend()
plt.grid(True)
plt.show()
