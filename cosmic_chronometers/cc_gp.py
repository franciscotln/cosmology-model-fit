import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, DotProduct
from y2005cc.data import get_data

legend, z_values, H_values, cov_matrix = get_data()
dH_values = np.sqrt(np.diag(cov_matrix))  # simple diagonal covariance matrix
z_values = z_values.reshape(-1, 1)

z_pred = np.linspace(0, z_values.max(), 500).reshape(-1, 1)

# Define DotProduct kernel
gp = GaussianProcessRegressor(
    kernel = C(34**2) * DotProduct(1.5)**1.5,
    alpha = dH_values**2,
    normalize_y = False,
    n_restarts_optimizer = 100,
)
gp.fit(z_values, H_values)

H_pred, sigma = gp.predict(z_pred, return_std=True)
print("H0", f"{H_pred[0]:.2f} ± {sigma[0]:.2f}") # H0 65.5 ± 5.0 km/s/Mpc
print("GP score:", gp.score(z_values, H_values)) # 0.9
print("GP marginal likelihood value", gp.log_marginal_likelihood_value_) # -137.51
print("GP kernel", gp.kernel_)

plt.style.use('dark_background')  # Enable dark mode
plt.errorbar(z_values, H_values, yerr=dH_values, fmt='.', label='CC Data', capsize=2)
plt.plot(z_pred, H_pred, label='GP Mean ($DotProduct() ^ 1.5$)')
plt.fill_between(z_pred.ravel(), H_pred - sigma, H_pred + sigma, alpha=0.5, label=r'$1\sigma$')
plt.fill_between(z_pred.ravel(), H_pred - 2*sigma, H_pred + 2*sigma, alpha=0.3, label=r'$2\sigma$')
plt.xlabel('z')
plt.xlim(0, np.max(z_values) + 0.1)
plt.ylabel('H(z) [km/s/Mpc]')
plt.title(f'Gaussian Process $H_0$: {H_pred[0]:.1f} ± {sigma[0]:.1f} km/s/Mpc')
plt.legend()
plt.grid(True)
plt.show()
