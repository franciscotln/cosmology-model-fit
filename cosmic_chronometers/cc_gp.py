import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, PairwiseKernel
from y2005cc.data import get_data

legend, z_values, H_values, cov_matrix = get_data()
dH_values = np.sqrt(np.diag(cov_matrix))  # simple diagonal covariance matrix
z_values = z_values.reshape(-1, 1)

z_pred = np.linspace(0, z_values.max(), 500).reshape(-1, 1)

# Define DotProduct kernel
gp = GaussianProcessRegressor(
    kernel = C(114**2) * PairwiseKernel(metric='poly', gamma=0.02),
    alpha = dH_values**2,
    normalize_y = False,
    n_restarts_optimizer = 10,
)
gp.fit(z_values.reshape(-1,1), H_values.reshape(-1,1))

H_pred, sigma = gp.predict(z_pred, return_std=True)
print("H0", f"{H_pred[0]:.2f} ± {sigma[0]:.2f}") # H0 65.40 ± 4.97 km/s/Mpc
print("GP score:", gp.score(z_values, H_values)) # 0.9
print("GP marginal likelihood value", gp.log_marginal_likelihood_value_) # -138.24
print("GP kernel", gp.kernel_)

plt.style.use('dark_background')
plt.errorbar(z_values, H_values, yerr=dH_values, fmt='.', label='CC Data', capsize=2)
plt.plot(z_pred, H_pred, label='GP Mean $PairwiseKernel(poly)$')
plt.fill_between(z_pred.ravel(), H_pred - sigma, H_pred + sigma, alpha=0.5, label=r'$1\sigma$')
plt.fill_between(z_pred.ravel(), H_pred - 2 * sigma, H_pred + 2 * sigma, alpha=0.3, label=r'$2\sigma$')
plt.xlabel('z')
plt.xlim(0, np.max(z_values) + 0.1)
plt.ylabel('H(z) [km/s/Mpc]')
plt.title(f'Gaussian Process $H_0$: {H_pred[0]:.1f} ± {sigma[0]:.1f} km/s/Mpc')
plt.legend()
plt.grid(True)
plt.savefig('cosmic_chronometers/cc_gp.png', dpi=500)
plt.close()

# Likelihood function
c0 = np.logspace(2, 7, num=100)
gamma = np.logspace(-5, 3, num=100)
c0_grid, gamma_grid = np.meshgrid(c0, gamma)

log_marginal_likelihood = [
    gp.log_marginal_likelihood(theta=np.log([c0_v, gamma_v]))
    for c0_v, gamma_v in zip(c0_grid.ravel(), gamma_grid.ravel())
]
log_marginal_likelihood = np.reshape(log_marginal_likelihood, gamma_grid.shape)
vmin, vmax = (-log_marginal_likelihood).min(), 150
level = np.around(np.logspace(np.log10(vmin), np.log10(vmax), num=20), decimals=1)
plt.contour(
    c0_grid,
    gamma_grid,
    -log_marginal_likelihood,
    levels=level,
    norm=LogNorm(vmin=vmin, vmax=vmax),
)
plt.colorbar()
plt.xscale("log")
plt.yscale("log")
plt.xlabel("$c_0$")
plt.ylabel("$\gamma$")
plt.title("Negative log-marginal-likelihood")
plt.savefig('cosmic_chronometers/cc_gp_likelihood.png', dpi=500)
plt.close()
