import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, WhiteKernel, DotProduct
from y2005cc.data import get_data

legend, z_values, H_values, cov_matrix = get_data()
dH_values = np.sqrt(np.diag(cov_matrix))  # simple diagonal covariance matrix
z_values = z_values.reshape(-1, 1)

H_std = np.std(H_values)
H_mean = np.mean(H_values)
normalized_H = (H_values - H_mean) / H_std
normalised_sigma = dH_values / H_std


kernel = C() * DotProduct()**3 + WhiteKernel(0.005, 'fixed')
gp = GaussianProcessRegressor(
    kernel=kernel,
    alpha=normalised_sigma**2,
    normalize_y=False,
    n_restarts_optimizer=10,
)
gp.fit(z_values.reshape(-1,1), normalized_H.reshape(-1,1))

z_pred = np.linspace(0, z_values.max(), 500).reshape(-1, 1)
H_pred, cov_pred = gp.predict(z_pred, return_cov=True)
cov_pred = cov_pred * H_std**2
H_pred = H_pred * H_std + H_mean
sigma = np.sqrt(np.diag(cov_pred))

print("H0", f"{H_pred[0]:.2f} ± {sigma[0]:.2f}") # H0 67.66 ± 5.79 km/s/Mpc
print("GP score:", gp.score(z_values, normalized_H)) # 0.897
print("GP marginal likelihood value", gp.log_marginal_likelihood_value_ - H_values.size * np.log(H_std)) # -137.5
print("GP kernel", gp.kernel_)
print("cov mat condition number", np.linalg.cond(cov_pred)) # 232.6

plt.style.use('dark_background')
plt.errorbar(z_values, H_values, yerr=dH_values, fmt='.', label='CC Data', capsize=2)
plt.plot(z_pred, H_pred, label='GP Mean $DotProduct(2.14)^3$')
plt.fill_between(z_pred.ravel(), H_pred - sigma, H_pred + sigma, alpha=0.5, label=r'$1\sigma$')
plt.fill_between(z_pred.ravel(), H_pred - 2 * sigma, H_pred + 2 * sigma, alpha=0.3, label=r'$2\sigma$')
plt.xlabel('z')
plt.xlim(0, np.max(z_values) + 0.1)
plt.ylabel('H(z) [km/s/Mpc]')
plt.title(f'Gaussian Process $H_0$: {H_pred[0]:.1f} ± {sigma[0]:.1f} km/s/Mpc')
plt.legend()
plt.grid(True)
plt.savefig('cosmic_chronometers/cc_gp.png', dpi=400)
plt.close()

plt.imshow(cov_pred, cmap='hot', interpolation='none')
plt.colorbar()
plt.title('Covariance Matrix')
plt.savefig('cosmic_chronometers/cc_gp_cov_matrix.png', dpi=400)
plt.close()

# Likelihood function
s0 = np.logspace(-5, 5, num=100)
offset = np.logspace(-5, 5, num=100)
s0_grid, offset_grid = np.meshgrid(s0, offset)

log_marginal_likelihood = [
    gp.log_marginal_likelihood(theta=np.log([c0_v, scale_v]))
    for c0_v, scale_v in zip(s0_grid.ravel(), offset_grid.ravel())
]
log_marginal_likelihood = np.reshape(log_marginal_likelihood, offset_grid.shape)
vmin, vmax = (-log_marginal_likelihood).min(), 150
level = np.around(np.logspace(np.log10(vmin), np.log10(vmax), num=20), decimals=1)
plt.contour(
    s0_grid,
    offset_grid,
    -log_marginal_likelihood,
    levels=level,
    norm=LogNorm(vmin=vmin, vmax=vmax),
)
plt.colorbar()
plt.xscale("log")
plt.yscale("log")
plt.xlabel("$\sigma_0$")
plt.ylabel("offset")
plt.title("Negative log-marginal-likelihood")
plt.savefig('cosmic_chronometers/cc_gp_likelihood.png', dpi=400)
plt.close()
