import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, Matern
from y2005cc.data import get_data

legend, z_values, H_values, cov_matrix = get_data()
dH_values = np.sqrt(np.diag(cov_matrix))  # simple diagonal covariance matrix
z_values = z_values.reshape(-1, 1)

gp = GaussianProcessRegressor(
    kernel=C(np.mean(H_values) ** 2) * Matern(length_scale=2.5, nu=4.5),
    alpha=dH_values**2,
    normalize_y=False,
    n_restarts_optimizer=20,
)
gp.fit(z_values.reshape(-1, 1), H_values.reshape(-1, 1))

z_pred = np.linspace(0, z_values.max(), 100).reshape(-1, 1)
H_pred, cov_pred = gp.predict(z_pred, return_cov=True)
sigma = np.sqrt(np.diag(cov_pred))
likelihood = gp.log_marginal_likelihood_value_

print("H0", f"{H_pred[0]:.1f} ± {sigma[0]:.1f}")  # H0 67.1 +- 5.8 km/s/Mpc
print("GP score:", gp.score(z_values, H_values))  # 0.902
print("Likelihood", likelihood)  # -138.46
print("GP kernel", gp.kernel_)  # 129**2 * Matern(length_scale=2.25, nu=4.5)

flat_z = z_pred.ravel()
plt.style.use("dark_background")
plt.errorbar(z_values, H_values, yerr=dH_values, fmt=".", label="CCH", capsize=2)
plt.plot(flat_z, H_pred, label="GP Mean")
plt.fill_between(flat_z, H_pred - sigma, H_pred + sigma, alpha=0.8, label=r"$1\sigma$")
plt.fill_between(
    flat_z,
    H_pred - 2 * sigma,
    H_pred + 2 * sigma,
    alpha=0.3,
    label=r"$2\sigma$",
)
plt.xlabel("z")
plt.xlim(0, np.max(z_values) + 0.1)
plt.ylabel("H(z) [km/s/Mpc]")
plt.title(f"GP $H_0$: {H_pred[0]:.1f} ± {sigma[0]:.1f} km/s/Mpc")
plt.legend()
plt.grid(True)
plt.savefig("cosmic_chronometers/cc_gp.png", dpi=300)
plt.close()

plt.imshow(cov_pred, cmap="hot", interpolation="none")
plt.colorbar()
plt.title("Covariance Matrix")
plt.show()
