import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel as C,
    WhiteKernel,
    DotProduct,
)
from y2005cc.data import get_data

legend, z_values, H_values, cov_matrix = get_data()
dH_values = np.sqrt(np.diag(cov_matrix))  # simple diagonal covariance matrix
z_values = z_values.reshape(-1, 1)

H_std = np.std(H_values)
H_mean = np.mean(H_values)
normalized_H = (H_values - H_mean) / H_std
normalised_sigma = dH_values / H_std

gp = GaussianProcessRegressor(
    kernel=C((1 / H_std) ** 2) * DotProduct() ** 4 + WhiteKernel(1e-4, "fixed"),
    alpha=normalised_sigma**2,
    normalize_y=False,
    n_restarts_optimizer=20,
)
gp.fit(z_values.reshape(-1, 1), normalized_H.reshape(-1, 1))

z_pred = np.linspace(0, z_values.max(), 100).reshape(-1, 1)
H_pred, cov_pred = gp.predict(z_pred, return_cov=True)
cov_pred = cov_pred * H_std**2
H_pred = H_pred * H_std + H_mean
sigma = np.sqrt(np.diag(cov_pred))
likelihood = gp.log_marginal_likelihood_value_ - H_values.size * np.log(H_std)

print("H0", f"{H_pred[0]:.1f} ± {sigma[0]:.1f}")  # H0 67.4 +- 4.9 km/s/Mpc
print("GP score:", gp.score(z_values, normalized_H))  # 0.898
print("Likelihood", likelihood)  # -137.29
print(
    "GP kernel", gp.kernel_
)  # 0.026**2 * DotProduct(sigma_0=2.64) ** 4 + WhiteKernel(noise_level=0.0001)

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
plt.savefig("cosmic_chronometers/cc_gp_cov_matrix.png", dpi=300)
plt.close()
