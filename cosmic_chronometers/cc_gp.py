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

fixed_noise_to_avoid_singular_matrix = WhiteKernel(
    noise_level=0.005, noise_level_bounds="fixed"
)
kernel = C(0.5**2) * DotProduct(sigma_0=3) ** 4 + fixed_noise_to_avoid_singular_matrix
gp = GaussianProcessRegressor(
    kernel=kernel,
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

print("H0", f"{H_pred[0]:.2f} ± {sigma[0]:.2f}")
# H0 67.28 +- 5.77 km/s/Mpc
print("GP score:", gp.score(z_values, normalized_H))
# 0.898
print(
    "GP marginal likelihood value",
    gp.log_marginal_likelihood_value_ - H_values.size * np.log(H_std),
)
# -137.56
print("GP kernel", gp.kernel_)
# 0.0258**2 * DotProduct(sigma_0=2.66) ** 4 + WhiteKernel(noise_level=0.005)
print("cov mat condition number", np.linalg.cond(cov_pred))
# 703

plt.style.use("dark_background")
plt.errorbar(z_values, H_values, yerr=dH_values, fmt=".", label="CC Data", capsize=2)
plt.plot(z_pred, H_pred, label="GP Mean")
plt.fill_between(
    z_pred.ravel(), H_pred - sigma, H_pred + sigma, alpha=0.5, label=r"$1\sigma$"
)
plt.fill_between(
    z_pred.ravel(),
    H_pred - 2 * sigma,
    H_pred + 2 * sigma,
    alpha=0.3,
    label=r"$2\sigma$",
)
plt.xlabel("z")
plt.xlim(0, np.max(z_values) + 0.1)
plt.ylabel("H(z) [km/s/Mpc]")
plt.title(f"Gaussian Process $H_0$: {H_pred[0]:.1f} ± {sigma[0]:.1f} km/s/Mpc")
plt.legend()
plt.grid(True)
plt.savefig("cosmic_chronometers/cc_gp.png", dpi=300)
plt.close()

plt.imshow(cov_pred, cmap="hot", interpolation="none")
plt.colorbar()
plt.title("Covariance Matrix")
plt.savefig("cosmic_chronometers/cc_gp_cov_matrix.png", dpi=300)
plt.close()
