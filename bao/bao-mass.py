import sys
sys.path.append('/Users/francisco.neto/Documents/private/cosmology-model-fit')

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

z_vals = np.array([0.38, 0.51, 0.70, 0.85, 1.48, 2.33, 2.33], dtype=np.float64)

distances_DM = np.array([10.27, 13.38, 17.65, 19.50, 30.21, 37.60, 37.30], dtype=np.float64)
sigma_distances_DM = np.array([0.15, 0.18, 0.30, 1.00, 0.79, 1.90, 1.70], dtype=np.float64)

distances_DH = np.array([24.89, 22.43, 19.78, 19.60, 13.23, 8.93, 9.08], dtype=np.float64)
sigma_distances_DH = np.array([0.58, 0.48, 0.46, 2.10, 0.47, 0.28, 0.34], dtype=np.float64)

# Speed of light (km/s)
C = 299792.458

# Hubble constant (km/s/Mpc)
h0 = 72

def integral_of_e_z(z, p):
    a0_over_ae = np.power(1 + z, 1 / (1 - p))
    return 2 * ((1 - p)**(-p/(3-p))) * (1 - 1 / np.sqrt(a0_over_ae))


def DM(z, p):
    # flat universe
    comoving_distance = (C / h0) * integral_of_e_z(z=z, p=p)
    return comoving_distance


def DH(z, p):
    return (C / h0) * np.power(1 + z, (-1.5 + p) / (1 - p))

def chi_squared(params):
    """Computes the chi-squared error for the combined transverse and radial fits."""
    p, r_d = params

    # Model predictions
    DM_model = DM(z_vals, p) / r_d
    DH_model = DH(z_vals, p) / r_d

    # Compute chi-squared
    chi2_DM = np.sum(((DM_model - distances_DM) / sigma_distances_DM) ** 2)
    chi2_DH = np.sum(((DH_model - distances_DH) / sigma_distances_DH) ** 2)

    return chi2_DM + chi2_DH  # Total chi-squared error

# Initial guess: (p, r_d)
initial_guess = [0, 100]

# Run the optimization
result = minimize(chi_squared, initial_guess, bounds=[(-1, 0.99), (20, 200)])

# Extract best-fit parameters
p_fit, r_d_fit = result.x
p_err, r_d_err = np.sqrt(np.diag(result.hess_inv.todense())) if result.success else (None, None)

print(f"Best-fit p: {p_fit:.3f} ± {p_err:.3f}")
print(f"Best-fit r_d: {r_d_fit:.2f} ± {r_d_err:.2f}")

# Plot results
z_range = np.linspace(0.01, 3, 100)
figure, (trans, rad) = plt.subplots(2, 1)

trans.plot(z_range, DM(z_range, p_fit) / r_d_fit, 'r-', label="Best Fit")
trans.scatter(x=z_vals, y=distances_DM, label="Data", marker='.', alpha=0.6)
trans.errorbar(x=z_vals, y=distances_DM, yerr=sigma_distances_DM, fmt='|', capsize=2)
trans.set_xscale('log')
trans.set_ylabel(r'$D_M / r_d$')
trans.legend()

rad.plot(z_range, DH(z_range, p_fit) / r_d_fit, 'g-', label="Best Fit")
rad.scatter(x=z_vals, y=distances_DH, label="Data", marker='.', alpha=0.6)
rad.errorbar(x=z_vals, y=distances_DH, yerr=sigma_distances_DH, fmt='|', capsize=2)
rad.set_xscale('log')
rad.set_ylabel(r'$D_H / r_d$')
rad.set_xlabel('z')
rad.legend()

plt.show()
