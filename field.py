import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d


# values from BAO + CC + DES5Y
G = 1  # 6.6743e-11 # m^3/kg/s^2 normalised
c = 1  # 299792458 # m/s normalised
Rho_de_0 = 1  # normalised
H0 = 1  # normalised. Fit gives: 67.1 # km/s/Mpc
O_m = 0.307
O_r = 4 * 10**-5  # Radiation densit
w0 = -0.833  # Equation of state parameter from fit

# Range of scale factors
a_vals = np.linspace(1e-3, 3, 2000)

# Dark energy equation of state
w_de = lambda a: -1 + 2 * (1 + w0) * a**3 / (1 + a**3)

Rho_de = lambda a: Rho_de_0 * (2 / (1 + a**3)) ** (2 * (1 + w0))

H = lambda a: H0 * np.sqrt(O_m * a**-3 + O_r * a**-4 + (1 - O_m) * Rho_de(a) / Rho_de_0)

V_phi = lambda a: (1 - w_de(a)) * Rho_de(a) / 2

d_phi_da = lambda a: np.sqrt(Rho_de(a) * (1 + w_de(a))) / (a * H(a))

phi_vals = cumulative_trapezoid(d_phi_da(a_vals), a_vals, initial=0)

a_of_phi = interp1d(phi_vals, a_vals, bounds_error=False, fill_value="extrapolate")

V_of_phi = lambda phi: V_phi(a_of_phi(phi))

phi_plot = np.linspace(min(phi_vals), max(phi_vals), 2000)

d_phi_dt_val = d_phi_da(a_vals) * H(a_vals) * a_vals

# Scalar field
plt.figure(figsize=(8, 5))
plt.plot(a_vals, phi_vals, label=r"$\phi(a)$")
plt.axvline(x=1, color="r", linestyle="--", label="Current time")
plt.xlabel(r"$a$")
plt.ylabel(r"$\phi(a)$")
plt.xlim(0, 2)
plt.ylim(0, max(phi_vals))
plt.title(r"Scalar Field $\phi(a)$")
plt.legend()
plt.grid(True)
plt.show()

# Potential
plt.figure(figsize=(8, 5))
plt.plot(phi_plot, V_of_phi(phi_plot), label=r"$V(\phi)$")
plt.xlabel(r"$\phi$")
plt.ylabel(r"$V(\phi)$")
plt.title(r"Scalar Field Potential $V(\phi)$")
plt.legend()
plt.grid(True)
plt.show()

# Plot V(a)
plt.figure(figsize=(8, 5))
plt.plot(a_vals, V_phi(a_vals), label=r"$V(a)$")
plt.xlabel(r"$a$")
plt.ylabel(r"$V(a)$")
plt.title(r"Scalar Field Potential $V(a)$")
plt.legend()
plt.grid(True)
plt.show()

dt_da = lambda a: 1 / (a * H(a))
t_vals = cumulative_trapezoid(dt_da(a_vals), a_vals, initial=0)
a_of_t = interp1d(t_vals, a_vals, bounds_error=False, fill_value="extrapolate")
phi_of_t = lambda t: np.interp(t, t_vals, phi_vals)
t_grid = np.linspace(min(t_vals), max(t_vals), 1000)

# Kinetic term
plt.figure(figsize=(8, 5))
plt.plot(t_vals, 0.5 * d_phi_dt_val**2, label=r"Kinetic term")
plt.xlabel("t")
plt.ylabel(r"0.5 * $\left(\frac{d\phi}{dt}\right)^2$")
plt.title(r"Scalar Field Kinetic Term $\left(\frac{d\phi}{dt}\right)^2$")
plt.axvline(
    x=t_vals[np.argmax(0.5 * d_phi_dt_val**2)],
    color="g",
    linestyle="--",
    label="Max speed",
    alpha=0.5,
)
plt.grid(True)
plt.legend()
plt.show()


plt.figure(figsize=(8, 5))
plt.plot(t_grid, phi_of_t(t_grid), label=r"$\phi(t)$")
plt.xlabel(r"$t$")
plt.ylabel(r"$\phi(t)$")
plt.title(r"Scalar Field $\phi$ vs Time $t$")
plt.legend()
plt.grid(True)
plt.show()
