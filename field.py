import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d

# values from DESI DR2 + rd_Planck + DES5Y
Rho_de_0 = 1  # normalised
H0 = 66.99  # Hubble constant in km/s/Mpc
Om = 0.309
Or = 4.1835e-05 / (H0 / 100) ** 2  # Radiation density
w0 = -0.799  # Equation of state parameter from fit

a_min = 1e-8
a_max = 4
N_a = 5000
a_vals = np.linspace(a_min, a_max, N_a)


w_de = lambda a: -1 + (1 + w0) * a**3

Rho_de = lambda a: Rho_de_0 * np.exp((1 + w0) * (1 - a**3))

H = (
    lambda a: H0
    * (Om * a**-3 + Or * a**-4 + (1 - Om - Or) * Rho_de(a) / Rho_de_0) ** 0.5
)

# Kinetic and Potential energy density split (properly normalized)
# For a scalar field: ρ = K + V, where K = (1/2)(dφ/dt)², V = potential
# With w = (K - V)/ρ, we get: K = (1+w)ρ/2, V = (1-w)ρ/2
K_phi = lambda a: (1 + w_de(a)) * Rho_de(a) / 2  # Kinetic energy density
V_phi = lambda a: (1 - w_de(a)) * Rho_de(a) / 2  # Potential energy density

# Dimensionless Hubble parameter
E_a = lambda a: H(a) / H0

# d_phi/da in reduced Planck units (dimensionless)
# From dφ/dt = √[2K] = √[(1+w)ρ] and da/dt = aH
# dφ/da = √[(1+w)ρ] / (aH)
d_phi_da = lambda a: np.sqrt((1 + w_de(a)) * Rho_de(a)) / (a * H(a))

phi_vals = cumulative_trapezoid(d_phi_da(a_vals), a_vals, initial=0)

a_of_phi = interp1d(phi_vals, a_vals, bounds_error=False, fill_value="extrapolate")

V_of_phi = lambda phi: V_phi(a_of_phi(phi))

phi_plot = np.linspace(min(phi_vals), max(phi_vals), 2000)

# d_phi/dt in physical units: multiply by H (in reduced Planck units)
# da/dt = H*a, so d_phi/dt = (d_phi/da) * (da/dt) = (d_phi/da) * H * a
# Note: d_phi_da already includes the proper normalization
d_phi_dt_val = d_phi_da(a_vals) * H(a_vals) * a_vals

# Scalar field
plt.figure(figsize=(8, 5))
plt.plot(a_vals, phi_vals, label=r"$\phi(a)$")
plt.axvline(x=1, color="r", linestyle="--", label="Present time")
plt.xlabel(r"$a$")
plt.ylabel(r"$\phi(a)$ [reduced Planck units]")
plt.xlim(0, None)
plt.title(r"Scalar Field $\phi(a)$")
plt.legend()
plt.grid(True)
plt.show()

# Potential
phi_today = np.interp(1, a_vals, phi_vals)  # Value of phi at a=1 (today)
plt.figure(figsize=(8, 5))
plt.plot(phi_plot, V_of_phi(phi_plot), label=r"$V(\phi)$")
plt.axvline(
    x=phi_today,
    color="r",
    linestyle="--",
    label=f"Present time ($\\phi_0$ = {phi_today:.3f})",
)
plt.xlabel(r"$\phi$ [reduced Planck units]")
plt.ylabel(r"$V(\phi)$ [normalized]")
plt.xlim(0, None)
plt.title(r"Scalar Field Potential $V(\phi)$")
plt.legend()
plt.grid(True)
plt.show()

# Plot V(a)
plt.figure(figsize=(8, 5))
plt.plot(a_vals, V_phi(a_vals), label=r"$V(a)$")
plt.xlabel(r"$a$")
plt.ylabel(r"$V(a)$ [normalized]")
plt.title(r"Scalar Field Potential $V(a)$")
plt.legend()
plt.grid(True)
plt.show()


# dt/da in units of 1/H0 (dimensionless time)
dt_da = lambda a: 1 / (a * E_a(a))
t_vals = cumulative_trapezoid(dt_da(a_vals), a_vals, initial=0)

# Convert time from 1/H0 units to Gyr
# H0 in km/s/Mpc, 1/H0 in seconds: 1 Mpc = 3.085677581e19 km
# Hubble time: 1/H0 = 9.77813 Gyr for H0 = 100 km/s/Mpc
Hubble_time_Gyr = 9.77813 / (H0 / 100)  # Hubble time in Gyr
t_vals_Gyr = t_vals * Hubble_time_Gyr
t_today_Gyr = np.interp(1, a_vals, t_vals_Gyr)

# Interpolation functions
a_of_t = interp1d(t_vals_Gyr, a_vals, bounds_error=False, fill_value="extrapolate")
phi_of_t = lambda t: np.interp(t, t_vals_Gyr, phi_vals)

# Create time grid from early times to near-future, avoiding extrapolation issues
# Use only the time range covered by our integration (not extrapolated)
t_future_Gyr = min(t_today_Gyr * 1.5, max(t_vals_Gyr) * 0.95)
t_grid_Gyr = np.linspace(t_vals_Gyr[10], t_future_Gyr, 1000)

t_at_a1 = np.interp(1, a_vals, t_vals)
print(f"Hubble time (Gyr): {Hubble_time_Gyr:.3f}")
print(f"t_today (Gyr): {t_today_Gyr:.3f}")
print(f"t_today (1/H0 units): {t_at_a1:.3f}")

# Kinetic term: (1/2)(dφ/dt)²
# This should equal the kinetic energy density K_phi = (1+w)ρ/2
kinetic_term = 0.5 * d_phi_dt_val**2
potential_term = V_phi(a_vals)

# Kinetic term vs time
plt.figure(figsize=(8, 5))
plt.plot(
    t_vals_Gyr, kinetic_term, label=r"Kinetic term $K = \frac{1}{2}(\dot{\phi})^2$"
)
plt.plot(t_vals_Gyr, potential_term, label=r"Potential term $V(\phi)$")
plt.xlabel("t [Gyr]")
plt.ylabel(r"Energy density [normalized]")
plt.title(r"Scalar Field Energy Components")
plt.axvline(x=t_today_Gyr, color="r", linestyle="--", label="Present time", alpha=0.5)
plt.grid(True)
plt.legend()
plt.show()


t_plot_range = t_grid_Gyr[t_grid_Gyr <= max(t_vals_Gyr)]

plt.figure(figsize=(8, 5))
plt.plot(t_plot_range, a_of_t(t_plot_range), label=r"$a(t)$", linewidth=2)
plt.axvline(x=t_today_Gyr, color="r", linestyle="--", label="Present time", alpha=0.7)
plt.xlabel(r"$t$ [Gyr]")
plt.ylabel(r"$a(t)$")
plt.title(r"Scale Factor $a$ vs Time $t$ [Gyr]")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


phi_of_t_quad = lambda t: 1.713572e-05 * t * (t - 1)


plt.figure(figsize=(8, 5))
plt.plot(t_plot_range, phi_of_t(t_plot_range), label=r"$\phi(t)$", linewidth=2)
plt.plot(t_plot_range, phi_of_t_quad(t_plot_range), label=r"quadratic", linestyle="--")
plt.axvline(x=t_today_Gyr, color="r", linestyle="--", label="Present time", alpha=0.7)
plt.xlabel(r"$t$ [Gyr]")
plt.ylabel(r"$\phi(t)$ [reduced Planck units]")
plt.title(r"Scalar Field $\phi$ vs Time $t$")
plt.legend()
plt.xlim(0, None)
plt.grid(True)
plt.show()
