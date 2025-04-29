import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from y2005cc.data import get_data


legend, z_values, H_values, dH_values = get_data()
z_values = z_values.to_numpy()
H_values = H_values.to_numpy()
dH_values = dH_values.to_numpy()


def H_lcdm_z(z, params = [67.9657, 0.3229]):
    h0, o_m = params
    return h0 * np.sqrt(o_m * (1 + z)**3 + (1 - o_m))


def H_wcdm_z(z, params = [71.3986, 0.3050, -1.3285]):
    h0, o_m, w0 = params
    exponent = 3 * (1 + w0)
    sum = 1 + z
    return h0 * np.sqrt(o_m * sum**3 + (1 - o_m) * sum**exponent)


def H_wxcdm_z(z, params = [72.4395, 0.3006, -1.4318]):
    h0, o_m, w0 = params
    exponent = 3 * (1 + w0)
    sum = 1 + z
    return h0 * np.sqrt(o_m * sum**3 + (1 - o_m) * ((2 * sum**2) / (1 + sum**2))**exponent)


def main():
    z_grid = np.linspace(0, np.max(z_values), 2000)
    n_bootstrap = 20_000
    H_fits = np.zeros((n_bootstrap, len(z_grid)))

    # Bootstrap loop: resampling with replacement
    for i in range(n_bootstrap):
        indices = np.random.choice(len(z_values), size=len(z_values), replace=True)
        z_boot = z_values[indices]
        H_boot = H_values[indices]
        dH_boot = dH_values[indices]

        sort_idx = np.argsort(z_boot)
        z_boot = z_boot[sort_idx]
        H_boot = H_boot[sort_idx]
        dH_boot = dH_boot[sort_idx]

        spline_i = UnivariateSpline(z_boot, H_boot, w=1/dH_boot**2, k=2)
        H_fits[i] = spline_i(z_grid)

    H0_16, H0_50, H0_84 = np.percentile(H_fits, [15.9, 50, 84.1], axis=0)
    upper = H0_84[0] - H0_50[0]
    lower = H0_50[0] - H0_16[0]
    print(f"H0: {H0_50[0]:.2f} +{upper:.2f} - {lower:.2f} km/s/Mpc")

    plt.figure(figsize=(10, 6))
    plt.errorbar(z_values, H_values, yerr=dH_values, fmt='.', label=legend, capsize=2)
    plt.plot(z_grid, H0_50, label=fr"Spline: $H_0 = {H0_50[0]:.2f}^{{+{upper:.2f}}}_{{-{lower:.2f}}}$ km/s/Mpc", color='black')
    plt.fill_between(z_grid, H0_16, H0_84, color='gray', alpha=0.3, label=r'spline $1\sigma$')
    plt.plot(z_grid, H_lcdm_z(z_grid), label=f'ΛCDM $H_0: 67.97^{{+{2.21}}}_{{-{2.24}}}$ km/s/Mpc', color='green', lw=1, alpha=0.6)
    plt.plot(z_grid, H_wcdm_z(z_grid), label='wCDM $H_0: 71.40^{{+{7.73}}}_{{-{5.70}}}$ km/s/Mpc', color='red', lw=1, alpha=0.6)
    plt.plot(z_grid, H_wxcdm_z(z_grid), label='wxCDM $H_0: 72.44^{{+{8.3109}}}_{{-{6.2978}}}$ km/s/Mpc', color='blue', lw=1, alpha=0.6)
    plt.title('Smoothing Spline 2deg with resampling uncertainties')
    plt.xlabel('z')
    plt.xlim(0, 2)
    plt.ylabel("H(z) - km/s/Mpc")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("cc_spline.png", dpi=800)
    plt.close()

if __name__ == "__main__":
    main()
