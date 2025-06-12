import matplotlib.pyplot as plt
import numpy as np


def plot_cc_predictions(H_z, z, H, H_err, label):
    z_smooth = np.linspace(0, max(z), 100)
    plt.figure(figsize=(8, 6))
    plt.errorbar(
        x=z,
        y=H,
        yerr=H_err,
        fmt=".",
        color="blue",
        alpha=0.4,
        label="CCH",
        capsize=2,
        linestyle="None",
    )
    plt.plot(z_smooth, H_z(z_smooth), color="red", alpha=0.5)
    plt.xlabel("Redshift (z)")
    plt.ylabel(r"$H(z)$")
    plt.xlim(0, np.max(z) + 0.2)
    plt.legend()
    plt.title(label)
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.errorbar(
        x=z,
        y=H - H_z(z),
        yerr=H_err,
        fmt=".",
        color="blue",
        alpha=0.4,
        label="Residuals",
        capsize=2,
        linestyle="None",
    )
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Redshift (z)")
    plt.ylabel(r"$H(z) - H_{model}(z)$")
    plt.xlim(0, np.max(z) + 0.2)
    plt.title(f"Residuals")
    plt.legend()
    plt.show()
