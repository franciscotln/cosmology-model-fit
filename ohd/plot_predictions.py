import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np


def plot_cc_predictions(H_z, z, H, H_err, label, err_scaling=None):
    residual = H - H_z(z)
    y_err = H_err if err_scaling is None else H_err / err_scaling
    z_smooth = np.linspace(0, max(z), 100)
    plt.figure(figsize=(8, 6))
    if err_scaling is not None:
        plt.errorbar(
            x=z,
            y=H,
            yerr=H_err,
            fmt=".",
            color="blue",
            alpha=0.15,
            label="CCH (unscaled)",
            capsize=2,
            linestyle="None",
        )
    plt.errorbar(
        x=z,
        y=H,
        yerr=y_err,
        fmt=".",
        color="blue",
        alpha=0.5,
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
    if err_scaling is not None:
        plt.errorbar(
            x=z,
            y=residual,
            yerr=H_err,
            fmt=".",
            color="blue",
            alpha=0.15,
            label="Residuals (unscaled)",
            capsize=2,
            linestyle="None",
        )
    plt.errorbar(
        x=z,
        y=residual,
        yerr=y_err,
        fmt=".",
        color="blue",
        alpha=0.5,
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
