import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np


def plot_cc_predictions(H_z, z, H, H_err, label, method=None, err_scaling=None):
    residual = H - H_z(z)
    y_err = H_err if err_scaling is None else H_err / err_scaling
    z_smooth = np.linspace(0, max(z), 100)
    method_colors = {"F": "tab:blue", "D": "tab:orange", "L": "tab:green"}
    if method is None:
        method_groups = [(None, np.ones(len(z), dtype=bool))]
    elif np.ndim(method) == 0:
        method_groups = [(str(method), np.ones(len(z), dtype=bool))]
    else:
        method = np.asarray(method)
        if len(method) != len(z):
            raise ValueError("method must have the same length as z")
        method_groups = [
            (method_name, method == method_name) for method_name in np.unique(method)
        ]

    def plot_data(y, errors, base_label, alpha):
        for method_name, mask in method_groups:
            color = method_colors.get(method_name, "tab:blue")
            data_label = base_label if method_name is None else method_name
            plt.errorbar(
                x=np.asarray(z)[mask],
                y=np.asarray(y)[mask],
                yerr=np.asarray(errors)[mask],
                fmt=".",
                color=color,
                alpha=alpha,
                label=data_label,
                capsize=2,
                linestyle="None",
            )

    plt.figure(figsize=(8, 6))
    if err_scaling is not None:
        plot_data(H, H_err, "CCH (unscaled)", 0.15)
    plot_data(H, y_err, "CCH", 0.5)
    plt.plot(z_smooth, H_z(z_smooth), color="red", alpha=0.5)
    plt.xlabel("Redshift (z)")
    plt.ylabel(r"$H(z)$")
    plt.xlim(0, np.max(z) + 0.2)
    plt.legend()
    plt.title(label)
    plt.show()

    plt.figure(figsize=(8, 6))
    if err_scaling is not None:
        plot_data(residual, H_err, "Residuals (unscaled)", 0.15)
    plot_data(residual, y_err, "Residuals", 0.5)
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Redshift (z)")
    plt.ylabel(r"$H(z) - H_{model}(z)$")
    plt.xlim(0, np.max(z) + 0.2)
    plt.title(f"Residuals")
    plt.legend()
    plt.show()
