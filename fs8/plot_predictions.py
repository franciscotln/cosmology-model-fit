import matplotlib.pyplot as plt
import numpy as np


def plot_predictions(fs8_theory, data, q, f_err=1.0):
    z_plot = np.linspace(0, np.max(data["z"]), 200)
    fs8_plot = fs8_theory(z_plot)

    scaled_fs8 = data["fs8"] * q
    scaled_err = data["fs8_err"] * q / f_err

    plt.errorbar(data["z"], scaled_fs8, yerr=scaled_err, fmt=".", label="data")
    plt.plot(z_plot, fs8_plot, label="theory", color="C1")
    plt.xlabel("z")
    plt.ylabel(r"$f\sigma_8(z)$")
    plt.legend()
    plt.show()

    residuals = scaled_fs8 - fs8_theory(data["z"])
    plt.errorbar(data["z"], residuals, yerr=scaled_err, fmt=".", label="residuals")
    plt.axhline(0, color="k", ls="--")
    plt.xlabel("z")
    plt.ylabel("residuals")
    plt.legend()
    plt.show()
