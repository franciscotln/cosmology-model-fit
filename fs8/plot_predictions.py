import matplotlib.pyplot as plt
import numpy as np


def plot_predictions(fs8_theory, data, q, f_err=1.0):
    PLANCK_MASK = data["omega_fid"] >= 0.3

    z_plot = np.linspace(0, np.max(data["z"]) + 0.5, 200)
    fs8_plot = fs8_theory(z_plot)

    scaled_fs8 = data["fs8"] * q
    scaled_err = data["fs8_err"] * q / f_err

    plt.errorbar(
        data["z"][PLANCK_MASK],
        scaled_fs8[PLANCK_MASK],
        yerr=scaled_err[PLANCK_MASK],
        fmt=".",
        label="Planck-based",
        color="C0",
    )
    plt.errorbar(
        data["z"][~PLANCK_MASK],
        scaled_fs8[~PLANCK_MASK],
        yerr=scaled_err[~PLANCK_MASK],
        fmt=".",
        label="WMAP-based",
        color="C1",
    )
    plt.plot(z_plot, fs8_plot, label="theory", color="C2")
    plt.xlabel("z")
    plt.ylabel(r"$f\sigma_8(z)$")
    plt.legend()
    plt.show()

    residuals = scaled_fs8 - fs8_theory(data["z"])
    plt.errorbar(
        data["z"][PLANCK_MASK],
        residuals[PLANCK_MASK],
        yerr=scaled_err[PLANCK_MASK],
        fmt=".",
        label="Planck-based residuals",
        color="C0",
    )
    plt.errorbar(
        data["z"][~PLANCK_MASK],
        residuals[~PLANCK_MASK],
        yerr=scaled_err[~PLANCK_MASK],
        fmt=".",
        label="WMAP-based residuals",
        color="C1",
    )
    plt.axhline(0, color="k", ls="--")
    plt.xlabel("z")
    plt.ylabel("residuals")
    plt.legend()
    plt.show()
