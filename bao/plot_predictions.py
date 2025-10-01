import numpy as np
import matplotlib.pyplot as plt

color_map = {
    "DV_over_rs": "red",
    "DM_over_rs": "blue",
    "DH_over_rs": "green",
}
latex_labels = {
    "DV_over_rs": "$D_V$ / $r_d$",
    "DM_over_rs": "$D_M$ / $r_d$",
    "DH_over_rs": "$D_H$ / $r_d$",
}
quantities_map = {"DV_over_rs": 0, "DM_over_rs": 1, "DH_over_rs": 2}


def plot_bao_predictions(theory_predictions, data, errors, title):
    z_smooth = np.linspace(0, max(data["z"]), 200)
    plt.figure(figsize=(8, 6))
    for q in set(data["quantity"]):
        quantity_mask = data["quantity"] == q
        plt.errorbar(
            x=data["z"][quantity_mask],
            y=data["value"][quantity_mask],
            yerr=errors[quantity_mask],
            fmt=".",
            color=color_map[q],
            label=latex_labels[q],
            capsize=2,
            linestyle="None",
        )
        plt.plot(
            z_smooth,
            theory_predictions(
                z_smooth, np.full_like(z_smooth, quantities_map[q], dtype=np.int32)
            ),
            color=color_map[q],
            alpha=0.5,
        )

    plt.xlabel("Redshift (z)")
    plt.ylabel(r"$O = \frac{D}{r_d}$")
    plt.legend()
    plt.grid(True)
    plt.title(title)
    plt.show()


def plot_bao_residuals(data, residuals, errors):
    for qtype, color in color_map.items():
        mask = data["quantity"] == qtype
        plt.errorbar(
            data["z"][mask],
            residuals[mask],
            yerr=errors[mask],
            fmt=".",
            color=color,
            ecolor=color,
            elinewidth=1,
            capsize=2,
            label=latex_labels[qtype],
        )
    plt.axhline(0, color="black", linestyle="--", alpha=0.5)
    plt.xlabel("Redshift $z$")
    plt.ylabel("BAO residuals (data - model)")
    plt.legend()
    plt.show()
