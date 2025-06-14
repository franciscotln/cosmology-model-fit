import numpy as np
import matplotlib.pyplot as plt


def plot_bao_predictions(theory_predictions, data, errors, title):
    config = {
        "colors": {"DV_over_rs": "red", "DM_over_rs": "blue", "DH_over_rs": "green"},
        "latex_labels": {
            "DV_over_rs": "$D_V$ / $r_d$",
            "DM_over_rs": "$D_M$ / $r_d$",
            "DH_over_rs": "$D_H$ / $r_d$",
        },
    }

    z_smooth = np.linspace(0, max(data["z"]), 100)
    plt.figure(figsize=(8, 6))
    for q in set(data["quantity"]):
        quantity_mask = data["quantity"] == q
        plt.errorbar(
            x=data["z"][quantity_mask],
            y=data["value"][quantity_mask],
            yerr=errors[quantity_mask],
            fmt=".",
            color=config["colors"][q],
            label=config["latex_labels"][q],
            capsize=2,
            linestyle="None",
        )
        plt.plot(
            z_smooth,
            theory_predictions(z_smooth, np.full_like(z_smooth, q, dtype="U10")),
            color=config["colors"][q],
            alpha=0.5,
        )

    plt.xlabel("Redshift (z)")
    plt.ylabel(r"$O = \frac{D}{r_d}$")
    plt.legend()
    plt.grid(True)
    plt.title(title)
    plt.show()
