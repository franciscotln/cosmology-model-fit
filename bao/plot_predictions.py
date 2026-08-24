import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

COLOR_MAP = {
    "DV_over_rs": "red",
    "DM_over_rs": "blue",
    "DH_over_rs": "green",
    "F_AP": "orange",
}

LATEX_LABELS = {
    "DV_over_rs": r"$D_V / r_d$",
    "DM_over_rs": r"$D_M / r_d$",
    "DH_over_rs": r"$D_H / r_d$",
    "F_AP": r"$F_{\text{AP}}$",
}

QUANTITIES_MAP = {"DV_over_rs": 0, "DM_over_rs": 1, "DH_over_rs": 2, "F_AP": 3}


def plot_bao_predictions(theory_predictions, data, errors, title="BAO Predictions"):
    fig, ax = plt.subplots(figsize=(8, 6))
    z_smooth = np.linspace(np.min(data["z"]), np.max(data["z"]), 200)

    for q in np.unique(data["quantity"]):
        if q not in COLOR_MAP:
            continue
        mask = data["quantity"] == q
        color = COLOR_MAP[q]
        label = LATEX_LABELS.get(q, q)

        # Plot data points with error bars
        ax.errorbar(
            x=data["z"][mask],
            y=data["value"][mask],
            yerr=errors[mask],
            fmt=".",
            color=color,
            label=label,
            capsize=2,
            linestyle="None",
        )

        q_code = QUANTITIES_MAP[q]
        ax.plot(
            z_smooth,
            theory_predictions(z_smooth, np.full_like(z_smooth, q_code, dtype=np.int32)),
            color=color,
            alpha=0.6,
        )

    ax.set_xlabel("Redshift $z$")
    ax.set_ylabel("BAO Observable")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    
    plt.show()
    return fig, ax


def plot_bao_residuals(data, residuals, errors, title="BAO Residuals"):
    fig, ax = plt.subplots(figsize=(8, 4))

    for qtype in np.unique(data["quantity"]):
        if qtype not in COLOR_MAP:
            continue
        mask = data["quantity"] == qtype
        color = COLOR_MAP[qtype]
        label = LATEX_LABELS.get(qtype, qtype)

        ax.errorbar(
            x=data["z"][mask],
            y=residuals[mask],
            yerr=errors[mask],
            fmt=".",
            color=color,
            ecolor=color,
            elinewidth=1,
            capsize=2,
            label=label,
        )

    ax.axhline(0, color="black", linestyle="--", alpha=0.7)
    ax.set_xlabel("Redshift $z$")
    ax.set_ylabel("Data - Model")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.show()
    return fig, ax
