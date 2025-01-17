import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib.widgets import CheckButtons
from scipy.stats import norm
import numpy as np


def plot_predictions(legend: str, x, y, y_model, label: str, y_err, x_scale='linear'):
    # real data
    plt.scatter(x=x, y=y, label=legend, marker='.', alpha=0.6)

    # predicted values
    plt.ylim(27, 50)
    plt.plot(x, y_model, 'r-', label=label)
    plt.xscale(value=x_scale)
    plt.xlabel(xlabel='Redshift (z)')
    plt.gca().xaxis.set_major_formatter(ScalarFormatter())
    plt.ylabel(ylabel='Distance modulus (mag)')
    plt.grid(visible=True, which='both', ls='--')
    plt.legend()

    # error bars from data
    [lines, caps, bars] = plt.errorbar(
        x=x,
        y=y,
        yerr=y_err,
        fmt='|',
        capsize=0,
    )
    [bar.set_alpha(0.4) for bar in bars]
    [cap.set_alpha(0.4) for cap in caps]
    checkbox_axes = plt.axes((0.01, 0.01, 0.15, 0.05))
    checkbox = CheckButtons(checkbox_axes, labels=['Error Bars'], actives=[True])
    plt.subplots_adjust(left=0.15, top=0.85, bottom=0.15)


    def update_error_bars(_):
        lines.set_visible(not lines.get_visible())
        [cap.set_visible(not cap.get_visible()) for cap in caps]
        [bar.set_visible(not bar.get_visible()) for bar in bars]
        plt.draw()

    checkbox.on_clicked(update_error_bars)
    plt.show()


def plot_residuals(z_values, residuals, y_err, bins):
    fig, [hist_plot, residuals_plot] = plt.subplots(2)
    fig.suptitle('Residual Analysis')

    # histogram
    hist_plot.hist(residuals, bins=bins, edgecolor='k', density=True)
    mu, std = norm.fit(residuals)
    x_min, x_max = hist_plot.get_xlim()
    x = np.linspace(x_min, x_max, 100)
    p = norm.pdf(x, mu, std)
    hist_plot.plot(x, p, 'r', linewidth=1)
    hist_plot.set_title('Normalised Histogram')
    hist_plot.set(xlabel='Residuals (mag)', ylabel='Density')
    hist_plot.set_ylim(0, 3.4)
    hist_plot.set_xlim(-1.5, 1.5)

    # residuals
    residuals_plot.scatter(z_values, residuals, marker='.', alpha=0.2)
    residuals_plot.errorbar(z_values, residuals, yerr=y_err, fmt='|', capsize=0, alpha=0.4)
    residuals_plot.axhline(y=0, color='red', linestyle='--')
    residuals_plot.set(xlabel='Redshift (z)', ylabel='Residuals (mag)', xscale='log')
    residuals_plot.xaxis.set_major_formatter(ScalarFormatter())
    residuals_plot.set_ylim(-1.6, 1.6)
    plt.show()


def print_color(key, value): print(f"\033[94m{key}: \033[00m \033[93m{value}\033[00m")
