import corner
import matplotlib.pyplot as plt
import numpy as np


def plot_corner_and_chains(labels, flat_samples, samples=None, weights=None):
    corner.corner(
        flat_samples,
        labels=labels,
        quantiles=[0.159, 0.5, 0.841],
        show_titles=True,
        title_fmt=".4f",
        bins=100,
        fill_contours=False,
        plot_datapoints=False,
        smooth=2.0,
        smooth1d=2.0,
        levels=(0.393, 0.864),
        range=np.repeat(0.9999, len(labels)),
        weights=weights,
    )
    plt.show()

    if samples is not None:
        ndim = samples.shape[2]
        plt.figure(figsize=(16, 1.5 * ndim))
        for n in range(ndim):
            plt.subplot2grid((ndim, 1), (n, 0))
            plt.plot(samples[:, :, n], alpha=0.3)
            plt.ylabel(labels[n])
            plt.xlim(0, None)
            mean_path = np.mean(samples[:, :, n], axis=1)
            plt.plot(mean_path, color="black", lw=1.5, label="mean" if n == 0 else "")
            if n == 0:
                plt.legend()
        plt.tight_layout()
        plt.show()
