from numba import njit
import numpy as np
from scipy.constants import c as c0
from scipy.linalg import cho_factor, solve_triangular
from y2005cc.data import get_data

c = c0 / 1000  # Speed of light in km/s

legend, z_values, H_values, cov_matrix = get_data()

cho = cho_factor(cov_matrix, lower=True)[0]
logdet = np.linalg.slogdet(cov_matrix)[1]


@njit
def H_z(z, params):
    H0, Om = params[0], params[1]
    return H0 * np.sqrt(Om * (1.0 + z) ** 3 + (1.0 - Om))


def solve_triang(cho_L, delta):
    y = solve_triangular(cho_L, delta, lower=True, check_finite=False)
    return np.dot(y, y)


def chi_squared(params):
    f = params[-1]

    delta = H_values - H_z(z_values, params)
    return f**2 * solve_triang(cho, delta)


def log_likelihood(params):
    N = len(z_values)
    normalization = N * np.log(2 * np.pi) + logdet - 2 * N * np.log(params[-1])
    return -0.5 * (chi_squared(params) + normalization)


def main():
    from nautilus import Sampler, Prior
    from corner import corner, quantile
    import matplotlib.pyplot as plt
    from multiprocessing import Pool
    from .plot_predictions import plot_cc_predictions

    prior = Prior()
    prior.add_parameter("H0", dist=(30, 100))
    prior.add_parameter("Om", dist=(0.0, 1.0))
    prior.add_parameter("f", dist=(0.1, 3.3))

    with Pool(5) as pool:
        sampler = Sampler(
            prior, log_likelihood, n_live=10_000, pool=pool, seed=42, pass_dict=False
        )
        sampler.run(verbose=True)

    samples, log_w, log_l = sampler.posterior()
    w = np.exp(log_w)
    log_evd = sampler.log_z
    one_sigma_ci = [0.159, 0.5, 0.841]

    corner(
        samples,
        weights=w,
        labels=prior.keys,
        quantiles=one_sigma_ci,
        show_titles=True,
        title_fmt=".4f",
        bins=100,
        fill_contours=False,
        plot_datapoints=False,
        smooth=2.0,
        smooth1d=2.0,
        levels=(0.393, 0.864),
        range=np.repeat(0.9999, len(prior.keys)),
    )
    plt.show()

    H0_16, H0_50, H0_84 = quantile(samples[:, 0], one_sigma_ci, weights=w)
    Om_16, Om_50, Om_84 = quantile(samples[:, 1], one_sigma_ci, weights=w)
    f_16, f_50, f_84 = quantile(samples[:, 2], one_sigma_ci, weights=w)

    best_fit = [H0_50, Om_50, f_50]

    print(f"H0: {H0_50:.1f} +{(H0_84 - H0_50):.1f} -{(H0_50 - H0_16):.1f}")
    print(f"Ωm: {Om_50:.3f} +{(Om_84 - Om_50):.3f} -{(Om_50 - Om_16):.3f}")
    print(f"f: {f_50:.2f} +{(f_84 - f_50):.2f} -{(f_50 - f_16):.2f}")
    print(f"Chi squared: {chi_squared(best_fit):.2f}")
    print(f"Log likelihood: {log_likelihood(best_fit):.2f}")
    print(f"Log evidence: {log_evd:.2f}")
    print(f"Degs of freedom: {len(z_values) - len(best_fit)}")

    plot_cc_predictions(
        H_z=lambda z: H_z(z, best_fit),
        z=z_values,
        H=H_values,
        H_err=np.sqrt(np.diag(cov_matrix)) / f_50,
        label=f"{legend} $H_0$: {H0_50:.1f} ± {(H0_84 - H0_50):.1f} km/s/Mpc",
    )


if __name__ == "__main__":
    main()

"""
*******************************
Results for data from
https://arxiv.org/pdf/2307.09501

4 data points from
https://arxiv.org/pdf/2506.03836
https://arxiv.org/pdf/2511.02730v1

1 data point from
https://arxiv.org/pdf/2606.07298v1

1 data point from
https://arxiv.org/pdf/2608.13178
*******************************
"""

"""
Flat ΛCDM

-------------------------------

Varying f in U(0.1, 3.3):
H0: 66.9 +3.6 -3.6 km/s/Mpc
Ωm: 0.329 +0.049 -0.042
f: 1.49 +0.18 -0.17
Chi squared: 36.31
Log likelihood: -149.17
Log evidence: -155.91 (diff: 2.35 in evidence favouring the model with f)
Degs of freedom: 35

-------------------------------

With fixed f = 1:
H0: 66.3 +5.4 -5.4 km/s/Mpc
Ωm: 0.34 +0.08 -0.06
Chi squared: 16.42
Log likelihood: -154.33
Log evidence: -158.26
Degs of freedom: 36
"""

"""
Log likelihood ratio test:
-2 * log(L0/L1) = -2 * log(L0) + 2 * log(L1)
-2 * (-154.33) + 2 * (-149.17) = 10.32

Degrees of freedom = 1
p-value = 0.0013
We are 99.87% confident that the model with f is better than the one without f.
So the uncertainties in the H(z) dataset are overestimated and should be scaled down.
3.21 sigma significance.
"""
