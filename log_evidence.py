import numpy as np


# Laplace approximation for Bayesian evidence (ln Z) using Hessian
def log_evidence(mc_samples, log_probs, log_probability, eps=1e-5):
    """
    Laplace approximation for Bayesian evidence (ln Z) using Hessian at MAP.
    """
    # 1. Find MAP estimate
    map_idx = np.argmax(log_probs)
    theta_map = mc_samples[map_idx]
    n_params = theta_map.shape[0]

    # 2. Compute Hessian of log_probability at MAP numerically
    def hessian(f, x, eps):
        x = np.asarray(x)
        n = x.size
        hess = np.zeros((n, n))
        fx = f(x)
        for i in range(n):
            x_i1 = x.copy()
            x_i1[i] += eps
            fxi1 = f(x_i1)
            x_i2 = x.copy()
            x_i2[i] -= eps
            fxi2 = f(x_i2)
            hess[i, i] = (fxi1 - 2 * fx + fxi2) / eps**2
            for j in range(i + 1, n):
                x_ij1 = x.copy()
                x_ij1[i] += eps
                x_ij1[j] += eps
                f_ij1 = f(x_ij1)

                x_ij2 = x.copy()
                x_ij2[i] += eps
                x_ij2[j] -= eps
                f_ij2 = f(x_ij2)

                x_ij3 = x.copy()
                x_ij3[i] -= eps
                x_ij3[j] += eps
                f_ij3 = f(x_ij3)

                x_ij4 = x.copy()
                x_ij4[i] -= eps
                x_ij4[j] -= eps
                f_ij4 = f(x_ij4)

                hess_ij = (f_ij1 - f_ij2 - f_ij3 + f_ij4) / (4 * eps**2)
                hess[i, j] = hess_ij
                hess[j, i] = hess_ij
        return hess

    H = hessian(log_probability, theta_map, eps)
    # Use negative Hessian (since log-prob is maximized at MAP)
    neg_H = -H
    # Add jitter for numerical stability
    neg_H += 1e-6 * np.eye(n_params)

    # 3. Log-posterior at MAP
    log_post_map = log_probability(theta_map)

    # 4. Laplace approximation ln(Z)
    return (
        log_post_map
        + 0.5 * n_params * np.log(2 * np.pi)
        - 0.5 * np.log(np.linalg.det(neg_H))
    )


"""
Scales describe in:
https://ptfonseca.github.io/pcal/reference/bfactor_log_interpret.html

Kass and Raftery (1995) suggest the following scale for interpreting the
difference in log-evidence (ln Z) between two models:
-inf < ln(Z) < 1: weak
1 <= ln(Z) < 3: positive
3 <= ln(Z) < 5: strong
ln(Z) >= 5: very strong

Jeffreys (1961) suggested the interpretation of Bayes factors
in half-units on the base 10 logarithmic scale, as indicated in the following:
0 < log10(z) < 0.5: weak
0.5 < log10(z) < 1: substantial
1 < log10(z) < 1.5: strong
1.5 < log10(z) < 2.0: very strong
log10(z) > 2.0: decisive
"""
