import numpy as np


# Laplace approximation for Bayesian evidence (ln Z)
def log_evidence(mc_samples, log_probs, log_probability):
    # 1. Find MAP estimate
    map_idx = np.argmax(log_probs)
    theta_map = mc_samples[map_idx]

    # 2. Covariance matrix from samples with jitter
    cov = np.cov(mc_samples, rowvar=False)
    cov += 1e-6 * np.eye(cov.shape[0])
    n_params = cov.shape[0]

    # 3. Log-posterior at MAP
    log_post_map = log_probability(theta_map)

    # 4. Laplace approximation ln(Z)
    return (
        log_post_map
        + 0.5 * n_params * np.log(2 * np.pi)
        + 0.5 * np.log(np.linalg.det(cov))
    )
