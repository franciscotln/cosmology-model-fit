import numpy as np


# Laplace approximation for Bayesian evidence (ln Z)
def log_evidence(mc_samples, log_probability):
    # 1. Find MAP estimate
    print("computing log evidence...")
    log_probs = np.array([log_probability(p) for p in mc_samples])
    map_idx = np.argmax(log_probs)
    theta_map = mc_samples[map_idx]

    # 2. Covariance matrix from samples
    print("computing covariance matrix...")
    cov = np.cov(mc_samples, rowvar=False)
    n_params = cov.shape[0]

    # 3. Log-posterior at MAP
    print("computing log posterior at MAP...")
    log_post_map = log_probability(theta_map)

    # 4. Laplace approximation
    logZ = (
        log_post_map
        + 0.5 * n_params * np.log(2 * np.pi)
        + 0.5 * np.log(np.linalg.det(cov))
    )
    return logZ
