import numpy as np


def gelman_rubin(chains):
    # chains shape: (nwalkers, nsamples, ndim)
    M, N, D = chains.shape

    # 1. Within-chain variance (W)
    # Calculate variance along the sample axis (axis=1)
    vars = np.var(chains, axis=1, ddof=1)  # shape: (M, D)
    W = np.mean(vars, axis=0)  # shape: (D,)

    # 2. Between-chain variance (B)
    means = np.mean(chains, axis=1)  # shape: (M, D)
    B = N * np.var(means, axis=0, ddof=1)  # shape: (D,)

    # 3. Estimated variance
    var_hat = ((N - 1) / N) * W + (1 / N) * B

    return np.sqrt(var_hat / W)
