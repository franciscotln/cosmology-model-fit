import numpy as np


def gelman_rubin(chains):
    nwalkers, nsamples, ndim = chains.shape
    M = nwalkers  # Number of chains
    N = nsamples  # Number of samples per chain

    rhat = np.zeros(ndim)

    # Calculate R-hat for each dimension (parameter)
    for i in range(ndim):
        samples = chains[:, :, i]

        # 1. Calculate within-chain variance (W)
        # s_j^2: variance of chain j (calculated with ddof=1)
        chain_vars = np.var(samples, axis=1, ddof=1)
        # W = (1/M) * sum(s_j^2)
        W = np.mean(chain_vars)

        # 2. Calculate between-chain variance (B)
        # bar(theta)_j: mean of chain j
        chain_means = np.mean(samples, axis=1)
        # bar(theta): grand mean across all chains and steps (used by np.var)
        # B/N = variance of chain means (calculated with ddof=1)
        # B = N * var(chain_means)
        B = N * np.var(chain_means, ddof=1)

        # 3. Calculate estimated marginal posterior variance (var_hat)
        # var_hat = ((N-1)/N) * W + (1/N) * B
        var_hat = ((N - 1) / N) * W + (1 / N) * B

        # 4. Calculate R-hat (Gelman-Rubin statistic)
        # rhat = sqrt(var_hat / W)
        rhat[i] = np.sqrt(var_hat / W)

    return rhat
