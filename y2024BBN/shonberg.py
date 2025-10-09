import numpy as np

# arXiv:2401.15054v2
# https://github.com/cosmodesi/desilike/blob/main/desilike/likelihoods/bbn/bbn.yaml
# https://github.com/schoeneberg/2024_bbn_results/blob/5d5079bed04ce747a83fcf9f8bc04904a6f490e7/covmats.txt#L14

# Ωb * h², N_eff with ∆Neff: -0.1
data = np.array([0.02196, 3.044 - 0.1], dtype=np.float64)

cov_matrix = np.array(
    [
        [4.03112260e-07, 7.30390042e-05],
        [7.30390042e-05, 4.52831584e-02],
    ],
    dtype=np.float64,
)
