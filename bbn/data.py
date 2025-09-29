import numpy as np

# arXiv:2401.15054v2
# https://github.com/cosmodesi/desilike/blob/main/desilike/likelihoods/bbn/bbn.yaml

data = np.array(
    [
        0.02196,  # Ωb h^2
        2.944,  # N_eff
    ],
    dtype=np.float64,
)

cov_matrix = np.array(
    [
        [4.03112260e-07, 7.30390042e-05],
        [7.30390042e-05, 4.52831584e-02],
    ],
    dtype=np.float64,
)
