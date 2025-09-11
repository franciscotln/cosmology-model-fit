import numpy as np

# Source: https://arxiv.org/pdf/2503.14738
# https://github.com/CobayaSampler/bao_data/tree/master/desi_bao_dr2
data = np.genfromtxt(
    "y2025BAO/raw-data/data.csv",
    dtype=[
        ("z", float),
        ("value", float),
        ("quantity", "U10"),
        ("std", float),
        ("corr", float),
    ],
    delimiter=",",
    names=True,
)

cov_matrix = np.zeros((len(data), len(data)))

for i in range(len(data)):
    cov_matrix[i, i] = data["std"][i] ** 2
    for j in range(i + 1, len(data)):
        if data["z"][i] == data["z"][j] and data["quantity"][i] != data["quantity"][j]:
            cov = data["corr"][i] * data["std"][i] * data["std"][j]
            cov_matrix[i, j] = cov
            cov_matrix[j, i] = cov


def get_data():
    return (
        "DESI BAO DR2",
        data[["z", "value", "quantity"]],
        cov_matrix,
    )
