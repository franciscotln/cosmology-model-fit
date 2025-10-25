import numpy as np

# DESY6 BAO (arXiv:2402.10696v1)
data = {
    "z": np.array([0.85]),
    "value": np.array([19.51]),
    "quantity": np.array(["DM_over_rs"]),
    "error": np.array([0.41]),
}


def get_data():
    return ("DES Y6 BAO", data)
