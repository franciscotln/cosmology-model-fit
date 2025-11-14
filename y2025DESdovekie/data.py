import pandas as pd
import numpy as np

data_frame = pd.read_csv("y2025DESdovekie/raw-data/distances.csv", sep="\s+")

selected_columns = data_frame[
    [
        "zHD",
        "zHEL",
        "MU",
        "MUERR",
        "PROBIA_BEAMS",
        "IDSURVEY",
    ]
]

covariance_matrix = np.load("y2025DESdovekie/raw-data/STAT+SYS.npy")


effective_sample_size = np.round(selected_columns["PROBIA_BEAMS"].sum()).astype(int)

z_values = selected_columns["zHD"].to_numpy(dtype=np.float64)
z_hel_values = selected_columns["zHEL"].to_numpy(dtype=np.float64)
mu = selected_columns["MU"].to_numpy(dtype=np.float64)
sort_indices = np.argsort(z_values)


def get_data():
    return (
        f"DES-SN5YR - effective: {effective_sample_size} SNe",
        z_values[sort_indices],
        z_hel_values[sort_indices],
        mu[sort_indices],
        covariance_matrix[sort_indices, :][:, sort_indices],
    )
