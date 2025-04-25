import os
import pandas as pd

# Source: https://arxiv.org/pdf/1709.00646

path_to_data = os.path.dirname(os.path.abspath(__file__)) + '/raw-data/'
data = pd.read_csv(
    path_to_data + 'data.txt',
    delimiter=",",
    dtype={ "z": float, "H": float, "sigma_H": float, "Reference": str },
)

def get_data():
    return (
        "Cosmic Chronometers compilation",
        data["z"],
        data["H"],
        data["sigma_H"],
    )
