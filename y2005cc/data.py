# source: https://arxiv.org/pdf/2307.09501
import os
import pandas as pd

path_to_data = os.path.dirname(os.path.abspath(__file__)) + '/raw-data/'
data = pd.read_csv(path_to_data + 'data2023.txt')

def get_data():
    return (
        f"Cosmic Chronometers ({data.shape[0]} data points)",
        data["z"],
        data["H"],
        data["sigma_H"]
    )
