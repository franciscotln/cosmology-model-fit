import os
import pandas as pd
import numpy as np

# Source: https://github.com/ja-vazquez/SimpleMC/blob/master/simplemc/data/HDiagramCompilacion-data.txt

path_to_data = os.path.dirname(os.path.abspath(__file__)) + '/raw-data/'
data = pd.read_csv(path_to_data + 'data.txt', sep='\s+')
covariance = pd.read_csv(path_to_data + 'covariance.txt', sep='\s+', header=None)

def get_data():
    return (
        f"Cosmic Chronometers compilation ({data.shape[0]} data points)",
        data["z"].to_numpy(),
        data["H"].to_numpy(),
        np.sqrt(np.diag(covariance)),
    )
