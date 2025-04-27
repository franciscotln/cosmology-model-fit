import os
import pandas as pd
import numpy as np

# Source: https://github.com/ja-vazquez/SimpleMC/blob/master/simplemc/data/HDiagramCompilacion-data.txt

path_to_data = os.path.dirname(os.path.abspath(__file__)) + '/raw-data/'
data = pd.read_csv(path_to_data + 'data.txt', sep='\s+')
covariance = pd.read_csv(path_to_data + 'covariance.txt', sep='\s+', header=None)

def get_data():
    return (
        "Cosmic Chronometers compilation (37 data points)",
        data["z"],
        data["H"],
        np.sqrt(np.diag(covariance)),
    )
