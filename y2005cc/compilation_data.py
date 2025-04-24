import os
import numpy as np

# Source: https://github.com/ja-vazquez/SimpleMC/blob/master/simplemc/data/HDiagramCompilacion-data.txt

path_to_data = os.path.dirname(os.path.abspath(__file__)) + '/raw-data/'
data = np.genfromtxt(path_to_data + 'data.txt')
covariance_matrix = np.genfromtxt(path_to_data + 'covariance.txt')

z_values = data[:, 0]
H_values = data[:, 1]
dH_values = np.sqrt(np.diag(covariance_matrix))

def get_data():
    return (
        "Cosmic Chronometers compilation",
        z_values,
        H_values,
        dH_values
    )
