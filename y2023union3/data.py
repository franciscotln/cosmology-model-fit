import os
import pandas as pd
import numpy as np

path_to_data = os.path.dirname(os.path.abspath(__file__)) + '/raw-data/'

data_frame = pd.read_csv(path_to_data + 'bins_union_3.txt', sep = ' ')
cov_data = pd.read_csv(path_to_data + 'cov_bins_union_3.txt')

n = data_frame['zcmb'].size
covariance_matrix = np.array(cov_data['cov_mu'], dtype=np.float64, copy=True).reshape((n, n))

z_values = np.array(data_frame['zcmb'], dtype = np.float64, copy = True)
distance_moduli_values = np.array(data_frame['mb'], dtype = np.float64, copy = True)

sort_indices = np.argsort(z_values)
bin = sort_indices[0:22]

def get_data():
    return (
        'Union 3 - Bins',
        z_values[bin],
        distance_moduli_values[bin],
        covariance_matrix[bin, :][:, bin],
    )
