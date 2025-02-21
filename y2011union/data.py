# Source: https://supernova.lbl.gov/Union/
import os
import pandas as pd
import numpy as np

path_to_data = os.path.dirname(os.path.abspath(__file__)) + '/raw-data/'
distances_file = pd.read_csv(path_to_data + 'distances.txt', sep = ' ')
selected_columns = distances_file[['z', 'mu', 'sigma_mu']]
covariance_matrix = np.loadtxt(path_to_data + 'covariance_stat_sys.txt', delimiter='\t')

z_values = np.array(selected_columns['z'], dtype = np.float64, copy = False)
sort_indices = np.argsort(z_values)

def get_data():
    return (
        'Union2.1',
        z_values[sort_indices],
        np.array(selected_columns['mu'], dtype = np.float64, copy = False)[sort_indices],
        covariance_matrix[sort_indices,:][:, sort_indices]
    )
