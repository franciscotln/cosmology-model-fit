# Source: https://github.com/dscolnic/Pantheon/blob/master/lcparam_full_long_zhel.txt
import os
import pandas as pd
import numpy as np

def get_data():
    path_to_data = os.path.dirname(os.path.abspath(__file__)) + '/raw-data/'
    df = pd.read_csv(path_to_data + 'mb.txt', sep = ' ')
    convariances_file = pd.read_csv(path_to_data + 'mb_covariance_sys.txt')
    n = int(np.sqrt(convariances_file.size))
    variances = np.array(convariances_file['cov_mu'].values, dtype=np.float64, copy=True).reshape((n, n))

    selected_columns = df[['zcmb', 'mb', 'dmb']]
    z_values = np.array(selected_columns['zcmb'], dtype = np.float64, copy = False)
    apparent_magnitude_vals = np.array(selected_columns['mb'], dtype = np.float64, copy = False)
    sigma_magnitudes = np.array(selected_columns['dmb'], dtype = np.float64, copy = False)
    covariance_matrix = variances + np.diag(selected_columns['dmb'].values ** 2)

    sort_indices = np.argsort(z_values)

    return (
        'Pantheon2018',
        z_values[sort_indices],
        apparent_magnitude_vals[sort_indices],
        sigma_magnitudes[sort_indices],
        covariance_matrix[sort_indices, :][:, sort_indices],
    )
