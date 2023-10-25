# Source: https://supernova.lbl.gov/Union/figures/SCPUnion2.1_mu_vs_z.txt
import os
import pandas as pd
import numpy as np

def get_data():
    path_to_data = os.path.dirname(os.path.abspath(__file__)) + '/raw-data/'
    distances_file = pd.read_csv(path_to_data + 'data.txt', sep = ' ')
    selected_columns = distances_file[['z', 'mu', 'sigma_mu']].sort_values(by = 'z')

    return (
        'Union2.1',
        np.array(selected_columns['z'], dtype = np.float64, copy = False),
        np.array(selected_columns['mu'], dtype = np.float64, copy = False),
        np.array(selected_columns['sigma_mu'], dtype = np.float64, copy = False)
    )
