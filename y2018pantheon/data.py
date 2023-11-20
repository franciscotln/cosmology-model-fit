# Source: https://github.com/dscolnic/Pantheon/blob/master/lcparam_full_long_zhel.txt
import os
import pandas as pd
import numpy as np

def get_data():
    path_to_data = os.path.dirname(os.path.abspath(__file__)) + '/raw-data/'
    df = pd.read_csv(path_to_data + 'data.txt', sep = ' ')
    selected_columns = df[['zcmb', 'mb', 'dmb']].sort_values(by = 'zcmb')

    # Absolute Magnitude according to Union2.1: https://supernova.lbl.gov/Union/figures/SCPUnion2.1_mu_vs_z.txt
    M0 = -19.3081547178

    return (
        'Pantheon2018',
        np.array(selected_columns['zcmb'], dtype = np.float64, copy = False),
        np.array(selected_columns['mb'] - M0, dtype = np.float64, copy = False),
        np.array(selected_columns['dmb'], dtype = np.float64, copy = False),
    )
