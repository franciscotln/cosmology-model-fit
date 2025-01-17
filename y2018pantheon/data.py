# Source: https://github.com/dscolnic/Pantheon/blob/master/lcparam_full_long_zhel.txt
import os
import pandas as pd
import numpy as np

def get_data():
    path_to_data = os.path.dirname(os.path.abspath(__file__)) + '/raw-data/'
    df = pd.read_csv(path_to_data + 'data.txt', sep = ' ')
    selected_columns = df[['zcmb', 'mb', 'dmb']].sort_values(by = 'zcmb')

    # Absolute Magnitude according to https://iopscience.iop.org/article/10.3847/1538-4357/ad8c21
    # bins [0:297] [297:630] [630:835] [835:None] (0 - 0.15, 0.15 - 0.3, 0.3 - 0.5, 0.5-end)
    M0 = -19.28

    return (
        'Pantheon2018',
        np.array(selected_columns['zcmb'], dtype = np.float64, copy = False),
        np.array(selected_columns['mb'] - M0, dtype = np.float64, copy = False),
        np.array(selected_columns['dmb'], dtype = np.float64, copy = False),
    )
