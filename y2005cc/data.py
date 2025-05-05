# source: https://arxiv.org/pdf/2307.09501
# Covariance components: https://arxiv.org/pdf/2003.07362
# Covariance matrix construction: https://gitlab.com/mmoresco/CCcovariance/-/blob/master/examples/CC_covariance.ipynb
import os
import pandas as pd
import numpy as np

path_to_data = os.path.dirname(os.path.abspath(__file__)) + '/raw-data/'
data = pd.read_csv(path_to_data + 'data.txt')
cov_components = pd.read_csv(path_to_data + 'cov_components.txt')

z = data['z'].to_numpy()
Hz = data['H'].to_numpy()
sigma_H = data['sigma_H'].to_numpy()

zmod = cov_components['z'].to_numpy()
imf = cov_components['imf'].to_numpy()
slib = cov_components['stlib'].to_numpy()
sps = cov_components['sps'].to_numpy()
spsooo = cov_components['spsooo'].to_numpy()

cov_mat_diag = np.zeros((z.size, z.size), dtype='float64') 

for i in range(z.size):
	cov_mat_diag[i,i] = sigma_H[i]**2

imf_intp = np.interp(z, zmod, imf)/100
slib_intp = np.interp(z, zmod, slib)/100
sps_intp = np.interp(z, zmod, sps)/100
spsooo_intp = np.interp(z, zmod, spsooo)/100

cov_mat_imf = np.zeros((len(z), len(z)), dtype='float64')
cov_mat_slib = np.zeros((len(z), len(z)), dtype='float64')
cov_mat_sps = np.zeros((len(z), len(z)), dtype='float64')
cov_mat_spsooo = np.zeros((len(z), len(z)), dtype='float64')

for i in range(len(z)):
	for j in range(len(z)):
		cov_mat_imf[i,j] = Hz[i] * imf_intp[i] * Hz[j] * imf_intp[j]
		cov_mat_slib[i,j] = Hz[i] * slib_intp[i] * Hz[j] * slib_intp[j]
		cov_mat_sps[i,j] = Hz[i] * sps_intp[i] * Hz[j] * sps_intp[j]
		cov_mat_spsooo[i,j] = Hz[i] * spsooo_intp[i] * Hz[j] * spsooo_intp[j]

cov_matrix = cov_mat_spsooo + cov_mat_imf + cov_mat_diag

def get_data():
    return (
        f"Cosmic Chronometers ({data.shape[0]} data points)",
        z,
		Hz,
        cov_matrix,
    )
