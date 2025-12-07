import numpy as np

# https://svn.sdss.org/public/data/eboss/DR16cosmo/tags/v1_0_1/likelihoods/BAO-plus/
# https://www.sdss4.org/science/final-bao-and-rsd-measurements

pathname = "y2020SDSSfs8/raw/"
data = np.genfromtxt(pathname + "fs8.csv", delimiter=",", names=True, dtype=np.float64)
cov_mat = np.genfromtxt(pathname + "fs8_cov.dat", dtype=np.float64)
