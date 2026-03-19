# cosmology-model-fit
## Necessary packages to run the code in python3
All files have to be executed as a module with the `-m` flag
```bash
pip3 install numpy getdist pandas matplotlib scipy corner numba numdifftools
```

## To run Hubble fits (example below for Pantheon+ sample) as a module
```bash
python3 -m sn.pantheon
```

## To run a joint fit combining DESI DR2 + DES5Y Dovekie + BBN + $\theta*$
```bash
python3 -m bao.desi_des5y_bbn_theta_star
```

## Cosmological tests
### Variable equation of state for dark energy, for example with this thawing model:

$$w_{DE}(z) = -1 + \frac{n(1 + w_0)}{1+(n-1)(1 + z)^3}$$

`n` is restricted by the equation `w(-1) = 1`, then:

$$n = \frac{2}{1+w_0}$$

This equation of state yields the evolution of energy density given by:

$$ \rho(z)_{DE} = \rho_0 [\frac{2(1+z)^3}{(1+w_0) + (1 - w_0)(1+z)^3}]^2$$

This model is within the limits of thawing quintessence models as described in https://arxiv.org/abs/astro-ph/0505494 for the considerend redshift range.

### More likely case: systematics in the standardisation of supernovae

$$z_{pec} \approx \frac{v_{pec}}{c} \times mask$$

$$mask = (z \le z_{turn}, 1, -1)$$

$$z_{cosmo} = -1 + \frac{1+z_{obs}}{1+z_{pec}} $$

For 3 different SN1a datasets the value of `z_turn` is respectively:
- `DES5Y` (previous and Dovekie analysis): 0.10563 (almost exactly where the low-z sample ends)
- `Union3` and `Union3.1`: node at z = 0.2
- `Pantheon+`: 0.15

Most relevant files (SNe + BAO + cmb):
- bao/desi_cmb_union3.py
- bao/desi_cmb_des5y.py
- bao/desi_cmb_pantheon.py

Most relevant files (SNe + BAO + gaussian prior on Omh2):
- bao/desi_union3_omh2.py
- bao/desi_des5y_omh2.py

Most relevant files (SNe alone)
- sn/union3_1.py
- sn/des5y.py
- sn/pantheon.py

The `DES5Y` results in the files above are related to the Dovekie re-analysis. The previous original analysis yielded an even higher bayesian evidence and frequentist significance such as:

$$ \Delta ln(Z) = 5.8 $$
$$ \Delta \chi^2 = 15.19 $$

when fitting `SNe+BAO+CMB compressed likelihood` adding the additional parameter `v`.

Adding `v` while also fitting any other model with evolving dark energy consistently falls right back into the predicted ΛCDM values for `w0` and/or `wa`.