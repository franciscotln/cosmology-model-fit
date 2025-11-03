# cosmology-model-fit
Probing variable equation of state for dark energy, for example with this thawing model:
$$w_{DE}(z) = -1 + \frac{4(1 + w_0)}{1+3(1 + z)^3}$$
This within the limits of thawing quintessence models as described in https://arxiv.org/abs/astro-ph/0505494 for the considerend range

## Necessary packages to run the code in python3
```bash
pip3 install numpy pandas matplotlib scipy corner numba
```

## To run Hubble fits (example below for Pantheon+ sample) as a module
```bash
python3 -m sn.pantheon
```

## To run a joint fit combining DESI DR2 + DES5Y + BBN + $\theta*$
```bash
python3 -m bao.desi_des5y_bbn_theta_star
```
