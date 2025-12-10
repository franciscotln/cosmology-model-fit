# cosmology-model-fit
Probing variable equation of state for dark energy, for example with this thawing model:
$$w_{DE}(z) = -1 + \frac{n(1 + w_0)}{1+(n-1)(1 + z)^3}$$
`n` is restricted by the equation `w(-1) = 1`, then:
$$n = \frac{2}{1+w_0}$$

This equation of state yields the evolution of energy density given by:
$$ \rho(z)_{DE} = \rho_0 [\frac{2(1+z)^3}{(1+w_0) + (1 - w_0)(1+z)^3}]^2$$

This model is within the limits of thawing quintessence models as described in https://arxiv.org/abs/astro-ph/0505494 for the considerend redshift range.

## Necessary packages to run the code in python3
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
