# cosmology-model-fit
Probing variable equation of state for dark energy, for example with my own:
$$w_{DE}(z) = w_0 - (1 + w_0) \frac{(1 + z)^3 - 1}{(1 + z)^3 + 1}$$

## Necessary packages to run the code in python3
```bash
pip3 install numpy pandas matplotlib scipy corner numba
```

## To run Hubble fits (example below for Pantheon+ sample) as a module
```bash
python3 -m sn.pantheon
```

## To run BAO fits
```bash
python3 -m bao.desi
```
