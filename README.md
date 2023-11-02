# cosmology-model-fit
redshift-scale factor model without approximation derived from General Relativity without the need for dark energy, for a purely matter-dominated flat universe.

## Necessary packages to run model.py in python3
```bash
pip3 install numpy pandas matplotlib scipy
```

## To run the model for mass only universe (Einstein - de Sitter)
```bash
python3 model-mass.py
```

It will print the fit results to the console and display one plot with the fit.
After closing the first plot, a second one will appear with the residual analysis.

## To run the model for cosmological constant only universe (de Sitter)
```bash
python3 model-lambda.py
```

## To run the current ΛCDM model
```bash
python3 lcdm.py
```
The same actions from above apply to this model.

## To run the model with differente datasets change the imports to the respective data package
```python
from y2022pantheonSHOES.data import get_data

# or
from y2018pantheon.data import get_data

# or
from y2011union.data import get_data
```
### only this line needs to be changed.
### Other datasets named grb concern gamma ray burst just for reference as they are not standard candles.
