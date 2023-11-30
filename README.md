# Wheen 23
## Necessary code to fit and run the model presented in the Wheen *et al.*, 2023
paper.

### **Files**
 - `benth_fit_I_coefs.py`: Fits the phenomenological model to Bintanja's ice
   volume data (`data/bin_isolated_ice_vol_km3`).
 - `extract_from_reduced_I.py`: Uses the data constraints presented in the paper
   to estimate the physical model parameters from `benth_fit_I_coefs.py`.
 - `wheen_I.py`: Solves the phenomenological model.
 - `wheen.py`: Solves the physical model.

For any questions, please email `liam.wheen@bristol.ac.uk`
