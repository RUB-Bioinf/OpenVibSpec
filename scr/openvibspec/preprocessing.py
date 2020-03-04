''' This submodule provides basic preprocessing functionality to work with hyperspectral data/images.

E.g.
- Normalization
- Baseline Correction/Removal
- RGB-Image standardization
- Scatter correction (especially RMieS-correction)
- Data transformations from 3D to 2D and reverse
- ...

@JoshuaButke
github.com/butkej
Mail: joshua.butke@rub.de
'''

# IMPORTS
#########

import numpy as np


# FUNCTIONS
###########

def _baseline_corr(data, lam=1000, p=0.05, n_iter=10):
    ''' Asymmetric least squares smoothing for baseline removal/correction.
    Adapted from Eilers and Boelens (2005) and with optimized memory usage. 

    Two parameters: lam (lambda) for smoothness and p for asymmetry.
    Generally for data with positive peaks 0.001 <= p <= 0.1 is a good choice and 10^2 <= lam <= 10^9

    Although iteration number is fixed at 10 your mileage may vary if the weights do not converge in this time.

    Returns the baseline corrected data.
    '''
    from scipy import sparse
    from scipy.sparse.linalg import spsolve

    data_length = len(data)
    D = sparse.diags([1,-2,1],[0,-1,-2], shape=(data_length, data_length-2))
    D = lam * D.dot(D.T)
    weights = np.ones(data_length)
    W = sparse.spdiags(weights, 0, data_length, data_length)
    for i in range(n_iter):
        W.setdiag(weights)
        Z = W + D 
        z = spsolve(Z, weights*data)
        weights = p * (data > z) + (1-p) * (data < z)
    
    return z

def baseline_als(data, lam=1000, p=0.05, n_iter=10):
    '''Checks input data shape. If it's a single spectrum defaults to calling subfunction _baseline_corr. Otherwise loops through the data of shape (number of spectra, data points) and applies correction to each spectrum.

    Returns the baseline corrected data as a numpy array.
    '''

    if len(data.shape) == 1:
        result = np.array(_baseline_corr(data,lam,p,n_iter))
        return result

    elif len(data.shape) == 2:
        result = np.array([_baseline_corr(i,lam,p,n_iter) for i in data])
        return result

    else:
        print('Data shape error! Please check your input values accordingly. Desired shape of (number of spectra, data points)')

