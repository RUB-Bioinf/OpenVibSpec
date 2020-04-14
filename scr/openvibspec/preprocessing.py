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

#---------------------------------------------------------------------------
# AS IMPLEMENTED BY SHUXIA GUO DURING THE INITIAL HACKATHON
#---------------------------------------------------------------------------

from typing import Union as U, Tuple as T, Optional
from sklearn.decomposition import PCA

import numpy as np

def emsc(spectra: np.ndarray, 
         wavenumbers: np.ndarray, 
         poly_order: Optional[int]=2,
         reference: np.ndarray = None, 
         constituents: np.ndarray = None,
         use_reference: bool=True,
         return_coefs: bool = False) -> U[np.ndarray, T[np.ndarray, np.ndarray]]:
    """
    Preprocess all spectra with EMSC
    :param spectra: ndarray of shape [n_samples, n_channels]
    :param wavenumbers: ndarray of shape [n_channels]
    :param poly_order: order of polynomial
    None: EMSC without polynomial components
    :param reference: reference spectrum 
    None: use average spectrum as reference;
    :param constituents: ndarray of shape [n_consituents, n_channels]
    Except constituents it can also take orthogonal vectors,
    for example from PCA.
    :param use_reference: if False not include reference in the model
    :param return_coefs: if True returns coefficients
    [n_samples, n_coeffs], where n_coeffs = 1 + len(costituents) + (order + 1).
    Order of returned coefficients:
    1) b*reference +                                    # reference coeff
    k) c_0*constituent[0] + ... + c_k*constituent[k] +  # constituents coeffs
    a_0 + a_1*w + a_2*w^2 + ...                         # polynomial coeffs
    :return: preprocessed spectra
    """
#    assert poly_order >= 0, 'poly_order must be >= 0'
    
    if reference is None:
        reference = np.mean(spectra, axis=0)
    
    reference = reference[:, np.newaxis]
    
    half_rng = np.abs(wavenumbers[0] - wavenumbers[-1]) / 2
    normalized_wns = (wavenumbers - np.mean(wavenumbers)) / half_rng
    
    if poly_order is None:
        if constituents is None:
            columns = (reference)
        else:
            columns = (reference, constituents.T)
    else:
        polynomial_columns = [np.ones(len(wavenumbers))]
        for j in range(1, poly_order + 1):
            polynomial_columns.append(normalized_wns ** j)
        polynomial_columns = np.stack(polynomial_columns, axis=1)
        # spectrum = X*coefs + residues
        # least squares -> A = (X.T*X)^-1 * X.T; coefs = A * spectrum
        if constituents is None:
            columns = (reference, polynomial_columns)
        else:
            columns = (reference, constituents.T, polynomial_columns)
    
    if not use_reference: columns = columns[1:]
    
    if isinstance(columns, tuple): X = np.concatenate(columns, axis=1)
    else: X = columns.copy()
    A = np.dot(np.linalg.pinv(np.dot(X.T, X)), X.T)

    spectra_columns = spectra.T
    coefs = np.dot(A, spectra_columns)
    residues = spectra_columns - np.dot(X, coefs)

    if use_reference:  preprocessed_spectra = residues/coefs[0] + reference
    else: preprocessed_spectra = residues.copy() 

    if return_coefs:
        return preprocessed_spectra.T, coefs.T

    return preprocessed_spectra.T

def rep_emsc(spectra: np.ndarray, 
             wavenumbers: np.ndarray, 
             poly_order: Optional[int]=2,
             reference: np.ndarray = None, 
             replicate: np.ndarray = None,
             n_comp: int=1,
             use_reference: bool=True,
             return_coefs: bool = False):
    """
    Preprocess all spectra with EMSC
    :param spectra: ndarray of shape [n_samples, n_channels]
    :param wavenumbers: ndarray of shape [n_channels]
    :param poly_order: order of polynomial
    None: EMSC without polynomial components
    :param reference: reference spectrum 
    None: use average spectrum as reference;
    :param replicate: ndarray of shape [n_samples] 
    :param n_comp: number of principal components used for replicate correction
    :param use_reference: if False not include reference in the model
    :param return_coefs: if True returns coefficients
    [n_samples, n_coeffs], where n_coeffs = 1 + n_comp + (order + 1).
    Order of returned coefficients:
    1) b*reference +                                    # reference coeff
    n) r_0*loading_rep[0] + ... + r_n*loading_rep[n] +  # replicate coeffs
    a_0 + a_1*w + a_2*w^2 + ...                         # polynomial coeffs
    :return: preprocessed spectra
    """
    constituents = cal_rep_matrix(spectra=spectra,
                                  wavenumbers=wavenumbers,
                                  replicate=replicate,
                                  do_PCA=True,
                                  n_comp=n_comp)[1]
    
    res = emsc(spectra=spectra, 
               wavenumbers=wavenumbers, 
               poly_order=poly_order,
               reference=reference, 
               constituents=constituents,
               use_reference=use_reference,
               return_coefs=return_coefs)
    
    return res
    
def cal_rep_matrix(spectra: np.ndarray, 
                   wavenumbers: np.ndarray, 
                   replicate: np.ndarray = None,
                   do_PCA: bool=False,
                   n_comp: int = 1):
    """
    Preprocess all spectra with EMSC
    :param spectra: ndarray of shape [n_samples, n_channels]
    :param wavenumbers: ndarray of shape [n_channels]
    :param replicate: ndarray of shape [n_samples] 
    :param do_PCA: if True returns loadings from PCA
    :param n_comp: number of principal components used for replicate correction
    :return: mean spectra of each replicate
    """
    n_rep = len(replicate)
    n_sample = np.shape(spectra)[0]
        
    assert n_rep == n_sample
        
    rep_mean = []
    rep_uni = np.unique(replicate)
    for j in range(len(rep_uni)):
        rep_mean.append(np.mean(spectra[replicate==rep_uni[j],:], axis=0))
        
    rep_mean = np.stack(rep_mean, axis=0)
    
    if do_PCA:
        n_comp = np.min((n_rep, n_comp))
        model_pca = PCA(n_comp)
        rep_mean_c = rep_mean - np.mean(rep_mean, axis=0)
        rep_columns = model_pca.fit(rep_mean_c).components_
        return rep_mean, rep_columns
    else:
        return rep_mean
