''' This submodule provides basic preprocessing functionality to work with hyperspectral data/images.

E.g.
- Normalization
- Baseline Correction/Removal
- RGB-Image standardization
- Scatter correction (especially RMieS-correction)
- Data transformations from 3D to 2D and reverse
- ...

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
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(data_length, data_length - 2))
    D = lam * D.dot(D.T)
    weights = np.ones(data_length)
    W = sparse.spdiags(weights, 0, data_length, data_length)
    for i in range(n_iter):
        W.setdiag(weights)
        Z = W + D
        z = spsolve(Z, weights * data)
        weights = p * (data > z) + (1 - p) * (data < z)

    return z


def baseline_als(data, lam=1000, p=0.05, n_iter=10):
    '''Checks input data shape. If it's a single spectrum defaults to calling subfunction _baseline_corr. Otherwise loops through the data of shape (number of spectra, data points) and applies correction to each spectrum.

    Returns the baseline corrected data as a numpy array.
    '''

    if len(data.shape) == 1:
        result = np.array(_baseline_corr(data, lam, p, n_iter))
        return result

    elif len(data.shape) == 2:
        result = np.array([_baseline_corr(i, lam, p, n_iter) for i in data])
        return result

    else:
        print(
            'Data shape error! Please check your input values accordingly. Desired shape of (number of spectra, data points)')


# ---------------------------------------------------------------------------
# AS IMPLEMENTED BY SHUXIA GUO DURING THE INITIAL HACKATHON
# ---------------------------------------------------------------------------

from typing import Union as U, Tuple as T, Optional
from sklearn.decomposition import PCA
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix

import numpy as np


def emsc(spectra: np.ndarray,
         wavenumbers: np.ndarray,
         poly_order: Optional[int] = 2,
         reference: np.ndarray = None,
         constituents: np.ndarray = None,
         use_reference: bool = True,
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
    :return: preprocessed spectra, [coefficients]
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

    if isinstance(columns, tuple):
        X = np.concatenate(columns, axis=1)
    else:
        X = columns.copy()
    A = np.dot(np.linalg.pinv(np.dot(X.T, X)), X.T)

    spectra_columns = spectra.T
    coefs = np.dot(A, spectra_columns)
    residues = spectra_columns - np.dot(X, coefs)

    if use_reference:
        preprocessed_spectra = residues / coefs[0] + reference
    else:
        preprocessed_spectra = residues.copy()

    if return_coefs:
        return preprocessed_spectra.T, coefs.T

    return preprocessed_spectra.T


def rep_emsc(spectra: np.ndarray,
             wavenumbers: np.ndarray,
             replicate: np.ndarray,
             poly_order: Optional[int] = 2,
             reference: np.ndarray = None,
             n_comp: int = 1,
             use_reference: bool = True,
             return_coefs: bool = False):
    """
    Preprocess all spectra with replicate EMSC
    :param spectra: ndarray of shape [n_samples, n_channels]
    :param wavenumbers: ndarray of shape [n_channels]
    :param replicate: ndarray of shape [n_samples] 
    :param poly_order: order of polynomial
    None: EMSC without polynomial components
    :param reference: reference spectrum 
    None: use average spectrum as reference;
    :param n_comp: number of principal components used for replicate correction
    :param use_reference: if False not include reference in the model
    :param return_coefs: if True returns coefficients
    [n_samples, n_coeffs], where n_coeffs = 1 + n_comp + (order + 1).
    Order of returned coefficients:
    1) b*reference +                                    # reference coeff
    n) r_0*loading_rep[0] + ... + r_n*loading_rep[n] +  # replicate coeffs
    a_0 + a_1*w + a_2*w^2 + ...                         # polynomial coeffs
    :return: preprocessed spectra, [coefficients]
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
                   replicate: np.ndarray,
                   do_PCA: bool = False,
                   n_comp: int = 1):
    """
    Calculate mean spectra for each replicate, and do PCA if required
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

    #### replace for loop with map ####
    rep_mean = list(map(lambda x: np.mean(spectra[replicate == x, :], axis=0), rep_uni))
    #    for j in range(len(rep_uni)):
    #        rep_mean.append(np.mean(spectra[replicate==rep_uni[j],:], axis=0))

    rep_mean = np.stack(rep_mean, axis=0)

    if do_PCA:
        n_comp = np.min((n_rep, n_comp))
        model_pca = PCA(n_comp)
        rep_mean_c = rep_mean - np.mean(rep_mean, axis=0)
        rep_columns = model_pca.fit(rep_mean_c).components_
        return rep_mean, rep_columns
    else:
        return rep_mean


def cal_merit_lda(spectra: np.ndarray,
                  wavenumbers: np.ndarray,
                  replicate: np.ndarray,
                  label: np.ndarray):
    """
    Benchmark of replicate EMSC correction based on LDA classification
    :param spectra: ndarray of shape [n_samples, n_channels]
    :param wavenumbers: ndarray of shape [n_channels]
    :param replicate: ndarray of shape [n_samples] 
    :param label: ndarray of shape [n_samples] 
    :return: mean sensitivity of leave-one-replicate-out cross-validation
    """

    logo = LeaveOneGroupOut()

    res_true = []
    res_pred = []
    for train, test in logo.split(spectra, label, groups=replicate):
        tmp_model = LinearDiscriminantAnalysis()
        tmp_model.fit(spectra[train], label[train])
        res_pred = np.append(res_pred, tmp_model.predict(spectra[test]))
        res_true = np.append(res_true, label[test])

    c_m = confusion_matrix(res_true, res_pred, labels=np.unique(label))

    res = np.mean(np.diag(c_m) / np.sum(c_m, axis=1))

    return res


def rep_emsc_opt(spectra: np.ndarray,
                 wavenumbers: np.ndarray,
                 replicate: np.ndarray,
                 label: np.ndarray,
                 poly_order: Optional[int] = 2,
                 reference: np.ndarray = None,
                 n_comp_all: np.ndarray = (1, 2, 3),
                 use_reference: bool = True,
                 return_coefs: bool = False,
                 fun_merit=cal_merit_lda,
                 do_correction: bool = True):
    """
    Preprocess all spectra with replicate EMSC, wit automatically optimization of n_comp 
    :param spectra: ndarray of shape [n_samples, n_channels]
    :param wavenumbers: ndarray of shape [n_channels]
    :param replicate: ndarray of shape [n_samples] 
    :param label: ndarray of shape [n_samples] 
    :param poly_order: order of polynomial
    None: EMSC without polynomial components
    :param reference: reference spectrum 
    None: use average spectrum as reference;
    :param n_comp_all: calidated number of principal components 
    used for replicate correction
    :param use_reference: if False not include reference in the model
    :param return_coefs: if True returns coefficients
    [n_samples, n_coeffs], where n_coeffs = 1 + n_comp + (order + 1).
    Order of returned coefficients:
    1) b*reference +                                    # reference coeff
    n) r_0*loading_rep[0] + ... + r_n*loading_rep[n] +  # replicate coeffs
    a_0 + a_1*w + a_2*w^2 + ...                         # polynomial coeffs
    :param fun_merit: function used to calculate the merits 
    benchmarking the goodness of replicate correction
    :param do_correction: if or not do replicate EMSC correction using optimal n_comp
    :return: [preprocessed spectra, [coefficients]], merits, opt_comp
    """

    uni_rep = np.unique(replicate)

    merits = []
    for n_comp in n_comp_all:

        if n_comp >= len(uni_rep): break

        prep_spectra = rep_emsc(spectra=spectra,
                                wavenumbers=wavenumbers,
                                replicate=replicate,
                                poly_order=poly_order,
                                reference=reference,
                                n_comp=n_comp,
                                use_reference=use_reference,
                                return_coefs=False)

        met = fun_merit(spectra=prep_spectra,
                        wavenumbers=wavenumbers,
                        replicate=replicate,
                        label=label)

        merits.append(met)

    opt_comp = n_comp_all[np.argmax(merits)]

    if do_correction:
        res = rep_emsc(spectra=spectra,
                       wavenumbers=wavenumbers,
                       replicate=replicate,
                       poly_order=poly_order,
                       reference=reference,
                       n_comp=opt_comp,
                       use_reference=use_reference,
                       return_coefs=return_coefs)
        return res, merits, opt_comp
    else:
        return merits, opt_comp


# ---------------------------------------------------------------------------------------------------
# The following Part Is Based On Norway Biospectools 
# ---------------------------------------------------------------------------------------------------


import numpy as np


class EmptyCriterionError(Exception):
    pass


class BaseStopCriterion:
    def __init__(self, max_iter):
        self.max_iter = max_iter
        self.reset()

    def reset(self):
        self.best_idx = 0
        self.scores = []
        self.values = []

    def add(self, score, value=None):
        self.scores.append(score)
        self.values.append(value)

        self._update_best(score)

    def _update_best(self, score):
        if score < self.scores[self.best_idx]:
            self.best_idx = len(self.scores) - 1

    def __bool__(self):
        return self.cur_iter == self.max_iter or bool(self._stop())

    @property
    def cur_iter(self):
        return len(self.scores)

    @property
    def best_score(self):
        if len(self.scores) == 0:
            raise EmptyCriterionError('No scores were added')
        return self.scores[self.best_idx]

    @property
    def best_value(self):
        if len(self.values) == 0:
            raise EmptyCriterionError('No scores were added')
        return self.values[self.best_idx]

    @property
    def best_iter(self):
        if len(self.scores) == 0:
            raise EmptyCriterionError('No scores were added')
        if self.best_idx >= 0:
            return self.best_idx + 1
        else:
            return len(self.scores) - self.best_idx + 1

    def _stop(self) -> bool:
        return False


class MatlabStopCriterion(BaseStopCriterion):
    def __init__(self, max_iter, precision=None):
        super().__init__(max_iter)
        self.precision = precision
        self._eps = np.finfo(float).eps

    def _stop(self) -> bool:
        if self.cur_iter <= 2:
            return False

        pp_rmse, p_rmse, rmse = [round(r, self.precision)
                                 for r in self.scores[-3:]]

        return rmse > p_rmse or pp_rmse - rmse <= self._eps


class TolStopCriterion(BaseStopCriterion):
    def __init__(self, max_iter, tol, patience):
        super().__init__(max_iter)
        self.tol = tol
        self.patience = patience

    def _update_best(self, score):
        if score + self.tol < self.best_score:
            self.best_idx = len(self.scores) - 1

    def _stop(self) -> bool:
        if self.cur_iter <= self.patience + 1:
            return False

        no_update_iters = self.cur_iter - self.best_iter
        return no_update_iters > self.patience


# ---------------------------------------------------------------------------------------

import logging
from typing import Union as U, Tuple as T, Optional as O

import numpy as np


class EMSCInternals:
    """Class that contains intermediate results of EMSC algorithm.
    Parameters
    ----------
    coefs : `(N_samples, 1 + N_constituents + (poly_order + 1) ndarray`
        All coefficients for each transformed sample. First column is a
        scaling parameter followed by constituent and polynomial coefs.
        This is a transposed solution of equation
        _model @ coefs.T = spectrum.
    scaling_coefs : `(N_samples,) ndarray`
        Scaling coefficients (reference to the first column of coefs_).
    polynomial_coefs : `(N_samples, poly_order + 1) ndarray`
        Coefficients for each polynomial order.
    constituents_coefs : `(N_samples, N_constituents) ndarray`
        Coefficients for each constituent.
    residuals : `(N_samples, K_channels) ndarray`
         Chemical residuals that were not fitted by EMSC model.
    Raises
    ------
    AttributeError
        When polynomial's or constituents' coeffs are not available.
    """

    def __init__(
            self,
            coefs: np.ndarray,
            residuals: np.ndarray,
            poly_order: O[int],
            constituents: O[np.ndarray]):
        assert len(coefs.T) == len(residuals), 'Inconsistent number of spectra'

        self.coefs = coefs.T
        self.residuals = residuals
        if constituents is not None:
            self._n_constituents = len(constituents)
        else:
            self._n_constituents = 0
        if poly_order is not None:
            self._n_polynomials = poly_order
        else:
            self._n_polynomials = 0

    @property
    def scaling_coefs(self) -> np.ndarray:
        return self.coefs[:, 0]

    @property
    def constituents_coefs(self) -> np.ndarray:
        if self._n_constituents == 0:
            raise AttributeError(
                'constituents were not set up. '
                'Did you forget to call transform?')
        return self.coefs[:, 1:1 + self._n_constituents]

    @property
    def polynomial_coefs(self) -> np.ndarray:
        if self._n_polynomials == 0:
            raise AttributeError(
                'poly_order was not set up. '
                'Did you forget to call transform?')
        return self.coefs[:, 1 + self._n_constituents:]


class EMSC:
    """Extended multiplicative signal correction (EMSC) [1]_.
    Parameters
    ----------
    reference : `(K_channels,) ndarray`
        Reference spectrum.
    wavenumbers : `(K_channels,) ndarray`, optional
        Wavenumbers must be passed if given polynomial order
        is greater than zero.
    poly_order : `int`, optional (default 2)
        Order of polynomial to be used in regression model. If None
        then polynomial will be not used.
    weights : `(K_channels,) ndarray`, optional
        Weights for spectra.
    constituents : `(N_constituents, K_channels) np.ndarray`, optional
        Chemical constituents for regression model [2]. Can be used to add
        orthogonal vectors.
    scale : `bool`, default True
        If True then spectra will be scaled to reference spectrum.
    rebuild_model : `bool`, default True
         If True, then model will be built each time transform is called,
         this allows to dynamically change parameters of EMSC class.
         Otherwise model will be built once (for speed).
    Other Parameters
    ----------------
    _model : `(K_channels, 1 + N_constituents + (poly_order + 1) ndarray`
        Matrix that is used to solve least squares. First column is a
        reference spectrum followed by constituents and polynomial columns.
    _norm_wns : `(K_channels,) ndarray`
        Normalized wavenumbers to -1, 1 range
    References
    ----------
    .. [1] A. Kohler et al. *EMSC: Extended multiplicative
           signal correction as a tool for separation and
           characterization of physical and chemical
           information  in  fourier  transform  infrared
           microscopy  images  of  cryo-sections  of
           beef loin.* Applied spectroscopy, 59(6):707–716, 2005.
    """

    # TODO: Add numpy typing for array-like objects?
    def __init__(
            self,
            reference,
            wavenumbers=None,
            poly_order: O[int] = 2,
            constituents=None,
            weights=None,
            scale: bool = True,
            rebuild_model: bool = True,
    ):
        self.reference = np.asarray(reference)
        self.wavenumbers = wavenumbers
        if self.wavenumbers is not None:
            self.wavenumbers = np.asarray(wavenumbers)
        self.poly_order = poly_order
        self.weights = weights
        if self.weights is not None:
            self.weights = np.asarray(weights)
        self.constituents = constituents
        if self.constituents is not None:
            self.constituents = np.asarray(constituents)
        self.scale = scale
        self.rebuild_model = rebuild_model

        # lazy init during transform
        # allows to change dynamically EMSC's parameters
        self._model = None
        self._norm_wns = None

    def transform(
            self,
            spectra,
            internals: bool = False,
            check_correlation: bool = True) \
            -> U[np.ndarray, T[np.ndarray, EMSCInternals]]:
        spectra = np.asarray(spectra)
        self._validate_inputs()
        if check_correlation:
            self._check_high_correlation(spectra)

        if self.rebuild_model or self._model is None:
            self._norm_wns = self._normalize_wns()
            self._model = self._build_model()

        coefs = self._solve_lstsq(spectra)
        residuals = spectra - np.dot(self._model, coefs).T

        scaling = coefs[0]
        corr = self.reference + residuals / scaling[:, None]
        if not self.scale:
            corr *= scaling[:, None]

        if internals:
            internals_ = EMSCInternals(
                coefs, residuals, self.poly_order, self.constituents)
            return corr, internals_
        return corr

    def clear_state(self):
        del self._model
        del self._norm_wns

    def _build_model(self):
        columns = [self.reference]
        if self.constituents is not None:
            columns.extend(self.constituents)
        if self.poly_order is not None:
            columns.append(np.ones_like(self.reference))
            if self.poly_order > 0:
                n = self.poly_order + 1
                columns.extend(self._norm_wns ** pwr for pwr in range(1, n))
        return np.stack(columns, axis=1)

    def _solve_lstsq(self, spectra):
        if self.weights is None:
            return np.linalg.lstsq(self._model, spectra.T, rcond=None)[0]
        else:
            w = self.weights[:, None]
            return np.linalg.lstsq(self._model * w, spectra.T * w, rcond=None)[0]

    def _validate_inputs(self):
        if (self.poly_order is not None
                and self.poly_order < 0):
            raise ValueError(
                'poly_order must be equal or greater than 0')
        if (self.poly_order is not None
                and self.poly_order > 0
                and self.wavenumbers is None):
            raise ValueError(
                'wavenumbers must be specified when poly_order is given')
        if (self.wavenumbers is not None
                and len(self.wavenumbers) != len(self.reference)):
            raise ValueError(
                "Shape of wavenumbers doesn't match reference spectrum")

    def _check_high_correlation(self, spectra):
        mean = spectra.mean(axis=0)
        score = np.corrcoef(mean, self.reference)[0, 1]
        if score < 0.7:
            logging.warning(
                f'Low pearson score {score:.2f} between mean and reference '
                f'spectrum. Make sure that reference spectrum given in '
                f'the same order as spectra.')

    def _normalize_wns(self):
        if self.wavenumbers is None:
            return None
        half_rng = np.abs(self.wavenumbers[0] - self.wavenumbers[-1]) / 2
        return (self.wavenumbers - np.mean(self.wavenumbers)) / half_rng


def emsc(
        spectra: np.ndarray,
        wavenumbers: np.ndarray,
        poly_order: O[int] = 2,
        reference: np.ndarray = None,
        weights: np.ndarray = None,
        constituents: np.ndarray = None,
        return_coefs: bool = False,
        return_residuals: bool = False) -> U[np.ndarray, T[np.ndarray, np.ndarray],
                                             T[np.ndarray, np.ndarray, np.ndarray]]:
    """Preprocess all spectra with EMSC algorithm [1]_.
    Parameters
    ----------
    spectra : `(N_samples, K_channels) np.ndarray`
        Spectra to be processed.
    wavenumbers : `(K_channels,) ndarray`
        Wavenumbers.
    poly_order : `int`, optional
        Order of polynomial to be used in regression model. If None
        then polynomial will be not used. (2, by default)
    reference : `(K_channels,) ndarray`, optional
        Reference spectrum. If None, then average will be computed.
    weights : `(K_channels,) ndarray`, optional
        Weights for spectra.
    constituents : `(N_constituents, K_channels) np.ndarray`, optional
        Chemical constituents for regression model [2]. Can be used to add
        orthogonal vectors.
    return_coefs : `bool`, optional
        Return coefficients.
    return_residuals : `bool`, optional
        Return residuals.
    Returns
    -------
    preprocessed_spectra : `(N_samples, K_channels) ndarray`
    coefficients : `(N_samples, 1 + N_constituents + (poly_order + 1) ndarray`, optional
        If ``return_coefs`` is true, then returns coefficients in the
        following order:
        #. Scaling parametes, b (related to reference spectrum)
        #. All constituents parameters in the same order as they given
        #. Polynomial coefficients (slope, quadratic effect and so on)
    residuals: `(N_samples, K_channels) ndarray`, optional
        If ``return_residuals`` is true, then returns residuals
    References
    ----------
    .. [1] A. Kohler et al. *EMSC: Extended multiplicative
           signal correction as a tool for separation and
           characterization of physical and chemical
           information  in  fourier  transform  infrared
           microscopy  images  of  cryo-sections  of
           beef loin.* Applied spectroscopy, 59(6):707–716, 2005.
    """
    assert poly_order is None or poly_order >= 0, 'poly_order must be >= 0'

    if reference is None:
        reference = np.mean(spectra, axis=0)

    # solve for coefs: X @ coefs = spectrum (column)
    # (1) build matrix X = [reference constituents polynomial]
    columns = [reference]
    if constituents is not None:
        columns.extend(constituents)
    if poly_order is not None:
        norm_wns = _normalize_wavenumbers(wavenumbers)
        columns.append(np.ones_like(norm_wns))
        for j in range(1, poly_order + 1):
            columns.append(norm_wns ** j)
    X = np.stack(columns, axis=1)

    # (2) Calculate coefs
    if weights is None:
        coefs = np.linalg.lstsq(X, spectra.T, rcond=None)[0]
    else:
        w = weights[:, None]
        coefs = np.linalg.lstsq(X * w, spectra.T * w, rcond=None)[0]

    # (3) Preprocessing
    residuals = spectra.T - np.dot(X, coefs)
    preprocessed_spectra = reference[:, None] + residuals / coefs[0]

    # (4) return results
    if return_residuals and return_coefs:
        return preprocessed_spectra.T, coefs.T, residuals.T
    elif return_coefs:
        return preprocessed_spectra.T, coefs.T
    elif return_residuals:
        return preprocessed_spectra.T, residuals.T

    return preprocessed_spectra.T


def _normalize_wavenumbers(wns: np.ndarray):
    half_rng = np.abs(wns[0] - wns[-1]) / 2
    return (wns - np.mean(wns)) / half_rng


# -----------------------------------------------------------------------------------------------
from typing import Optional, List, Union as U, Tuple as T
import copy

import numpy as np
from sklearn.decomposition import TruncatedSVD
from scipy.signal import hilbert
from scipy.interpolate import interp1d
import numexpr as ne


# from biospectools.preprocessing import EMSC
# from biospectools.preprocessing.emsc import EMSCInternals
# from biospectools.preprocessing.criterions import \
#    BaseStopCriterion, TolStopCriterion, EmptyCriterionError


class MeEMSCInternals:
    coefs: np.ndarray
    residuals: np.ndarray
    emscs: List[Optional[EMSC]]
    criterions: List[BaseStopCriterion]
    rmses: np.ndarray
    n_iterations: np.ndarray
    n_mie_components: int

    def __init__(
            self,
            criterions: List[BaseStopCriterion],
            n_mie_components: int):
        self.criterions = criterions
        self.n_mie_components = n_mie_components
        if self.n_mie_components <= 0:
            raise ValueError('n_components must be greater than 0')

        self._extract_from_criterions()

    def _extract_from_criterions(self):
        self.emscs = []
        np_arrs = [[] for _ in range(4)]
        rmses, iters, coefs, resds = np_arrs
        for c in self.criterions:
            try:
                self.emscs.append(c.best_value['emsc'])
                emsc_inns: EMSCInternals = c.best_value['internals']
                coefs.append(emsc_inns.coefs[0])
                resds.append(emsc_inns.residuals[0])
                rmses.append(c.best_score)
                iters.append(c.best_iter)
            except EmptyCriterionError:
                self.emscs.append(None)
                coefs.append(np.nan)
                resds.append(np.nan)
                rmses.append(np.nan)
                iters.append(0)

        self.rmses, self.n_iterations, self.coefs, self.residuals = \
            [np.array(np.broadcast_arrays(*arr)) for arr in np_arrs]

    @property
    def scaling_coefs(self) -> np.ndarray:
        return self.coefs[:, 0]

    @property
    def mie_components_coefs(self) -> np.ndarray:
        assert self.n_mie_components > 0, \
            'Number of mie components must be greater than zero'
        return self.coefs[:, 1:1 + self.n_mie_components]

    @property
    def polynomial_coefs(self) -> np.ndarray:
        return self.coefs[:, -1:]


class MeEMSC:
    def __init__(
            self,
            reference: np.ndarray,
            wavenumbers: np.ndarray,
            n_components: Optional[int] = None,
            n0s: np.ndarray = None,
            radiuses: np.ndarray = None,
            h: float = 0.25,
            weights: np.ndarray = None,
            max_iter: int = 30,
            tol: float = 1e-4,
            patience: int = 1,
            positive_ref: bool = True,
            verbose: bool = False):
        self.reference = reference
        self.wavenumbers = wavenumbers
        self.weights = weights

        self.mie_generator = MatlabMieCurvesGenerator(n0s, radiuses, h)
        self.mie_decomposer = MatlabMieCurvesDecomposer(n_components)
        self.stop_criterion = TolStopCriterion(max_iter, tol, patience)

        self.positive_ref = positive_ref
        self.verbose = verbose

    def transform(self, spectra: np.ndarray, internals=False) \
            -> U[np.ndarray, T[np.ndarray, MeEMSCInternals]]:
        ref_x = self.reference
        if self.positive_ref:
            ref_x[ref_x < 0] = 0
        basic_emsc = EMSC(ref_x, self.wavenumbers, rebuild_model=False)

        correcteds = []
        criterions = []
        for spectrum in spectra:
            try:
                result = self._correct_spectrum(basic_emsc, ref_x, spectrum)
            except np.linalg.LinAlgError:
                result = np.full_like(self.wavenumbers, np.nan)
            correcteds.append(result)
            if internals:
                criterions.append(copy.copy(self.stop_criterion))

        if internals:
            inns = MeEMSCInternals(criterions, self.mie_decomposer.n_components)
            return np.array(correcteds), inns
        return np.array(correcteds)

    def _correct_spectrum(self, basic_emsc, pure_guess, spectrum):
        self.stop_criterion.reset()
        while not self.stop_criterion:
            emsc = self._build_emsc(pure_guess, basic_emsc)
            pure_guess, inn = emsc.transform(
                spectrum[None], internals=True, check_correlation=False)
            pure_guess = pure_guess[0]
            rmse = np.sqrt(np.mean(inn.residuals ** 2))
            iter_result = \
                {'corrected': pure_guess, 'internals': inn, 'emsc': emsc}
            self.stop_criterion.add(rmse, iter_result)
        return self.stop_criterion.best_value['corrected']

    def _build_emsc(self, reference, basic_emsc: EMSC) -> EMSC:
        # scale with basic EMSC:
        reference = basic_emsc.transform(
            reference[None], check_correlation=False)[0]
        if np.all(np.isnan(reference)):
            raise np.linalg.LinAlgError()

        if self.weights is not None:
            reference *= self.weights
        if self.positive_ref:
            reference[reference < 0] = 0

        qexts = self.mie_generator.generate(reference, self.wavenumbers)
        qexts = self._orthogonalize(qexts, reference)
        components = self.mie_decomposer.find_orthogonal_components(qexts)

        emsc = EMSC(reference, poly_order=0, constituents=components)
        return emsc

    def _orthogonalize(self, qext: np.ndarray, reference: np.ndarray):
        rnorm = reference / np.linalg.norm(reference)
        s = np.dot(qext, rnorm)[:, None]
        qext_orthogonalized = qext - s * rnorm
        return qext_orthogonalized


class MatlabMieCurvesGenerator:
    def __init__(self, n0s=None, rs=None, h=0.25):
        self.rs = rs if rs is not None else np.linspace(2, 7.1, 10)
        self.n0s = n0s if n0s is not None else np.linspace(1.1, 1.4, 10)
        self.h = h

        self.rs = self.rs * 1e-6
        self.alpha0s = 4 * np.pi * self.rs * (self.n0s - 1)

        optical_depths = 0.5 * np.pi * self.rs
        fs = self.h * np.log(10) / (4 * np.pi * optical_depths)
        self.gammas = fs / (self.n0s - 1)

        # prepare for broadcasting (alpha, gamma, wns)
        self.alpha0s = self.alpha0s[:, None, None]
        self.gammas = self.gammas[None, :, None]

    def generate(self, pure_absorbance, wavenumbers):
        wavenumbers = wavenumbers * 100
        nprs, nkks = self._get_refractive_index(pure_absorbance, wavenumbers)
        qexts = self._calculate_qext_curves(nprs, nkks, wavenumbers)
        return qexts

    def _calculate_qext_curves(self, nprs, nkks, wavenumbers):
        rho = self.alpha0s * (1 + self.gammas * nkks) * wavenumbers
        tanbeta = nprs / (1 / self.gammas + nkks)
        beta = np.arctan(tanbeta)
        qexts = ne.evaluate(
            '2 - 4 * exp(-rho * tanbeta) * cos(beta) / rho * sin(rho - beta)'
            '- 4 * exp(-rho * tanbeta) * (cos(beta) / rho) ** 2 * cos(rho - 2 * beta)'
            '+ 4 * (cos(beta) / rho) ** 2 * cos(2 * beta)')
        return qexts.reshape(-1, len(wavenumbers))

    def _get_refractive_index(self, pure_absorbance, wavenumbers):
        pad_size = 200
        # Extend absorbance spectrum
        wns_ext = self._extrapolate_wns(wavenumbers, pad_size)
        pure_ext = np.pad(pure_absorbance, pad_size, mode='edge')

        # Calculate refractive index
        nprs_ext = pure_ext / wns_ext
        nkks_ext = hilbert(nprs_ext).imag
        if wns_ext[0] < wns_ext[1]:
            nkks_ext *= -1

        nprs = nprs_ext[pad_size:-pad_size]
        nkks = nkks_ext[pad_size:-pad_size]
        return nprs, nkks

    def _extrapolate_wns(self, wns, pad_size):
        f = interp1d(np.arange(len(wns)), wns, fill_value='extrapolate')
        idxs_ext = np.arange(-pad_size, len(wns) + pad_size)
        wns_ext = f(idxs_ext)
        return wns_ext


class MatlabMieCurvesDecomposer:
    def __init__(self, n_components: Optional[int]):
        self.n_components = n_components
        self.svd = TruncatedSVD(self.n_components, n_iter=7)

        self.max_components = 30
        self.explained_thresh = 99.96

    def find_orthogonal_components(self, qexts: np.ndarray):
        if self.n_components is None:
            self.n_components = self._estimate_n_components(qexts)
            self.svd.n_components = self.n_components
            # do not refit svd, since it was fitted during _estimation
            return self.svd.components_[:self.n_components]

        self.svd.fit(qexts)
        return self.svd.components_

    def _estimate_n_components(self, qexts: np.ndarray):
        self.svd.n_components = min(self.max_components, qexts.shape[-1] - 1)
        self.svd.fit(qexts)

        # svd.explained_variance_ is not used since
        # it is not consistent with matlab code
        lda = self.svd.singular_values_ ** 2
        explained_var = np.cumsum(lda / np.sum(lda)) * 100
        n_comp = np.argmax(explained_var > self.explained_thresh) + 1
        return n_comp


# ----------------------------------------------------------------------------------------------------------
def savitzky_golay_filter(y, window_size, order, deriv=0, rate=1):
    """
    Based on this : https://scipy.github.io/old-wiki/pages/Cookbook/SavitzkyGolay

    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothi
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
   W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
   Cambridge University Press ISBN-13: 9780521880688

    """
    import numpy as np
    from math import factorial

    # TODO:
    # import jax
    # import jax.numpy as jnp
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))

    except ValueError:
        raise ValueError("window_size and order have to be of type int")

    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")

    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")

    order_range = range(order + 1)

    half_window = (window_size - 1) // 2

    # precompute coefficients

    b = np.mat([[k ** i for i in order_range] for k in range(-half_window, half_window + 1)])
    m = np.linalg.pinv(b).A[deriv] * rate ** deriv * factorial(deriv)

    # pad the signal at the extremes with
    # values taken from the signal itself

    firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])

    lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])

    y = np.concatenate((firstvals, y, lastvals))

    return np.convolve(m[::-1], y, mode='valid')


def savitzky_golay(y, window_size, order, deriv=0, rate=1, array=True):
    # def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    """
    

    """

    if array == True:
        return np.array([savitzky_golay_filter(xi, window_size, order, deriv=0, rate=1) for xi in y])
    else:
        return savitzky_golay_filter(y, window_size, order, deriv=0, rate=1)
