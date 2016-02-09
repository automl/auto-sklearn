from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.utils.extmath import row_norms
from sklearn.utils.sparsefuncs_fast import inplace_csr_row_normalize_l1
from sklearn.utils.sparsefuncs_fast import inplace_csr_row_normalize_l2

from autosklearn.util.common import warn_if_not_float
import numpy as np
from scipy import sparse

def normalize(X, norm='l2', axis=1, copy=True):
    """Scale input vectors individually to unit norm (vector length).

    Parameters
    ----------
    X : array or scipy.sparse matrix with shape [n_samples, n_features]
        The data to normalize, element by element.
        scipy.sparse matrices should be in CSR format to avoid an
        un-necessary copy.

    norm : 'l1' or 'l2', optional ('l2' by default)
        The norm to use to normalize each non zero sample (or each non-zero
        feature if axis is 0).

    axis : 0 or 1, optional (1 by default)
        axis used to normalize the data along. If 1, independently normalize
        each sample, otherwise (if 0) normalize each feature.

    copy : boolean, optional, default True
        set to False to perform inplace row normalization and avoid a
        copy (if the input is already a numpy array or a scipy.sparse
        CSR matrix and if axis is 1).

    See also
    --------
    :class:`sklearn.preprocessing.Normalizer` to perform normalization
    using the ``Transformer`` API (e.g. as part of a preprocessing
    :class:`sklearn.pipeline.Pipeline`)
    """
    if norm not in ('l1', 'l2'):
        raise ValueError("'%s' is not a supported norm" % norm)

    if axis == 0:
        sparse_format = 'csc'
    elif axis == 1:
        sparse_format = 'csr'
    else:
        raise ValueError("'%d' is not a supported axis" % axis)

    X = check_array(X, sparse_format, copy=copy)
    warn_if_not_float(X, 'The normalize function')
    if axis == 0:
        X = X.T

    if sparse.issparse(X):
        X = check_array(X, accept_sparse=sparse_format, dtype=np.float64)
        if norm == 'l1':
            inplace_csr_row_normalize_l1(X)
        elif norm == 'l2':
            inplace_csr_row_normalize_l2(X)
    else:
        if norm == 'l1':
            norms = np.abs(X).sum(axis=1)
            norms[norms == 0.0] = 1.0
        elif norm == 'l2':
            norms = row_norms(X)
            norms[norms == 0.0] = 1.0
        X /= norms[:, np.newaxis]

    if axis == 0:
        X = X.T

    return X


class Normalizer(BaseEstimator, TransformerMixin):
    """Normalize samples individually to unit norm.

    Each sample (i.e. each row of the data matrix) with at least one
    non zero component is rescaled independently of other samples so
    that its norm (l1 or l2) equals one.

    This transformer is able to work both with dense numpy arrays and
    scipy.sparse matrix (use CSR format if you want to avoid the burden of
    a copy / conversion).

    Scaling inputs to unit norms is a common operation for text
    classification or clustering for instance. For instance the dot
    product of two l2-normalized TF-IDF vectors is the cosine similarity
    of the vectors and is the base similarity metric for the Vector
    Space Model commonly used by the Information Retrieval community.

    Parameters
    ----------
    norm : 'l1' or 'l2', optional ('l2' by default)
        The norm to use to normalize each non zero sample.

    copy : boolean, optional, default True
        set to False to perform inplace row normalization and avoid a
        copy (if the input is already a numpy array or a scipy.sparse
        CSR matrix).

    Notes
    -----
    This estimator is stateless (besides constructor parameters), the
    fit method does nothing but is useful when used in a pipeline.

    See also
    --------
    :func:`sklearn.preprocessing.normalize` equivalent function
    without the object oriented API
    """

    def __init__(self, norm='l2', copy=True):
        self.norm = norm
        self.copy = copy

    def fit(self, X, y=None):
        """Do nothing and return the estimator unchanged

        This method is just there to implement the usual API and hence
        work in pipelines.
        """
        X = check_array(X, accept_sparse='csr')
        return self

    def transform(self, X, y=None, copy=None):
        """Scale each non zero row of X to unit norm

        Parameters
        ----------
        X : array or scipy.sparse matrix with shape [n_samples, n_features]
            The data to normalize, row by row. scipy.sparse matrices should be
            in CSR format to avoid an un-necessary copy.
        """
        copy = copy if copy is not None else self.copy
        X = check_array(X, accept_sparse='csr')
        return normalize(X, norm=self.norm, axis=1, copy=copy)
