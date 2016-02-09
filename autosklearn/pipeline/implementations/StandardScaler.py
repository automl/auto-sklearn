import numpy as np
from scipy import sparse

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.utils.sparsefuncs import inplace_column_scale, \
    mean_variance_axis

from autosklearn.util.common import warn_if_not_float

def _mean_and_std(X, axis=0, with_mean=True, with_std=True):
    """Compute mean and std deviation for centering, scaling.
    Zero valued std components are reset to 1.0 to avoid NaNs when scaling.
    """
    X = np.asarray(X)
    Xr = np.rollaxis(X, axis)

    if with_mean:
        mean_ = Xr.mean(axis=0)
    else:
        mean_ = None

    if with_std:
        std_ = Xr.std(axis=0)
        if isinstance(std_, np.ndarray):
            std_[std_ == 0.] = 1.0
        elif std_ == 0.:
            std_ = 1.
    else:
        std_ = None

    return mean_, std_


class StandardScaler(BaseEstimator, TransformerMixin):
    """Standardize features by removing the mean and scaling to unit variance
    Centering and scaling happen independently on each feature by computing
    the relevant statistics on the samples in the training set. Mean and
    standard deviation are then stored to be used on later data using the
    `transform` method.
    Standardization of a dataset is a common requirement for many
    machine learning estimators: they might behave badly if the
    individual feature do not more or less look like standard normally
    distributed data (e.g. Gaussian with 0 mean and unit variance).
    For instance many elements used in the objective function of
    a learning algorithm (such as the RBF kernel of Support Vector
    Machines or the L1 and L2 regularizers of linear models) assume that
    all features are centered around 0 and have variance in the same
    order. If a feature has a variance that is orders of magnitude larger
    that others, it might dominate the objective function and make the
    estimator unable to learn from other features correctly as expected.
    Parameters
    ----------
    with_mean : boolean, True by default
        If True, center the data before scaling.
        This does not work (and will raise an exception) when attempted on
        sparse matrices, because centering them entails building a dense
        matrix which in common use cases is likely to be too large to fit in
        memory.

    with_std : boolean, True by default
        If True, scale the data to unit variance (or equivalently,
        unit standard deviation).

    copy : boolean, optional, default True
        If False, try to avoid a copy and do inplace scaling instead.
        This is not guaranteed to always work inplace; e.g. if the data is
        not a NumPy array or scipy.sparse CSR matrix, a copy may still be
        returned.

    Attributes
    ----------
    mean_ : array of floats with shape [n_features]
        The mean value for each feature in the training set.

    std_ : array of floats with shape [n_features]
        The standard deviation for each feature in the training set.

    See also
    --------
    :func:`sklearn.preprocessing.scale` to perform centering and
    scaling without using the ``Transformer`` object oriented API

    :class:`sklearn.decomposition.RandomizedPCA` with `whiten=True`
    to further remove the linear correlation across features.
    """


    def __init__(self, copy=True, with_mean=True, with_std=True,
                 center_sparse=True):
        self.with_mean = with_mean
        self.with_std = with_std
        self.copy = copy
        self.center_sparse = center_sparse

    def fit(self, X, y=None):
        """Don't trust the documentation of this module!

        Compute the mean and std to be used for later scaling.

        Parameters
        ----------
        X : array-like or CSR matrix with shape [n_samples, n_features]
            The data used to compute the mean and standard deviation
            used for later scaling along the features axis.
        """
        X = check_array(X, copy=self.copy, accept_sparse="csc",
                         ensure_2d=False)
        if warn_if_not_float(X, estimator=self):
            # Costly conversion, but otherwise the pipeline will break:
            # https://github.com/scikit-learn/scikit-learn/issues/1709
            X = X.astype(np.float32)
        if sparse.issparse(X):
            if self.center_sparse:
                means = []
                vars = []

                # This only works for csc matrices...
                for i in range(X.shape[1]):
                    if X.indptr[i] == X.indptr[i + 1]:
                        means.append(0)
                        vars.append(1)
                    else:
                        vars.append(
                            X.data[X.indptr[i]:X.indptr[i + 1]].var())
                        # If the variance is 0, set all occurences of this
                        # features to 1
                        means.append(
                            X.data[X.indptr[i]:X.indptr[i + 1]].mean())
                        if 0.0000001 >= vars[-1] >= -0.0000001:
                            means[-1] -= 1

                self.std_ = np.sqrt(np.array(vars))
                self.std_[np.array(vars) == 0.0] = 1.0
                self.mean_ = np.array(means)

                return self
            elif self.with_mean:
                raise ValueError(
                    "Cannot center sparse matrices: pass `with_mean=False` "
                    "instead. See docstring for motivation and alternatives.")
            else:
                self.mean_ = None

            if self.with_std:
                var = mean_variance_axis(X, axis=0)[1]
                self.std_ = np.sqrt(var)
                self.std_[var == 0.0] = 1.0
            else:
                self.std_ = None
            return self
        else:
            self.mean_, self.std_ = _mean_and_std(
                X, axis=0, with_mean=self.with_mean, with_std=self.with_std)
            return self

    def transform(self, X, y=None, copy=None):
        """Perform standardization by centering and scaling

        Parameters
        ----------
        X : array-like with shape [n_samples, n_features]
            The data used to scale along the features axis.
        """
        check_is_fitted(self, 'std_')

        copy = copy if copy is not None else self.copy
        X = check_array(X, copy=copy, accept_sparse="csc", ensure_2d=False)
        if warn_if_not_float(X, estimator=self):
            X = X.astype(np.float)
        if sparse.issparse(X):
            if self.center_sparse:
                for i in range(X.shape[1]):
                    X.data[X.indptr[i]:X.indptr[i + 1]] -= self.mean_[i]

            elif self.with_mean:
                raise ValueError(
                    "Cannot center sparse matrices: pass `with_mean=False` "
                    "instead. See docstring for motivation and alternatives.")

            else:
                pass

            if self.std_ is not None:
                inplace_column_scale(X, 1 / self.std_)
        else:
            if self.with_mean:
                X -= self.mean_
            if self.with_std:
                X /= self.std_
        return X


    def inverse_transform(self, X, copy=None):
        """Scale back the data to the original representation

        Parameters
        ----------
        X : array-like with shape [n_samples, n_features]
            The data used to scale along the features axis.
        """
        check_is_fitted(self, 'std_')

        copy = copy if copy is not None else self.copy
        if sparse.issparse(X):
            if self.with_mean:
                raise ValueError(
                    "Cannot uncenter sparse matrices: pass `with_mean=False` "
                    "instead See docstring for motivation and alternatives.")
            if not sparse.isspmatrix_csr(X):
                X = X.tocsr()
                copy = False
            if copy:
                X = X.copy()
            if self.std_ is not None:
                inplace_column_scale(X, self.std_)
        else:
            X = np.asarray(X)
            if copy:
                X = X.copy()
            if self.with_std:
                X *= self.std_
            if self.with_mean:
                X += self.mean_
        return X
