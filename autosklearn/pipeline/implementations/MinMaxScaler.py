import numpy as np
from scipy import sparse

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, check_array

from autosklearn.util.common import warn_if_not_float


class MinMaxScaler(BaseEstimator, TransformerMixin):
    """Transforms features by scaling each feature to a given range.

    This estimator scales and translates each feature individually such
    that it is in the given range on the training set, i.e. between
    zero and one.

    The transformation is given by::

        X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
        X_scaled = X_std * (max - min) + min

    where min, max = feature_range.

    This transformation is often used as an alternative to zero mean,
    unit variance scaling.

    Read more in the :ref:`User Guide <preprocessing_scaler>`.

    Parameters
    ----------
    feature_range: tuple (min, max), default=(0, 1)
        Desired range of transformed data.

    copy : boolean, optional, default True
        Set to False to perform inplace row normalization and avoid a
        copy (if the input is already a numpy array).

    Attributes
    ----------
    min_ : ndarray, shape (n_features,)
        Per feature adjustment for minimum.

    scale_ : ndarray, shape (n_features,)
        Per feature relative scaling of the data.
    """

    def __init__(self, feature_range=(0, 1), copy=True):
        self.feature_range = feature_range
        self.copy = copy

    def fit(self, X, y=None):
        """Compute the minimum and maximum to be used for later scaling.
        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data used to compute the per-feature minimum and maximum
            used for later scaling along the features axis.
        """
        X = check_array(X, copy=self.copy, ensure_2d=True,
                        accept_sparse="csc", dtype=np.float32,
                        ensure_min_samples=2)

        if warn_if_not_float(X, estimator=self):
            # Costly conversion, but otherwise the pipeline will break:
            # https://github.com/scikit-learn/scikit-learn/issues/1709
            X = X.astype(np.float)

        feature_range = self.feature_range
        if feature_range[0] >= feature_range[1]:
            raise ValueError("Minimum of desired feature range must be smaller"
                             " than maximum. Got %s." % str(feature_range))
        if sparse.issparse(X):
            data_min = []
            data_max = []
            data_range = []
            for i in range(X.shape[1]):
                if X.indptr[i] == X.indptr[i+1]:
                    data_min.append(0)
                    data_max.append(0)
                    data_range.append(0)
                else:
                    data_min.append(X.data[X.indptr[i]:X.indptr[i + 1]].min())
                    data_max.append(X.data[X.indptr[i]:X.indptr[i + 1]].max())
            data_min = np.array(data_min, dtype=np.float32)
            data_max = np.array(data_max, dtype=np.float32)
            data_range = data_max - data_min

        else:
            data_min = np.min(X, axis=0)
            data_range = np.max(X, axis=0) - data_min

        # Do not scale constant features
        if isinstance(data_range, np.ndarray):
            # For a sparse matrix, constant features will be set to one!
            if sparse.issparse(X):
                for i in range(len(data_min)):
                    if data_range[i] == 0.0:
                        data_min[i] = data_min[i] - 1
            data_range[data_range == 0.0] = 1.0
        elif data_range == 0.:
            data_range = 1.

        self.scale_ = (feature_range[1] - feature_range[0]) / data_range
        self.min_ = feature_range[0] - data_min * self.scale_
        self.data_range = data_range
        self.data_min = data_min
        return self

    def transform(self, X):
        """Scaling features of X according to feature_range.

        Parameters
        ----------
        X : array-like with shape [n_samples, n_features]
            Input data that will be transformed.
        """
        check_is_fitted(self, 'scale_')

        X = check_array(X, accept_sparse="csc", copy=self.copy)

        if sparse.issparse(X):
            for i in range(X.shape[1]):
                X.data[X.indptr[i]:X.indptr[i + 1]] *= self.scale_[i]
                X.data[X.indptr[i]:X.indptr[i + 1]] += self.min_[i]
        else:
            X *= self.scale_
            X += self.min_
        return X

    def inverse_transform(self, X):
        """Undo the scaling of X according to feature_range.

        Parameters
        ----------
        X : array-like with shape [n_samples, n_features]
            Input data that will be transformed.
        """
        check_is_fitted(self, 'scale_')

        X = check_array(X, copy=self.copy, accept_sparse="csc", ensure_2d=False)
        X -= self.min_
        X /= self.scale_
        return X
