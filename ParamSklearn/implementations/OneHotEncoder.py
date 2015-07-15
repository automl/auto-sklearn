import numpy as np
from scipy import sparse

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.utils import check_array


def _transform_selected(X, transform, selected="all", copy=True):
    """Apply a transform function to portion of selected features

    Parameters
    ----------
    X : array-like or sparse matrix, shape=(n_samples, n_features)
        Dense array or sparse matrix.

    transform : callable
        A callable transform(X) -> X_transformed

    copy : boolean, optional
        Copy X even if it could be avoided.

    selected: "all" or array of indices or mask
        Specify which features to apply the transform to.

    Returns
    -------
    X : array or sparse matrix, shape=(n_samples, n_features_new)
    """

    if selected == "all":
        return transform(X)

    X = check_array(X, accept_sparse='csc', force_all_finite=False)
    n_features = X.shape[1]
    ind = np.arange(n_features)
    sel = np.zeros(n_features, dtype=bool)
    sel[np.asarray(selected)] = True
    not_sel = np.logical_not(sel)
    n_selected = np.sum(sel)

    # Add 1 to all categorical colums to avoid loosing them due to slicing
    subtract = False
    if sparse.isspmatrix_csr(X):
        X.data += 1
        subtract = True
    X = check_array(X, copy=copy, force_all_finite=False, accept_sparse="csc")
    if subtract:
        X.data -= 1

    if len(selected) == 0:
        return X

    if n_selected == 0:
        # No features selected.
        return X
    elif n_selected == n_features:
        # All features selected.
        return transform(X)
    else:
        # Add 1 to all categorical columns to avoid loosing them due to slicing
        if sparse.issparse(X):
            for idx in range(n_features):
                if idx in ind[sel]:
                    X.data[X.indptr[idx]:X.indptr[idx + 1]] += 1
            X_ = X[:, ind[sel]]
            for idx in range(n_features):
                if idx in ind[sel]:
                    X.data[X.indptr[idx]:X.indptr[idx + 1]] -= 1
            X_.data -= 1
        else:
            X_ = X[:, ind[sel]]

        X_sel = transform(X_)
        X_not_sel = X[:, ind[not_sel]]

        if sparse.issparse(X_sel) or sparse.issparse(X_not_sel):
            return sparse.hstack((X_sel, X_not_sel)).tocsr()
        else:
            return np.hstack((X_sel, X_not_sel))


class OneHotEncoder(BaseEstimator, TransformerMixin):
    """Don't trust the documentation of this module!

    Encode categorical integer features using a one-hot aka one-of-K scheme.

    The input to this transformer should be a matrix of integers, denoting
    the values taken on by categorical (discrete) features. The output will be
    a sparse matrix were each column corresponds to one possible value of one
    feature. It is assumed that input features take on values in the range
    [0, n_values).

    This encoding is needed for feeding categorical data to many scikit-learn
    estimators, notably linear models and SVMs with the standard kernels.

    Parameters
    ----------
    categorical_features: "all" or array of indices or mask
        Specify what features are treated as categorical.

        - 'all' (default): All features are treated as categorical.
        - array of indices: Array of categorical feature indices.
        - mask: Array of length n_features and with dtype=bool.

        Non-categorical features are always stacked to the right of the matrix.

    dtype : number type, default=np.float
        Desired dtype of output.

    sparse : boolean, default=True
        Will return sparse matrix if set True else will return an array.

    Attributes
    ----------
    `active_features_` : array
        Indices for active features, meaning values that actually occur
        in the training set. Only available when n_values is ``'auto'``.

    `feature_indices_` : array of shape (n_features,)
        Indices to feature ranges.
        Feature ``i`` in the original data is mapped to features
        from ``feature_indices_[i]`` to ``feature_indices_[i+1]``
        (and then potentially masked by `active_features_` afterwards)

    `n_values_` : array of shape (n_features,)
        Maximum number of values per feature.

    Examples
    --------
    Given a dataset with three features and two samples, we let the encoder
    find the maximum value per feature and transform the data to a binary
    one-hot encoding.

    >>> from sklearn.preprocessing import OneHotEncoder
    >>> enc = OneHotEncoder()
    >>> enc.fit([[0, 0, 3], [1, 1, 0], [0, 2, 1], \
[1, 0, 2]])  # doctest: +ELLIPSIS
    OneHotEncoder(categorical_features='all', dtype=<... 'float'>,
           n_values='auto', sparse=True)
    >>> enc.n_values_
    array([2, 3, 4])
    >>> enc.feature_indices_
    array([0, 2, 5, 9])
    >>> enc.transform([[0, 1, 1]]).toarray()
    array([[ 1.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  0.]])

    See also
    --------
    sklearn.feature_extraction.DictVectorizer : performs a one-hot encoding of
      dictionary items (also handles string-valued features).
    sklearn.feature_extraction.FeatureHasher : performs an approximate one-hot
      encoding of dictionary items or strings.
    """

    def __init__(self, categorical_features="all",
                 dtype=np.float, sparse=True):
        self.categorical_features = categorical_features
        self.dtype = dtype
        self.sparse = sparse

    def fit(self, X, y=None):
        """Fit OneHotEncoder to X.

        Parameters
        ----------
        X : array-like, shape=(n_samples, n_feature)
            Input array of type int.

        Returns
        -------
        self
        """
        self.fit_transform(X)
        return self

    def _fit_transform(self, X):

        # Add 1 to all categorical colums to avoid loosing them due to slicing
        subtract = False
        if sparse.isspmatrix_csr(X):
            X.data += 1
            subtract = True
        X = check_array(X, accept_sparse="csc", force_all_finite=False)
        if subtract:
            X.data -= 1

        n_samples, n_features = X.shape

        # By replacing NaNs (which means a column full on NaNs in the
        # original data matrix) with a 1, we add a column full of zeros to
        # the array
        if sparse.isspmatrix_csc(X):
            n_values = [0]
            for idx in range(n_features):
                if X.indptr[idx] == X.indptr[idx+1]:
                    values_for_idx = 1
                else:
                    values_for_idx = np.nanmax(
                        X.data[X.indptr[idx]:X.indptr[idx + 1]]) + 1
                n_values.append(values_for_idx if
                                np.isfinite(values_for_idx) else 1)
            row_indices = X.indices
        else:
            n_values = np.hstack([[0], np.nanmax(X, axis=0) + 1])
            n_values[~np.isfinite(n_values)] = 1
            row_indices = np.tile(np.arange(n_samples, dtype=np.int32),
                                  n_features)

        total_num_values = np.nansum(n_values)

        column_indices = []
        data = []
        feature_indices = []

        for idx in range(X.shape[1]):
            if sparse.isspmatrix_csc(X):
                values_ = X.getcol(idx).data
            else:
                values_ = X[:, idx]

            offset = np.nansum(n_values[:idx+1])
            column_indices_idx = [offset + value if np.isfinite(value)
                                  else offset for value in values_]
            data_idx = [1 if np.isfinite(value) else 0 for value in values_]
            feature_indices_idx = {value: value + offset
                                   for value in values_
                                   if np.isfinite(value)}

            column_indices.extend(column_indices_idx)
            data.extend(data_idx)
            feature_indices.append(feature_indices_idx)

        self.feature_indices_ = feature_indices
        self.n_values = n_values
        # tocsr() removes zeros in the data which represent NaNs
        out = sparse.coo_matrix((data, (row_indices, column_indices)),
                                shape=(n_samples, total_num_values),
                                dtype=self.dtype).tocsr()
        return out if self.sparse else out.toarray()

    def fit_transform(self, X, y=None):
        """Fit OneHotEncoder to X, then transform X.

        Equivalent to self.fit(X).transform(X), but more convenient and more
        efficient. See fit for the parameters, transform for the return value.
        """
        return _transform_selected(X, self._fit_transform,
                                   self.categorical_features, copy=True)

    def _transform(self, X):
        """Assumes X contains only categorical features."""
        X = check_array(X, dtype=np.int, accept_sparse='csc')

        # Add 1 to all categorical colums to avoid loosing them due to slicing
        subtract = False
        if sparse.isspmatrix_csr(X):
            X.data += 1
            subtract = True
        X = check_array(X, accept_sparse="csc", force_all_finite=False)
        if subtract:
            X.data -= 1
        n_samples, n_features = X.shape

        indices = self.feature_indices_
        if n_features != len(indices):
            raise ValueError("X has different shape than during fitting."
                             " Expected %d, got %d."
                             % (len(indices), n_features))

        if sparse.isspmatrix_csc(X):
            row_indices = X.indices
        else:
            row_indices = np.tile(np.arange(n_samples, dtype=np.int32),
                                  n_features)

        data = []
        column_indices = []

        for idx, feature in enumerate(range(n_features)):
            if sparse.isspmatrix_csc(X):
                values_ = X.getcol(idx).data
            else:
                values_ = X[:, idx]

            offset = np.sum(self.n_values[:idx+1])
            feature_indices_idx = self.feature_indices_[idx]
            column_indices_idx = [feature_indices_idx.get(x, offset)
                                  for x in values_]
            data_idx = [1 if feature_indices_idx.get(x) is not None else 0
                        for x in values_]

            column_indices.extend(column_indices_idx)
            data.extend(data_idx)

        out = sparse.coo_matrix((data, (row_indices, column_indices)),
                                shape=(n_samples, np.sum(self.n_values)),
                                dtype=self.dtype).tocsr()

        return out if self.sparse else out.toarray()

    def transform(self, X):
        """Transform X using one-hot encoding.

        Parameters
        ----------
        X : array-like, shape=(n_samples, n_features)
            Input array of type int.

        Returns
        -------
        X_out : sparse matrix if sparse=True else a 2-d array, dtype=int
            Transformed input.
        """
        return _transform_selected(X, self._transform,
                                   self.categorical_features, copy=True)
