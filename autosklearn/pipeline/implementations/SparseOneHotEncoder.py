import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array


class SparseOneHotEncoder(BaseEstimator, TransformerMixin):
    """Encode categorical integer features using a one-hot aka one-of-K scheme.

    The input to this transformer should be a sparse matrix of integers, denoting
    the values taken on by categorical (discrete) features. The output will be
    a sparse matrix were each column corresponds to one possible value of one
    feature. It is assumed that input features take on values in the range
    [0, n_values).

    Attributes
    ----------
    `feature_indices_` : array of shape (n_features,)
        Indices to feature ranges.
        Feature ``i`` in the original data is mapped to features
        from ``feature_indices_[i]`` to ``feature_indices_[i+1]``
        (and then potentially masked by `active_features_` afterwards)

    `n_values_` : array of shape (n_features,)
        Maximum number of values per feature.
    """

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

    def _check_X(self, X):
        if not sparse.issparse(X):
            raise TypeError("SparseOneHotEncoder requires X to be sparse")
        X = check_array(X, accept_sparse='csc', force_all_finite=False,
                        dtype=np.int32)
        if X.min() < 0:
            raise ValueError("X needs to contain only non-negative integers.")

        return X

    def fit_transform(self, X, y=None):
        X = self._check_X(X)

        n_samples, n_features = X.shape
        n_values = X.max(axis=0).toarray().flatten() + 2
        self.n_values_ = n_values
        n_values = np.hstack([[0], n_values])
        indices = np.cumsum(n_values)
        self.feature_indices_ = indices

        row_indices = X.indices
        column_indices = []
        for i in range(len(X.indptr) - 1):
            nbr = X.indptr[i+1] - X.indptr[i]
            column_indices_ = [indices[i]] * nbr
            column_indices_ += X.data[X.indptr[i]:X.indptr[i+1]]
            column_indices.extend(column_indices_)
        data = np.ones(X.data.size)

        out = sparse.coo_matrix((data, (row_indices, column_indices)),
                                shape=(n_samples, indices[-1]),
                                dtype=np.int32).tocsc()

        mask = np.array(out.sum(axis=0)).ravel() != 0
        active_features = np.where(mask)[0]
        out = out[:, active_features]
        self.active_features_ = active_features
        return out.tocsr()

    def transform(self, X):
        X = self._check_X(X)

        n_samples, n_features = X.shape
        indices = self.feature_indices_
        if n_features != indices.shape[0] - 1:
            raise ValueError("X has different shape than during fitting."
                             " Expected %d, got %d."
                             % (indices.shape[0] - 1, n_features))

        n_values_check = X.max(axis=0).toarray().flatten() + 1

        # Ignore all indicators which are out of bounds (i.e. assign index 0)
        # This strategy is analogous with using handle_unkwon='ignore' on a sklearn's
        # one hot encoder.
        if (n_values_check > self.n_values_).any():
            # raise ValueError("Feature out of bounds. Try setting n_values.")
            for i, n_value_check in enumerate(n_values_check):
                if (n_value_check - 1) >= self.n_values_[i]:
                    indptr_start = X.indptr[i]
                    indptr_end = X.indptr[i+1]
                    zeros_mask = X.data[indptr_start:indptr_end] >= self.n_values_[i]
                    X.data[indptr_start:indptr_end][zeros_mask] = 0

        row_indices = X.indices
        column_indices = []
        for i in range(len(X.indptr) - 1):
            nbr = X.indptr[i + 1] - X.indptr[i]
            column_indices_ = [indices[i]] * nbr
            column_indices_ += X.data[X.indptr[i]:X.indptr[i + 1]]
            column_indices.extend(column_indices_)
        data = np.ones(X.data.size)

        out = sparse.coo_matrix((data, (row_indices, column_indices)),
                                shape=(n_samples, indices[-1]),
                                dtype=np.int32).tocsc()

        out = out[:, self.active_features_]
        return out.tocsr()
