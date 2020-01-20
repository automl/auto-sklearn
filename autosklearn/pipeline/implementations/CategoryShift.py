import numpy as np
from scipy import sparse
from sklearn.utils import check_array
from sklearn.base import BaseEstimator, TransformerMixin


class CategoryShift(BaseEstimator, TransformerMixin):
    """ Add 3 to every category.
    """

    def _convert_and_check_X(self, X):
        X_data = X.data if sparse.issparse(X) else X

        # Check if data is numeric and positive
        if X_data.dtype.kind not in set('buif') or np.nanmin(X_data) < 0:
            raise ValueError('Categories should be non-negative numbers. '
                             'NOTE: floats will be casted to integers.')

        # Use check_array to make sure we are using the right kind of sparse array
        # Notice that we cannot convert the array to integer right now. That would get
        # rid of the np.nans and we need them later on for the imputation.
        X = check_array(X, accept_sparse='csc', force_all_finite=False, copy=True)
        return X

    def fit(self, X, y=None):
        self._convert_and_check_X(X)
        return self

    def transform(self, X):
        X = self._convert_and_check_X(X)
        # Increment everything by three to account for the fact that
        # np.NaN will get an index of two, and coalesced values will get index of
        # one, index of zero is not assigned to also work with sparse data
        X_data = X.data if sparse.issparse(X) else X
        X_data += 3
        return X
