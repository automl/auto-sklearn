import numpy as np
from scipy import sparse
from sklearn.utils import check_array

class CategoryShift:
    """ Add 3 to every category.
    """
    
    def _convert_and_check_X(self, X, copy):
        X_data = X.data if sparse.issparse(X) else X
        if np.nanmin(X_data) < 0:
            raise ValueError("X needs to contain only non-negative integers.")
        X = check_array(X, accept_sparse='csc', force_all_finite=False, copy=copy)
        return X
    
    def fit(self, X, y=None):
        self._convert_and_check_X(X, copy=True)
        return self 

    def transform(self, X):
        self._convert_and_check_X(X, copy=False)
        # Increment everything by three to account for the fact that
        # np.NaN will get an index of two, and coalesced values will get index of
        # one, index of zero is not assigned to also work with sparse data
        X_data = X.data if sparse.issparse(X) else X
        X_data += 3
        return X

    def fit_transform(self, X, y=None):
        return self.fit(X,y).transform(X)