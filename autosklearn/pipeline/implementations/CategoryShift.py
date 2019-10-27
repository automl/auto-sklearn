import numpy as np
from scipy import sparse
from sklearn.utils import check_array


class CategoryShift:
    """ A transformer to be used as first step on data preprocessing of categorical 
    features. It makes integer categories greater or equal to three. Categories one 
    and two are reserved for special purposes (imputation and coalescence)
    """

    def fit(self, X, y=None):
        return self 

    def transform(self, X):
        # Check if integers are positive
        if np.nanmin(X) < 0:
            raise ValueError("X needs to contain only non-negative integers.")

        #TODO: check_array is removing the NaNs! Is this happening also in the master branch?
        #X = check_array(X, accept_sparse='csc', force_all_finite=False,
        #                dtype=np.int32)

        # Then increment everything by three to account for the fact that
        # np.NaN will get an index of two, and coalesced values will get index of
        # one, index of zero is not assigned to also work with sparse data
        if sparse.issparse(X):
            X.data += 3
        else:
            X += 3
        return X


    def fit_transform(self, X, y=None):
        return self.fit(X,y).transform(X)