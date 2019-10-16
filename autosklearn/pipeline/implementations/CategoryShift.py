import numpy as np
from scipy import sparse


class CategoryShift:
    def fit(self, X, y=None):
        return self 

    def transform(self, X):
        #TODO check if there are no negative values


        # First increment everything by three to account for the fact that
        # np.NaN will get an index of two, and coalesced values will get index of
        # one, index of zero is not assigned to also work with sparse data
        if sparse.issparse(X):
            X.data += 3
        else:
            X += 3
        return X


    def fit_transform(self, X, y=None):
        return self.fit(X,y).transform(X)