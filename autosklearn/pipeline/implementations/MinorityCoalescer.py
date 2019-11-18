import numpy as np
from scipy import sparse

class MinorityCoalescer:
    """ Group together categories which occurence is less than a specified minimum fraction.
    """
    
    def __init__(self, minimum_fraction=None):
        self.minimum_fraction = minimum_fraction

    def fit(self, X, y=None):
        return self 

    def transform(self, X):
        if self.minimum_fraction is None:
            return X

        # Remember which values should not be coalesced
        do_not_coalesce = list()
        for column in range(X.shape[1]):
            do_not_coalesce.append(set())

            if sparse.issparse(X):
                indptr_start = X.indptr[column]
                indptr_end = X.indptr[column + 1]
                unique, counts = np.unique(
                    X.data[indptr_start:indptr_end], return_counts=True)
                colsize = indptr_end - indptr_start
            else:
                unique, counts = np.unique(X[:, column], return_counts=True)
                colsize = X.shape[0]

            for unique_value, count in zip(unique, counts):
                fraction = float(count) / colsize
                if fraction >= self.minimum_fraction:
                    do_not_coalesce[-1].add(unique_value)

            for unique_value in unique:
                if unique_value not in do_not_coalesce[-1]:
                    if sparse.issparse(X):
                        indptr_start = X.indptr[column]
                        indptr_end = X.indptr[column + 1]
                        X.data[indptr_start:indptr_end][
                            X.data[indptr_start:indptr_end] == unique_value] = 1
                    else:
                        X[:, column][X[:, column] == unique_value] = 1

        self.do_not_coalesce_ = do_not_coalesce
        return X

    def fit_transform(self, X, y=None):
        return self.fit(X,y).transform(X)