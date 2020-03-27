import numpy as np
from scipy import sparse


class MinorityCoalescer:
    """ Group together categories which occurence is less than a specified
    minimum fraction. Coalesced categories get index of one.
    """

    def __init__(self, minimum_fraction=None):
        self.minimum_fraction = minimum_fraction

    def check_X(self, X):
        X_data = X.data if sparse.issparse(X) else X
        if np.nanmin(X_data) < 2:
            raise ValueError("X needs to contain only integers greater than two.")

    def fit(self, X, y=None):
        self.check_X(X)

        if self.minimum_fraction is None:
            return self

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

        self.do_not_coalesce_ = do_not_coalesce
        return self

    def transform(self, X):
        self.check_X(X)

        if self.minimum_fraction is None:
            return X

        for column in range(X.shape[1]):
            if sparse.issparse(X):
                indptr_start = X.indptr[column]
                indptr_end = X.indptr[column + 1]
                unique = np.unique(X.data[indptr_start:indptr_end])
                for unique_value in unique:
                    if unique_value not in self.do_not_coalesce_[column]:
                        indptr_start = X.indptr[column]
                        indptr_end = X.indptr[column + 1]
                        X.data[indptr_start:indptr_end][
                            X.data[indptr_start:indptr_end] == unique_value] = 1
            else:
                unique = np.unique(X[:, column])
                unique_values = [unique_value for unique_value in unique
                                 if unique_value not in self.do_not_coalesce_[column]]
                mask = np.isin(X[:, column], unique_values)
                X[mask, column] = 1
        return X

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
