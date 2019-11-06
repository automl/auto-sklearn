import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array

class SparseOneHotEncoder(BaseEstimator, TransformerMixin):
    """Encode categorical integer features using a one-hot aka one-of-K scheme.
    The input to this transformer should be a SPARSE matrix of integers. 
    """

    #def __init__(self, dtype=np.float, sparse=True):
    def __init__(self, dtype=np.float, sparse=True):
        self.dtype = dtype
        #self.sparse = sparse

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not sparse.issparse(X):
            raise TypeError("SparseOneHotEncoder requires X to be sparse")

        n_samples = X.shape[0]
        n_values = X.max(axis=0).toarray().flatten() + 2
        indices = np.cumsum(np.hstack([[0], n_values]))

        self.n_values_ = n_values
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
        #return out.tocsr() if self.sparse else out.toarray()
        return out.tocsr()
