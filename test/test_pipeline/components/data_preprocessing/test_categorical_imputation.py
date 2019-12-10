import numpy as np
from scipy import sparse

from autosklearn.pipeline.components.data_preprocessing.imputation.categorical_imputation\
    import CategoricalImputation
from autosklearn.pipeline.util import PreprocessingTestCase


class CategoricalImputationTest(PreprocessingTestCase):
    def _get_dataset(self):
        size = (50, 20)
        X = np.array(np.random.randint(3, 10, size=size), dtype=float)
        mask = np.logical_not(np.random.randint(0, 5, size=size), dtype=bool)
        X[mask] = np.nan
        return X, mask

    def test_default(self):
        X, mask = self._get_dataset()
        Y = CategoricalImputation().fit_transform(X)
        self.assertTrue((np.argwhere(Y == 2) == np.argwhere(mask)).all())
        self.assertTrue((np.argwhere(Y != 2) == np.argwhere(np.logical_not(mask))).all())

    def test_default_sparse(self):
        X, mask = self._get_dataset()
        X = sparse.csc_matrix(X)
        Y = CategoricalImputation().fit_transform(X)
        Y = Y.todense()
        self.assertTrue((np.argwhere(Y == 2) == np.argwhere(mask)).all())
        self.assertTrue((np.argwhere(Y != 2) == np.argwhere(np.logical_not(mask))).all())
