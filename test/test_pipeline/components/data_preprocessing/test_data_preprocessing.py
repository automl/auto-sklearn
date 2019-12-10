import unittest
import numpy as np
from scipy import sparse

from autosklearn.pipeline.components.data_preprocessing.data_preprocessing \
    import DataPreprocessor


class PreprocessingPipelineTest(unittest.TestCase):

    def do_a_fit_transform(self, sparse):
        # Numerical dataset (as used in NumericalPreprocessingPipelineTest)
        X_num = np.array([
            [3.14, 1.,     1.],   # noqa : matrix legibility
            [3.14, 2., np.nan],   # noqa : matrix legibility
            [3.14, 3.,     3.]])  # noqa : matrix legibility
        sdev = (2 / 3) ** .5
        Y_num = np.array([
            [-1/sdev, -1/sdev],   # noqa : matrix legibility
            [     0.,      0.],   # noqa : matrix legibility
            [ 1/sdev,  1/sdev]])  # noqa : matrix legibility
        # Categorical dataset (as used in CategoricalPreprocessingPipelineTest)
        X_cat = np.array([
            [1, 2, 0],
            [3, 0, 0],
            [2, 9, np.nan]])
        Y_cat = np.array([
            [1, 0, 0, 0, 1, 0, 0, 1],
            [0, 0, 1, 1, 0, 0, 0, 1],
            [0, 1, 0, 0, 0, 1, 1, 0]])
        # Combined dataset with shuffled columns:
        X_comb = np.hstack((X_num, X_cat))
        categ_feat = np.array([False] * 3 + [True] * 3)
        random_order = np.random.choice(np.arange(6), size=6, replace=False)
        X_comb = X_comb[:, random_order]
        categ_feat = categ_feat[random_order]
        # fit_transform
        Y_comb = DataPreprocessor(
            categorical_features=categ_feat, sparse=sparse).fit_transform(X_comb)
        Y_comb = Y_comb.todense() if sparse else Y_comb
        # check shape
        self.assertEquals(Y_comb.shape, (3, 10))
        # check content. Categorical columns appear first in Y.
        for row in range(3):
            self.assertAlmostEqual(np.sum(Y_cat[row]), np.sum(Y_comb[row,:8]))
            self.assertAlmostEqual(np.sum(Y_num[row]), np.sum(Y_comb[row,8:]))

    def test_fit_transform(self):
        self.do_a_fit_transform(sparse=False)

    def test_fit_transform_sparse(self):
        self.do_a_fit_transform(sparse=True)
