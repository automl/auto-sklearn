import unittest
import numpy as np

from autosklearn.pipeline.components.data_preprocessing.data_preprocessing \
    import DataPreprocessor


class PreprocessingPipelineTest(unittest.TestCase):

    def do_a_fit_transform(self, sparse_output):
        # Numerical dataset (as used in NumericalPreprocessingPipelineTest)
        X_num = np.array([
            [3.14, 1.,     1.],   # noqa : matrix legibility
            [3.14, 2., np.nan],   # noqa : matrix legibility
            [3.14, 3.,     3.]])  # noqa : matrix legibility
        # After the preprocessing of X_num, we expect that:
        # Feature 1 get dropped due to lack of variance
        # The missing value on feature 3 gets imputed by the mean (2.)
        # All features get normalized by subtracting the mean and dividing by the
        # standard deviation.
        # Therefore, Y_num is what we should get after data preprocessing X_num:
        sdev = np.sqrt(2 / 3)
        Y_num = np.array([
            [-1/sdev, -1/sdev],   # noqa : matrix legibility
            [     0.,      0.],   # noqa : matrix legibility
            [ 1/sdev,  1/sdev]])  # noqa : matrix legibility
        # Categorical dataset (as used in CategoricalPreprocessingPipelineTest)
        X_cat = np.array([
            [1, 2, 0],
            [3, 0, 0],
            [2, 9, np.nan]])
        # After the preprocessing of X_cat, we expect that:
        # The missing value in feature 3 gets imputed as category on its own.
        # Features get encoded by one hot encoding.
        # Therefore, Y_cat is what we should get after data preprocessing X_cat:
        Y_cat = np.array([
            [1, 0, 0, 0, 1, 0, 0, 1],
            [0, 0, 1, 1, 0, 0, 0, 1],
            [0, 1, 0, 0, 0, 1, 1, 0]])
        # Combine datasets and shuffle columns:
        X_comb = np.hstack((X_num, X_cat))
        categ_feat = np.array([False] * 3 + [True] * 3)
        random_order = np.random.choice(np.arange(6), size=6, replace=False)
        X_comb = X_comb[:, random_order]
        categ_feat = categ_feat[random_order]
        # Data preprocessing
        DPP = DataPreprocessor(categorical_features=categ_feat, sparse=sparse_output)
        Y_comb = DPP.fit_transform(X_comb)
        Y_comb = Y_comb.todense() if sparse_output else Y_comb
        # check shape
        self.assertEquals(Y_comb.shape, (3, 10))
        # check content. Categorical columns appear first in Y.
        for row in range(3):
            self.assertAlmostEqual(np.sum(Y_cat[row]), np.sum(Y_comb[row, :8]))
            self.assertAlmostEqual(np.sum(Y_num[row]), np.sum(Y_comb[row, 8:]))
        #
        # Now we do everything again, but using a different dataset and the already
        # fitted data preprocessor
        #

        # Numerical dataset
        X_num = np.array([
            [1., 5.,     1.],   # noqa : matrix legibility
            [2., np.nan, 0.],   # noqa : matrix legibility
            [3., 1.,     3.]])  # noqa : matrix legibility
        # Y_num is what we should get after data preprocessing X_num
        Y_num = np.array([
            [ 3/sdev, -1/sdev],   # noqa : matrix legibility
            [     0., -2/sdev],   # noqa : matrix legibility
            [-1/sdev,  1/sdev]])  # noqa : matrix legibility
        # Categorical dataset
        X_cat = np.array([
            [1, 2,      1],
            [3, 0,      np.nan],
            [2, np.nan, 0]])
        # Y_cat is what we should get after data preprocessing X_cat
        Y_cat = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 1]])
        # Combine datasetd and shuffle columns:
        X_comb = np.hstack((X_num, X_cat))
        X_comb = X_comb[:, random_order]
        # Data preprocessing with already fitted transformer
        Y_comb = DPP.transform(X_comb)
        Y_comb = Y_comb.todense() if sparse_output else Y_comb
        # check shape
        self.assertEquals(Y_comb.shape, (3, 10))
        # check content. Categorical columns appear first in Y.
        for row in range(3):
            self.assertAlmostEqual(np.sum(Y_cat[row]), np.sum(Y_comb[row, :8]))
            self.assertAlmostEqual(np.sum(Y_num[row]), np.sum(Y_comb[row, 8:]))

    def test_fit_transform(self):
        self.do_a_fit_transform(sparse_output=False)

    def test_fit_transform_sparse(self):
        self.do_a_fit_transform(sparse_output=True)

    def test_string_categories(self):
        # Numerical dataset (as used in NumericalPreprocessingPipelineTest)
        X_num = np.array([
            [3.14, 1.,     1.],   # noqa : matrix legibility
            [3.14, 2., np.nan],   # noqa : matrix legibility
            [3.14, 3.,     3.]])  # noqa : matrix legibility
        # Categorical string dataset
        X_cat = np.array([
            ['red', 'medium', 'small'],
            ['blue', 'short', 'big'],
            ['white', 'tall', np.nan]])
        # Combined dataset with shuffled columns:
        X_comb = np.hstack((X_num, X_cat))
        categ_feat = np.array([False] * 3 + [True] * 3)
        random_order = np.random.choice(np.arange(6), size=6, replace=False)
        X_comb = X_comb[:, random_order]
        categ_feat = categ_feat[random_order]
        # Strings are not allowed, therefore:
        with self.assertRaises(ValueError):
            DataPreprocessor(categorical_features=categ_feat).fit_transform(X_comb)
