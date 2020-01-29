import unittest
import numpy as np
from scipy import sparse

from autosklearn.pipeline.components.data_preprocessing.data_preprocessing_categorical \
    import CategoricalPreprocessingPipeline


class CategoricalPreprocessingPipelineTest(unittest.TestCase):

    def test_data_type_consistency(self):
        X = np.random.randint(3, 6, (3, 4))
        Y = CategoricalPreprocessingPipeline().fit_transform(X)
        self.assertFalse(sparse.issparse(Y))

        X = sparse.csc_matrix(
            ([3, 6, 4, 5], ([0, 1, 2, 1], [3, 2, 1, 0])), shape=(3, 4))
        Y = CategoricalPreprocessingPipeline().fit_transform(X)
        self.assertTrue(sparse.issparse(Y))

    def test_fit_transform(self):
        X = np.array([
            [1, 2, 1],
            [3, 1, 1],
            [2, 9, np.nan]])
        Y = np.array([
            [1, 0, 0, 0, 1, 0, 0, 1],
            [0, 0, 1, 1, 0, 0, 0, 1],
            [0, 1, 0, 0, 0, 1, 1, 0]])
        # dense input
        Yt = CategoricalPreprocessingPipeline().fit_transform(X)
        np.testing.assert_array_equal(Yt, Y)
        # sparse input
        X_sparse = sparse.csc_matrix(X)
        Yt = CategoricalPreprocessingPipeline().fit_transform(X_sparse)
        Yt = Yt.todense()
        np.testing.assert_array_equal(Yt, Y)

    def test_transform(self):
        X1 = np.array([
            [1, 2, 0],
            [3, 0, 0],
            [2, 9, np.nan]])
        Y1 = np.array([
            [1, 0, 0, 0, 1, 0, 0, 1],
            [0, 0, 1, 1, 0, 0, 0, 1],
            [0, 1, 0, 0, 0, 1, 1, 0]])
        X2 = np.array([
            [2, 2, 1],
            [3, 0, 0],
            [2, np.nan, np.nan]])
        Y2 = np.array([
            [0, 1, 0, 0, 1, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 0, 1],
            [0, 1, 0, 0, 0, 0, 1, 0]])
        X3 = np.array([
            [3, np.nan, 0],
            [3, 9, np.nan],
            [2, 2, 5]])
        Y3 = np.array([
            [0, 0, 1, 0, 0, 0, 0, 1],
            [0, 0, 1, 0, 0, 1, 1, 0],
            [0, 1, 0, 0, 1, 0, 0, 0]])
        # "fit"
        CPPL = CategoricalPreprocessingPipeline()
        CPPL.fit_transform(X1)
        # Transform what was fitted
        Y1t = CPPL.transform(X1)
        np.testing.assert_array_equal(Y1t, Y1)
        # Transform a new dataset with categories not seen during fit
        Y2t = CPPL.transform(X2)
        np.testing.assert_array_equal(Y2t, Y2)
        # And again with yet a different dataset
        Y3t = CPPL.transform(X3)
        np.testing.assert_array_equal(Y3t, Y3)

    def test_transform_with_coalescence(self):
        # Generates an array with categories 0, 20, 5, 6, 10, and occurences of 60%,
        # 30%, 19% 0.5% and 0.5% respectively
        X = np.vstack((
            np.ones((120, 10)) * 0,
            np.ones((60, 10)) * 20,
            np.ones((18, 10)) * 5,
            np.ones((1, 10)) * 6,
            np.ones((1, 10)) * 10,
        ))
        for col in range(X.shape[1]):
            np.random.shuffle(X[:, col])

        CPPL = CategoricalPreprocessingPipeline()
        Y1t = CPPL.fit_transform(X)
        # From the 5 original categories, 2 are coalesced, remaining 4.
        # Dataset has 10 cols, therefore Y must have 40 (i.e. 4 x 10) cols
        self.assertEqual(Y1t.shape, (200, 40))
        # Consistency check:
        Y2t = CPPL.transform(X)
        np.testing.assert_array_equal(Y1t, Y2t)
