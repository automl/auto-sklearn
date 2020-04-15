import unittest
import numpy as np

import scipy.sparse

from autosklearn.pipeline.implementations.MinorityCoalescer import MinorityCoalescer


class MinorityCoalescerTest(unittest.TestCase):

    @property
    def X1(self):
        # Generates an array with categories 3, 4, 5, 6, 7 and occurences of 30%,
        # 30%, 30%, 5% and 5% respectively
        X = np.vstack((
            np.ones((30, 10)) * 3,
            np.ones((30, 10)) * 4,
            np.ones((30, 10)) * 5,
            np.ones((5, 10)) * 6,
            np.ones((5, 10)) * 7,
        ))
        for col in range(X.shape[1]):
            np.random.shuffle(X[:, col])
        return X

    @property
    def X2(self):
        # Generates an array with categories 3, 4, 5, 6, 7 and occurences of 5%,
        # 5%, 5%, 35% and 50% respectively
        X = np.vstack((
            np.ones((5, 10)) * 3,
            np.ones((5, 10)) * 4,
            np.ones((5, 10)) * 5,
            np.ones((35, 10)) * 6,
            np.ones((50, 10)) * 7,
        ))
        for col in range(X.shape[1]):
            np.random.shuffle(X[:, col])
        return X

    def test_default(self):
        X = self.X1
        X_copy = np.copy(X)
        Y = MinorityCoalescer().fit_transform(X)
        np.testing.assert_array_almost_equal(Y, X_copy)
        # Assert no copies were made
        self.assertEqual(id(X), id(Y))

    def test_coalesce_10_percent(self):
        X = self.X1
        Y = MinorityCoalescer(minimum_fraction=.1).fit_transform(X)
        for col in range(Y.shape[1]):
            hist = np.histogram(Y[:, col], bins=np.arange(1, 7))
            np.testing.assert_array_almost_equal(hist[0], [10, 0, 30, 30, 30])
        # Assert no copies were made
        self.assertEqual(id(X), id(Y))

    def test_coalesce_10_percent_sparse(self):
        X = scipy.sparse.csc_matrix(self.X1)
        Y = MinorityCoalescer(minimum_fraction=.1).fit_transform(X)
        # Assert no copies were made
        self.assertEqual(id(X), id(Y))
        Y = Y.todense()
        for col in range(Y.shape[1]):
            hist = np.histogram(Y[:, col], bins=np.arange(1, 7))
            np.testing.assert_array_almost_equal(hist[0], [10, 0, 30, 30, 30])

    def test_invalid_X(self):
        X = self.X1 - 2
        with self.assertRaises(ValueError):
            MinorityCoalescer().fit_transform(X)

    def test_transform_after_fit(self):
        # On both X_fit and X_transf, the categories 3, 4, 5, 6, 7 are present.
        X_fit = self.X1  # Here categories 3, 4, 5 have ocurrence above 10%
        X_transf = self.X2  # Here it is the opposite, just categs 6 and 7 are above 10%

        mc = MinorityCoalescer(minimum_fraction=.1).fit(X_fit)

        # transform() should coalesce categories as learned during fit.
        # Category distribution in X_transf should be irrelevant.
        Y = mc.transform(X_transf)
        for col in range(Y.shape[1]):
            hist = np.histogram(Y[:, col], bins=np.arange(1, 7))
            np.testing.assert_array_almost_equal(hist[0], [85, 0, 5, 5, 5])
