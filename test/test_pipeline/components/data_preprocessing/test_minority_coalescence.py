import unittest
import numpy as np
import scipy.sparse
from sklearn.utils.testing import assert_array_almost_equal

from autosklearn.pipeline.components.data_preprocessing.minority_coalescense\
    .minority_coalescer import MinorityCoalescer
from autosklearn.pipeline.components.data_preprocessing.minority_coalescense\
    .no_coalescense import NoCoalescence


class MinorityCoalescerTest(unittest.TestCase):

    @property
    def X(self):
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

    def test_no_coalescence(self):
        X = self.X
        Y = NoCoalescence().fit_transform(X)
        assert_array_almost_equal(Y, X)
        # Assert no copies were made
        self.assertEqual(id(X), id(Y))

    def test_default(self):
        X = self.X
        X_copy = np.copy(X)
        Y = MinorityCoalescer().fit_transform(X)
        assert_array_almost_equal(Y, X_copy)
        # Assert no copies were made
        self.assertEqual(id(X), id(Y))

    def test_coalesce_10_percent(self):
        X = self.X
        Y = MinorityCoalescer(minimum_fraction=.1).fit_transform(X)
        for col in range(Y.shape[1]):
            hist = np.histogram(Y[:, col], bins=np.arange(1, 7))
            assert_array_almost_equal(hist[0], [10, 0, 30, 30, 30])
        # Assert no copies were made
        self.assertEqual(id(X), id(Y))

    def test_coalesce_10_percent_sparse(self):
        X = scipy.sparse.csc_matrix(self.X)
        Y = MinorityCoalescer(minimum_fraction=.1).fit_transform(X)
        # Assert no copies were made
        self.assertEqual(id(X), id(Y))
        Y = Y.todense()
        for col in range(Y.shape[1]):
            hist = np.histogram(Y[:, col], bins=np.arange(1, 7))
            assert_array_almost_equal(hist[0], [10, 0, 30, 30, 30])

    def test_invalid_X(self):
        X = self.X - 2
        with self.assertRaises(ValueError):
            MinorityCoalescer().fit_transform(X)
