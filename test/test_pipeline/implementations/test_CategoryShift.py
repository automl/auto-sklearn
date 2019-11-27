import unittest
import numpy as np
import scipy.sparse
from sklearn.utils.testing import assert_array_almost_equal

from autosklearn.pipeline.implementations.CategoryShift import CategoryShift


class CategoryShiftTest(unittest.TestCase):

    def test_dense(self):
        X = np.random.randint(0, 255, (3,4))
        X_copy = np.copy(X)
        Y = CategoryShift().fit_transform(X_copy)
        assert_array_almost_equal(X_copy, X + 3)
        # Check if no copies were made
        self.assertEqual(id(X_copy), id(Y))
    
    def test_sparse(self):
        X = scipy.sparse.csc_matrix(([1, 2, 0, 4], ([0, 1, 2, 1], [3, 2, 1, 0])), shape=(3, 4))
        X_copy = scipy.sparse.csc_matrix.copy(X)
        Y = CategoryShift().fit_transform(X_copy)
        X.data += 3
        assert_array_almost_equal(X.todense(), X_copy.todense())
        # Check if no copies were made
        self.assertEqual(id(X_copy), id(Y))

    def test_negative(self):
        X = np.array([[-1,2], [3, 4]])
        with self.assertRaises(ValueError):
            CategoryShift().fit_transform(X)
    
    #def test_string(self):
    #    X = np.array([['blue', 'flat'], ['red', 'convex']])
    #    with self.assertRaises(ValueError):
    #        CategoryShift().fit_transform(X)

    def test_fit_doesnt_modify_array(self):
        X = np.random.rand(3,4) * 10.
        X_copy = np.copy(X)
        CategoryShift().fit(X)
        assert_array_almost_equal(X, X_copy)


        