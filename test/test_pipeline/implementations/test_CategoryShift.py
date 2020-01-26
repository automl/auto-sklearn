import unittest
import numpy as np
import scipy.sparse

from autosklearn.pipeline.implementations.CategoryShift import CategoryShift


class CategoryShiftTest(unittest.TestCase):

    def test_dense(self):
        X = np.random.randint(0, 255, (3, 4))
        Y = CategoryShift().fit_transform(X)
        self.assertTrue((Y == X + 3).all())

    def test_sparse(self):
        X = scipy.sparse.csc_matrix(
            ([1, 2, 0, 4], ([0, 1, 2, 1], [3, 2, 1, 0])), shape=(3, 4))
        Y = CategoryShift().fit_transform(X)
        X.data += 3
        self.assertTrue((Y.todense() == X.todense()).all())
        # Check if the sparsity stays the same before and after the transformation
        self.assertEqual(X.data.shape, Y.data.shape)
        self.assertTrue((X.indices == Y.indices).all())
        self.assertTrue((X.indptr == Y.indptr).all())

    def test_negative(self):
        X = np.array([[-1, 2], [3, 4]])
        with self.assertRaises(ValueError):
            CategoryShift().fit_transform(X)

    def test_string(self):
        X = np.array([['a', 'b'], ['c', 'd']])
        with self.assertRaises(ValueError):
            CategoryShift().fit_transform(X)
