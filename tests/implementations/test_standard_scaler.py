import unittest

import numpy as np
import scipy.sparse
from sklearn.utils.testing import assert_array_almost_equal
from sklearn.preprocessing.data import scale
from sklearn.utils.sparsefuncs import inplace_column_scale, \
    mean_variance_axis0

from AutoSklearn.implementations.StandardScaler import StandardScaler

matrix1 = [[0, 1, 2],
           [0, 1, 2],
           [0, 1, 2]]


class TestStandardScaler(unittest.TestCase):
    def test_scaler_1d(self):
        """Test scaling of dataset along single axis"""
        rng = np.random.RandomState(0)
        X = rng.randn(5)
        X_orig_copy = X.copy()

        scaler = StandardScaler()
        X_scaled = scaler.fit(X).transform(X, copy=False)
        assert_array_almost_equal(X_scaled.mean(axis=0), 0.0)
        assert_array_almost_equal(X_scaled.std(axis=0), 1.0)

        # check inverse transform
        X_scaled_back = scaler.inverse_transform(X_scaled)
        assert_array_almost_equal(X_scaled_back, X_orig_copy)

        # Test with 1D list
        X = [0., 1., 2, 0.4, 1.]
        scaler = StandardScaler()
        X_scaled = scaler.fit(X).transform(X, copy=False)
        assert_array_almost_equal(X_scaled.mean(axis=0), 0.0)
        assert_array_almost_equal(X_scaled.std(axis=0), 1.0)

        X_scaled = scale(X)
        assert_array_almost_equal(X_scaled.mean(axis=0), 0.0)
        assert_array_almost_equal(X_scaled.std(axis=0), 1.0)

        # Test with sparse list
        X = scipy.sparse.coo_matrix((np.random.random((10,)),
                                     ([i**2 for i in range(10)],
                                      [0 for i in range(10)])))
        X = X.tocsr()
        scaler = StandardScaler()
        X_scaled = scaler.fit(X).transform(X, copy=False)

        self.assertFalse(np.any(np.isnan(X_scaled.data)))
        self.assertAlmostEqual(X_scaled.mean(axis=0), 0)
        assert_array_almost_equal(np.sqrt([X.data[X.indices == i].var()
                                           for i in range(X.shape[1])]), 1)

        # Check that X has not been copied
        self.assertTrue(X_scaled is X)
        # Check that the matrix is still sparse
        self.assertEqual(len(X.indices), 10)

    def test_scaler_2d_arrays(self):
        """Test scaling of 2d array along first axis"""
        rng = np.random.RandomState(0)
        X = rng.randn(4, 5)
        X[:, 0] = 0.0  # first feature is always of zero

        scaler = StandardScaler()
        X_scaled = scaler.fit(X).transform(X, copy=True)
        self.assertFalse(np.any(np.isnan(X_scaled)))

        assert_array_almost_equal(X_scaled.mean(axis=0), 5 * [0.0])
        assert_array_almost_equal(X_scaled.std(axis=0), [0., 1., 1., 1., 1.])
        # Check that X has been copied
        self.assertTrue(X_scaled is not X)

        # check inverse transform
        X_scaled_back = scaler.inverse_transform(X_scaled)
        self.assertTrue(X_scaled_back is not X)
        self.assertTrue(X_scaled_back is not X_scaled)
        assert_array_almost_equal(X_scaled_back, X)

        X_scaled = scale(X, axis=1, with_std=False)
        self.assertFalse(np.any(np.isnan(X_scaled)))
        assert_array_almost_equal(X_scaled.mean(axis=1), 4 * [0.0])
        X_scaled = scale(X, axis=1, with_std=True)
        self.assertFalse(np.any(np.isnan(X_scaled)))
        assert_array_almost_equal(X_scaled.mean(axis=1), 4 * [0.0])
        assert_array_almost_equal(X_scaled.std(axis=1), 4 * [1.0])
        # Check that the data hasn't been modified
        self.assertTrue(X_scaled is not X)

        X_scaled = scaler.fit(X).transform(X, copy=False)
        self.assertFalse(np.any(np.isnan(X_scaled)))
        assert_array_almost_equal(X_scaled.mean(axis=0), 5 * [0.0])
        assert_array_almost_equal(X_scaled.std(axis=0), [0., 1., 1., 1., 1.])
        # Check that X has not been copied
        self.assertTrue(X_scaled is X)

        X = rng.randn(4, 5)
        X[:, 0] = 1.0  # first feature is a constant, non zero feature
        scaler = StandardScaler()
        X_scaled = scaler.fit(X).transform(X, copy=True)
        self.assertFalse(np.any(np.isnan(X_scaled)))
        assert_array_almost_equal(X_scaled.mean(axis=0), 5 * [0.0])
        assert_array_almost_equal(X_scaled.std(axis=0), [0., 1., 1., 1., 1.])
        # Check that X has not been copied
        self.assertTrue(X_scaled is not X)

        # Same thing for sparse matrices...
        X = scipy.sparse.coo_matrix((np.random.random((12,)),
                                     ([i for i in range(12)],
                                      [int(i / 3) for i in range(12)])))
        X = X.tocsr()
        scaler = StandardScaler()
        X_scaled = scaler.fit(X).transform(X, copy=False)

        self.assertFalse(np.any(np.isnan(X_scaled.data)))
        assert_array_almost_equal(X_scaled.mean(axis=0),
                                  np.zeros((1, 4), dtype=np.float64))
        assert_array_almost_equal(np.sqrt([X.data[X.indices == i].var()
                                           for i in range(X.shape[1])]).reshape((1, 4)),
                                  np.ones((1, 4), dtype=np.float64))

        # Check that X has not been copied
        self.assertTrue(X_scaled is X)
        # Check that the matrix is still sparse
        self.assertEqual(len(X.indices), 12)

