import unittest

import numpy as np
from scipy import sparse
from sklearn.utils.testing import assert_array_almost_equal
from sklearn.datasets import load_iris

from ParamSklearn.util import get_dataset
from ParamSklearn.implementations.MinMaxScaler import MinMaxScaler


class MinMaxScalerTest(unittest.TestCase):
    def test_min_max_scaler_iris(self):
        iris = load_iris()
        X = iris.data

        scaler = MinMaxScaler()
        # default params
        X_trans = scaler.fit_transform(X)
        assert_array_almost_equal(X_trans.min(axis=0), 0)
        assert_array_almost_equal(X_trans.min(axis=0), 0)
        assert_array_almost_equal(X_trans.max(axis=0), 1)
        X_trans_inv = scaler.inverse_transform(X_trans)
        assert_array_almost_equal(X, X_trans_inv)

        # not default params: min=1, max=2
        scaler = MinMaxScaler(feature_range=(1, 2))
        X_trans = scaler.fit_transform(X)
        assert_array_almost_equal(X_trans.min(axis=0), 1)
        assert_array_almost_equal(X_trans.max(axis=0), 2)
        X_trans_inv = scaler.inverse_transform(X_trans)
        assert_array_almost_equal(X, X_trans_inv)

        # min=-.5, max=.6
        scaler = MinMaxScaler(feature_range=(-.5, .6))
        X_trans = scaler.fit_transform(X)
        assert_array_almost_equal(X_trans.min(axis=0), -.5)
        assert_array_almost_equal(X_trans.max(axis=0), .6)
        X_trans_inv = scaler.inverse_transform(X_trans)
        assert_array_almost_equal(X, X_trans_inv)

        # raises on invalid range
        scaler = MinMaxScaler(feature_range=(2, 1))
        self.assertRaises(ValueError, scaler.fit, X)

    def test_min_max_scaler_zero_variance_features(self):
        """Check min max scaler on toy data with zero variance features"""
        X = [[0., 1., +0.5],
             [0., 1., -0.1],
             [0., 1., +1.1]]

        X_new = [[+0., 2., 0.5],
                 [-1., 1., 0.0],
                 [+0., 1., 1.5]]
        # default params
        scaler = MinMaxScaler()
        X_trans = scaler.fit_transform(X)
        X_expected_0_1 = [[0., 0., 0.5],
                          [0., 0., 0.0],
                          [0., 0., 1.0]]
        assert_array_almost_equal(X_trans, X_expected_0_1)
        X_trans_inv = scaler.inverse_transform(X_trans)
        assert_array_almost_equal(X, X_trans_inv)

        X_trans_new = scaler.transform(X_new)
        X_expected_0_1_new = [[+0., 1., 0.500],
                              [-1., 0., 0.083],
                              [+0., 0., 1.333]]
        assert_array_almost_equal(X_trans_new, X_expected_0_1_new, decimal=2)

        # not default params
        scaler = MinMaxScaler(feature_range=(1, 2))
        X_trans = scaler.fit_transform(X)
        X_expected_1_2 = [[1., 1., 1.5],
                          [1., 1., 1.0],
                          [1., 1., 2.0]]
        assert_array_almost_equal(X_trans, X_expected_1_2)


    @unittest.skip("I don't understand the original unit test. Thus, I forbid "
                   "1d input to the scaling function.")
    def test_min_max_scaler_1d(self):
        """Test scaling of dataset along single axis"""
        rng = np.random.RandomState(0)
        X = rng.randn(5)
        X_orig_copy = X.copy()

        scaler = MinMaxScaler()
        X_scaled = scaler.fit(X).transform(X)
        assert_array_almost_equal(X_scaled.min(axis=0), 0.0)
        assert_array_almost_equal(X_scaled.max(axis=0), 1.0)

        # check inverse transform
        X_scaled_back = scaler.inverse_transform(X_scaled)
        assert_array_almost_equal(X_scaled_back, X_orig_copy)

        # Test with 1D list
        X = [0., 1., 2, 0.4, 1.]
        scaler = MinMaxScaler()
        X_scaled = scaler.fit(X).transform(X)
        assert_array_almost_equal(X_scaled.min(axis=0), 0.0)
        assert_array_almost_equal(X_scaled.max(axis=0), 1.0)

    def test_min_max_scaler_sparse_boston_data(self):
        # Use the boston housing dataset, because column three is 1HotEncoded!
        # This is important to test; because the normal sklearn rescaler
        # would set all values of the 1Hot Encoded column to zero, while we
        # keep the values at 1.
        X_train, Y_train, X_test, Y_test = get_dataset('boston',
                                                       make_sparse=True)
        num_data_points = len(X_train.data)
        expected_max_values = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        expected_max_values = np.array(expected_max_values).reshape((1, -1))

        scaler = MinMaxScaler()
        scaler.fit(X_train, Y_train)
        transformation = scaler.transform(X_train)

        assert_array_almost_equal(np.array(transformation.todense().min(axis=0)),
                                  np.zeros((1, 13)))
        assert_array_almost_equal(np.array(transformation.todense().max(axis=0)),
                                  expected_max_values)
        # Test that the matrix is still sparse
        self.assertTrue(sparse.issparse(transformation))
        self.assertEqual(num_data_points, len(transformation.data))