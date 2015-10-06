import os
import unittest

import numpy as np
from sklearn.utils.testing import assert_array_almost_equal

from autosklearn.data.abstract_data_manager import AbstractDataManager


dataset_train = [[2.5, 3.3, 2, 5, 1, 1],
                 [1.0, 0.7, 1, 5, 1, 0],
                 [1.3, 0.8, 1, 4, 1, 1]]
dataset_train = np.array(dataset_train)
dataset_valid = [[1.5, 1.7, 1, 4, 1, 1],
                 [2.0, 2.1, 1, 5, 1, 0],
                 [1.9, 1.8, 2, 4, 0, 1]]
dataset_valid = np.array(dataset_valid)
dataset_test = [[0.9, 2.2, 2, 4, 1, 1],
                [0.7, 3.1, 1, 5, 1, 1],
                [2.4, 2.6, 2, 5, 0, 1]]
dataset_test = np.array(dataset_test)

N = "Numerical"
B = "Binary"
C = "Categorical"


class InitFreeDataManager(AbstractDataManager):
    def __init__(self):
        pass


class CompetitionDataManagerTest(unittest.TestCase):
    _multiprocess_can_split_ = True

    def setUp(self):
        self.D = InitFreeDataManager()
        self.D._data = {}
        self.D._data['X_train'] = dataset_train.copy()
        self.D._data['X_valid'] = dataset_valid.copy()
        self.D._data['X_test'] = dataset_test.copy()

    def test_perform1HotEncoding(self):
        self.D.feat_type = [N, N, N, N, N, N]
        self.D._info = {'is_sparse': 0, 'has_missing': 0}
        self.D.perform1HotEncoding()

        assert_array_almost_equal(dataset_train, self.D.data['X_train'])
        assert_array_almost_equal(dataset_valid, self.D.data['X_valid'])
        assert_array_almost_equal(dataset_test, self.D.data['X_test'])
        self.assertIsInstance(self.D.data['X_train'], np.ndarray)
        self.assertIsInstance(self.D.data['X_valid'], np.ndarray)
        self.assertIsInstance(self.D.data['X_test'], np.ndarray)

    def test_perform1HotEncoding_binary_data(self):
        self.D.feat_type = [N, N, N, N, B, B]
        self.D._info = {'is_sparse': 0, 'has_missing': 0}
        self.D.perform1HotEncoding()

        # Nothing should have happened to the array...
        assert_array_almost_equal(dataset_train, self.D.data['X_train'])
        assert_array_almost_equal(dataset_valid, self.D.data['X_valid'])
        assert_array_almost_equal(dataset_test, self.D.data['X_test'])
        self.assertIsInstance(self.D.data['X_train'], np.ndarray)
        self.assertIsInstance(self.D.data['X_valid'], np.ndarray)
        self.assertIsInstance(self.D.data['X_test'], np.ndarray)

    def test_perform1HotEncoding_categorical_data(self):
        self.D.feat_type = [N, N, C, C, B, B]
        self.D._info = {'is_sparse': 0, 'has_missing': 0}
        self.D.perform1HotEncoding()

        # Check if converted back to dense array
        self.assertIsInstance(self.D.data['X_train'], np.ndarray)
        self.assertIsInstance(self.D.data['X_valid'], np.ndarray)
        self.assertIsInstance(self.D.data['X_test'], np.ndarray)
        # Check if the dimensions are correct
        self.assertEqual((3, 8), self.D.data['X_train'].shape)
        self.assertEqual((3, 8), self.D.data['X_valid'].shape)
        self.assertEqual((3, 8), self.D.data['X_test'].shape)
        # Some tests if encoding works
        self.assertEqual(self.D.data['X_train'][:, :4].max(), 1)
        self.assertEqual(self.D.data['X_valid'][:, :4].min(), 0)
        self.assertEqual(self.D.data['X_test'][:, :4].min(), 0)
        # Test that other stuff is not encoded
        self.assertEqual(self.D.data['X_train'][0, 4], 2.5)

    def test_perform1HotEncoding_binary_data_with_missing_values(self):
        # self.D.feat_type = [N, N, N, N, B, B]
        #self.D.info = {'is_sparse': 0, 'has_missing': 1}
        #self.D.perform1HotEncoding()
        #self.assertEqual((3, 8), self.D.data['X_train'].shape)
        pass
