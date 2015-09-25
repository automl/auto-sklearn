# -*- encoding: utf-8 -*-
"""Created on Dec 16, 2014.

@author: Aaron Klein

"""
from __future__ import print_function
import unittest

import numpy as np

from autosklearn.evaluation.resampling import split_data


class ResamplingTest(unittest.TestCase):
    _multiprocess_can_split_ = True

    def test_split_data_regression(self):
        n_points = 1000
        np.random.seed(42)
        n_dims = np.random.randint(1, 100)
        X = np.random.rand(n_points, n_dims)
        y = np.random.rand(n_points)

        X_train, X_valid, Y_train, Y_valid = split_data(X, y)

        self.assertEqual(X_train.shape[0], 670)
        self.assertEqual(X_valid.shape[0], 330)
        self.assertEqual(Y_train.shape[0], 670)
        self.assertEqual(Y_valid.shape[0], 330)
        self.assertEqual(X_train.shape[1], n_dims)
        self.assertEqual(X_valid.shape[1], n_dims)

        # Random checks
        self.assertAlmostEqual(X_train[4, 2], 0.5986584841970366)
        self.assertAlmostEqual(X_valid[4, 2], 0.63911512838980322)

    def test_split_not_same_shape(self):
        X = np.array([[3, 4], [1, 2], [3, 4]])
        y = np.array([0, 0, 1, 1])
        self.assertRaises(ValueError, split_data, X, y)

    def test_stratify(self):
        for i in range(5):
            self._split_regular()
            self._split_regular_classification()
            self._stratify()

    def _split_regular(self):
        X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4]])
        y = np.array([0, 0, 0, 1, 1, 2])
        X_train, X_valid, Y_train, Y_valid = split_data(X, y)

        # Check shapes
        self.assertEqual(X_train.shape, (4, 2))
        self.assertEqual(Y_train.shape, (4, ))
        self.assertEqual(X_valid.shape, (2, 2))
        self.assertEqual(Y_valid.shape, (2, ))

        self.assertListEqual(list(Y_valid), [0, 0])
        self.assertListEqual(list(Y_train), [2, 0, 1, 1])

    def _split_regular_classification(self):
        X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4]])
        y = np.array([0, 0, 2, 1, 1, 0])
        X_train, X_valid, Y_train, Y_valid = split_data(X, y,
                                                        classification=True)

        # Check shapes
        self.assertEqual(X_train.shape, (4, 2))
        self.assertEqual(Y_train.shape, (4, ))
        self.assertEqual(X_valid.shape, (2, 2))
        self.assertEqual(Y_valid.shape, (2, ))

        self.assertListEqual(list(Y_valid), [0, 1])
        self.assertListEqual(list(Y_train), [0, 0, 1, 2])

    def _stratify(self):
        X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4]])
        y = np.array([0, 0, 0, 0, 1, 1])
        X_train, X_valid, Y_train, Y_valid = split_data(X, y)

        # Check shapes
        self.assertEqual(X_train.shape[0], 4)
        self.assertEqual(X_train.shape[1], 2)
        self.assertEqual(Y_train.shape[0], 4)

        self.assertEqual(X_valid.shape[0], 2)
        self.assertEqual(X_valid.shape[1], 2)
        self.assertEqual(Y_valid.shape[0], 2)

        self.assertListEqual(list(Y_valid), [1, 0])
        self.assertListEqual(list(Y_train), [0, 0, 0, 1])

    def test_split_classification_many_imbalanced_classes(self):
        for i in range(10):
            X = np.array([range(20), range(20)]).transpose()
            y = np.array((0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3,
                          4, 5))
            np.random.shuffle(y)
            X_train, X_valid, Y_train, Y_valid = split_data(
                X, y,
                classification=True)
            print(X_train, Y_train)
            self.assertLessEqual(max(Y_valid), 1)


if __name__ == '__main__':
    unittest.main()
