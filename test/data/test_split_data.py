'''
Created on Dec 16, 2014

@author: Aaron Klein
'''
import unittest
import numpy as np

from autosklearn.data.split_data import split_data


class Test(unittest.TestCase):

    def test_split_data(self):
        n_points = 1000
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

    def test_split_not_same_shape(self):
        X = np.array([[3, 4], [1, 2], [3, 4]])
        y = np.array([0, 0, 1, 1])
        self.assertRaises(ValueError, split_data, X, y)

    def test_stratify(self):
        for i in range(5):
            self._split_regular()
            self._stratify()

    def _split_regular(self):
        X = np.array([[1, 2], [3, 4], [1, 2]])
        y = np.array([0, 1, 2])
        X_train, X_valid, Y_train, Y_valid = split_data(X, y)

        # Check shapes
        self.assertEqual(X_train.shape[0], 2)
        self.assertEqual(X_train.shape[1], 2)
        self.assertEqual(Y_train.shape[0], 2)

        self.assertEqual(X_valid.shape[0], 1)
        self.assertEqual(X_valid.shape[1], 2)
        self.assertEqual(Y_valid.shape[0], 1)

        self.assertListEqual(list(Y_valid), [0, ])
        self.assertListEqual(list(Y_train), [1, 2])

    def _stratify(self):
        X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
        y = np.array([0, 0, 1, 1])
        X_train, X_valid, Y_train, Y_valid = split_data(X, y)

        # Check shapes
        self.assertEqual(X_train.shape[0], 2)
        self.assertEqual(X_train.shape[1], 2)
        self.assertEqual(Y_train.shape[0], 2)

        self.assertEqual(X_valid.shape[0], 2)
        self.assertEqual(X_valid.shape[1], 2)
        self.assertEqual(Y_valid.shape[0], 2)

        self.assertListEqual(list(Y_valid), [0, 1])
        self.assertListEqual(list(Y_train), [0, 1])

if __name__ == "__main__":
    unittest.main()
