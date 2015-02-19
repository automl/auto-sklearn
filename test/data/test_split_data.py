'''
Created on Dec 16, 2014

@author: Aaron Klein
'''
import unittest
import numpy as np

from AutoML2015.data.split_data import split_data


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

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
