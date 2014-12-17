'''
Created on Dec 16, 2014

@author: Aaron Klein
'''
import unittest
import numpy as np

from AutoML2015.util.split_data import split_data


class Test(unittest.TestCase):


    def test_split_data(self):
        n_points = np.random.randint()
        n_dims = np.random.randint()
        X = np.random.rand(n_points, n_dims)
        y = np.random.rand(n_points)

        X_train, X_valid, Y_train, Y_valid  = split_data(X,y)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()