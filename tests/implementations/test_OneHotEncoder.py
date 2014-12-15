import unittest

import numpy as np

from AutoSklearn.implementations.OneHotEncoder import OneHotEncoder

dense1 = [[1, 5, 9],
          [1, 3, 9]]
dense1_1h = [[1, 0, 1, 1],
             [1, 1, 0, 1]]

dense2 = [[1, np.NaN, 9],
          [np.NaN, 3, 9],
          [2, 1, 7]]
dense2_1h = [[1, 0, 0, 0, 0, 1],
             [0, 0, 0, 1, 0, 1],
             [0, 1, 1, 0, 1, 0]]


class OneHotEncoderTest(unittest.TestCase):
    def test_dense1(self):
        self.fit_then_transform(dense1_1h, dense1)

    def test_dense2(self):
        self.fit_then_transform(dense2_1h, dense2)

    def fit_then_transform(self, expected, input):
        ohe = OneHotEncoder()
        ohe.fit(input)
        transformation = ohe.transform(input)
        transformation = transformation.todense()
        self.assertTrue((expected == transformation).all())