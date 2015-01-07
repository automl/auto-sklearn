import unittest

import numpy as np
import scipy.sparse

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
dense2_partial_1h = [[1., 0., 0., 0., 9.],
                     [0., 0., 0., 1., 9.],
                     [0., 1., 1., 0., 7.]]

with_string = [("Black", 5, 9),
               ("Blue", 3, 7),
               ("Red", 2, 5),
               (np.NaN, 3, 1),
               ("Black", 1, 1)]

with_string_1h = [[1, 0, 0, 5, 9],
                  [0, 1, 0, 3, 7],
                  [0, 0, 1, 2, 5],
                  [0, 0, 0, 3, 1],
                  [1, 0, 0, 1, 1]]


class OneHotEncoderTest(unittest.TestCase):
    def test_dense1(self):
        self.fit_then_transform(dense1_1h, dense1)
        self.fit_then_transform_dense(dense1_1h, dense1)

    def test_dense2(self):
        self.fit_then_transform(dense2_1h, dense2)
        self.fit_then_transform_dense(dense2_1h, dense2)

    def test_dense2_with_non_sparse_components(self):
        self.fit_then_transform(dense2_partial_1h, dense2,
                                categorical_features=[True, True, False])
        self.fit_then_transform_dense(dense2_partial_1h, dense2,
                                      categorical_features=[True, True, False])

    def test_with_string(self):
        self.fit_then_transform(with_string_1h, with_string,
                                categorical_features=[True, False, False])
        self.fit_then_transform_dense(with_string_1h, with_string,
                                      categorical_features=[True, False, False])

    def fit_then_transform(self, expected, input, categorical_features='all'):
        ohe = OneHotEncoder(categorical_features=categorical_features)
        ohe.fit(input)
        transformation = ohe.transform(input)
        self.assertIsInstance(transformation, scipy.sparse.csr_matrix)
        transformation = transformation.todense()
        self.assertTrue((expected == transformation).all())

    def fit_then_transform_dense(self, expected, input, categorical_features='all'):
        ohe = OneHotEncoder(categorical_features=categorical_features,
                            sparse=False)
        ohe.fit(input)
        transformation = ohe.transform(input)
        self.assertIsInstance(transformation, np.ndarray)
        self.assertTrue((expected == transformation).all())
