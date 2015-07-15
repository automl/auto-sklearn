import unittest

import numpy as np
import scipy.sparse
from sklearn.utils.testing import assert_array_almost_equal

from ParamSklearn.implementations.OneHotEncoder import OneHotEncoder

dense1 = np.array([[0, 1, 0],
                   [0, 0, 0]])
dense1_1h = np.array([[1, 0, 1, 1],
                     [1, 1, 0 , 1]])

dense2 = np.array([[0, np.NaN, 0],
                   [np.NaN, 0, 2],
                   [1, 1, 1]])
dense2_1h = np.array([[1, 0, 0, 0, 1, 0, 0],
                      [0, 0, 1, 0, 0, 0, 1],
                      [0, 1, 0, 1, 0, 1, 0]])
dense2_partial_1h = np.array([[1., 0., 0., 0., 0.],
                              [0., 0., 1., 0., 2.],
                              [0., 1., 0., 1., 1.]])

# All NaN slice
dense3 = np.array([[0, 1, np.NaN],
                   [1, 0, np.NaN]])
dense3_1h = np.array([[1, 0, 0, 1, 0],
                      [0, 1, 1, 0, 0]])

sparse1 = scipy.sparse.csc_matrix(([2, 1, 0, 0, 1, 2],
                                   ((1, 4, 5, 2, 3, 5),
                                    (0, 0, 0, 1, 1, 1))), shape=(6, 2))
sparse1_1h = scipy.sparse.csc_matrix(([1, 1, 1, 1, 1, 1],
                                      ((5, 4, 1, 2, 3, 5),
                                       (0, 1, 2, 3, 4, 5))), shape=(6, 6))
sparse1_paratial_1h = scipy.sparse.csc_matrix(([1, 1, 1, 0, 1, 2],
                                               ((5, 4, 1, 2, 3, 5),
                                                (0, 1, 2, 3, 3, 3))),
                                              shape=(6, 4))

# All zeros slice
sparse2 = scipy.sparse.csc_matrix(([2, 1, 0, 0, 0, 0],
                                   ((1, 4, 5, 2, 3, 5),
                                    (0, 0, 0, 1, 1, 1))), shape=(6, 2))
sparse2_1h = scipy.sparse.csc_matrix(([1, 1, 1, 1, 1, 1],
                                      ((5, 4, 1, 2, 3, 5),
                                       (0, 1, 2, 3, 3, 3))), shape=(6, 4))

sparse2_csr = scipy.sparse.csr_matrix(([2, 1, 0, 0, 0, 0],
                                      ((1, 4, 5, 2, 3, 5),
                                       (0, 0, 0, 1, 1, 1))), shape=(6, 2))
sparse2_csr_1h = scipy.sparse.csr_matrix(([1, 1, 1, 1, 1, 1],
                                         ((5, 4, 1, 2, 3, 5),
                                          (0, 1, 2, 3, 3, 3))), shape=(6, 4))


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

    def test_dense3(self):
        self.fit_then_transform(dense3_1h, dense3)
        self.fit_then_transform_dense(dense3_1h, dense3)

    def test_sparse1(self):
        self.fit_then_transform(sparse1_1h.todense(), sparse1)
        self.fit_then_transform_dense(sparse1_1h.todense(), sparse1)

    def test_sparse1_with_non_sparse_components(self):
        self.fit_then_transform(sparse1_paratial_1h.todense(), sparse1,
                                categorical_features=[True, False])
        # This test does not apply here. The sparse matrix will be cut into a
        #  continouos and a categorical part, after one hot encoding only the
        #  categorical part is an array, the continuous part will still be a
        # sparse matrix. Therefore, the OHE will only return a sparse matrix
        #self.fit_then_transform_dense(sparse1_paratial_1h.todense(), sparse1,
        #                              categorical_features=[True, False])

    def test_sparse2(self):
        self.fit_then_transform(sparse2_1h.todense(), sparse2)
        self.fit_then_transform_dense(sparse2_1h.todense(), sparse2)

    def test_sparse2_csr(self):
        self.fit_then_transform(sparse2_csr_1h.todense(), sparse2_csr)
        self.fit_then_transform_dense(sparse2_csr_1h.todense(), sparse2_csr)

    def fit_then_transform(self, expected, input, categorical_features='all'):
        ohe = OneHotEncoder(categorical_features=categorical_features)
        transformation = ohe.fit_transform(input.copy())
        self.assertIsInstance(transformation, scipy.sparse.csr_matrix)
        assert_array_almost_equal(expected, transformation.todense())

        ohe2 = OneHotEncoder(categorical_features=categorical_features)
        ohe2.fit(input.copy())
        transformation = ohe2.transform(input.copy())
        self.assertIsInstance(transformation, scipy.sparse.csr_matrix)
        assert_array_almost_equal(expected, transformation.todense())

    def fit_then_transform_dense(self, expected, input, categorical_features='all'):
        ohe = OneHotEncoder(categorical_features=categorical_features,
                            sparse=False)
        transformation = ohe.fit_transform(input.copy())
        self.assertIsInstance(transformation, np.ndarray)
        assert_array_almost_equal(expected, transformation)

        ohe2 = OneHotEncoder(categorical_features=categorical_features,
                             sparse=False)
        ohe2.fit(input.copy())
        transformation = ohe2.transform(input.copy())
        self.assertIsInstance(transformation, np.ndarray)
        assert_array_almost_equal(expected, transformation)

    def test_transform_with_unknown_value(self):
        input = np.array(((0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))).transpose()
        ohe = OneHotEncoder()
        ohe.fit(input)
        test_data = np.array(((0, 1, 2, 6), (0, 1, 6, 7))).transpose()
        output = ohe.transform(test_data).todense()
        self.assertEqual(5, np.sum(output))


