import unittest

from scipy import sparse

from ParamSklearn.components.preprocessing.imputation import Imputation
from ParamSklearn.util import _test_preprocessing


class ImputationTest(unittest.TestCase):
    def test_default_configuration(self):
        transformations = []
        for i in range(10):
            transformation, original = _test_preprocessing(Imputation)
            self.assertEqual(transformation.shape, original.shape)
            self.assertTrue((transformation == original).all())
            transformations.append(transformation)
            if len(transformations) > 1:
                self.assertTrue(
                    (transformations[-1] == transformations[-2]).all())

    def test_default_configuration_sparse_data(self):
        transformations = []
        transformation, original = _test_preprocessing(Imputation,
                                                       make_sparse=True)
        self.assertEqual(transformation.shape, original.shape)
        self.assertTrue((transformation.data == original.data).all())
        self.assertIsInstance(transformation, sparse.csc_matrix)
        transformations.append(transformation)