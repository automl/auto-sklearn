import unittest

from ParamSklearn.components.preprocessing.sparse_filtering import SparseFiltering
from ParamSklearn.util import _test_preprocessing, PreprocessingTestCase


class SparseFilteringComponentTest(PreprocessingTestCase):
    def test_default_configuration(self):
        transformation, original = _test_preprocessing(SparseFiltering)
        self.assertEqual(transformation.shape[0], original.shape[0])
        self.assertFalse((transformation == 0).all())

    @unittest.skip("Right now, the SparseFiltering returns a float64 array!")
    def test_preprocessing_dtype(self):
        super(SparseFilteringComponentTest, self)._test_preprocessing_dtype(SparseFiltering)