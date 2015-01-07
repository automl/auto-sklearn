import unittest

from AutoSklearn.components.preprocessing.sparse_filtering import SparseFiltering
from AutoSklearn.util import _test_preprocessing


class SparseFilteringComponentTest(unittest.TestCase):
    def test_default_configuration(self):
        transformation, original = _test_preprocessing(SparseFiltering)
        self.assertEqual(transformation.shape[0], original.shape[0])
        self.assertFalse((transformation == 0).all())
