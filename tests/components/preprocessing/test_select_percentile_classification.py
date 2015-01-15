import unittest

import scipy.sparse

from AutoSklearn.components.preprocessing.select_percentile_classification import SelectPercentileClassification
from AutoSklearn.util import _test_preprocessing


class SelectPercentileClassificationTest(unittest.TestCase):
    def test_default_configuration(self):
        transformation, original = _test_preprocessing(SelectPercentileClassification)
        self.assertEqual(transformation.shape[0], original.shape[0])
        self.assertEqual(transformation.shape[1], int(original.shape[1]/2))
        self.assertFalse((transformation == 0).all())

        transformation, original = _test_preprocessing(SelectPercentileClassification, make_sparse=True)
        self.assertTrue(scipy.sparse.issparse(transformation))
        self.assertEqual(transformation.shape[0], original.shape[0])
        self.assertEqual(transformation.shape[1], int(original.shape[1]/2))
