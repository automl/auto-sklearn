import unittest

from ParamSklearn.components.preprocessing.select_percentile_regression import SelectPercentileRegression
from ParamSklearn.util import _test_preprocessing


class SelectPercentileRegressionTest(unittest.TestCase):
    def test_default_configuration(self):
        transformation, original = _test_preprocessing(dataset="boston", Preprocessor=SelectPercentileRegression)
        self.assertEqual(transformation.shape[0], original.shape[0])
        self.assertEqual(transformation.shape[1], int(original.shape[1]/2))
        self.assertFalse((transformation == 0).all())
