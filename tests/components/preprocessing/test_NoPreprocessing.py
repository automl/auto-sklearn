import numpy as np
import unittest

from ParamSklearn.components.preprocessing.no_preprocessing import NoPreprocessing
from ParamSklearn.util import _test_preprocessing


class NoneComponentTest(unittest.TestCase):
    def test_default_configuration(self):
        transformation, original = _test_preprocessing(NoPreprocessing)
        self.assertEqual(transformation.shape[0], original.shape[0])
        self.assertEqual(transformation.shape[1], original.shape[1])
        self.assertFalse((transformation == 0).all())
        self.assertEqual(np.sum(original), np.sum(transformation))
        self.assertEqual(np.min(original), np.min(transformation))
        self.assertEqual(np.max(original), np.max(transformation))
        self.assertEqual(np.std(original), np.std(transformation))
        self.assertEqual(np.mean(original), np.mean(transformation))


