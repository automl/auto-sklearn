import numpy as np
import unittest

from autosklearn.pipeline.components.feature_preprocessing.no_preprocessing import NoPreprocessing
from autosklearn.pipeline.util import _test_preprocessing, PreprocessingTestCase


class NoneComponentTest(PreprocessingTestCase):
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

    def test_preprocessing_dtype(self):
        super(NoneComponentTest, self)._test_preprocessing_dtype(NoPreprocessing)


