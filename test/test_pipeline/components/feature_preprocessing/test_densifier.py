import unittest

import numpy as np

from autosklearn.pipeline.components.feature_preprocessing.densifier import Densifier
from autosklearn.pipeline.util import _test_preprocessing, PreprocessingTestCase


class DensifierComponentTest(PreprocessingTestCase):
    def test_default_configuration(self):
        transformation, original = _test_preprocessing(Densifier, make_sparse=True)
        self.assertIsInstance(transformation, np.ndarray)
        self.assertEqual(transformation.shape, original.shape)
        self.assertIsInstance(transformation, np.ndarray)

    def test_preprocessing_dtype(self):
        super(DensifierComponentTest, self)._test_preprocessing_dtype(Densifier)

