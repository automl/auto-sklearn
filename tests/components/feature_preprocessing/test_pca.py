import unittest

import numpy as np

from ParamSklearn.components.feature_preprocessing.pca import PCA
from ParamSklearn.util import _test_preprocessing, PreprocessingTestCase


class PCAComponentTest(PreprocessingTestCase):
    def test_default_configuration(self):
        transformations = []
        for i in range(10):
            transformation, original = _test_preprocessing(PCA)
            self.assertEqual(transformation.shape, original.shape)
            self.assertFalse((transformation == original).all())
            transformations.append(transformation)
            if len(transformations) > 1:
                self.assertTrue((transformations[-1] == transformations[-2]).all())

    def test_preprocessing_dtype(self):
        super(PCAComponentTest, self)._test_preprocessing_dtype(PCA,
                                                                test_sparse=False)