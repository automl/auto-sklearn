import unittest

from AutoSklearn.components.preprocessing.pca import PCA
from AutoSklearn.util import _test_preprocessing


class LibLinearComponentTest(unittest.TestCase):
    def test_default_configuration(self):
        transformations = []
        for i in range(10):
            transformation, original = _test_preprocessing(PCA)
            self.assertEqual(transformation.shape, original.shape)
            self.assertFalse((transformation == original).all())
            transformations.append(transformation)
            if len(transformations) > 1:
                self.assertTrue((transformations[-1] == transformations[-2]).all())