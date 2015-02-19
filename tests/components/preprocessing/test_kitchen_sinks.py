import unittest

from ParamSklearn.components.preprocessing.kitchen_sinks import RandomKitchenSinks
from ParamSklearn.util import _test_preprocessing


class KitchenSinkComponent(unittest.TestCase):
    def test_default_configuration(self):
        transformation, original = _test_preprocessing(RandomKitchenSinks)
        self.assertEqual(transformation.shape[0], original.shape[0])
        self.assertEqual(transformation.shape[1], 100)
        self.assertFalse((transformation == 0).all())
