import unittest

import numpy as np

from ParamSklearn.components.preprocessing.densifier import Densifier
from ParamSklearn.util import _test_preprocessing


class DensifierComponentTest(unittest.TestCase):
    def test_default_configuration(self):
        transformation, original = _test_preprocessing(Densifier, make_sparse=True)
        self.assertEqual(transformation.shape, original.shape)
        self.assertIsInstance(transformation, np.ndarray)
