import unittest

from AutoSklearn.components.classification.liblinear import LibLinear_SVC
from AutoSklearn.util import _test_classifier_with_iris


class LibLinearComponentTest(unittest.TestCase):
    def test_default_configuration(self):
        for i in range(10):
            predictions, targets = _test_classifier_with_iris(LibLinear_SVC)
            self.assertTrue(all(targets == predictions))