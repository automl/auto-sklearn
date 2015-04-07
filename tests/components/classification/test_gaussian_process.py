import unittest

from ParamSklearn.components.classification.gaussian_process import GPyClassifier
from ParamSklearn.util import _test_classifier

import sklearn.metrics


class GPyClassifierComponentTest(unittest.TestCase):
    def test_default_configuration(self):
        for i in range(2):
            predictions, targets = _test_classifier(GPyClassifier)
            self.assertGreaterEqual(
                sklearn.metrics.accuracy_score(predictions, targets), 0.958)
            self.assertLessEqual(
                sklearn.metrics.accuracy_score(predictions, targets), 0.98)

