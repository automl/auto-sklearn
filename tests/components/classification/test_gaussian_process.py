import unittest

from ParamSklearn.components.classification.gaussian_process import GPyClassifier
from ParamSklearn.util import _test_classifier

import sklearn.metrics


class GPyClassifierComponentTest(unittest.TestCase):
    def test_default_configuration(self):
        for i in range(10):
            predictions, targets = _test_classifier(GPyClassifier)
            self.assertAlmostEqual(0.95999999999999996,
                sklearn.metrics.accuracy_score(predictions, targets))

