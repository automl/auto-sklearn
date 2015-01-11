import unittest

from AutoSklearn.components.classification.gradient_boosting import \
    GradientBoostingClassifier
from AutoSklearn.util import _test_classifier

import sklearn.metrics


class GradientBoostingComponentTest(unittest.TestCase):
    def test_default_configuration(self):
        for i in range(10):
            predictions, targets = \
                _test_classifier(GradientBoostingClassifier)
            self.assertAlmostEqual(0.93999999999999995,
                sklearn.metrics.accuracy_score(predictions, targets))