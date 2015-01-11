import unittest

from AutoSklearn.components.classification.extra_trees import \
    ExtraTreesClassifier
from AutoSklearn.util import _test_classifier

import sklearn.metrics


class ExtraTreesComponentTest(unittest.TestCase):
    def test_default_configuration(self):
        for i in range(10):
            predictions, targets = \
                _test_classifier(ExtraTreesClassifier)
            self.assertAlmostEqual(0.95999999999999996,
                sklearn.metrics.accuracy_score(predictions, targets))