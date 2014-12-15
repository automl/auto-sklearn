import unittest

from AutoSklearn.components.classification.random_forest import RandomForest
from AutoSklearn.util import _test_classifier_with_iris

import sklearn.metrics


class RandomForestComponentTest(unittest.TestCase):
    def test_default_configuration(self):
        for i in range(10):
            predictions, targets = _test_classifier_with_iris(RandomForest)
            self.assertAlmostEqual(0.94,
                sklearn.metrics.accuracy_score(predictions, targets))