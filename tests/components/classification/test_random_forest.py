import unittest

from AutoSklearn.components.classification.random_forest import RandomForest
from AutoSklearn.util import _test_classifier

import sklearn.metrics


class RandomForestComponentTest(unittest.TestCase):
    def test_default_configuration(self):
        for i in range(10):
            predictions, targets = _test_classifier(RandomForest, dataset='iris')
            self.assertAlmostEqual(0.92,
                sklearn.metrics.accuracy_score(predictions, targets))