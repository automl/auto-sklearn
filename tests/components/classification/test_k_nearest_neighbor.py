import unittest

from AutoSklearn.components.classification.k_neighbors_classifier import \
    KNearestNeighborsClassifier
from AutoSklearn.util import _test_classifier_with_iris

import sklearn.metrics


class KNearestNeighborsComponentTest(unittest.TestCase):
    def test_default_configuration(self):
        for i in range(10):
            predictions, targets = \
                _test_classifier_with_iris(KNearestNeighborsClassifier)
            self.assertAlmostEqual(0.95999999999999,
                sklearn.metrics.accuracy_score(predictions, targets))