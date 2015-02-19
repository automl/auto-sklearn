import unittest

from ParamSklearn.components.classification.k_nearest_neighbors import \
    KNearestNeighborsClassifier
from ParamSklearn.util import _test_classifier

import sklearn.metrics


class KNearestNeighborsComponentTest(unittest.TestCase):
    def test_default_configuration(self):
        for i in range(10):
            predictions, targets = \
                _test_classifier(KNearestNeighborsClassifier)
            self.assertAlmostEqual(0.959999999999999,
                sklearn.metrics.accuracy_score(predictions, targets))