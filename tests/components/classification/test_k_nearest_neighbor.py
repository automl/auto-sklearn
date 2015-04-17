import unittest

from ParamSklearn.components.classification.k_nearest_neighbors import \
    KNearestNeighborsClassifier
from ParamSklearn.util import _test_classifier, _test_classifier_predict_proba

import sklearn.metrics


class KNearestNeighborsComponentTest(unittest.TestCase):
    def test_default_configuration(self):
        for i in range(10):
            predictions, targets = \
                _test_classifier(KNearestNeighborsClassifier)
            self.assertAlmostEqual(0.959999999999999,
                sklearn.metrics.accuracy_score(predictions, targets))

    def test_default_configuration_sparse_data(self):
        for i in range(10):
            predictions, targets = \
                _test_classifier(KNearestNeighborsClassifier, sparse=True)
            self.assertAlmostEqual(0.82,
                                   sklearn.metrics.accuracy_score(predictions,
                                                                  targets))

    def test_default_configuration_predict_proba(self):
        for i in range(10):
            predictions, targets = \
                _test_classifier_predict_proba(KNearestNeighborsClassifier)
            self.assertAlmostEqual(1.381551055796429,
                sklearn.metrics.log_loss(targets, predictions))