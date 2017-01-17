import unittest

from autosklearn.pipeline.components.classification.k_nearest_neighbors import \
    KNearestNeighborsClassifier
from autosklearn.pipeline.util import _test_classifier, _test_classifier_predict_proba

import numpy as np
import sklearn.metrics
import sklearn.neighbors


class KNearestNeighborsComponentTest(unittest.TestCase):
    def test_default_configuration(self):
        for i in range(2):
            predictions, targets = \
                _test_classifier(KNearestNeighborsClassifier)
            self.assertAlmostEqual(0.959999999999999,
                sklearn.metrics.accuracy_score(predictions, targets))

    def test_default_configuration_sparse_data(self):
        for i in range(2):
            predictions, targets = \
                _test_classifier(KNearestNeighborsClassifier, sparse=True)
            self.assertAlmostEqual(0.82,
                                   sklearn.metrics.accuracy_score(predictions,
                                                                  targets))

    def test_default_configuration_predict_proba(self):
        for i in range(2):
            predictions, targets = \
                _test_classifier_predict_proba(KNearestNeighborsClassifier)
            self.assertAlmostEqual(1.381551055796429,
                sklearn.metrics.log_loss(targets, predictions))

    def test_default_configuration_binary(self):
        for i in range(2):
            predictions, targets = \
                _test_classifier(KNearestNeighborsClassifier, make_binary=True)
            self.assertAlmostEqual(1.0,
                                   sklearn.metrics.accuracy_score(predictions,
                                                                  targets))

    def test_default_configuration_multilabel(self):
        for i in range(2):
            predictions, targets = \
                _test_classifier(KNearestNeighborsClassifier,
                                 make_multilabel=True)
            self.assertAlmostEqual(0.959999999999999,
                                   sklearn.metrics.accuracy_score(predictions,
                                                                  targets))

    def test_default_configuration_predict_proba_multilabel(self):
        for i in range(2):
            predictions, targets = \
                _test_classifier_predict_proba(KNearestNeighborsClassifier,
                                               make_multilabel=True)
            self.assertEqual(predictions.shape, ((50, 3)))
            self.assertAlmostEqual(0.97060428849902536,
                                   sklearn.metrics.average_precision_score(
                                       targets, predictions))

    def test_target_algorithm_multioutput_multiclass_support(self):
        cls = sklearn.neighbors.KNeighborsClassifier()
        X = np.random.random((10, 10))
        y = np.random.randint(0, 1, size=(10, 10))
        # Running this without an exception is the purpose of this test!
        cls.fit(X, y)