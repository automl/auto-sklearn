import unittest

from autosklearn.pipeline.components.classification.random_forest import RandomForest
from autosklearn.pipeline.util import _test_classifier, \
    _test_classifier_iterative_fit, _test_classifier_predict_proba

import numpy as np
import sklearn.ensemble
import sklearn.metrics


class RandomForestComponentTest(unittest.TestCase):
    def test_default_configuration(self):
        for i in range(2):
            predictions, targets = _test_classifier(RandomForest)
            self.assertAlmostEqual(0.95999999999999996,
                sklearn.metrics.accuracy_score(predictions, targets))

    def test_default_configuration_sparse(self):
        for i in range(2):
            predictions, targets = _test_classifier(RandomForest, sparse=True)
            self.assertAlmostEqual(0.85999999999999999,
                                   sklearn.metrics.accuracy_score(predictions,
                                                                  targets))

    def test_default_configuration_iterative_fit(self):
        for i in range(2):
            predictions, targets = \
                _test_classifier_iterative_fit(RandomForest)
            self.assertAlmostEqual(0.95999999999999996,
                                   sklearn.metrics.accuracy_score(
                                       predictions, targets))

    def test_default_configuration_binary(self):
        for i in range(2):
            predictions, targets = _test_classifier(RandomForest,
                                                    make_binary=True)
            self.assertAlmostEqual(1.0,
                                   sklearn.metrics.accuracy_score(
                                       predictions, targets))

    def test_default_configuration_multilabel(self):
        for i in range(2):
            predictions, targets = _test_classifier(RandomForest,
                                                    make_multilabel=True)
            self.assertAlmostEqual(0.95999999999999996,
                                   sklearn.metrics.accuracy_score(
                                       predictions, targets))

    def test_default_configuration_predict_proba_multilabel(self):
        for i in range(2):
            predictions, targets = \
                _test_classifier_predict_proba(RandomForest,
                                               make_multilabel=True)
            self.assertEqual(predictions.shape, ((50, 3)))
            self.assertAlmostEqual(0.99252721833266977,
                                   sklearn.metrics.average_precision_score(
                                       targets, predictions))

    def test_target_algorithm_multioutput_multiclass_support(self):
        cls = sklearn.ensemble.RandomForestClassifier()

        X = np.random.random((10, 10))
        y = np.random.randint(0, 1, size=(10, 10))
        # Running this without an exception is the purpose of this test!
        cls.fit(X, y)