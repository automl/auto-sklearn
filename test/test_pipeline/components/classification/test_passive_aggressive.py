import unittest

from autosklearn.pipeline.components.classification.passive_aggressive import \
    PassiveAggressive
from autosklearn.pipeline.util import _test_classifier, \
    _test_classifier_iterative_fit, _test_classifier_predict_proba

import numpy as np
import sklearn.metrics
import sklearn.linear_model


class PassiveAggressiveComponentTest(unittest.TestCase):
    def test_default_configuration(self):
        for i in range(2):
            predictions, targets = _test_classifier(PassiveAggressive)
            self.assertAlmostEqual(0.71999999999999997,
                                   sklearn.metrics.accuracy_score(predictions,
                                                                  targets))

    def test_default_configuration_iterative_fit(self):
        for i in range(2):
            predictions, targets = _test_classifier_iterative_fit(
                PassiveAggressive)
            self.assertAlmostEqual(0.81999999999999995,
                                   sklearn.metrics.accuracy_score(
                                       predictions, targets))

    def test_default_configuration_digits(self):
        for i in range(2):
            predictions, targets = \
                _test_classifier(classifier=PassiveAggressive, dataset='digits')
            self.assertAlmostEqual(0.92046144505160898,
                                   sklearn.metrics.accuracy_score(predictions,
                                                                  targets))

    def test_default_configuration_digits_iterative_fit(self):
        for i in range(2):
            predictions, targets = _test_classifier_iterative_fit(classifier=PassiveAggressive,
                                                    dataset='digits')
            self.assertAlmostEqual(0.92349726775956287,
                                   sklearn.metrics.accuracy_score(
                                       predictions, targets))

    def test_default_configuration_binary(self):
        for i in range(2):
            predictions, targets = _test_classifier(PassiveAggressive,
                                                    make_binary=True)
            self.assertAlmostEqual(1.0,
                                   sklearn.metrics.accuracy_score(predictions,
                                                                  targets))

    def test_default_configuration_multilabel(self):
        for i in range(2):
            predictions, targets = \
                _test_classifier(classifier=PassiveAggressive,
                                 dataset='digits',
                                 make_multilabel=True)
            self.assertAlmostEqual(0.8975269956947447,
                                   sklearn.metrics.average_precision_score(
                                       targets, predictions))

    def test_default_configuration_multilabel_predict_proba(self):
        for i in range(2):
            predictions, targets = \
                _test_classifier_predict_proba(classifier=PassiveAggressive,
                                               make_multilabel=True)
            self.assertEqual(predictions.shape, ((50, 3)))
            self.assertAlmostEqual(0.99703892466326138,
                                   sklearn.metrics.average_precision_score(
                                       targets, predictions))

    def test_target_algorithm_multioutput_multiclass_support(self):
        cls = sklearn.linear_model.PassiveAggressiveClassifier()

        X = np.random.random((10, 10))
        y = np.random.randint(0, 1, size=(10, 10))
        self.assertRaisesRegexp(ValueError, 'bad input shape \(10, 10\)',
                                cls.fit, X, y)