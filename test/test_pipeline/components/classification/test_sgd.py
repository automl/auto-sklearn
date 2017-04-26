import unittest

from autosklearn.pipeline.components.classification.sgd import SGD
from autosklearn.pipeline.util import _test_classifier, \
    _test_classifier_iterative_fit, _test_classifier_predict_proba

import numpy as np
import sklearn.metrics
import sklearn.linear_model


class SGDComponentTest(unittest.TestCase):
    def test_default_configuration(self):
        for i in range(2):
            predictions, targets = _test_classifier(SGD)
            self.assertAlmostEqual(0.64000000000000001,
                                   sklearn.metrics.accuracy_score(predictions,
                                                                  targets))

    def test_default_configuration_iterative_fit(self):
        for i in range(2):
            predictions, targets = _test_classifier_iterative_fit(
                SGD)
            self.assertAlmostEqual(0.76000000000000001,
                                   sklearn.metrics.accuracy_score(
                                       predictions, targets))

    def test_default_configuration_digits(self):
        for i in range(2):
            predictions, targets = \
                _test_classifier(SGD, dataset='digits')
            self.assertAlmostEqual(0.89981785063752273,
                                   sklearn.metrics.accuracy_score(predictions,
                                                                  targets))

    def test_default_configuration_digits_iterative_fit(self):
        for i in range(2):
            predictions, targets = _test_classifier_iterative_fit(
                SGD,
                dataset='digits')
            self.assertAlmostEqual(0.89981785063752273,
                                   sklearn.metrics.accuracy_score(
                                       predictions, targets))

    def test_default_configuration_binary(self):
        for i in range(2):
            predictions, targets = _test_classifier(SGD, make_binary=True)
            self.assertAlmostEqual(1.0,
                                   sklearn.metrics.accuracy_score(predictions,
                                                                  targets))

    def test_target_algorithm_multioutput_multiclass_support(self):
        cls = sklearn.linear_model.SGDClassifier()
        X = np.random.random((10, 10))
        y = np.random.randint(0, 1, size=(10, 10))
        self.assertRaisesRegexp(ValueError, 'bad input shape \(10, 10\)',
                                cls.fit, X, y)
