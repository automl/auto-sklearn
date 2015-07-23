import unittest

from ParamSklearn.components.classification.sgd import SGD
from ParamSklearn.util import _test_classifier, _test_classifier_iterative_fit

import sklearn.metrics


class SGDComponentTest(unittest.TestCase):
    def test_default_configuration(self):
        for i in range(10):
            predictions, targets = _test_classifier(SGD)
            self.assertAlmostEqual(1.0,
                                   sklearn.metrics.accuracy_score(predictions,
                                                                  targets))

    def test_default_configuration_iterative_fit(self):
        for i in range(10):
            predictions, targets = _test_classifier_iterative_fit(
                SGD)
            self.assertAlmostEqual(1.0,
                                   sklearn.metrics.accuracy_score(
                                       predictions, targets))

    def test_default_configuration_digits(self):
        for i in range(10):
            predictions, targets = \
                _test_classifier(SGD, dataset='digits')
            self.assertAlmostEqual(0.89313904068002425,
                                   sklearn.metrics.accuracy_score(predictions,
                                                                  targets))

    def test_default_configuration_digits_iterative_fit(self):
        for i in range(10):
            predictions, targets = _test_classifier_iterative_fit(
                SGD,
                dataset='digits')
            self.assertAlmostEqual(0.89313904068002425,
                                   sklearn.metrics.accuracy_score(
                                       predictions, targets))