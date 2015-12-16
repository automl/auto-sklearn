import unittest

from ParamSklearn.components.classification.passive_aggressive import \
    PassiveAggressive
from ParamSklearn.util import _test_classifier, _test_classifier_iterative_fit

import sklearn.metrics


class PassiveAggressiveComponentTest(unittest.TestCase):
    def test_default_configuration(self):
        for i in range(10):
            predictions, targets = _test_classifier(PassiveAggressive)
            self.assertAlmostEqual(0.97999999999999998,
                                   sklearn.metrics.accuracy_score(predictions,
                                                                  targets))

    def test_default_configuration_iterative_fit(self):
        for i in range(10):
            predictions, targets = _test_classifier_iterative_fit(
                PassiveAggressive)
            self.assertAlmostEqual(0.97999999999999998,
                                   sklearn.metrics.accuracy_score(
                                       predictions, targets))

    def test_default_configuration_digits(self):
        for i in range(10):
            predictions, targets = \
                _test_classifier(classifier=PassiveAggressive, dataset='digits')
            self.assertAlmostEqual(0.91924711596842745,
                                   sklearn.metrics.accuracy_score(predictions,
                                                                  targets))

    def test_default_configuration_digits_iterative_fit(self):
        for i in range(10):
            predictions, targets = _test_classifier_iterative_fit(classifier=PassiveAggressive,
                                                    dataset='digits')
            self.assertAlmostEqual(0.91924711596842745,
                                   sklearn.metrics.accuracy_score(
                                       predictions, targets))