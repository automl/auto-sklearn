import unittest

from autosklearn.pipeline.components.classification.xgradient_boosting import \
    XGradientBoostingClassifier
from autosklearn.pipeline.util import _test_classifier, \
    _test_classifier_iterative_fit

import sklearn.metrics
import sklearn.ensemble


class XGradientBoostingComponentTest(unittest.TestCase):
    def test_default_configuration(self):
        for i in range(2):
            predictions, targets = \
                _test_classifier(XGradientBoostingClassifier)
            self.assertAlmostEqual(0.92,
                sklearn.metrics.accuracy_score(predictions, targets))

    def test_default_configuration_sparse(self):
        for i in range(2):
            predictions, targets = _test_classifier(XGradientBoostingClassifier,
                                                    sparse=True)
            self.assertAlmostEqual(0.88,
                                   sklearn.metrics.accuracy_score(predictions,
                                                                  targets))

    def test_default_configuration_binary(self):
        for i in range(2):
            predictions, targets = _test_classifier(
                XGradientBoostingClassifier, make_binary=True)
            self.assertAlmostEqual(1.0,
                                   sklearn.metrics.accuracy_score(predictions,
                                                                  targets))

    def test_default_configuration_binary_sparse(self):
        for i in range(2):
            predictions, targets = _test_classifier(
                XGradientBoostingClassifier, make_binary=True, sparse=True)
            self.assertAlmostEqual(0.95999999999999996,
                                   sklearn.metrics.accuracy_score(predictions,
                                                                  targets))