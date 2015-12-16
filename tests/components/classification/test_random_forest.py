import unittest

from ParamSklearn.components.classification.random_forest import RandomForest
from ParamSklearn.util import _test_classifier, _test_classifier_iterative_fit

import sklearn.metrics


class RandomForestComponentTest(unittest.TestCase):
    def test_default_configuration(self):
        for i in range(10):
            predictions, targets = _test_classifier(RandomForest)
            self.assertAlmostEqual(0.95999999999999996,
                sklearn.metrics.accuracy_score(predictions, targets))

    def test_default_configuration_sparse(self):
        for i in range(10):
            predictions, targets = _test_classifier(RandomForest, sparse=True)
            self.assertAlmostEqual(0.85999999999999999,
                                   sklearn.metrics.accuracy_score(predictions,
                                                                  targets))

    def test_default_configuration_iterative_fit(self):
        for i in range(10):
            predictions, targets = \
                _test_classifier_iterative_fit(RandomForest)
            self.assertAlmostEqual(0.95999999999999996,
                                   sklearn.metrics.accuracy_score(
                                       predictions, targets))