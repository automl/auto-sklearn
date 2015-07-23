import unittest

from ParamSklearn.components.classification.adaboost import \
    AdaboostClassifier
from ParamSklearn.util import _test_classifier

import sklearn.metrics


class AdaBoostComponentTest(unittest.TestCase):
    def test_default_configuration_iris(self):
        for i in range(10):
            predictions, targets = \
                _test_classifier(AdaboostClassifier)
            self.assertAlmostEqual(0.93999999999999995,
                                   sklearn.metrics.accuracy_score(predictions, targets))

    def test_default_configuration_iris_sparse(self):
        for i in range(10):
            predictions, targets = \
                _test_classifier(AdaboostClassifier, sparse=True)
            self.assertAlmostEqual(0.88,
                                   sklearn.metrics.accuracy_score(predictions,
                                                                  targets))

    def test_default_configuration_digits(self):
        for i in range(10):
            predictions, targets = \
                _test_classifier(classifier=AdaboostClassifier,
                                 dataset='digits')
            self.assertAlmostEqual(0.6915604128718883,
                                   sklearn.metrics.accuracy_score(predictions, targets))
