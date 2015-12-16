import unittest

from ParamSklearn.components.classification.extra_trees import \
    ExtraTreesClassifier
from ParamSklearn.util import _test_classifier, _test_classifier_iterative_fit

import sklearn.metrics


class ExtraTreesComponentTest(unittest.TestCase):
    def test_default_configuration(self):
        for i in range(10):
            predictions, targets = \
                _test_classifier(ExtraTreesClassifier)
            self.assertAlmostEqual(0.95999999999999996,
                sklearn.metrics.accuracy_score(predictions, targets))

    def test_default_configuration_sparse(self):
        for i in range(10):
            predictions, targets = \
                _test_classifier(ExtraTreesClassifier, sparse=True)
            self.assertAlmostEqual(0.71999999999999997,
                                   sklearn.metrics.accuracy_score(predictions,
                                                                  targets))

    def test_default_configuration_iterative_fit(self):
        for i in range(10):
            predictions, targets = \
                _test_classifier_iterative_fit(ExtraTreesClassifier)
            self.assertAlmostEqual(0.95999999999999996,
                                   sklearn.metrics.accuracy_score(predictions,
                                                                  targets))