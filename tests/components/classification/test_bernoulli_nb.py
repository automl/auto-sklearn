import unittest

from ParamSklearn.components.classification.bernoulli_nb import \
    BernoulliNB
from ParamSklearn.util import _test_classifier, _test_classifier_iterative_fit

import sklearn.metrics


class BernoulliNBComponentTest(unittest.TestCase):
    def test_default_configuration(self):
        for i in range(10):
            predictions, targets = \
                _test_classifier(BernoulliNB)
            self.assertAlmostEqual(0.26000000000000001,
                                   sklearn.metrics.accuracy_score(predictions,
                                                                  targets))

    def test_default_configuration_iterative_fit(self):
        for i in range(10):
            predictions, targets = \
                _test_classifier_iterative_fit(BernoulliNB)
            self.assertAlmostEqual(0.26000000000000001,
                                   sklearn.metrics.accuracy_score(predictions,
                                                                  targets))