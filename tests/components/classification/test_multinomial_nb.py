import unittest

from ParamSklearn.components.classification.multinomial_nb import \
    MultinomialNB
from ParamSklearn.util import _test_classifier, _test_classifier_iterative_fit

import sklearn.metrics


class MultinomialNBComponentTest(unittest.TestCase):
    def test_default_configuration(self):
        for i in range(10):
            predictions, targets = \
                _test_classifier(MultinomialNB)
            self.assertAlmostEqual(0.97999999999999998,
                                   sklearn.metrics.accuracy_score(predictions,
                                                                  targets))

    def test_default_configuration_iterative_fit(self):
        for i in range(10):
            predictions, targets = \
                _test_classifier_iterative_fit(MultinomialNB)
            self.assertAlmostEqual(0.97999999999999998,
                                   sklearn.metrics.accuracy_score(predictions,
                                                                  targets))