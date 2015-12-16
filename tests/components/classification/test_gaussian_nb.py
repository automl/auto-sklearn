import unittest

from ParamSklearn.components.classification.gaussian_nb import \
    GaussianNB
from ParamSklearn.util import _test_classifier, _test_classifier_iterative_fit

import sklearn.metrics


class GaussianNBComponentTest(unittest.TestCase):
    def test_default_configuration(self):
        for i in range(10):
            predictions, targets = \
                _test_classifier(GaussianNB)
            self.assertAlmostEqual(0.95999999999999996,
                                   sklearn.metrics.accuracy_score(predictions,
                                                                  targets))

    def test_default_configuration_iterative_fit(self):
        for i in range(10):
            predictions, targets = \
                _test_classifier_iterative_fit(GaussianNB)
            self.assertAlmostEqual(0.95999999999999996,
                                   sklearn.metrics.accuracy_score(predictions,
                                                                  targets))