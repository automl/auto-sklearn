import unittest

from ParamSklearn.components.classification.sgd import SGD
from ParamSklearn.util import _test_classifier

import sklearn.metrics


class SGDComponentTest(unittest.TestCase):
    def test_default_configuration(self):
        for i in range(10):
            predictions, targets = _test_classifier(SGD, dataset='iris')
            self.assertAlmostEqual(0.96,
                                   sklearn.metrics.accuracy_score(predictions,
                                                                  targets))

    def test_default_configuration_digits(self):
        for i in range(10):
            predictions, targets = \
                _test_classifier(classifier=SGD, dataset='digits')
            self.assertAlmostEqual(0.89313904068002425,
                                   sklearn.metrics.accuracy_score(predictions,
                                                                  targets))