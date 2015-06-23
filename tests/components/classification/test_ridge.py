import unittest

from ParamSklearn.components.classification.ridge import Ridge
from ParamSklearn.util import _test_classifier

import sklearn.metrics


class SGDComponentTest(unittest.TestCase):
    def test_default_configuration(self):
        for i in range(10):
            predictions, targets = _test_classifier(Ridge, dataset='iris')
            self.assertAlmostEqual(0.88,
                                   sklearn.metrics.accuracy_score(predictions,
                                                                  targets))

    def test_default_configuration_digits(self):
        for i in range(10):
            predictions, targets = \
                _test_classifier(classifier=Ridge, dataset='digits')
            self.assertAlmostEqual(0.87553126897389189,
                                   sklearn.metrics.accuracy_score(predictions,
                                                                  targets))