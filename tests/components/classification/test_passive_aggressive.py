import unittest

from ParamSklearn.components.classification.passive_aggresive import PassiveAggressive
from ParamSklearn.util import _test_classifier

import sklearn.metrics


class SGDComponentTest(unittest.TestCase):
    def test_default_configuration(self):
        for i in range(10):
            predictions, targets = _test_classifier(PassiveAggressive, dataset='iris')
            self.assertAlmostEqual(0.92,
                                   sklearn.metrics.accuracy_score(predictions,
                                                                  targets))

    def test_default_configuration_digits(self):
        for i in range(10):
            predictions, targets = \
                _test_classifier(classifier=PassiveAggressive, dataset='digits')
            self.assertAlmostEqual(0.91317547055251969,
                                   sklearn.metrics.accuracy_score(predictions,
                                                                  targets))