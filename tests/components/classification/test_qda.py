import unittest

from ParamSklearn.components.classification.qda import QDA
from ParamSklearn.util import _test_classifier

import sklearn.metrics


class QDAComponentTest(unittest.TestCase):
    def test_default_configuration_iris(self):
        for i in range(10):
            predictions, targets = \
                _test_classifier(QDA)
            self.assertAlmostEqual(1.0,
                                   sklearn.metrics.accuracy_score(predictions,
                                                                  targets))

    #@unittest.skip("QDA fails on this one")
    def test_default_configuration_digits(self):
        for i in range(10):
            predictions, targets = \
                _test_classifier(classifier=QDA, dataset='digits')
            self.assertAlmostEqual(0.18882817243472982,
                                   sklearn.metrics.accuracy_score(predictions,
                                                                  targets))
