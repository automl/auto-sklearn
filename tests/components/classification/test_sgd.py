import unittest

from ParamSklearn.components.classification.sgd import SGD
from ParamSklearn.util import _test_classifier

import sklearn.metrics


class RandomForestComponentTest(unittest.TestCase):
    def test_default_configuration(self):
        for i in range(10):
            predictions, targets = _test_classifier(SGD, dataset='iris')
            self.assertAlmostEqual(0.96,
                                   sklearn.metrics.accuracy_score(predictions,
                                                                  targets))