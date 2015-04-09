import unittest

from ParamSklearn.components.classification.proj_logit import ProjLogitCLassifier
from ParamSklearn.util import _test_classifier

import sklearn.metrics


class ProjLogitComponentTest(unittest.TestCase):
    def test_default_configuration(self):
        for i in range(10):
            predictions, targets = _test_classifier(ProjLogitCLassifier, dataset='iris')
            self.assertAlmostEqual(0.98,
                sklearn.metrics.accuracy_score(predictions, targets))
