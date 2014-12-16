import unittest

from AutoSklearn.components.classification.libsvm_svc import LibSVM_SVC
from AutoSklearn.util import _test_classifier

import sklearn.metrics


class LibSVM_SVCComponentTest(unittest.TestCase):
    def test_default_configuration(self):
        for i in range(10):
            predictions, targets = _test_classifier(LibSVM_SVC, dataset='iris')
            self.assertAlmostEqual(0.96,
                sklearn.metrics.accuracy_score(predictions, targets))
