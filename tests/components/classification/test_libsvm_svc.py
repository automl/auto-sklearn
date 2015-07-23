import unittest

from ParamSklearn.components.classification.libsvm_svc import LibSVM_SVC
from ParamSklearn.util import _test_classifier, _test_classifier_predict_proba

import sklearn.metrics


class LibSVM_SVCComponentTest(unittest.TestCase):
    def test_default_configuration(self):
        for i in range(10):
            predictions, targets = _test_classifier(LibSVM_SVC, dataset='iris')
            self.assertAlmostEqual(0.96,
                sklearn.metrics.accuracy_score(predictions, targets))

    def test_default_configuration_predict_proba(self):
        for i in range(10):
            predictions, targets = _test_classifier_predict_proba(
                LibSVM_SVC, sparse=True)
            self.assertAlmostEqual(1.3028778322629093,
                                   sklearn.metrics.log_loss(targets,
                                                            predictions))
