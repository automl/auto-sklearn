import unittest

from ParamSklearn.components.classification.decision_tree import DecisionTree
from ParamSklearn.util import _test_classifier, _test_classifier_predict_proba

import sklearn.metrics


class DecisionTreetComponentTest(unittest.TestCase):
    def test_default_configuration(self):
        for i in range(10):
            predictions, targets = _test_classifier(DecisionTree,
                                                    dataset='iris')
            self.assertAlmostEqual(0.92,
                                   sklearn.metrics.accuracy_score(predictions,
                                                                  targets))

    def test_default_configuration_sparse(self):
        for i in range(10):
            predictions, targets = _test_classifier(DecisionTree, sparse=True)
            self.assertAlmostEqual(0.69999999999999996,
                                   sklearn.metrics.accuracy_score(predictions,
                                                              targets))

    def test_default_configuration_predict_proba(self):
        for i in range(10):
            predictions, targets = _test_classifier_predict_proba(
                DecisionTree, dataset='iris')
            self.assertAlmostEqual(0.28069887755912964,
                sklearn.metrics.log_loss(targets, predictions))