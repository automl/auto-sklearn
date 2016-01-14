import unittest

from autosklearn.pipeline.components.classification.decision_tree import DecisionTree
from autosklearn.pipeline.util import _test_classifier, _test_classifier_predict_proba

import numpy as np
import sklearn.metrics
import sklearn.tree


class DecisionTreetComponentTest(unittest.TestCase):
    def test_default_configuration(self):
        for i in range(10):
            predictions, targets = _test_classifier(DecisionTree)
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
                DecisionTree)
            self.assertAlmostEqual(0.28069887755912964,
                sklearn.metrics.log_loss(targets, predictions))

    def test_default_configuration_binary(self):
        for i in range(10):
            predictions, targets = _test_classifier(
                DecisionTree, make_binary=True)
            self.assertAlmostEqual(1.0,
                                   sklearn.metrics.accuracy_score(
                                       targets, predictions))

    def test_default_configuration_multilabel(self):
        for i in range(10):
            predictions, targets = _test_classifier(
                DecisionTree, make_multilabel=True)
            print(predictions, targets)
            self.assertAlmostEqual(0.94120857699805072,
                                   sklearn.metrics.average_precision_score(
                                       targets, predictions))

    def test_default_configuration_multilabel_predict_proba(self):
        for i in range(10):
            predictions, targets = _test_classifier_predict_proba(
                DecisionTree, make_multilabel=True)
            self.assertEqual(predictions.shape, ((50, 3)))
            self.assertAlmostEqual(0.94589326168273546,
                                   sklearn.metrics.average_precision_score(
                                       targets, predictions))

    def test_target_algorithm_multioutput_multiclass_support(self):
        cls = sklearn.tree.DecisionTreeClassifier()
        X = np.random.random((10, 10))
        y = np.random.randint(0, 1, size=(10, 10))
        # Running this without an exception is the purpose of this test!
        cls.fit(X, y)