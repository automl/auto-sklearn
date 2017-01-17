import unittest

from autosklearn.pipeline.components.classification.extra_trees import \
    ExtraTreesClassifier
from autosklearn.pipeline.util import _test_classifier, \
    _test_classifier_iterative_fit, _test_classifier_predict_proba

import numpy as np
import sklearn.metrics
import sklearn.ensemble


class ExtraTreesComponentTest(unittest.TestCase):
    def test_default_configuration(self):
        for i in range(2):
            predictions, targets = \
                _test_classifier(ExtraTreesClassifier)
            self.assertAlmostEqual(0.95999999999999996,
                sklearn.metrics.accuracy_score(targets, predictions))

    def test_default_configuration_predict_proba(self):
        for i in range(2):
            predictions, targets = \
                _test_classifier_predict_proba(ExtraTreesClassifier)
            self.assertAlmostEqual(0.1086791056721286,
                                   sklearn.metrics.log_loss(
                                       targets, predictions))

    def test_default_configuration_sparse(self):
        for i in range(2):
            predictions, targets = \
                _test_classifier(ExtraTreesClassifier, sparse=True)
            self.assertAlmostEqual(0.71999999999999997,
                                   sklearn.metrics.accuracy_score(targets,
                                                                  predictions))

    def test_default_configuration_iterative_fit(self):
        for i in range(2):
            predictions, targets = \
                _test_classifier_iterative_fit(ExtraTreesClassifier)
            self.assertAlmostEqual(0.93999999999999995,
                                   sklearn.metrics.accuracy_score(targets,
                                                                  predictions))

    def test_default_configuration_binary(self):
        for i in range(2):
            predictions, targets = \
                _test_classifier(ExtraTreesClassifier, make_binary=True)
            self.assertAlmostEqual(1,
                                   sklearn.metrics.accuracy_score(targets,
                                                                  predictions))

    def test_default_configuration_multilabel(self):
        for i in range(2):
            predictions, targets = \
                _test_classifier(ExtraTreesClassifier, make_multilabel=True)
            self.assertAlmostEqual(0.97060428849902536,
                                   sklearn.metrics.average_precision_score(
                                       targets, predictions))

    def test_default_configuration_predict_proba_multilabel(self):
        for i in range(2):
            predictions, targets = \
                _test_classifier_predict_proba(ExtraTreesClassifier,
                                               make_multilabel=True)
            self.assertEqual(predictions.shape, ((50, 3)))
            self.assertAlmostEqual(0.99401797442008899,
                                   sklearn.metrics.average_precision_score(
                                       targets, predictions))

    def test_target_algorithm_multioutput_multiclass_support(self):
        cls = sklearn.ensemble.ExtraTreesClassifier()
        X = np.random.random((10, 10))
        y = np.random.randint(0, 1, size=(10, 10))
        # Running this without an exception is the purpose of this test!
        cls.fit(X, y)