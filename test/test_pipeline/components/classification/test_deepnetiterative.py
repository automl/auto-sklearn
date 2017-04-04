import unittest

from autosklearn.pipeline.components.classification.DeepNetIterative import \
    DeepNetIterative
from autosklearn.pipeline.util import _test_classifier, \
    _test_classifier_iterative_fit, _test_classifier_predict_proba

import sklearn.metrics
import sklearn.ensemble


class DeepNetIterativeComponentTest(unittest.TestCase):
    def test_default_configuration(self):
        for i in range(2):
            predictions, targets = \
                _test_classifier(DeepNetIterative)
            self.assertAlmostEqual(0.57999999999999996,
                                   sklearn.metrics.accuracy_score(targets,
                                                                  predictions))

    def test_default_configuration_predict_proba(self):
        for i in range(2):
            predictions, targets = \
                _test_classifier_predict_proba(DeepNetIterative)
            self.assertAlmostEqual(0.76018262995220975,
                                   sklearn.metrics.log_loss(
                                       targets, predictions))

    def test_default_configuration_sparse(self):
        for i in range(2):
            predictions, targets = \
                _test_classifier(DeepNetIterative, sparse=True)
            self.assertAlmostEqual(0.38,
                                   sklearn.metrics.accuracy_score(targets,
                                                                  predictions))

    def test_default_configuration_iterative_fit(self):
        for i in range(2):
            predictions, targets = \
                _test_classifier_iterative_fit(DeepNetIterative)
            self.assertAlmostEqual(0.62,
                                   sklearn.metrics.accuracy_score(targets,
                                                                  predictions))

    def test_default_configuration_binary(self):
        for i in range(2):
            predictions, targets = \
                _test_classifier(DeepNetIterative, make_binary=True)
            self.assertAlmostEqual(0.9599999999999,
                                   sklearn.metrics.accuracy_score(targets,
                                                                  predictions))

    def test_default_configuration_multilabel(self):
        for i in range(2):
            predictions, targets = \
                _test_classifier(DeepNetIterative, make_multilabel=True)
            self.assertAlmostEqual(0.71361111111111108,
                                   sklearn.metrics.average_precision_score(
                                       targets, predictions))

    def test_default_configuration_predict_proba_multilabel(self):
        for i in range(2):
            predictions, targets = \
                _test_classifier_predict_proba(DeepNetIterative,
                                               make_multilabel=True)
            self.assertEqual(predictions.shape, ((50, 3)))
            self.assertAlmostEqual(0.76835649552496521,
                                   sklearn.metrics.average_precision_score(
                                       targets, predictions))
