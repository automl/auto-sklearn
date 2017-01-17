import unittest

from autosklearn.pipeline.components.classification.qda import QDA
from autosklearn.pipeline.util import _test_classifier, _test_classifier_predict_proba

import numpy as np
import sklearn.metrics
import sklearn.discriminant_analysis


class QDAComponentTest(unittest.TestCase):
    def test_default_configuration_iris(self):
        for i in range(2):
            predictions, targets = \
                _test_classifier(QDA)
            self.assertAlmostEqual(1.0,
                                   sklearn.metrics.accuracy_score(predictions,
                                                                  targets))

    #@unittest.skip("QDA fails on this one")
    def test_default_configuration_digits(self):
        for i in range(2):
            predictions, targets = \
                _test_classifier(classifier=QDA, dataset='digits')
            self.assertAlmostEqual(0.18882817243472982,
                                   sklearn.metrics.accuracy_score(predictions,
                                                                  targets))

    def test_default_configuration_binary(self):
        for i in range(2):
            predictions, targets = \
                _test_classifier(QDA, make_binary=True)
            self.assertAlmostEqual(1.0,
                                   sklearn.metrics.accuracy_score(predictions,
                                                                  targets))

    def test_default_configuration_multilabel(self):
        for i in range(2):
            predictions, targets = \
                _test_classifier(QDA, make_multilabel=True)
            self.assertAlmostEqual(0.99456140350877187,
                                   sklearn.metrics.average_precision_score(
                                       predictions, targets))

    def test_default_configuration_predict_proba_multilabel(self):
        for i in range(2):
            predictions, targets = \
                _test_classifier_predict_proba(QDA,
                                               make_multilabel=True)
            self.assertEqual(predictions.shape, ((50, 3)))
            self.assertAlmostEqual(1.0,
                                   sklearn.metrics.average_precision_score(
                                       targets, predictions))

    def test_target_algorithm_multioutput_multiclass_support(self):
        cls = sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis()
        X = np.random.random((10, 10))
        y = np.random.randint(0, 1, size=(10, 10))
        self.assertRaisesRegexp(ValueError, 'bad input shape \(10, 10\)',
                                cls.fit, X, y)
