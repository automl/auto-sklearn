import unittest

from autosklearn.pipeline.components.classification.bernoulli_nb import \
    BernoulliNB
from autosklearn.pipeline.util import _test_classifier, \
    _test_classifier_iterative_fit, _test_classifier_predict_proba

import numpy as np
import sklearn.metrics
import sklearn.naive_bayes


class BernoulliNBComponentTest(unittest.TestCase):
    def test_default_configuration(self):
        for i in range(2):
            predictions, targets = \
                _test_classifier(BernoulliNB)
            self.assertAlmostEqual(0.26000000000000001,
                                   sklearn.metrics.accuracy_score(predictions,
                                                                  targets))

    def test_default_configuration_iterative_fit(self):
        for i in range(2):
            predictions, targets = \
                _test_classifier_iterative_fit(BernoulliNB)
            self.assertAlmostEqual(0.26000000000000001,
                                   sklearn.metrics.accuracy_score(predictions,
                                                                  targets))

    def test_default_configuration_binary(self):
        for i in range(2):
            predictions, targets = \
                _test_classifier(BernoulliNB, make_binary=True)
            self.assertAlmostEqual(0.73999999999999999,
                                   sklearn.metrics.accuracy_score(
                                       predictions, targets))

    def test_default_configuration_multilabel(self):
        for i in range(2):
            predictions, targets = \
                _test_classifier(classifier=BernoulliNB,
                                 dataset='digits',
                                 make_multilabel=True)
            self.assertAlmostEqual(0.73112394623587451,
                                   sklearn.metrics.average_precision_score(
                                       targets, predictions))

    def test_default_configuration_multilabel_predict_proba(self):
        for i in range(2):
            predictions, targets = \
                _test_classifier_predict_proba(classifier=BernoulliNB,
                                               make_multilabel=True)
            self.assertEqual(predictions.shape, ((50, 3)))
            self.assertAlmostEqual(0.66666666666666663,
                                   sklearn.metrics.average_precision_score(
                                       targets, predictions))

    def test_target_algorithm_multioutput_multiclass_support(self):
        cls = sklearn.naive_bayes.BernoulliNB()
        X = np.random.random((10, 10))
        y = np.random.randint(0, 1, size=(10, 10))
        self.assertRaisesRegexp(ValueError, 'bad input shape \(10, 10\)',
                                cls.fit, X, y)