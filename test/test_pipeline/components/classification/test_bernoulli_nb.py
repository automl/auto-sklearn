import unittest

from autosklearn.pipeline.components.classification.bernoulli_nb import \
    BernoulliNB
from autosklearn.pipeline.util import _test_classifier, _test_classifier_iterative_fit

import numpy as np
import sklearn.metrics
import sklearn.naive_bayes


class BernoulliNBComponentTest(unittest.TestCase):
    def test_default_configuration(self):
        for i in range(10):
            predictions, targets = \
                _test_classifier(BernoulliNB)
            self.assertAlmostEqual(0.26000000000000001,
                                   sklearn.metrics.accuracy_score(predictions,
                                                                  targets))

    def test_default_configuration_iterative_fit(self):
        for i in range(10):
            predictions, targets = \
                _test_classifier_iterative_fit(BernoulliNB)
            self.assertAlmostEqual(0.26000000000000001,
                                   sklearn.metrics.accuracy_score(predictions,
                                                                  targets))

    def test_default_configuration_binary(self):
        for i in range(10):
            predictions, targets = \
                _test_classifier(BernoulliNB, make_binary=True)
            self.assertAlmostEqual(0.73999999999999999,
                                   sklearn.metrics.accuracy_score(
                                       predictions, targets))

    def test_target_algorithm_multioutput_multiclass_support(self):
        cls = sklearn.naive_bayes.BernoulliNB()
        X = np.random.random((10, 10))
        y = np.random.randint(0, 1, size=(10, 10))
        self.assertRaisesRegexp(ValueError, 'bad input shape \(10, 10\)',
                                cls.fit, X, y)