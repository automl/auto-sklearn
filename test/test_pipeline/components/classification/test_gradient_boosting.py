import unittest

from autosklearn.pipeline.components.classification.gradient_boosting import \
    GradientBoostingClassifier
from autosklearn.pipeline.util import _test_classifier, \
    _test_classifier_iterative_fit, _test_classifier_predict_proba

import sklearn.metrics
import sklearn.ensemble
import numpy as np


class GradientBoostingComponentTest(unittest.TestCase):
    def test_default_configuration(self):
        for i in range(2):
            predictions, targets = \
                _test_classifier(GradientBoostingClassifier)
            self.assertAlmostEqual(0.93999999999999995,
                sklearn.metrics.accuracy_score(predictions, targets))

    def test_default_configuration_iterative_fit(self):
        for i in range(2):
            predictions, targets = \
                _test_classifier_iterative_fit(GradientBoostingClassifier)
            self.assertAlmostEqual(0.95999999999999996,
                                   sklearn.metrics.accuracy_score(predictions,
                                                                  targets))

    def test_default_configuration_binary(self):
        for i in range(2):
            predictions, targets = _test_classifier(
                GradientBoostingClassifier, make_binary=True)
            self.assertAlmostEqual(1.0,
                                   sklearn.metrics.accuracy_score(predictions,
                                                                  targets))

    def test_target_algorithm_multioutput_multiclass_support(self):
        cls = sklearn.ensemble.GradientBoostingClassifier()
        X = np.random.random((10, 10))
        y = np.random.randint(0, 1, size=(10, 10))
        self.assertRaisesRegexp(ValueError, 'bad input shape \(10, 10\)',
                                cls.fit, X, y)