import unittest

from autosklearn.pipeline.components.classification.xgradient_boosting import \
    XGradientBoostingClassifier
from autosklearn.pipeline.util import _test_classifier,\
    _test_classifier_predict_proba

import sklearn.metrics
import sklearn.ensemble
import numpy as np


class XGradientBoostingComponentTest(unittest.TestCase):
    def test_default_configuration(self):
        for i in range(10):
            predictions, targets = \
                _test_classifier(XGradientBoostingClassifier)
            self.assertAlmostEqual(0.92,
                sklearn.metrics.accuracy_score(predictions, targets))

    def test_default_configuration_binary(self):
        for i in range(10):
            predictions, targets = _test_classifier(
                XGradientBoostingClassifier, make_binary=True)
            self.assertAlmostEqual(1.0,
                                   sklearn.metrics.accuracy_score(predictions,
                                                                  targets))

    def test_target_algorithm_multioutput_multiclass_support(self):
        cls = sklearn.ensemble.GradientBoostingClassifier()
        X = np.random.random((10, 10))
        y = np.random.randint(0, 1, size=(10, 10))
        self.assertRaisesRegexp(ValueError, 'bad input shape \(10, 10\)',
                                cls.fit, X, y)