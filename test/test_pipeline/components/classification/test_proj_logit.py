import unittest

from autosklearn.pipeline.components.classification.proj_logit import ProjLogitCLassifier
from autosklearn.pipeline.util import _test_classifier

import numpy as np
import sklearn.metrics
import autosklearn.pipeline.implementations.ProjLogit


class ProjLogitComponentTest(unittest.TestCase):
    def test_default_configuration(self):
        for i in range(10):
            predictions, targets = _test_classifier(ProjLogitCLassifier, dataset='iris')
            self.assertAlmostEqual(0.98,
                sklearn.metrics.accuracy_score(predictions, targets))

    def test_default_configuration_digits(self):
        for i in range(10):
            predictions, targets = _test_classifier(ProjLogitCLassifier,
                                                    dataset='digits')
            self.assertAlmostEqual(0.8986035215543412,
                sklearn.metrics.accuracy_score(predictions, targets))

    def test_default_configuration_binary(self):
        for i in range(10):
            predictions, targets = _test_classifier(ProjLogitCLassifier,
                                                    make_binary=True)
            self.assertAlmostEqual(1.0,
                                   sklearn.metrics.accuracy_score(predictions,
                                                                  targets))

    @unittest.skip('Cannot be tested ATM. Wait for Tobias')
    def test_target_algorithm_multioutput_multiclass_support(self):
        cls = autosklearn.pipeline.implementations.ProjLogit.ProjLogit()

        X = np.random.random((10, 10))
        y = np.random.randint(0, 1, size=(10, 10))
        self.assertRaisesRegex(ValueError, 'bad input shape \(10, 10\)',
                               cls.fit, X, y)