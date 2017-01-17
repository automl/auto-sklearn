import unittest

from autosklearn.pipeline.components.regression.ard_regression import \
    ARDRegression
from autosklearn.pipeline.util import _test_regressor

import sklearn.metrics


class ARDRegressionComponentTest(unittest.TestCase):
    def test_default_configuration(self):
        for i in range(2):
            predictions, targets = \
                _test_regressor(ARDRegression, dataset='boston')
            self.assertAlmostEqual(0.70316694175513961,
                                   sklearn.metrics.r2_score(targets,
                                                            predictions))
