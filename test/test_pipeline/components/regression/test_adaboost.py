import unittest

from autosklearn.pipeline.components.regression.adaboost import \
    AdaboostRegressor
from autosklearn.pipeline.util import _test_regressor

import sklearn.metrics


class AdaBoostComponentTest(unittest.TestCase):
    def test_default_configuration(self):
        for i in range(10):
            predictions, targets = \
                _test_regressor(AdaboostRegressor, dataset='boston')
            self.assertAlmostEqual(0.53163164558327014,
                                   sklearn.metrics.r2_score(targets,
                                                            predictions))

    def test_default_configuration_sparse(self):
        for i in range(10):
            predictions, targets = \
                _test_regressor(AdaboostRegressor, sparse=True, dataset='boston')
            self.assertAlmostEqual(0.22163148231037866,
                                   sklearn.metrics.r2_score(targets,
                                                            predictions))
