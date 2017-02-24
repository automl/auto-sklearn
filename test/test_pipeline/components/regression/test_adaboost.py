import unittest

from autosklearn.pipeline.components.regression.adaboost import \
    AdaboostRegressor
from autosklearn.pipeline.util import _test_regressor

import sklearn.metrics


class AdaBoostComponentTest(unittest.TestCase):
    def test_default_configuration(self):
        for i in range(2):
            predictions, targets = \
                _test_regressor(AdaboostRegressor, dataset='boston')
            self.assertAlmostEqual(0.59461560848921158,
                                   sklearn.metrics.r2_score(targets,
                                                            predictions))

    def test_default_configuration_sparse(self):
        for i in range(2):
            predictions, targets = \
                _test_regressor(AdaboostRegressor, sparse=True, dataset='boston')
            self.assertAlmostEqual(0.2039634989252479,
                                   sklearn.metrics.r2_score(targets,
                                                            predictions))
