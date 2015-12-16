import unittest

from ParamSklearn.components.regression.adaboost import \
    AdaboostRegressor
from ParamSklearn.util import _test_regressor

import sklearn.metrics


class AdaBoostComponentTest(unittest.TestCase):
    def test_default_configuration(self):
        for i in range(10):
            predictions, targets = \
                _test_regressor(AdaboostRegressor, dataset='boston')
            self.assertAlmostEqual(0.11053868761882502,
                                   sklearn.metrics.r2_score(targets,
                                                            predictions))

    def test_default_configuration_sparse(self):
        for i in range(10):
            predictions, targets = \
                _test_regressor(AdaboostRegressor, sparse=True, dataset='boston')
            self.assertAlmostEqual(-0.077540100211211049,
                                   sklearn.metrics.r2_score(targets,
                                                            predictions))
