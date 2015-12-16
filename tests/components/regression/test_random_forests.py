import unittest

from ParamSklearn.components.regression.random_forest import RandomForest
from ParamSklearn.util import _test_regressor, _test_regressor_iterative_fit

import sklearn.metrics


class RandomForestComponentTest(unittest.TestCase):
    def test_default_configuration(self):
        for i in range(10):

            predictions, targets = _test_regressor(RandomForest)
            self.assertAlmostEqual(0.41224692924630502,
                sklearn.metrics.r2_score(y_true=targets, y_pred=predictions))


    def test_default_configuration_sparse(self):
        for i in range(10):
            predictions, targets = _test_regressor(RandomForest, sparse=True)
            self.assertAlmostEqual(0.24117530425422551,
                sklearn.metrics.r2_score(y_true=targets, y_pred=predictions))

    def test_default_configuration_iterative_fit(self):
        for i in range(10):
            predictions, targets = \
                _test_regressor_iterative_fit(RandomForest)
            self.assertAlmostEqual(0.41224692924630502,
                sklearn.metrics.r2_score(y_true=targets, y_pred=predictions))
