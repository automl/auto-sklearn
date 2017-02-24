import unittest

from autosklearn.pipeline.components.regression.xgradient_boosting import \
    XGradientBoostingRegressor
from autosklearn.pipeline.util import _test_regressor, \
    _test_regressor_iterative_fit


import sklearn.metrics
import sklearn.ensemble


class XGradientBoostingComponentTest(unittest.TestCase):
    def test_default_configuration(self):
        for i in range(2):
            predictions, targets = _test_regressor(XGradientBoostingRegressor)
            self.assertAlmostEqual(0.34009199992306871,
                sklearn.metrics.r2_score(y_true=targets, y_pred=predictions))

    def test_default_configuration_sparse(self):
        for i in range(2):
            predictions, targets = _test_regressor(XGradientBoostingRegressor,
                                                   sparse=True)
            self.assertAlmostEqual(0.20743694821393754,
                sklearn.metrics.r2_score(y_true=targets, y_pred=predictions))

    #def test_default_configuration_iterative_fit(self):
    #    for i in range(10):
    #        predictions, targets = \
    #            _test_regressor_iterative_fit(XGradientBoostingRegressor)
    #        self.assertAlmostEqual(0.40965687834764064,
    #            sklearn.metrics.r2_score(y_true=targets, y_pred=predictions))

    #def test_default_configuration_iterative_fit_sparse(self):
    #    for i in range(10):
    #        predictions, targets = \
    #            _test_regressor_iterative_fit(XGradientBoostingRegressor,
    #                                          sparse=True)
    #        self.assertAlmostEqual(0.40965687834764064,
    #            sklearn.metrics.r2_score(y_true=targets, y_pred=predictions))