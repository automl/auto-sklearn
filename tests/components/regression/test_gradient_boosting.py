import unittest

from ParamSklearn.components.regression.gradient_boosting import GradientBoosting
from ParamSklearn.util import _test_regressor, _test_regressor_iterative_fit

import sklearn.metrics


class GradientBoostingComponentTest(unittest.TestCase):
    def test_default_configuration(self):
        for i in range(10):

            predictions, targets = _test_regressor(GradientBoosting)
            self.assertAlmostEqual(0.35273007696557712,
                sklearn.metrics.r2_score(y_true=targets, y_pred=predictions))

    def test_default_configuration_iterative_fit(self):
        for i in range(10):
            predictions, targets = _test_regressor(GradientBoosting)
            self.assertAlmostEqual(0.35273007696557712,
                sklearn.metrics.r2_score(y_true=targets, y_pred=predictions))
