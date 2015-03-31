import unittest

from ParamSklearn.components.regression.gradient_boosting import GradientBoosting
from ParamSklearn.util import _test_regressor

import sklearn.metrics


class GradientBoostingComponentTest(unittest.TestCase):
    def test_default_configuration(self):
        for i in range(10):

            predictions, targets = _test_regressor(GradientBoosting,
                                                   dataset='diabetes')
            self.assertAlmostEqual(0.38851325425603489,
                sklearn.metrics.r2_score(y_true=targets, y_pred=predictions))
