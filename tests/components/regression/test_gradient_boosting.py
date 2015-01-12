import unittest

from AutoSklearn.components.regression.gradient_boosting import GradientBoosting
from AutoSklearn.util import _test_regressor

import sklearn.metrics


class GradientBoostingComponentTest(unittest.TestCase):
    def test_default_configuration(self):
        for i in range(10):

            predictions, targets = _test_regressor(GradientBoosting,
                                                   dataset='diabetes')
            self.assertAlmostEqual(0.39056015252360077,
                sklearn.metrics.r2_score(y_true=targets, y_pred=predictions))
