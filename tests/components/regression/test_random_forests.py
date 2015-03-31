import unittest

from ParamSklearn.components.regression.random_forest import RandomForest
from ParamSklearn.util import _test_regressor

import sklearn.metrics


class RandomForestComponentTest(unittest.TestCase):
    def test_default_configuration(self):
        for i in range(10):

            predictions, targets = _test_regressor(RandomForest,
                                                   dataset='diabetes')
            self.assertAlmostEqual(0.41224692924630502,
                sklearn.metrics.r2_score(y_true=targets, y_pred=predictions))
