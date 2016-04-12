import unittest

from autosklearn.pipeline.components.regression.gaussian_process import GaussianProcess
from autosklearn.pipeline.util import _test_regressor

import sklearn.metrics


class GaussianProcessComponentTest(unittest.TestCase):
    def test_default_configuration(self):
        # Only twice to reduce the number of warning printed to the command line
        for i in range(2):
            # Float32 leads to numeric instabilities
            predictions, targets = _test_regressor(GaussianProcess,
                                                   dataset='boston')
            self.assertAlmostEqual(0.83362335184173442,
                sklearn.metrics.r2_score(y_true=targets, y_pred=predictions),
                places=2)

