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
            # My machine: 0.574913739659292
            # travis-ci: 0.49562471963524557
            self.assertLessEqual(
                sklearn.metrics.r2_score(y_true=targets, y_pred=predictions),
                0.6)
            self.assertGreaterEqual(
                sklearn.metrics.r2_score(y_true=targets, y_pred=predictions),
                0.4)

