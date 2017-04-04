import unittest

from autosklearn.pipeline.components.regression.RegDeepNet import \
    RegDeepNet
from autosklearn.pipeline.util import _test_regressor, \
    _test_regressor_iterative_fit

import sklearn.metrics


class RegDeepNetComponentTest(unittest.TestCase):
    def test_default_configuration(self):
        for i in range(2):
            predictions, targets = _test_regressor(RegDeepNet)
            self.assertAlmostEqual(0.39563713703628156,
                                   sklearn.metrics.r2_score(y_true=targets,
                                                            y_pred=predictions),
                                   places=2)
