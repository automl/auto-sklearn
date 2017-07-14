import unittest

from autosklearn.pipeline.components.regression.RegDeepNetIterative import \
    RegDeepNetIterative
from autosklearn.pipeline.util import _test_regressor, \
    _test_regressor_iterative_fit

import sklearn.metrics


class RegDeepNetIterativeComponentTest(unittest.TestCase):
    def test_default_configuration(self):
        for i in range(2):
            predictions, targets = _test_regressor(RegDeepNetIterative)
            self.assertAlmostEqual(0.41222280019741031,
                                   sklearn.metrics.r2_score(y_true=targets,
                                                            y_pred=predictions),
                                   places=2)

        for i in range(2):
            predictions, targets = _test_regressor_iterative_fit(RegDeepNetIterative)
            self.assertAlmostEqual(0.41222280019741031,
                                   sklearn.metrics.r2_score(y_true=targets,
                                                            y_pred=predictions),
                                   places=2)
