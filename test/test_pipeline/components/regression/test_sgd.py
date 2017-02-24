import unittest

from autosklearn.pipeline.components.regression.sgd import SGD
from autosklearn.pipeline.util import _test_regressor, _test_regressor_iterative_fit

import sklearn.metrics


class SGDComponentTest(unittest.TestCase):
    def test_default_configuration(self):
        for i in range(2):
            predictions, targets = _test_regressor(SGD)
            self.assertAlmostEqual(0.066576586105546731,
                                   sklearn.metrics.r2_score(y_true=targets,
                                                            y_pred=predictions))

    def test_default_configuration_iterative_fit(self):
        for i in range(2):
            predictions, targets = _test_regressor_iterative_fit(SGD)
            self.assertAlmostEqual(0.066576586105546731,
                                   sklearn.metrics.r2_score(y_true=targets,
                                                            y_pred=predictions))