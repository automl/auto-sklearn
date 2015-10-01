import unittest

from ParamSklearn.components.regression.sgd import SGD
from ParamSklearn.util import _test_regressor, _test_regressor_iterative_fit

import sklearn.metrics


class SGDComponentTest(unittest.TestCase):
    def test_default_configuration(self):
        for i in range(10):
            predictions, targets = _test_regressor(SGD)
            self.assertAlmostEqual(0.092460881802630235,
                                   sklearn.metrics.r2_score(y_true=targets,
                                                            y_pred=predictions))

    def test_default_configuration_iterative_fit(self):
        for i in range(10):
            predictions, targets = _test_regressor_iterative_fit(SGD)
            self.assertAlmostEqual(0.092460881802630235,
                                   sklearn.metrics.r2_score(y_true=targets,
                                                            y_pred=predictions))

    def test_default_configuration_digits(self):
        for i in range(10):
            predictions, targets = _test_regressor(SGD, dataset='boston')
            self.assertAlmostEqual(-2.9165866511775519e+31,
                                   sklearn.metrics.r2_score(y_true=targets,
                                                            y_pred=predictions))

    def test_default_configuration_digits_iterative_fit(self):
        for i in range(10):
            predictions, targets = _test_regressor_iterative_fit(SGD,
                                                                 dataset='boston')
            self.assertAlmostEqual(-2.9165866511775519e+31,
                                   sklearn.metrics.r2_score(y_true=targets,
                                                            y_pred=predictions))