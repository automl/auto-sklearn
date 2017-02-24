import unittest

from autosklearn.pipeline.components.regression.extra_trees import \
    ExtraTreesRegressor
from autosklearn.pipeline.util import _test_regressor, _test_regressor_iterative_fit

import sklearn.metrics


class ExtraTreesComponentTest(unittest.TestCase):
    def test_default_configuration(self):
        for i in range(2):
            predictions, targets = \
                _test_regressor(ExtraTreesRegressor)
            self.assertAlmostEqual(0.43258995365114405,
                                   sklearn.metrics.r2_score(targets,
                                                             predictions))

    def test_default_configuration_sparse(self):
        for i in range(2):
            predictions, targets = \
                _test_regressor(ExtraTreesRegressor, sparse=True)
            self.assertAlmostEqual(0.28016012771570553,
                                   sklearn.metrics.r2_score(targets,
                                                            predictions))

    def test_default_configuration_iterative_fit(self):
        for i in range(2):
            predictions, targets = \
                _test_regressor_iterative_fit(ExtraTreesRegressor)
            self.assertAlmostEqual(0.43258995365114405,
                                   sklearn.metrics.r2_score(targets,
                                                            predictions))