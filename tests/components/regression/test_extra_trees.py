import unittest

from ParamSklearn.components.regression.extra_trees import \
    ExtraTreesRegressor
from ParamSklearn.util import _test_regressor, _test_regressor_iterative_fit

import sklearn.metrics


class ExtraTreesComponentTest(unittest.TestCase):
    def test_default_configuration(self):
        for i in range(10):
            predictions, targets = \
                _test_regressor(ExtraTreesRegressor)
            self.assertAlmostEqual(0.4269923975466271,
                                   sklearn.metrics.r2_score(targets,
                                                             predictions))

    def test_default_configuration_sparse(self):
        for i in range(10):
            predictions, targets = \
                _test_regressor(ExtraTreesRegressor, sparse=True)
            self.assertAlmostEqual(0.26287621251507987,
                                   sklearn.metrics.r2_score(targets,
                                                            predictions))

    def test_default_configuration_iterative_fit(self):
        for i in range(10):
            predictions, targets = \
                _test_regressor_iterative_fit(ExtraTreesRegressor)
            self.assertAlmostEqual(0.4269923975466271,
                                   sklearn.metrics.r2_score(targets,
                                                            predictions))