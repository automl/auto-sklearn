import unittest

from ParamSklearn.components.regression.k_nearest_neighbors import \
    KNearestNeighborsRegressor
from ParamSklearn.util import _test_regressor

import sklearn.metrics


class KNearestNeighborsComponentTest(unittest.TestCase):
    def test_default_configuration(self):
        for i in range(10):
            predictions, targets = \
                _test_regressor(KNearestNeighborsRegressor)
            self.assertAlmostEqual(0.068600456340847438,
                                   sklearn.metrics.r2_score(targets,
                                                            predictions))

    def test_default_configuration_sparse_data(self):
        for i in range(10):
            predictions, targets = \
                _test_regressor(KNearestNeighborsRegressor, sparse=True)
            self.assertAlmostEqual(-0.16321841460809972,
                                   sklearn.metrics.r2_score(targets,
                                                            predictions))
