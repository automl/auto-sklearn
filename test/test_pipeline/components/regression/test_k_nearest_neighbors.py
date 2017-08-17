import sklearn.neighbors

from autosklearn.pipeline.components.regression.k_nearest_neighbors import \
    KNearestNeighborsRegressor
from .test_base import BaseRegressionComponentTest


class KNearestNeighborsComponentTest(BaseRegressionComponentTest):

    __test__ = True

    res = dict()
    res["default_boston"] = 0.18393287980040374
    res["default_boston_iterative"] = None
    res["default_boston_sparse"] = -0.23029229186279609
    res["default_boston_iterative_sparse"] = None
    res["default_diabetes"] = 0.068600456340847438
    res["default_diabetes_iterative"] = None
    res["default_diabetes_sparse"] = -0.16321841460809972
    res["default_diabetes_iterative_sparse"] = None

    sk_mod = sklearn.neighbors.KNeighborsRegressor

    module = KNearestNeighborsRegressor